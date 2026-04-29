---
id: 02-gpu-memory-sizing
title: "2 · GPU Memory Sizing: The Hard Constraint"
---

GPU VRAM is not a soft limit you can negotiate around with clever configuration. It is a binary constraint. The system either fits or it does not start. No batching strategy, no parallelism trick, no serving framework feature helps you if the total memory footprint exceeds available VRAM. This makes memory sizing the first engineering calculation, not the last.

When an LLM inference engine starts up, it must allocate four distinct categories of GPU memory before it can serve a single request. These four categories have completely different characteristics — some are fixed at load time, some grow dynamically with traffic, some are model-determined, some are workload-determined — and conflating them is the root cause of most production memory miscalculations.

**1. [Model weights](https://www.ibm.com/think/topics/llm-parameters)** are the parameters loaded from disk. This is the number everyone calculates. It is fixed the moment the model loads and does not change regardless of traffic.

**2. [KV cache](https://www.linkedin.com/pulse/kv-cache-hidden-optimization-behind-real-time-ai-vinay-jayanna-cvfec/)** is the stored key and value tensors from the attention mechanism, maintained for every token in every active request. Unlike weights, KV cache is not a property of the model — it grows dynamically at runtime as a function of sequence length, concurrency, and the number of attention layers. At high concurrency with long context, KV cache routinely exceeds model weight memory by a significant margin.

**3. Activations** are the intermediate tensors computed during the forward pass. They are not stored across requests like KV cache; they exist only during the computation of a single forward pass. At large batch sizes they accumulate across all layers simultaneously — typically 15-30 GB for a 70B model at production concurrency.

**4. Framework overhead** is everything the serving runtime consumes that has nothing to do with the model or the requests. CUDA runtime, driver context, internal memory pools, pre-allocated buffers, memory fragmentation. In practice it runs 10-20 GB for mature frameworks on H100s.

These four components sum to your total VRAM requirement:

:::info
`Total VRAM = Model Weights + KV Cache + Activations + Framework Overhead`
:::

The formula is simple. The difficulty is that three of the four terms are functions of runtime variables — concurrency, sequence length, batch size — not static model properties. You cannot calculate this from a model card. You must calculate it from your workload characterization.

To make this concrete, consider Llama-3 70B at 50 concurrent requests with 8K context:

![Figure 2.1 — GPU memory breakdown: FP16 vs FP8 for Llama-3 70B](/img/figures/fig-2-1-gpu-memory-breakdown-fp16-fp8.png)

<figcaption>

**Figure 2.1 — GPU memory breakdown: FP16 vs FP8 for Llama-3 70B (50 concurrent requests, 8K context)**

*Quantization from FP16 to FP8 halves both model weights and KV cache, reducing total memory from 287 GB to 165 GB. Activations and framework overhead are unaffected by weight precision. The GPU count drops from 4-6 H100s to 2-3 — the single highest-leverage memory decision available before touching throughput.*

</figcaption>

---

## 2.1 Model Weights: The Floor

Model weight memory is determined by parameter count and numerical precision:

:::info
`Model memory (bytes) = num_parameters × bytes_per_dtype`
:::

| Precision | Bytes per parameter |
|-----------|-------------------|
| FP32 | 4 |
| FP16 / BF16 | 2 |
| INT8 | 1 |
| INT4 | 0.5 |

In practice: Llama-3 8B in FP16 needs 16 GB. Llama-3 70B in FP16 needs 140 GB. Llama-3 70B in INT4 needs 35 GB — the difference between requiring a full DGX node and fitting on a single GPU with room to spare.

**Five properties of model weight memory that production engineers learn the hard way:**

**The alignment tax is real.** GPU memory allocators pad each tensor to align to memory boundaries. In practice, resident weight memory runs 3-8% above what the formula predicts. On a 140 GB model that is 4-11 GB of memory you never explicitly allocated.

**Weight memory is a loading event, not just a budget line.** The memory has to travel from storage to GPU before the first request can be served.

| Storage path | Effective bandwidth | 70B FP8 (70 GB) cold start | 70B FP16 (140 GB) cold start |
|---|---|---|---|
| Local NVMe (PCIe Gen4) | ~12.5 GB/s | ~6 seconds | ~11 seconds |
| Network-attached (10 GbE) | ~1.1 GB/s | ~64 seconds | ~2 minutes |
| Object storage (S3/GCS) | ~0.5-1.0 GB/s effective | 2-5 minutes | 4-10 minutes |

Scale-from-zero is almost never the right pattern for LLM fleets. Unlike stateless web services, LLM fleets pay a 2-10 minute cold start penalty on every scale-up event from zero — a penalty that makes reactive autoscaling operationally untenable for any workload with predictable traffic.

**Precision mismatches create transient spikes.** Loading FP16 weights and casting to BF16 at runtime briefly holds both representations in memory simultaneously. A configuration that fits at steady state can OOM before it ever serves a token.

**LoRA fundamentally changes the multi-tenant weight equation.** Base weights are loaded once and shared across every adapter variant. Each LoRA adapter adds only its delta — typically 1-5% of base model size. Serving ten fine-tuned variants of Llama-3 70B costs approximately the same weight memory as serving one.

**Serialization format affects load behavior, not resident size.** Safetensors supports memory-mapped loading — the OS pages in only what is needed, and the process never holds a full second copy during deserialization. Pickle-based formats require a full deserialization pass that temporarily allocates a second buffer alongside the model being loaded.

---

## 2.2 KV Cache: Where Production Systems Run Out of Memory

This is the component most teams underestimate, often catastrophically. KV cache does not appear in model cards. It is not a static property of the model. It grows dynamically at runtime with every token in every active request.

:::info
`KV_cache (bytes) = 2 × num_layers × num_KV_heads × head_dim × seq_len × concurrency × bytes_per_element`

The factor of 2 is for both the key and value vectors stored per layer.
:::

A concrete example: Llama-3 70B in FP16 has 80 layers, 8 KV heads, and a head dimension of 128 — working out to approximately `0.26 MB` of KV cache per token per request.

At 50 concurrent requests with 8K-token inputs: `50 × 8,000 × 0.26 MB = 104 GB` of KV cache alongside 140 GB of model weights. Already more than 70% of the model's own footprint.

At 32K tokens the KV cache grows to 416 GB at the same concurrency. At 128K context, just 4 concurrent requests push KV cache past model weights entirely.

![Figure 2.2 — KV cache memory vs concurrency for Llama-3 70B (FP16)](/img/figures/fig-2-2-kv-cache-memory-vs-concurrency.png)

<figcaption>

**Figure 2.2 — KV cache memory vs concurrency for Llama-3 70B (FP16)**

*Each line represents a different context length. KV cache grows linearly with concurrency — but the slope scales directly with context length, making long-context workloads disproportionately memory-intensive. At 128K context, model weights are exceeded at just 4 concurrent requests. At 32K context, the crossover happens at 16. Only short-context workloads (4K-8K) stay below model weight memory at typical production concurrency. Size for your P95 context length, not your average.*

</figcaption>

**MHA vs GQA — why KV head count matters more than attention head count.**

[Multi-Head Attention (MHA)](https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/) is the standard: every query head has its own dedicated KV head. A model with 64 attention heads stores 64 KV head pairs per token.

[Grouped Query Attention (GQA)](https://www.ibm.com/think/topics/grouped-query-attention) allows multiple query heads to share the same Key and Value vectors. [Llama-3 70B has 64 query heads but only 8 KV heads](https://arxiv.org/abs/2407.21783) — an 8× reduction in KV cache size relative to a standard MHA model of the same scale. Modern architectures including Llama-3, Mistral, Qwen, and Gemma all use GQA for exactly this reason.

:::warning
When estimating KV cache memory, always check the KV head count explicitly — not the total attention head count. They are often not the same number, and using the wrong one will cause you to overestimate KV cache memory significantly.
:::

---

## 2.3 Activations: The Necessary but Manageable Component

Every forward pass produces intermediate tensors that must stay resident in memory until the next operation consumes them. They exist only for the duration of a single forward pass — but at large batch sizes they accumulate across all layers simultaneously.

The historical problem was severe. Standard attention produces an N×N attention matrix — at 8K tokens that is 64 million entries, at 32K tokens over a billion. Memory scaled as `O(seq²)`.

[**FlashAttention**](https://github.com/dao-ailab/flash-attention) eliminates the quadratic scaling. The insight: you never need the full N×N matrix in memory at once — you need the result of multiplying it by the value vectors. FlashAttention computes this incrementally in blocks that fit in the GPU's on-chip SRAM. The full N×N matrix is never materialized. Activation scaling drops from `O(seq²)` to approximately `O(seq)`.

![Figure 2.3 — Standard attention vs FlashAttention memory model](/img/figures/fig-2-3-standard-vs-flashattention.png)

<figcaption>

**Figure 2.3 — Standard attention vs FlashAttention memory model**

*Standard attention materializes the full N×N score matrix in HBM before softmax can run — memory grows quadratically with sequence length. At 32K tokens that is roughly 4 billion matrix entries. FlashAttention eliminates this by computing attention in tiles that fit entirely in on-chip SRAM, maintaining only two running scalars per row. Memory drops from O(N²) to O(N). This is what makes 32K and 128K context workloads physically viable on current hardware.*

</figcaption>

FlashAttention is now standard across all major inference frameworks — vLLM, TensorRT-LLM, and Triton all use it by default.

:::note
Budget approximately **20% of model weight memory** for activations under FlashAttention. For Llama-3 70B that is roughly 28 GB — significant, but predictable and stable across context lengths in a way that KV cache is not.
:::

---

## 2.4 Framework Overhead: The Remainder You Cannot Ignore

Every inference framework is a runtime system that manages memory pools, request queues, block tables, batching state, and CUDA contexts. All of that infrastructure lives in GPU memory — before your first request arrives.

What drives the number varies by framework. vLLM pre-allocates block tables for PagedAttention proportional to your configured memory pool size. TensorRT-LLM pre-allocates engine execution buffers at build time, sized to your declared maximum batch size and sequence length. SGLang maintains a radix cache structure for prefix reuse.

:::note
Budget **5-10% of total VRAM** for framework overhead. On an 80 GB H100 that is 4-8 GB structurally unavailable regardless of what the model card says.
:::

---

## 2.5 Multi-LoRA Serving: Memory Implications

Most enterprise deployments serve multiple fine-tuned variants of the same base model. Multi-LoRA serving shares base model weights across all tenants while loading only lightweight adapter weights per request.

The memory formula:

:::info
`Total memory = base_model_memory + (max_simultaneous_adapters × adapter_size) + KV_cache + activations`
:::

A LoRA adapter for a 70B model with `rank r=16` typically adds 50-200 MB per adapter. The rank controls expressive capacity — `r=8` to `r=64` covers most production use cases. Crucially, only adapters for currently active requests need to reside in GPU memory — inactive adapters can be paged out to CPU memory.

![Figure 2.5 — Multi-LoRA serving memory architecture](/img/figures/fig-2-5-multi-lora-serving-memory.png)

<figcaption>

**Figure 2.5 — Multi-LoRA serving memory architecture**

*The base model loads once and is shared across all tenants. Only adapters for currently active requests reside in GPU memory — inactive adapters are paged to CPU DRAM and loaded on demand via LRU eviction. The binding constraint is not the total number of adapters but the maximum number simultaneously active in the batch. The failure mode: bursty traffic causes all adapters to load concurrently, exhausting GPU memory — not from the base model or KV cache, but from adapter accumulation.*

</figcaption>

:::warning
**The failure mode teams hit in production.** A deployment serving 100 LoRA adapters assumes adapter memory is negligible because each adapter is small. Under bursty traffic, requests for many distinct adapters arrive simultaneously, all 100 adapters get loaded concurrently, and the deployment OOMs — not because of the base model or KV cache, but because of adapter accumulation. Fix: cap `max_simultaneous_adapters` and implement LRU eviction.
:::

[vLLM supports multi-LoRA serving](https://docs.vllm.ai/en/latest/features/lora.html) natively with configurable `max_loras` and `max_lora_rank`. [SGLang](https://docs.sglang.io/advanced_features/lora.html) supports multi-LoRA with similar controls.

---

## 2.6 Putting It Together: The Minimum GPU Count Formula

:::info
`Min GPUs = ceil((model_weights + KV_cache_at_peak_concurrency + activations + overhead + (max_simultaneous_adapters × adapter_size)) / GPU_VRAM)`
:::

The KV cache term is the variable that changes everything. Two teams running the same model on the same GPU will get completely different minimum GPU counts if their concurrency and context length assumptions differ. This is why workload characterization in Section 1 comes before GPU selection, not after.

**Worked example:** Llama-3 70B in FP16, 50 concurrent requests at P95 input length of 8K tokens, H100 80 GB SXM.

| Component | Memory |
|---|---|
| Model weights (FP16) | 140 GB |
| KV cache (`50 × 8K × 0.26 MB`) | 104 GB |
| Activations (~20% of weights) | 28 GB |
| Framework overhead | 15 GB |
| **Total** | **287 GB** |
| **Minimum GPUs** | **`ceil(287/80) = 4 H100s`** |

Now apply FP8 quantization:

| Component | FP16 | FP8 |
|---|---|---|
| Model weights | 140 GB | 70 GB |
| KV cache | 104 GB | 52 GB |
| Activations | 28 GB | 28 GB |
| Framework overhead | 15 GB | 15 GB |
| **Total** | **287 GB** | **165 GB** |
| **Minimum GPUs** | **4** | **`ceil(165/80) = 3`** |

FP8 is a GPU count decision, not a quality tuning decision.

:::warning
**On headroom.** The minimum GPU count is not your deployment GPU count. A system running at 95% VRAM utilization is one traffic burst away from an OOM cascade. Add **30-50% headroom** beyond the calculated minimum. For the FP16 example above: 4 GPUs minimum → 6 GPUs deployed. For FP8: 3 GPUs minimum → 4 GPUs deployed.
:::
