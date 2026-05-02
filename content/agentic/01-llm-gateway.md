---
id: 01-llm-gateway
title: "LLM Gateway and Multi-Provider Routing"
---

# LLM Gateway and Multi-Provider Routing

<div class="chapter-summary">
Scaling agentic systems across an organization surfaces a class of failures that have nothing to do with retrieval quality or agent loop design. Many of them originate at the entry point - the layer every request passes through before it touches a model, a vector index, or a tool. At team scale this layer is often invisible: a single provider, a single API key, traffic flowing through without ceremony. At enterprise scale it becomes the foundation everything else depends on. Get it wrong and the failures propagate downstream into every system that depends on it.
</div>

---

## What the gateway actually is

The LLM gateway is the control plane between your application and every model provider you depend on. It is not a proxy. A proxy forwards requests. A gateway governs them - routing intelligently, enforcing cost budgets, caching responses, shaping requests before they hit the model, and giving you unified observability across every provider, model version, and tenant in your system.

Think of it as the database connection pool problem, replayed for LLMs. Every service talking directly to an external provider, no abstraction layer, no visibility, no fallback. The database connection pool solved this for databases a decade ago. The LLM gateway is the same pattern. The difference is that LLM provider calls are orders of magnitude more expensive per request, more variable in latency, and more exposed to provider-level failures than database connections ever were.

The LLM gateway has moved from an optional convenience layer to load-bearing infrastructure. As enterprise AI deployments scale past tens of millions of daily requests, the absence of a proper control plane between applications and providers is the single most common source of unexpected cost spikes, provider-caused outages, and unattributable quality regressions.

A production gateway owns more than routing. At enterprise scale it becomes the control plane for every concern that cuts across your entire AI stack:

| Concern              | What it means at scale                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Routing**          | Which model, provider, and region handles each request - and why                                                               |
| **Caching**          | Semantic and prefix caching to eliminate redundant inference calls entirely                                                    |
| **Request shaping**  | Token budget enforcement, context compression, prompt optimization before the request leaves your infrastructure               |
| **Rate limiting**    | Absorbing provider rate limits without cascading failures into the application layer                                           |
| **Backpressure**     | Signaling upstream callers to slow down when the system is under stress - shedding low-priority traffic before queues overflow |
| **Cost attribution** | Per-tenant, per-model, per-feature spend tracking at request granularity                                                       |
| **Resilience**       | Fallback chains, circuit breakers, and retry logic that behaves correctly inside an agentic loop                               |
| **Observability**    | Unified tracing, latency distributions, and cost dashboards across every provider                                              |
| **Model versioning** | Version pinning, shadow testing, and canary deployments against live traffic                                                   |

We will go through them in order.

---

## Routing: from passthrough to intelligence

The simplest gateway routes all requests to a single model at a single provider. This works until the provider has an outage, deprecates the model, raises prices, or your request volume exceeds their rate limits. At enterprise scale, any of these events is a matter of when, not if.

### Latency-based routing

The most immediate routing improvement is distributing requests across multiple providers or model instances based on measured latency. The gateway maintains a real-time latency table - P50, P95, P99 per provider per model - and routes new requests to the currently fastest option.

If Provider A has a current P95 of 1.2s and Provider B has a P95 of 0.8s for equivalent models, and your SLO is 1.0s P95, routing to Provider B keeps you in SLO while Provider A does not. The routing decision is made per request based on the rolling window, not a static configuration.

The gateway needs a running sense of each provider's current latency - not a static average computed at startup, and not the raw last-observed value which overreacts to a single slow request.

The approach used in production: weight recent observations more heavily than older ones, so the estimate gradually tracks real changes without being thrown off by noise. A provider that was fast all day but had one slow request five minutes ago should still be considered fast. A provider that has been consistently slow for the last ten requests should be considered slow.

The standard implementation is an exponentially-weighted moving average:

```
latency_estimate = w × latest_response_time + (1 - w) × previous_estimate
```

`latest_response_time` is what you just measured. `previous_estimate` is what you believed before this request. `w` is the weight you give to new evidence - a value between 0 and 1. At `w = 0.1`, new observations count for 10% of the updated estimate and history counts for 90% - stable, slow to react. At `w = 0.3`, new observations count for 30% - faster to catch genuine degradations but more sensitive to transient spikes. Start at 0.1 and tune upward if you find that real provider degradations persist long enough that your estimate is still pointing traffic at a struggling provider.

### Cost-based routing

Output tokens cost 3-5x more than input tokens across most providers. This asymmetry matters for routing decisions. A request that generates a long response costs significantly more than one that generates a short response, independent of the input length. Cost-based routing makes the provider selection decision based on the expected total cost of the request, not just the input token count.

The cost model per request:

```
cost(request) = (input_tokens × input_price_per_M / 1_000_000)
              + (estimated_output_tokens × output_price_per_M / 1_000_000)
```

Estimating output tokens before generation is imprecise - you can use historical averages by request type, or use a fast classifier to bucket requests into short/medium/long output categories and apply category-level averages. The estimate doesn't need to be perfect; it needs to be good enough to make the routing decision directionally correct.

[LLM API spending doubled](https://www.pluralsight.com/resources/blog/ai-and-data/how-cut-llm-costs-with-metering) from $3.5 billion to $8.4 billion between late 2024 and mid-2025, and 72% of organizations plan to increase their AI budgets further in 2026. Yet most teams have no systematic strategy to control those costs. Cost-based routing is that strategy at the infrastructure layer.

### Complexity-based routing (model selection)

The highest-leverage routing decision is not which provider to use - it is which model to use. Without routing, cost scales with volume. With routing, cost scales with complexity. This distinction determines whether your token spend grows predictably or accelerates uncontrollably.

The pattern: a lightweight classifier at the gateway evaluates each incoming request and buckets it into a complexity tier. Simple requests route to a small, fast, cheap model. Complex requests - multi-step reasoning, long-context synthesis, ambiguous queries - route to a frontier model.

A three-tier model hierarchy works well in practice:

| Tier | Model class | Use case | Typical cost ratio |
|------|-------------|----------|--------------------|
| Fast | 7B-13B parameter models (Llama 3 8B, Mistral 7B, Gemma) | Simple Q&A, classification, short summaries | 1× |
| Mid | Mid-tier frontier (GPT-4o mini, Claude Haiku, Gemini Flash) | Structured extraction, moderate reasoning | 5-10× |
| Frontier | Full frontier (GPT-4o, Claude Sonnet/Opus, Gemini Pro) | Complex reasoning, long-context synthesis, code generation | 20-50× |

The classifier itself should be fast and cheap - a fine-tuned small model, a heuristic based on query length and keyword signals, or a simple embedding similarity approach. It does not need to be perfect. Routing 80% of requests correctly to the right tier delivers most of the cost benefit.

At enterprise scale with millions of daily requests, the math compounds significantly. If 60% of your requests are genuinely simple and you route them to a Tier 1 model at 1/20th the cost of your frontier model, you have reduced your total inference spend by roughly 57% on that traffic without any quality degradation on the simple requests.

```
Baseline cost (all frontier): 1,000,000 requests × $0.01 = $10,000/day
With routing (60% to Tier 1, 40% frontier):
  Tier 1: 600,000 × $0.0005 = $300
  Frontier: 400,000 × $0.01  = $4,000
  Total: $4,300/day
  Savings: 57%
```

Reducing LLM cost and latency by 40-70% is achievable without sacrificing output quality, provided the optimization happens at the right layer in your stack.

---

## Semantic caching: eliminating redundant inference entirely

Routing optimizes which model handles a request. Semantic caching eliminates the model call  for requests that are meaningfully similar to something you have already answered.

### Why exact-match caching fails for LLMs

Traditional caches hash the input string and match on byte equality. "How do I reset my password?" and "What's the process to change my password?" produce different hashes and different cache keys, even though the correct answer is identical. The gateway dispatches two full LLM calls at full cost. At enterprise scale, this inefficiency is significant. [Studies](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73) show that a significant share of LLM queries - often 30-50% depending on the workload - are semantically identical to previous requests, just phrased differently. 

### Semantic caching architecture

The mechanics are straightforward: when a request arrives, the gateway embeds the query into a vector and searches the cache for a semantically similar previous query. If the similarity score clears a threshold, the cached response is returned immediately - no model call, no token spend, near-zero latency. If it doesn't, the request routes to the model and the response is stored for future hits.

![Semantic cache flow - hit and miss paths](/img/agentic/fig-1-1-semantic-cache-flow.svg)

<figcaption>

**Figure 1-1: Semantic Cache Flow**
*A query that clears the similarity threshold returns a cached response with near-zero latency and zero token cost. A cache miss routes to the model, generates a response, and stores it asynchronously for future hits.*

</figcaption>

The threshold is the critical engineering parameter and it deserves careful tuning. Set it too high - 0.99 or above - and you only catch near-exact duplicates, capturing a fraction of the potential savings. Set it too low - 0.80 - and you start returning cached responses to queries that are similar in wording but different in intent, which is a quality failure that is hard to detect and harder to explain to users. The right threshold depends on your domain. A customer support bot where most queries cluster around a handful of intents can run a lower threshold safely. A general-purpose enterprise assistant with diverse, open-ended queries needs a higher one. Start at 0.92-0.95 and tune based on measured cache hit rates and whether you see quality regressions in evaluation.

The economics at scale make this worth the engineering investment. Take a mid-size enterprise deployment: 1 million daily requests, average model cost of $0.01 per call (roughly Claude Haiku or GPT-4o mini at moderate context length), and a 30% semantic cache hit rate - which is conservative given production data showing 30-50% of enterprise queries are semantically redundant.

```
Baseline (no cache):  1,000,000 × $0.01       = $10,000/day

With semantic cache:
  Cache misses:         700,000 × $0.01        =  $7,000/day
  Cache hits (model):   300,000 × $0.00        =      $0/day
  Cache lookup cost:  1,000,000 × $0.00002     =     $20/day
                                                 ───────────
  Total:                                        =  $7,020/day

Daily saving:  $2,980  →  ~$1.09M/year
```

The lookup cost - embedding the incoming query plus a vector similarity search - is $20/day across all 1 million requests, including the cache hits that never reach the model. It is negligible against the $3,000/day in model calls it eliminates. The net saving is essentially the full cost of every cache hit, minus a rounding error.

### Prefix caching

Semantic caching matches queries by meaning. Prefix caching operates at a completely different layer - it matches the _shared structure_ of prompts rather than their semantic content.

Every request in a production agentic system carries a significant amount of fixed content: the system prompt, tool definitions, few-shot examples, organizational context, and safety instructions. For a typical enterprise deployment this fixed prefix runs 2,000 to 8,000 tokens - sometimes more. Without prefix caching, every single request pays the full input token cost for this fixed content, even though it is identical across millions of requests. That is not a retrieval problem or a generation problem. It is a structural inefficiency that compounds directly with request volume.

The distinction that matters for gateway design:

**What the inference engine owns** - the actual KV cache reuse mechanics. When the inference engine sees a new request whose prompt starts with bytes it has seen before, it can reuse the computed key-value tensors for that prefix instead of recomputing them. This eliminates the compute cost for the prefix tokens entirely. vLLM, TensorRT-LLM, and provider-side implementations (Anthropic, OpenAI) all handle this at the serving layer. The mechanics - block allocation, memory management, eviction policies - are covered in depth in [Sizing LLM Inference for Production](https://vinayj.com/sizing/07-kv-cache-optimization).

**What the gateway owns** - the conditions that make prefix cache hits possible in the first place. The inference engine can only reuse a cached prefix if the bytes are identical across requests. If one request puts the system prompt first and tool definitions second, and another request reverses that order, the prefix doesn't match - even though the content is semantically identical. The gateway enforces prompt structure consistency: system prompt always first, tool definitions always in the same order, organizational context always in the same position. This is a configuration and enforcement problem, not a caching problem, and it belongs at the gateway layer.

The gateway also decides how to partition each prompt into cacheable and non-cacheable segments. A well-structured partition:

```
[CACHEABLE PREFIX - fixed across all requests]
  System prompt
  Tool definitions
  Organizational context
  Few-shot examples

[NON-CACHEABLE SUFFIX - varies per request]
  Conversation history
  Retrieved context chunks
  Current user query
```

The boundary between these segments is a gateway configuration decision. Push too much into the non-cacheable suffix and you lose cache coverage. Push retrieved context into the cacheable prefix and you create cache key explosion - a unique cache entry per unique retrieval result, which defeats the purpose entirely.

The economics at provider level are significant. Anthropic's prompt caching costs $0.30 per million tokens on cache reads versus $3.00 per million on fresh input - a 90% cost reduction on the cached portion. For a system prompt and tool definitions totalling 4,000 tokens, running 1 million requests per day:

```
Without prefix caching:
  1,000,000 × 4,000 tokens × $3.00/M  =  $12,000/day

With prefix caching (gateway enforces consistent structure):
  1,000,000 × 4,000 tokens × $0.30/M  =   $1,200/day

Saving:  $10,800/day  →  ~$3.9M/year
```

This saving requires zero changes to application logic. It requires one thing from the gateway: consistent, deterministic prompt structure on every request.

:::info Prefix caching vs. semantic caching 
These two caching layers are complementary, not alternatives. Prefix caching eliminates the cost of fixed prompt structure that repeats across all requests. Semantic caching eliminates the cost of semantically redundant user queries. A production gateway deploys both - prefix caching handles the structural redundancy, semantic caching handles the query redundancy. 
:::

---

## Request shaping: optimizing before the model sees it

The gateway is the right place to enforce constraints on what reaches the model - not inside application code where enforcement is inconsistent across teams, not inside prompts where it's invisible to infrastructure, but at the layer where every request passes regardless of which team, product, or feature generated it.

#### Token budget enforcement
Every request has a token budget: input tokens consumed plus maximum output tokens authorized. At enterprise scale, unbounded requests are simultaneously a cost liability, a latency liability, and a fairness problem - a single runaway request consuming a large context window blocks capacity from other tenants in the same serving pool.

The gateway enforces hard limits on both dimensions before the request reaches the model. The engineering decisions are in the enforcement strategy, not the limit itself.

**Input token enforcement** has three approaches with meaningfully different quality implications:

_Head truncation_ - drop tokens from the beginning of the input. Preserves recency, which matters for conversational context, but destroys the system prompt and early instructions that establish the model's behavior. Almost never the right choice.

_Tail truncation_ - drop tokens from the end. Preserves the system prompt and instructions but loses the most recent retrieved context and conversational turns. Slightly better than head truncation for RAG workloads but still crude.

_Relevance-ranked truncation_ - the gateway uses retrieval scores attached to each context chunk as metadata to drop the lowest-relevance chunks first. The system prompt, tool definitions, and high-relevance retrieved chunks are preserved. Low-relevance chunks are dropped in ascending relevance order until the budget is satisfied. This requires the retrieval pipeline to pass relevance scores to the gateway alongside the retrieved text - a contract between retrieval and gateway that needs to be designed explicitly.

The relevance-ranked approach consistently outperforms naive truncation on answer quality within the same token budget. The cost is architectural: the gateway needs to parse structured input from the retrieval layer, not treat the assembled prompt as an opaque string.

**Output token enforcement** is simpler but has a subtle interaction with agentic systems. Capping `max_output_tokens` at the gateway prevents runaway generation on a single request. But in a multi-turn agentic loop, each turn's output becomes the next turn's input. A hard per-turn output cap that is too aggressive causes the model to truncate its reasoning mid-thought, producing incomplete tool call specifications or malformed JSON that causes the next step to fail. The per-turn output budget and the total loop token budget are separate constraints - both need to be enforced at the gateway layer, and the total loop budget needs to track cumulative token consumption across all turns, not just the current one.

#### Context compression
Retrieved context in RAG workloads is rarely uniformly dense. A typical retrieval pass returns chunks that range from highly relevant to marginally relevant, and even the relevant chunks often contain surrounding prose that adds length without adding signal. Compressing this before it reaches the model reduces input token cost without degrading the information the model actually needs.

**Extractive compression** identifies and retains the highest-signal sentences within each chunk, discarding surrounding context. Fast, deterministic, and predictable in its behavior. [LLMLingua](https://github.com/microsoft/LLMLingua) and similar token-level compression approaches operate on this principle and can reduce context length by 2-4× with minimal quality degradation on most RAG workloads.

**The latency tradeoff is real and needs to be measured, not assumed.** A compression pass that adds 80ms of latency to a request with a 1,200ms model call time is a 6.7% latency increase - almost always worth the 2-4× token reduction. The same 80ms compression pass on a request with a 200ms model call time is a 40% latency increase - almost never worth it unless your input token costs are unusually high. The gateway needs per-request-class latency budgets to make this decision correctly, not a global on/off configuration.

**Deduplication before compression** catches a common RAG failure mode: multiple retrieved chunks from the same source document containing near-identical content. Embedding-based deduplication at the gateway - before the compression pass - removes redundant chunks that would otherwise consume budget without adding signal. A cosine similarity check between retrieved chunks at a threshold of 0.97+ catches near-duplicates with negligible overhead.

The compression pipeline in order:
![Context compression pipeline|333](/img/agentic/fig-1-2-context-compression-pipeline.svg)

<figcaption>

**Figure 1-2: Context Compression Pipeline**
*When retrieved chunks fit within the token budget, they pass through ranked by relevance score. When they exceed it, the gateway applies extractive compression to low-relevance chunks first, then drops the lowest-scoring chunks until the budget is satisfied - preserving the highest-signal content at every step.*

</figcaption>
#### Prompt standardization and schema enforcement

At team scale, prompt quality is a craft problem - each team writes prompts that work for their use case. At enterprise scale, it becomes an infrastructure problem. Without gateway-level enforcement, you get: system prompts that vary from 200 to 8,000 tokens across teams, missing safety instructions on some request paths, tool definitions formatted inconsistently causing parse failures, and no visibility into what instructions are actually reaching the model in production.

The gateway enforces a prompt schema - a contract that every request must satisfy before it reaches the model:

json

```json
{
  "required_fields": ["system_prompt", "user_message"],
  "system_prompt": {
    "must_include": ["safety_instructions", "output_format"],
    "max_tokens": 2000
  },
  "tool_definitions": {
    "format": "openai_tools_schema",
    "max_tools": 20
  }
}
```

Schema validation at the gateway catches misconfigured requests before they reach the model - a malformed tool definition that would cause a silent tool call failure, a missing safety instruction that bypasses organizational policy, a system prompt that exceeds the budget allocation for its tier.

Beyond validation, the gateway applies organization-wide template injection - prepending standard safety instructions, compliance notices, or behavioral guidelines to every system prompt regardless of what the calling application provides. This is the only reliable way to ensure that organizational requirements are enforced consistently across every team, every product, and every model version - because application code can be misconfigured, but the gateway layer cannot be bypassed.

---

## Resilience: fallback, circuit breakers, and 429 handling

A resilient gateway doesn't just survive provider failures - it makes them invisible to the systems that depend on it. The difference between a gateway that returns errors when Anthropic has an outage and one that silently routes to OpenAI is entirely in how the resilience layer is engineered.

#### The fallback chain

A fallback chain is an ordered sequence of alternatives the gateway traverses when a provider fails. Designing one correctly requires thinking through three constraints simultaneously.

**Capability equivalence** is the first constraint. Not all models are interchangeable. A fallback that routes a 128K-context synthesis request to a model with a 32K context window will truncate silently and produce wrong answers. The fallback chain is not just a list of providers - it is a list of providers that can handle the specific request type. This means the gateway needs to understand request classification well enough to select fallbacks that are genuinely equivalent, not just available.

**Latency budget** is the second. Switching providers adds latency - the time to detect the failure, select the fallback, and establish a new connection. A gateway that waits 30 seconds for a primary provider to timeout before switching to a fallback has effectively failed its SLO regardless of the fallback's speed. Timeout thresholds on the primary path need to be set aggressively enough that the fallback has room to succeed within the overall SLO.

**Cost attribution** is the third. The fallback provider may be more expensive than the primary. At enterprise scale, fallback events need to be logged with provider, model, reason, and cost delta - both for financial attribution and for identifying patterns. If a specific request class is falling back consistently, that is a signal to adjust the primary routing, not just absorb the cost.

A production fallback chain for a mid-tier routing tier:

```
Primary:   Claude Sonnet (Anthropic)
  → on timeout/5xx:        GPT-4o mini (OpenAI)
    → on timeout/5xx:      Gemini Flash (Google)
      → on all providers exhausted: return graceful degradation response
```

The chain terminates at a graceful degradation response - not an error. At enterprise scale, returning a 503 to a user whose request traversed three providers and failed all of them is a worse outcome than returning a pre-computed response that covers the most common failure cases. The degradation response is part of the resilience design, not an afterthought.

#### 429 handling in agentic loops

Rate limit handling in agentic systems is fundamentally different from standard API client retry logic, and treating it the same way is one of the most common sources of runaway cost at scale.

The standard pattern - exponential backoff with jitter - is correct as a foundation:

python

```python
def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    # Exponential base delay - doubles each attempt, capped at 60s
    delay = min(cap, base * (2 ** attempt))
    
    # Equal jitter: guaranteed minimum of delay/2, random spread up to delay
    # Prevents thundering herd without allowing near-zero retries at high attempts
    # Preferred over full jitter for agentic loops where some minimum wait is needed
    jitter = delay / 2
    return jitter + random.uniform(0, jitter)

# Actual retry timing (range due to jitter):
# attempt 0: 0.5s  – 1.0s   (base 1s)
# attempt 1: 1.0s  – 2.0s   (base 2s)
# attempt 2: 2.0s  – 4.0s   (base 4s)
# attempt 3: 4.0s  – 8.0s   (base 8s)
# attempt 4: 8.0s  – 16.0s  (base 16s)
# attempt 5: 16.0s – 32.0s  (base 32s, approaching cap)
#
# Equal jitter ensures retries spread across the window - concurrent agent loops
# retrying attempt 3 will fire somewhere between 4-8s, not all at exactly 8s.
# Full jitter (0 to cap) risks near-zero delays at high attempt numbers in agentic
# loops where accumulated context makes each retry progressively more expensive.
```

The jitter term prevents thundering herd - multiple concurrent agent loops hitting the same 429 and retrying at exactly the same moment, generating a second collision. Without jitter, exponential backoff at scale produces synchronized retry waves that are almost as damaging as the original burst.

But in agentic contexts, two behaviors beyond standard backoff are required:

**Provider switching on 429, not same-provider retry.** Rate limits are time-windowed - per minute, per hour, per day depending on the provider and tier. Retrying the same provider on a 429 burns the backoff time waiting for a window that may not reset for minutes. The correct behavior is to treat a 429 as a routing signal - immediately move to the next provider in the fallback chain rather than waiting. The primary provider recovers on its own timeline; you route around it.

**Context accumulation tracking across retries.** This is the failure mode that catches most teams off guard. In a multi-turn agentic loop, each retry carries the full accumulated context of all previous turns - every tool call result, every intermediate reasoning step, every previous exchange. A request that was 2,000 tokens on attempt 1 may be 5,000 tokens on attempt 3 as the loop accumulates state. The gateway must track token count growth across retries and enforce the budget ceiling against the running total, not the initial request size. A retry that would push the total loop token spend above budget should be rejected at the gateway before it reaches the provider - not after the provider processes it and charges for it.

```python
class AgentLoopRetryPolicy:
    def should_retry(self, attempt: int, error: ProviderError,
                     cumulative_tokens: int, token_budget: int) -> tuple[bool, bool]:
        """Returns (should_retry, switch_provider)."""

        # Hard stop - token budget consumed across all turns, not just this one.
        # Context grows with each retry; enforce the ceiling on the running total.
        if cumulative_tokens >= token_budget:
            return False, False

        # 429: quota window not reset yet - switch provider immediately,
        # never wait on the same provider that just rejected you.
        if isinstance(error, RateLimitError):
            return attempt < self.max_attempts, True   # switch=True

        # Timeout: route around, same logic as rate limit.
        if isinstance(error, TimeoutError):
            return attempt < self.max_attempts, True   # switch=True

        # 5xx: transient server error, retry same provider.
        # 4xx: client error, retrying won't fix it.
        if isinstance(error, ServerError):
            return (attempt < self.max_attempts and
                    error.status_code in {500, 502, 503, 504}), False

        return False, False
```

#### Circuit breakers

A circuit breaker prevents a degraded provider from receiving traffic until it recovers - the same pattern used in microservices, applied to LLM provider connections. The state machine has three states:

**CLOSED** - normal operation, requests flow through, failure rate is monitored over a sliding window.

**OPEN** - failure rate exceeded the threshold, requests are rejected immediately without attempting the provider. The gateway routes to fallbacks. The circuit stays open for a configured timeout period.

**HALF-OPEN** - after the timeout, one test request is allowed through. If it succeeds, the circuit closes and normal routing resumes. If it fails, the circuit reopens and the timeout resets.

The engineering decisions that matter in production:

**What counts as a failure?** 5xx errors clearly do. Timeouts should too - a provider that takes 45 seconds to return an error is operationally equivalent to one that returns a 503. Rate limit responses (429) are different - they indicate the provider is healthy but your quota is exhausted, so they should trigger provider switching, not circuit breaking.

**What window size?** A 60-second sliding window balances responsiveness - catching genuine degradation quickly - against noise from transient spikes. A window that's too short opens the circuit on a single bad request. A window that's too long lets sustained degradation pass through for too long.

**What failure threshold?** 50% failure rate over the window is a conservative starting point. For providers where any failure is unusual (sub-1% baseline), a lower threshold - 20-25% - is more appropriate. Calibrate against your provider's historical baseline error rates.

**What granularity?** Circuit breakers per provider are table stakes. The real value at enterprise scale is per-provider per-model per-region granularity. An Anthropic outage in us-east-1 should not close the circuit for Anthropic in eu-west-1. A specific model version being degraded should not cut off other models from the same provider. Coarse-grained circuit breakers that treat an entire provider as a single entity will route around healthy capacity unnecessarily.

```
failure_threshold:    50% failure rate
evaluation_window:    60 seconds
minimum_requests:     10  # don't open on low sample size
open_timeout:         30 seconds before testing recovery
granularity:          per provider × per model × per region
```




---

## Cost attribution at enterprise scale

Without cost attribution at the gateway layer, cost optimization is blind. You cannot optimize what you cannot measure, and you cannot measure what you do not attribute. At enterprise scale, "the AI bill went up this month" is not actionable. "The customer-support RAG feature on team-alpha consumed 340% of its token budget because a retrieval regression started returning 8K-token chunks instead of 512-token chunks" is.

### Attribution dimensions

A production attribution model tracks spend across four dimensions at minimum. Each dimension serves a different optimization purpose.

**Tenant** - which team, product, or customer generated the cost. This is the dimension that enables chargeback, budget enforcement, and identifying which teams are burning disproportionate spend. Without tenant attribution, cost anomalies are organization-wide noise. With it, they're localized signals.

**Model and provider** - which model and version was used, on which provider. This dimension drives model routing optimization. If GPT-4o is generating 60% of your spend but Claude Haiku handles 80% of those requests with equivalent quality at one-fifth the cost, you have a routing misconfiguration, not a cost problem. You can only see this with model-level attribution.

**Feature and request type** - which product feature or workflow generated the request. Not all features have the same cost tolerance. A background document summarization job can afford higher latency and can route to cheaper models. A real-time customer-facing RAG response cannot. Feature-level attribution lets you apply different routing policies per feature rather than one global policy.

**Cache status** - whether the response was a cache hit or a model call. This dimension directly measures the ROI of your caching layer. If your semantic cache shows a 12% hit rate when you expected 35%, either your similarity threshold is too tight, your query distribution has shifted, or your cache is undersized. You cannot debug this without cache status in the attribution model.

The gateway tags every request at ingress and records the complete attribution record on completion:

```json
{
  "request_id": "req_7f3a9c2e",
  "timestamp": "2026-04-30T10:23:45Z",

  "tenant_id": "team-alpha",
  "feature": "customer-support-rag",
  "request_type": "retrieval_augmented_generation",

  "provider": "anthropic",
  "model": "claude-sonnet-4-5",
  "model_version": "20251001",

  "input_tokens": 2847,
  "output_tokens": 412,
  "cache_hit": false,
  "cache_tier": null,

  "cost_usd": 0.0412,
  "latency_ms": 1240,

  "fallback_used": false,
  "fallback_reason": null,
  "routing_tier": "mid"
}
```

Two fields worth highlighting. `cache_tier` distinguishes between a semantic cache hit and a prefix cache hit - they have different cost profiles and different optimization levers. `fallback_used` combined with `fallback_reason` tells you whether your fallback events are concentrated in a specific provider, time window, or request type - which is the signal you need to decide whether to adjust primary routing or negotiate better rate limits with the provider.

### Budget controls

Budget enforcement at the gateway layer is the only reliable way to prevent a single team, feature, or runaway loop from consuming organizational resources beyond what was planned. Application-level budget checks are bypassable, inconsistent, and invisible to the teams that need visibility. Gateway-level enforcement is universal.

The engineering decision is not whether to have limits - it is how they behave when triggered.

**Soft limits** fire before the hard limit to give engineering time to investigate and respond. 80% of monthly budget is a reasonable threshold. The soft limit generates an alert, logs the event in the attribution system, and optionally sends a notification to the team that owns the tenant. It does not change request behavior. The goal is human awareness, not automated intervention.

**Hard limits** change behavior. The naive implementation rejects the request with an error. At enterprise scale this is almost always the wrong choice - a user whose request is rejected because their team exhausted its monthly budget has a poor experience that has nothing to do with the quality of your AI system.

The correct hard limit behavior is degradation, not rejection:

```
Request arrives → tenant budget check
  → Under soft limit (0-80%):    route normally
  → Over soft limit (80-100%):   route normally + alert team
  → Over hard limit (100%+):     route to cheapest available model tier
  → Hard limit + no cheap model: return graceful degradation response
  → Never:                       return raw budget error to end user
```

The cheapest available model tier serves as a natural throttle - the team's requests still succeed but at lower quality and lower cost, which creates organic pressure to investigate and fix the underlying issue without degrading the user experience.

**Per-request budget tracking** adds a second enforcement dimension beyond monthly limits. An individual request that would consume more than a defined token ceiling - typically set per routing tier - is rejected at the gateway before reaching the provider. This catches runaway agentic loops before they become billing events, not after.

```
per_request_token_ceiling:
  fast_tier:     4,096 tokens total
  mid_tier:     16,384 tokens total
  frontier_tier: 65,536 tokens total
```

These ceilings should be set based on your actual P99 request sizes, not round numbers. A ceiling that's too low generates false positives on legitimate requests. A ceiling that's too high doesn't catch runaway loops until they've already consumed significant budget.

---

## Model versioning and canary deployments

Providers deprecate models on their own timelines, with notice periods that have ranged from six months to six weeks. When OpenAI deprecated GPT-4 in favor of GPT-4o, teams that had hardcoded model strings in application code needed to find and update every service that referenced `gpt-4`. Teams with a gateway updated one routing table entry. This is the simplest argument for logical model naming - it converts a multi-service incident into a single configuration change.

### Version pinning and the model catalog

The gateway maintains a model catalog - a versioned registry of every available model across all providers, their current status, their capability profile, and their cost per token. Application code references logical tier names, not provider-specific model strings:

```yaml
# Gateway model catalog
tiers:
  fast:
    primary:
      provider: anthropic
      model: claude-haiku-4-5         # resolved at routing time
      context_window: 200_000
      input_cost_per_M: 0.80
      output_cost_per_M: 4.00
      status: active
    fallback:
      provider: openai
      model: gpt-4o-mini-2024-07-18
      status: active

  mid:
    primary:
      provider: anthropic
      model: claude-sonnet-4-5
      context_window: 200_000
      input_cost_per_M: 3.00
      output_cost_per_M: 15.00
      status: active

  frontier:
    primary:
      provider: anthropic
      model: claude-opus-4-5
      context_window: 200_000
      input_cost_per_M: 15.00
      output_cost_per_M: 75.00
      status: active
```

Application code calls `route(tier="mid")`. The gateway resolves this to `claude-sonnet-4-5` at the time of the request. When Anthropic releases a successor and deprecates `claude-sonnet-4-5`, you update the catalog. No application code changes. No cross-team coordination. No incident.

The catalog also carries capability metadata that routing decisions depend on. A request requiring a 150K-token context window cannot route to a model with a 32K window regardless of tier assignment. The gateway checks capability constraints at routing time, not at configuration time - which means adding a new model to the catalog automatically makes it available to requests it can handle without any additional configuration.

**Deprecation handling** in the catalog follows a three-status lifecycle: `active` → `deprecated` (still routable but flagged for migration) → `sunset` (removed from routing, requests blocked). The `deprecated` status generates alerts to teams still routing to that model, giving them a migration window before the hard cutoff.

### Shadow mode testing

Shadow mode is the practice of running a candidate model against live production traffic without exposing users to its responses. It is the only way to answer the questions that matter before a rollout - not "does this model score better on our eval set" but "how does it behave on the actual distribution of requests our system receives."

The infrastructure requirement is an async duplication path that is strictly non-blocking on the primary response path:

```
Incoming request
       │
       ▼
Primary model ──────────────────────► Response to user
       │                               (P99 latency unaffected)
       │ (async fork, fire-and-forget)
       ▼
Shadow model ──► Log {request, shadow_response, latency, cost}
                 ──► Evaluation queue
```

The shadow path must be genuinely non-blocking. If the shadow model call can delay the primary response - through shared thread pools, shared connection limits, or any synchronous dependency - the shadow infrastructure is affecting production latency, which defeats the purpose. Implement the shadow fork as a background task dispatched after the primary response is returned to the caller.

**What to log from shadow runs** determines what questions you can answer during evaluation. At minimum:

```json
{
  "request_id": "req_7f3a9c2e",
  "shadow_model": "claude-sonnet-4-6",
  "primary_model": "claude-sonnet-4-5",
  "shadow_response": "...",
  "primary_response": "...",
  "shadow_latency_ms": 980,
  "primary_latency_ms": 1240,
  "shadow_cost_usd": 0.038,
  "primary_cost_usd": 0.041,
  "shadow_error": null,
  "request_tier": "mid",
  "feature": "customer-support-rag"
}
```

**Offline evaluation** against shadow logs should answer four questions before a canary launch: quality equivalence (are shadow responses semantically equivalent to primary responses on a sampled subset?), latency delta (does the shadow model meet the SLO on P95 and P99, not just P50?), cost delta (what is the per-request cost change at full traffic volume?), and failure mode coverage (does the shadow model fail on any request class that the primary handles correctly?). The last question requires evaluating shadow errors against the request distribution - a model that fails on a rare but important request class will not show up in aggregate quality metrics but will surface in failure mode analysis.

### Canary deployments

After shadow validation establishes that the candidate model is safe to expose to users, canary deployment routes a controlled fraction of live traffic to it and measures real-world impact.

The ramp schedule and rollback criteria are the engineering decisions that matter:

```python
CANARY_RAMP = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
HOLD_PERIOD_HOURS = 24   # minimum observation time at each fraction

ROLLBACK_THRESHOLDS = {
    "error_rate_delta":     0.005,   # >0.5pp increase triggers rollback
    "p99_latency_delta_ms": 200,     # >200ms P99 regression triggers rollback
    "cost_delta_fraction":  0.15,    # >15% cost increase triggers rollback
    "quality_score_delta":  -0.03,   # >3pp quality drop triggers rollback
}

def should_rollback(canary_metrics: dict, baseline_metrics: dict) -> bool:
    # Evaluate each threshold independently - any breach triggers rollback
    if (canary_metrics["error_rate"] - baseline_metrics["error_rate"]
            > ROLLBACK_THRESHOLDS["error_rate_delta"]):
        return True
    if (canary_metrics["p99_latency_ms"] - baseline_metrics["p99_latency_ms"]
            > ROLLBACK_THRESHOLDS["p99_latency_delta_ms"]):
        return True
    if ((canary_metrics["cost_per_request"] - baseline_metrics["cost_per_request"])
            / baseline_metrics["cost_per_request"]
            > ROLLBACK_THRESHOLDS["cost_delta_fraction"]):
        return True
    return False
```

Two details in the ramp design that matter in practice.

**Start at 1%, not 5%.** The first canary fraction should be small enough that a catastrophic failure - the model breaks on a specific request class that is 2% of your traffic - affects almost no users before the error rate threshold fires. 1% exposes the model to enough traffic to generate meaningful metrics while limiting blast radius.

**Hold period is non-negotiable.** Skipping or shortening the hold period at each ramp fraction is the most common canary failure mode. Some degradations - quality regressions on low-frequency request types, cost increases that only manifest on longer sessions - take hours to appear in aggregate metrics. A 24-hour hold at each fraction catches these before they become organization-wide.

**Automatic rollback must be implemented, not just designed.** A rollback procedure that requires a human to notice the metric breach, decide to roll back, and execute the configuration change is not a safety net at enterprise scale. The canary infrastructure monitors its own thresholds continuously and reverts the routing table automatically when any threshold is breached.

:::info Shadow mode vs. canary Shadow mode answers: 
is it safe to expose users to this model? Canary answers: does it perform as expected under real user load? They are sequential, not interchangeable. Running a canary without prior shadow validation exposes users to failure modes that could have been caught offline. 
:::

---

## Observability at the gateway layer

The gateway has a property no other layer in your stack shares - it sees every request, from every tenant, to every provider, across every model. That makes it the most information-dense observability point in the entire system. The question is not what you _can_ instrument here - it is which signals actually drive operational decisions and which ones just fill dashboards.

The discipline is to instrument for action, not for coverage. Every metric should have a defined alert threshold and a defined response. Metrics without those are noise.

### Signals that drive decisions

|Signal|What it tells you|Action when it degrades|
|---|---|---|
|**Latency P99 per provider × model**|Which provider is degrading and on which model specifically|Shift routing weight away from degrading provider before users notice|
|**Cache hit rate by request type**|Whether your caching layer is working for your actual query distribution|Retune similarity threshold, rewarm cache, or investigate query distribution shift|
|**Provider error rate**|Provider health before it manifests as user-facing failures|Trigger circuit breaker, shift to fallback chain|
|**Rate limit event rate per provider**|Whether you're approaching quota ceilings|Renegotiate limits, adjust routing weights, add provider capacity|
|**Fallback rate**|Whether primary routing is stable|Investigate primary provider degradation or misconfigured routing logic|
|**Cost per request by feature × model**|Whether cost anomalies are localized to a specific feature or model|Adjust routing policy for that feature, investigate retrieval regression|
|**Routing tier distribution**|Whether complexity-based routing is classifying correctly|Retune classifier if distribution shifts unexpectedly|
|**Token budget utilization per tenant**|Which tenants are approaching hard limits|Alert team, investigate request size growth|

Two signals deserve emphasis because they are the ones most commonly absent from gateway dashboards despite being the most actionable.

**Cache hit rate by request type, not overall.** An overall cache hit rate of 28% can mask a hit rate of 52% on your highest-volume request type and 4% on a long-tail type that is burning disproportionate cost. The overall number looks acceptable; the decomposed number reveals where to focus. Instrument hit rate per feature and per request tier separately.

**Routing decision logs.** Every routing decision - which tier was selected, which provider resolved it, whether a fallback fired, and what drove the decision - should be logged as a structured event. This is the data that lets you answer "why did this request cost $0.18 when the average is $0.02" without guesswork. Without routing decision logs, cost and latency anomalies are opaque. With them, they are traceable to a specific routing path within minutes.

### What a healthy gateway looks like at steady state

These are production-calibrated thresholds, not theoretical targets. Breaching any of them is a signal worth investigating, not ignoring:

|Metric|Healthy range|What a breach signals|
|---|---|---|
|Cache hit rate|≥ 25-35%|Query distribution shift, threshold misconfiguration, cache undersize|
|Gateway overhead (P99)|< 20ms|Synchronous operations on critical path, undersized vector index for semantic cache|
|Provider error rate|< 0.5% per provider|Provider degradation, misconfigured timeout thresholds|
|Fallback rate|< 2%|Primary routing instability, provider quota exhaustion|
|Cost per request trend|Flat or decreasing|Routing misconfiguration, retrieval regression inflating context size, cache decay|
|Routing tier distribution|Stable week-over-week|Query distribution shift requiring classifier retuning|

The gateway overhead number deserves a specific note. Production-grade gateways written in Go achieve sub-11 microsecond overhead at 5,000 requests per second. If your gateway is adding 100ms or more of overhead, the problem is almost always one of two things: a synchronous operation - database call, external API check, blocking lock - on the request critical path, or a semantic cache embedding step that is not fast enough for the latency budget it has been given. Both are architectural problems, not tuning problems. No amount of hardware scaling fixes a synchronous database call on every request path.

:::info Alerting vs. dashboarding 
A gateway metric that lives only on a dashboard requires a human to notice it has degraded. At enterprise scale, gateway metrics should have programmatic alert thresholds that fire before the degradation reaches users - not after. Latency P99 crossing 2× baseline, cache hit rate dropping 10 percentage points in an hour, provider error rate exceeding 1% - all of these should page before they appear in a post-incident review. 
:::

---

## Build vs. buy

The right answer depends on three axes: performance requirements, customization depth, and how fast you need to move. Most teams get this wrong by evaluating options against current needs rather than 18-month needs.

The gateway market has matured significantly. Open-source foundations that were experimental two years ago are now running at billion-request scale in production. Building from scratch is justified in two cases only: your routing requirements are genuinely novel in ways no existing solution supports, or your data residency and security constraints eliminate every other option. For everyone else, building from scratch means spending engineering months on infrastructure that others have already solved - and solving it worse on the first iteration.

**What to evaluate, not which tool to pick:**

The gateway landscape changes faster than any handbook should try to track. Evaluate candidates against these criteria at the time of your decision:

| Criterion                 | Why it matters                                      | What to measure                                                            |
| ------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------- |
| **Gateway overhead**      | Adds to every request's P99 latency                 | Benchmark at your expected RPS, not vendor claims                          |
| **Provider coverage**     | Determines fallback flexibility                     | Which providers you need today and in 12 months                            |
| **Routing customization** | Off-the-shelf routing may not match your logic      | Can you implement complexity-based and cost-based routing without forking? |
| **Semantic caching**      | Major cost lever - not all gateways support it      | Native support vs. bolt-on vs. you build it                                |
| **Data residency**        | Determines self-hosted vs. managed viability        | Can it run fully within your VPC?                                          |
| **SSO / RBAC**            | Enterprise requirement - often paywalled            | Is it included or a paid tier?                                             |
| **MCP support**           | Table stakes for agentic systems in 2026            | Native or plugin-based?                                                    |
| **Operational model**     | Managed vs. self-hosted has different failure modes | Who is on call when it breaks?                                             |

**The practical path:** start with an open-source foundation that meets your performance and customization requirements. Run it self-hosted. Customize from that foundation. The specific tool you choose matters less than the architecture you build around it - logical model names, consistent attribution, and a well-designed fallback chain work the same way regardless of which gateway implements them.

The chapters that follow assume a gateway that can implement the patterns described here. The choice of which gateway is yours to make based on your constraints at the time.

---

## Production readiness checklist

The sections above cover each gateway concern in depth. Before the gateway carries enterprise-scale load, the checklist below consolidates the verification points across all of them. An item that cannot be checked off is a known risk - the question is whether you are carrying it deliberately or unknowingly.

| Area                       | Verification                                                                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Routing**                | Multi-provider fallback chain tested with synthetic provider failures - not assumed to work                                               |
|                            | Complexity-based routing validated against actual production query distribution, not a synthetic benchmark                                |
|                            | Latency-based routing tracking real-time provider health with EWMA estimates per provider × model                                         |
|                            | 429 handling switches provider immediately - same-provider retry on a rate limit is never correct                                         |
| **Caching**                | Semantic cache similarity threshold tuned against your query distribution and validated with manual hit review                            |
|                            | Prefix caching enabled for all providers that support it - zero application code change required                                          |
|                            | Cache hit rate dashboarded and alerted - a significant drop week-over-week is a signal, not noise                                         |
|                            | Cache invalidation strategy defined for every time-sensitive content source feeding the index                                             |
| **Resilience**             | Circuit breakers configured per provider × model × region, thresholds calibrated to that provider's baseline error rate                   |
|                            | Exponential backoff with equal jitter on all retry paths - full jitter risks near-zero delays at high attempt counts                      |
|                            | Hard budget limits degrade to cheaper model tier - returning an error to the user is the wrong terminal behavior                          |
|                            | Load tested at 2× expected peak with fallback chain under sustained load, not just single-request failure scenarios                       |
| **Cost and observability** | Per-tenant, per-model, per-feature attribution live on every request path - including cache hits, fallbacks, and error paths              |
|                            | Soft and hard budget limits configured per tenant with degradation behavior verified, not just configured                                 |
|                            | Real-time cost dashboard with day-over-day and week-over-week comparison - invoice surprises mean observability gaps                      |
|                            | Provider health alerts programmatically fire on error rate and P99 latency breach - dashboards without alerts require a human to notice   |
| **Model versioning**       | Logical model names in application code, physical names resolved at gateway catalog - deprecation is a catalog update, not an incident    |
|                            | Shadow mode infrastructure verified non-blocking - a shadow call that can delay the primary response path is a production latency problem |
|                            | Canary rollback automatic and tested - a rollback procedure that requires human intervention is not a safety net at enterprise scale      |

The gateway is the entry point - the layer that governs every request before it touches a model or a retrieval system. With that foundation in place, the next question is what happens when a request reaches the retrieval layer: how embedding models are selected, benchmarked, and adapted for your domain, and what retrieval quality actually means at production scale.

---

*Next: [Chapter 2 - Embedding Models and Retrieval Quality](/agentic/02-embedding-models)*
