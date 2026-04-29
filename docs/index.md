---
id: index
title: Sizing LLM Inference for Production
slug: /
pagination_next: null
pagination_prev: null
---

import Link from '@docusaurus/Link';

<div style={{display:'flex', gap:'1.25rem', alignItems:'flex-end', margin:'0 0 2.5rem 0', flexWrap:'wrap'}}>
  <a href="/pdf/sizing-llm-inference-systems.pdf" target="_blank" rel="noopener noreferrer" title="Download the full field guide PDF">
    <img src="/img/covers/cover-front.png" alt="Sizing LLM Inference for Production - Cover" style={{width:'195px', borderRadius:'8px', boxShadow:'0 20px 48px rgba(0,0,0,0.45)', border:'none', display:'block', cursor:'pointer'}} />
  </a>
  <a href="/img/covers/cover-contents.png" target="_blank" rel="noopener noreferrer" title="What's inside">
    <img src="/img/covers/cover-contents.png" alt="What's Inside" style={{width:'158px', borderRadius:'8px', boxShadow:'0 14px 36px rgba(0,0,0,0.3)', border:'none', display:'block', cursor:'pointer'}} />
  </a>
  <a href="/img/covers/cover-author.png" target="_blank" rel="noopener noreferrer" title="About the author">
    <img src="/img/covers/cover-author.png" alt="About the Author" style={{width:'158px', borderRadius:'8px', boxShadow:'0 14px 36px rgba(0,0,0,0.3)', border:'none', display:'block', cursor:'pointer'}} />
  </a>
</div>

<p style={{fontFamily:'Inter,system-ui,sans-serif', fontSize:'12px', fontWeight:'700', textTransform:'uppercase', letterSpacing:'0.1em', color:'#2563EB', marginBottom:'0.75rem'}}>Field Guide · v1.1 · May 2026</p>

# Sizing LLM Inference for Production

<p style={{fontFamily:"'Source Serif 4',Georgia,serif", fontSize:'1.15rem', color:'var(--ifm-color-content-secondary)', lineHeight:'1.65', maxWidth:'640px', margin:'0.5rem 0 1.75rem'}}>
Inference is where AI infrastructure spend compounds indefinitely with usage growth. Most production LLM fleets are paying 2-3× what they need to, not from hardware limits but from sizing decisions made without a disciplined framework. This guide provides that framework: a complete decision sequence from workload characterization to production fleet sizing, grounded in the roofline model, queueing theory, and empirical benchmarking.
</p>

<div style={{display:'flex', gap:'1rem', margin:'0 0 1rem', flexWrap:'wrap'}}>
  <a href="/pdf/sizing-llm-inference-systems.pdf" target="_blank" rel="noopener noreferrer" style={{display:'inline-flex', alignItems:'center', gap:'0.4rem', fontFamily:'Inter,system-ui,sans-serif', fontSize:'14px', fontWeight:'600', color:'#FFFFFF', background:'#2563EB', border:'2px solid #2563EB', borderRadius:'6px', padding:'0.6rem 1.25rem', textDecoration:'none'}}>
    ↓ Download PDF (107 pages)
  </a>
  <Link to="/preface" style={{display:'inline-flex', alignItems:'center', gap:'0.4rem', fontFamily:'Inter,system-ui,sans-serif', fontSize:'14px', fontWeight:'600', color:'#2563EB', background:'transparent', border:'2px solid #2563EB', borderRadius:'6px', padding:'0.6rem 1.25rem', textDecoration:'none'}}>
    Read Online →
  </Link>
</div>

<p style={{fontFamily:'Inter,system-ui,sans-serif', fontSize:'12.5px', color:'var(--ifm-color-content-secondary)', margin:'0 0 2rem'}}>
  107 pages · 9 chapters · 13-step sizing algorithm · Staff / Principal ML Engineers
</p>

---

## Contents

| | Chapter | What you'll be able to do |
|---|---------|-------------|
| → | [Why Inference Sizing Is a Capital Allocation Problem](/sizing/why-inference-sizing) | Understand why inference - not training - now dominates AI infrastructure spend, and why "just add GPUs" is a compounding financial mistake |
| 1 | [Workload Characterization](/sizing/01-workload-characterization) | Derive the P95 input/output length distributions, peak RPS, and latency targets that every downstream calculation depends on - before touching a single config parameter |
| 2 | [GPU Memory Sizing](/sizing/02-gpu-memory-sizing) | Calculate the minimum GPU count from first principles: model weights, KV cache at peak concurrency, activations, and framework overhead - using your workload numbers, not model card estimates |
| 3 | [The Roofline Model](/sizing/03-roofline-model) | Explain precisely why prefill and decode require different hardware, and why buying H100s for their TFLOPS fails decode-dominated workloads |
| 4 | [Quantization](/sizing/04-quantization) | Sequence weight, activation, and KV cache quantization decisions correctly - and understand why FP8 is a GPU count decision, not a quality tuning decision |
| 5 | [Parallelism Strategy](/sizing/05-parallelism-strategy) | Select TP, PP, DP, and EP degrees in the right order, understand why TP=4 on PCIe can be slower than TP=2 on NVLink, and size MoE models correctly |
| 6 | [Batching Strategy](/sizing/06-batching-strategy) | Choose between continuous batching, chunked prefill, disaggregated P/D, and speculative decoding based on your specific TTFT/ITL tradeoff - not framework defaults |
| 7 | [KV Cache Optimization](/sizing/07-kv-cache-optimization) | Deploy PagedAttention, prefix caching, and cache-aware routing as a coordinated stack - and diagnose the memory pressure cascade before it presents as a latency problem |
| 8 | [Latency-Throughput Curve](/sizing/08-latency-throughput-curve) | Generate the empirical curve for your deployment, find the SLO-constrained operating point, and size the fleet from measured data rather than theoretical peaks |
| 9 | [Sizing Algorithm & Monitoring](/sizing/09-sizing-algorithm) | Execute the full 13-step sizing sequence in the correct dependency order, instrument the metrics that fire early enough to intervene, and avoid the utilization trap in TCO |
| ✦ | [First Principles, Last](/sizing/first-principles) | The reasoning framework that outlasts every hardware generation and framework version in this guide |

---

<div style={{fontFamily:'Inter,system-ui,sans-serif', fontSize:'13.5px', color:'var(--ifm-color-content-secondary)', background:'var(--ifm-background-surface-color)', border:'1px solid var(--ifm-color-emphasis-300)', borderLeft:'3px solid var(--ifm-color-emphasis-300)', borderRadius:'0 6px 6px 0', padding:'0.85rem 1.1rem', lineHeight:'1.6', fontStyle:'italic'}}>
This field moves fast - specific numbers, framework defaults, and GPU specs will age. The reasoning framework will not. Read the numbers as illustrations of the method, not as configuration targets to copy verbatim.
</div>
