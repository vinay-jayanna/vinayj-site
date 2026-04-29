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
    <img src="/img/covers/cover-front.png" alt="Sizing LLM Inference for Production — Cover" style={{width:'195px', borderRadius:'8px', boxShadow:'0 20px 48px rgba(0,0,0,0.45)', border:'none', display:'block', cursor:'pointer'}} />
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
From first principles to cost-efficient scale. A complete decision framework for GPU capacity planning — workload characterization, memory budgeting, parallelism, KV cache, and a 13-step sizing algorithm.
</p>

<div style={{display:'flex', gap:'1rem', margin:'0 0 1rem', flexWrap:'wrap'}}>
  <a href="/pdf/sizing-llm-inference-systems.pdf" target="_blank" rel="noopener noreferrer" style={{display:'inline-flex', alignItems:'center', gap:'0.4rem', fontFamily:'Inter,system-ui,sans-serif', fontSize:'14px', fontWeight:'600', color:'#FFFFFF', background:'#2563EB', border:'2px solid #2563EB', borderRadius:'6px', padding:'0.6rem 1.25rem', textDecoration:'none'}}>
    ↓ Download PDF (107 pages)
  </a>
  <Link to="/sizing/why-inference-sizing" style={{display:'inline-flex', alignItems:'center', gap:'0.4rem', fontFamily:'Inter,system-ui,sans-serif', fontSize:'14px', fontWeight:'600', color:'#2563EB', background:'transparent', border:'2px solid #2563EB', borderRadius:'6px', padding:'0.6rem 1.25rem', textDecoration:'none'}}>
    Read Online →
  </Link>
</div>

<p style={{fontFamily:'Inter,system-ui,sans-serif', fontSize:'12.5px', color:'var(--ifm-color-content-secondary)', margin:'0 0 2rem'}}>
  107 pages · 9 chapters · 13-step sizing algorithm · Staff / Principal ML Engineers
</p>

---

## Contents

| | Chapter | Key Question |
|---|---------|-------------|
| → | [Why Inference Sizing Is a Capital Allocation Problem](/sizing/why-inference-sizing) | Why GPU spend is now an engineering decision |
| 1 | [Workload Characterization](/sizing/01-workload-characterization) | What exactly am I serving? |
| 2 | [GPU Memory Sizing](/sizing/02-gpu-memory-sizing) | How many GPUs do I need at minimum? |
| 3 | [The Roofline Model](/sizing/03-roofline-model) | Why do prefill and decode behave differently? |
| 4 | [Quantization](/sizing/04-quantization) | How do I trade precision for scale? |
| 5 | [Parallelism Strategy](/sizing/05-parallelism-strategy) | How do I split the model across GPUs? |
| 6 | [Batching Strategy](/sizing/06-batching-strategy) | How do I maximize throughput without hurting latency? |
| 7 | [KV Cache Optimization](/sizing/07-kv-cache-optimization) | What is my most under-used lever? |
| 8 | [Latency-Throughput Curve](/sizing/08-latency-throughput-curve) | Where is my operating point? |
| 9 | [Sizing Algorithm & Monitoring](/sizing/09-sizing-algorithm) | How do I put it all together? |
| ✦ | [First Principles, Last](/sizing/first-principles) | The mental model underneath it all |

---

<div style={{fontFamily:'Inter,system-ui,sans-serif', fontSize:'13.5px', color:'var(--ifm-color-content-secondary)', background:'var(--ifm-background-surface-color)', border:'1px solid var(--ifm-color-emphasis-300)', borderLeft:'3px solid var(--ifm-color-emphasis-300)', borderRadius:'0 6px 6px 0', padding:'0.85rem 1.1rem', lineHeight:'1.6', fontStyle:'italic'}}>
This field moves fast — specific numbers, framework defaults, and GPU specs will age. The reasoning framework will not. Read the numbers as illustrations of the method, not as configuration targets to copy verbatim.
</div>
