---
id: index
title: Sizing LLM Inference for Production
slug: /
---

import Link from '@docusaurus/Link';

<div className="field-guide-cover">

<div className="field-guide-eyebrow">Field Guide · v1.0 · April 2026</div>

# Sizing LLM Inference for Production

<div className="field-guide-subtitle">
From first principles to cost-efficient scale. A complete decision framework for GPU capacity planning — workload characterization, memory budgeting, parallelism, KV cache, and a 13-step sizing algorithm.
</div>

<div className="field-guide-actions">
  <Link
    to="/pdf/sizing-llm-inference-systems.pdf"
    className="field-guide-btn-primary"
    target="_blank">
    ↓ Download PDF (107 pages)
  </Link>
  <Link
    to="/sizing/preface"
    className="field-guide-btn-secondary">
    Read Online →
  </Link>
</div>

<div className="field-guide-meta">
  <span>107 pages</span>
  <span>·</span>
  <span>9 chapters</span>
  <span>·</span>
  <span>13-step sizing algorithm</span>
  <span>·</span>
  <span>Staff / Principal ML Engineers</span>
</div>

</div>

---

## Preface

LLM inference has become one of the most consequential infrastructure engineering problems in the industry. The individual techniques are well-documented — quantization, parallelism, batching, KV cache optimization each have deep literature behind them. What is harder to find is a single framework that shows how these pieces fit together, in what order to apply them, and how to turn the output into a GPU count and cost estimate you can stand behind.

This guide is built around that framework. Every decision covered here — memory budgeting, parallelism strategy, quantization format, KV cache configuration — follows from measurable properties of your workload and your hardware. The goal is not to give you configurations to copy. It is to give you the reasoning to derive the right configuration for your specific model, traffic, and constraints — and to understand it well enough to adapt it when things change.

**Who this is for.** Staff and Principal ML Engineers, ML Platform Engineers, AI Infrastructure Architects, and Applied Scientists moving into production ownership. It assumes you are comfortable with transformer fundamentals, have hands-on GPU experience, and understand distributed systems concepts like memory hierarchies, throughput, and latency. Prior inference optimization experience is not required — that is what the guide builds from the ground up.

Engineers coming from adjacent areas — distributed systems, cloud infrastructure, or ML platform work — will find the systems reasoning familiar even if the LLM-specific concepts are new. Researchers and applied scientists who want to understand what happens to their models after training — why production behavior differs from evaluation, what drives serving cost, and how architectural choices like Mixture-of-Experts affect deployment economics — will find this a practical bridge between model design and system reality.

**How to use this guide.** The sections form a decision sequence — each one produces an output that feeds the next. Workload characterization informs memory sizing. Memory sizing constrains parallelism selection. Parallelism selection determines the benchmark configuration. The benchmark produces the operating point. The operating point sizes the fleet. Reading in order builds the full reasoning chain. After that the guide works as a reference — each section stands alone for readers returning to a specific production problem.

**A note on the pace of change.** This field moves fast — specific numbers, framework defaults, and GPU specs will age. The reasoning framework will not. Read the numbers as illustrations of the method, not as configuration targets to copy verbatim.

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

<div className="field-guide-note">
This field moves fast — specific numbers, framework defaults, and GPU specs will age. The reasoning framework will not. Read the numbers as illustrations of the method, not as configuration targets to copy verbatim.
</div>
