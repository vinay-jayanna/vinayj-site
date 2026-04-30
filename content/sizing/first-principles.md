---
id: first-principles
title: First Principles, Last
---

LLM inference infrastructure is one of the few places in modern software engineering where the gap between "it works" and "it works efficiently" translates directly into millions of dollars a year. A fleet running at 40% utilization with naive batching and no KV cache optimization is not a failing system - it serves requests, passes health checks, and looks fine on dashboards. It is just paying 3× what it needs to.

Most of the levers in this guide are not new ideas. Quantization, batching, caching, parallelism - these concepts predate LLMs. What is new is the specific way they interact in autoregressive generation: the prefill/decode duality, the quadratic KV cache growth with context length, the memory-bandwidth bottleneck that makes batch size the primary throughput lever rather than clock speed. Understanding these interactions, sequencing the decisions correctly, and measuring at each step is where most production deployments leave significant efficiency on the table - not because the engineers are inexperienced, but because the system is genuinely complex and the interactions are non-obvious until you have seen them in production.

The field will keep moving fast. New model architectures will change the memory math - Mixture-of-Experts already did, and whatever comes next will too. New hardware generations will shift the roofline ridge point and reopen hardware selection decisions that felt settled. New serving patterns will emerge the way disaggregated prefill-decode did - from a research paper to a production default in under two years. Specific numbers in this guide will age: GPU specs, framework defaults, benchmark results. Some already have between when sections were drafted and when you are reading this.

What does not age is the reasoning framework. The roofline model will correctly characterize any GPU workload regardless of the hardware generation. The four-component memory formula will correctly bound any transformer model regardless of its architecture. Little's Law will correctly describe any queuing system regardless of the serving framework. The latency-throughput curve will have a knee on any finite hardware system regardless of how much the knee moves rightward with each optimization. These are not LLM-specific insights - they are physics and queueing theory applied to a specific domain. They will outlast every framework version and hardware generation covered in this guide.

This guide is a starting point, not a finishing line. What stays constant is the value of reasoning from first principles: characterize the workload, understand the constraints, measure empirically, and size from data. Hopefully this guide makes that a little more principled.
