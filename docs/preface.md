---
id: preface
title: preface
---

**Preface**

LLM inference has become one of the most consequential infrastructure engineering problems in the industry. The individual techniques are well-documented \- quantization, parallelism, batching, KV cache optimization each have deep literature behind them. What is harder to find is a single framework that shows how these pieces fit together, in what order to apply them, and how to turn the output into a GPU count and cost estimate you can stand behind.

This guide is built around that framework. Every decision covered here \- memory budgeting, parallelism strategy, quantization format, KV cache configuration \- follows from measurable properties of your workload and your hardware. The goal is not to give you configurations to copy. It is to give you the reasoning to derive the right configuration for your specific model, traffic, and constraints \- and to understand it well enough to adapt it when things change.

**Who this is for.** This guide is written for engineers working at the intersection of ML and systems \- Staff and Principal ML Engineers, ML Platform Engineers, AI Infrastructure Architects, and Applied Scientists moving into production ownership. It assumes you are comfortable with transformer fundamentals, have hands-on GPU experience, and understand distributed systems concepts like memory hierarchies, throughput, and latency. Prior inference optimization experience is not required \- that is what the guide builds from the ground up.

Engineers coming from adjacent areas \- distributed systems, cloud infrastructure, or ML platform work \- will find the systems reasoning familiar even if the LLM-specific concepts are new. Researchers and applied scientists who want to understand what happens to their models after training \- why production behavior differs from evaluation, what drives serving cost, and how architectural choices like Mixture-of-Experts affect deployment economics \- will find this a practical bridge between model design and system reality.

**How to use this guide.** The sections form a decision sequence \- each one produces an output that feeds the next. Workload characterization informs memory sizing. Memory sizing constrains parallelism selection. Parallelism selection determines the benchmark configuration. The benchmark produces the operating point. The operating point sizes the fleet. Reading in order builds the full reasoning chain. After that the guide works as a reference \- each section on optimization, monitoring, and guardrails stands alone for readers returning to a specific production problem.

**A note on the pace of change.** This field moves fast \- specific numbers, framework defaults, and GPU specs will age. The reasoning framework will not. Read the numbers as illustrations of the method, not as configuration targets to copy verbatim.
