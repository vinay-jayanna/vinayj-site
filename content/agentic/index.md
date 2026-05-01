---
id: index
title: Agentic Systems in Production
slug: /
---

import Link from '@docusaurus/Link';

<p style={{fontFamily:'Inter,system-ui,sans-serif', fontSize:'12px', fontWeight:'700', textTransform:'uppercase', letterSpacing:'0.1em', color:'#2563EB', marginBottom:'0.75rem'}}>Engineering Handbook · In Progress · 2026</p>

# Agentic Systems in Production

<p style={{fontFamily:"'Source Serif 4',Georgia,serif", fontSize:'1.15rem', color:'var(--ifm-color-content-secondary)', lineHeight:'1.65', maxWidth:'640px', margin:'0.5rem 0 1.75rem'}}>
An Engineering Handbook for Reliability, Observability, and Scale
</p>

*By Vinay Jayanna - Staff ML Engineer, LLM Inference and GenAI Platform*

---

Most RAG and agentic systems work at team scale. This handbook is about what happens when the business runs on them.

Written for engineers who hold the pager - Principal and Staff ML Engineers building systems that serve tens of thousands of queries per day, where a non-deterministic loop going wrong, a retrieval pipeline going stale, or an agent exceeding its cost budget has real consequences.

:::info Download
PDF available on publication. The web version is the live, versioned edition.
:::

---

## What This Engineering Handbook Covers

**Part I - The Production Landscape** establishes the failure taxonomy and the infrastructure entry point. Non-determinism as a first-class engineering problem. The LLM gateway as the foundation everything else depends on.

**Part II - High-Throughput Retrieval** covers the full retrieval stack at production scale: embedding models, vector database internals and real-time indexing, document pipeline reliability, context engineering as a managed resource, and hybrid search with semantic caching.

**Part III - RAG System Design** covers production failure modes and architecture patterns, evaluation harnesses that catch regressions, and the cost, latency, and security concerns that multi-tenant RAG creates.

**Part IV - Agentic System Design** opens with the Agentic RAG loop as the bridge between retrieval and agency, then covers agent loop architecture starting at state machines, tool design and the MCP protocol layer, memory and state management, multi-agent orchestration and failure isolation, and the cost and security concerns unique to agentic systems.

**Part V - Reliability and Governance** covers DAG-based observability for non-deterministic systems, deterministic testing and agentic CI/CD, continuous improvement pipelines, and production guardrails with the full-stack sizing algorithm.

---

## Who This Is For

- Principal and Staff ML Engineers building or owning production RAG or agentic systems
- Platform engineers designing the infrastructure layer these workloads run on
- Tech leads making architecture decisions for AI products at scale

This handbook assumes working knowledge of LLM inference, distributed systems, and production ML operations. It does not explain what a transformer is, or what an agent is.

---

## Chapters

|     | Chapter                                                                                          | Focus                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| →   | [Why RAG and Agentic Systems Break at Scale](/agentic/why-rag-agentic-hard)                      | Failure taxonomy, production ops model                                                                                       |
| 1   | [LLM Gateway and Multi-Provider Routing](/agentic/01-llm-gateway)                                | Entry point infrastructure, routing, circuit breakers                                                                        |
| 2   | [Embedding Models and Retrieval Quality](/agentic/02-embedding-models)                           | Domain adaptation, benchmarking, fine-tuning                                                                                 |
| 3   | [Vector Database Architecture, Scaling, and Real-Time Indexing](/agentic/03-vector-database)     | ANN internals, sharding, real-time indexing, MTTI, hot-shard management                                                      |
| 4   | [Chunking, Context Construction, and Document Pipelines](/agentic/04-chunking-context)           | Pipeline reliability, metadata, freshness at scale                                                                           |
| 5   | [Context Engineering: Budget, Assembly, and Governance](/agentic/05-context-engineering)         | Context window as a managed resource, long context vs. RAG trade-off, budget enforcement, provenance, multi-tenant isolation |
| 6   | [Hybrid Search, Query Routing, and Semantic Caching](/agentic/06-hybrid-search)                  | Sparse+dense fusion, semantic cache, COGS reduction                                                                          |
| 7   | [RAG Architectures: Production Failure Modes and Design Patterns](/agentic/07-rag-architectures) | Failure taxonomy, modular RAG, GraphRAG, decision framework                                                                  |
| 8   | [RAG Evaluation: Metrics That Survive Production](/agentic/08-rag-evaluation)                    | RAGAS, eval pipelines, CI regression guards                                                                                  |
| 9   | [RAG Cost, Latency, and Security](/agentic/09-rag-cost-security)                                 | End-to-end latency decomposition, token budgets, caching layers, multi-tenant access control, adversarial retrieval          |
| 10  | [Agentic RAG: When Retrieval and Agency Interleave](/agentic/10-agentic-rag)                     | Retrieve→reason→rewrite loops, compounding failures, stopping conditions                                                     |
| 11  | [Agent Loop Design: State Machines, Re-planning, and Failure Isolation](/agentic/11-agent-loops) | State machines, re-planning, non-termination, runaway loop prevention                                                        |
| 12  | [Tool Design, MCP, and the Agentic Protocol Layer](/agentic/12-tool-mcp)                         | MCP production infrastructure, OWASP MCP Top 10, OAuth 2.1, tool poisoning                                                   |
| 13  | [Memory, State, and Context Across Agent Turns](/agentic/13-memory-state)                        | Memory architectures, state machines, eviction, session recovery                                                             |
| 14  | [Multi-Agent Orchestration and Failure Isolation](/agentic/14-multi-agent)                       | Supervisor patterns, deadlocks and shared state corruption, blast radius, circuit breakers, inter-agent protocols            |
| 15  | [Latency Budgets, Cost Control, and Agentic Security](/agentic/15-latency-cost-security)         | Token burn rate, model routing, inference-time compute trade-offs, prompt injection, least-privilege, sandboxing             |
| 16  | [DAG-Based Observability for Non-Deterministic Systems](/agentic/16-observability)               | Where OpenTelemetry fails, DAG trace schemas, drift detection                                                                |
| 17  | [Deterministic Testing and Agentic CI/CD](/agentic/17-testing-cicd)                              | Simulation environments, reproducible agent tests, regression-guarding pipelines                                             |
| 18  | [Continuous Improvement: Feedback Loops and Online Learning](/agentic/18-continuous-improvement) | Signal collection, A/B testing, safe fine-tuning, index freshness                                                            |
| 19  | [Production Guardrails, Compliance, and Sizing](/agentic/19-guardrails-sizing)                   | Classifiers at throughput, HIPAA/SOC2/GDPR, full-stack sizing algorithm, production readiness checklist                      |

---

## About the Author

Currently a Staff ML Engineer leading LLM inference optimization for one of the most consequential AI systems in the world - reaching hundreds of millions of users. Before that, spent nearly a decade at **AWS** building and scaling core inference infrastructure for **SageMaker** from its earliest days. Founded **Vipas.AI**, an AI inference marketplace that reached 25K daily visitors and received a VC term sheet. Earlier career spans building large-scale distributed systems and cloud infrastructure at Ericsson, Pegasystems, and global enterprises. Holder of a USPTO-pending patent in dynamic hierarchical storage and GPU optimization for LLM serving.

[vinayj.com](https://vinayj.com) · [LinkedIn](https://linkedin.com/in/vinayjayanna) · [GitHub](https://github.com/vinay-jayanna)
