---
id: 03-vector-database
title: "Vector Database Architecture, Scaling, and Real-Time Indexing"
---
# Vector Database Architecture, Scaling, and Real-Time Indexing

<div class="chapter-summary"> The previous chapter treated the vector index as a black box — the application submitted a query, got back a ranked candidate set, and moved on. In production, that black box is part of the agent's state and correctness boundary. It may hold enterprise knowledge, episodic memory, prior tool results, workflow observations, and evidence scoped to specific users, tenants, and permissions. A stale or wrongly-scoped result does not just weaken an answer — it can cause an agent to repeat an action, act on revoked access, ignore work that already completed, or build a multi-step plan on a state that stopped being true minutes ago. This chapter is the architecture that keeps retrieval correct while the corpus changes, traffic goes uneven, and infrastructure fails. </div>

---

## The production contract: correctness before performance

Most teams start a vector database design by comparing HNSW, IVF, and disk-oriented indexes. That is usually too early. An index is a mechanism, not a requirement, and choosing the mechanism before defining what the system must guarantee is the most common reason architectures that work cleanly in a proof of concept become unpredictable under real load.

A structural engineer does not pick steel, concrete, or timber and then figure out what the building needs to withstand. The design starts with occupancy, wind, seismic zone, and fire code — the material choice falls out of those constraints almost mechanically. Vector database architecture works the same way. HNSW, IVF, DiskANN, sharding, and replication are materials. The production contract is the load spec.

The contract has to define more than expected scale. It has to define what the system means by a successful write, a complete query, an authorized result, an acceptable stale read, and a recoverable failure. Without those definitions, every component can behave exactly as its local design intends while the system as a whole violates what the application assumed was true.

### The workload envelope

The workload envelope describes the conditions the retrieval system has to hold its guarantees under — sized to the planning horizon, not to what launch day looks like.

Start by separating two numbers that get conflated constantly: the **logical live set** — records that should currently be retrievable — and the **physical index population** — everything the index actually carries, including tombstoned nodes, superseded versions, mutable segments still being compacted, and replica copies. Memory pressure, traversal cost, and rebuild time are governed by the physical population, which is almost always larger than what your source system reports as "active." The gap between the two numbers is itself a signal worth monitoring — a widening gap usually means compaction is falling behind.

Embedding dimensionality drives storage and memory bandwidth, but the raw vector array is only one line item. Graph edges, metadata indexes, document identifiers, replica copies, and compaction headroom often add up to more memory than the vectors themselves. Multiplying vector count by dimensionality and element size gives you an encoding estimate, not a production memory model.

The workload also has a temperature. A billion-vector corpus where most queries hit a small, stable working set behaves nothing like the same corpus under uniformly distributed traffic — the hot set determines whether your cache strategy actually works and whether disk-oriented search stays inside the latency budget. Average query volume hides the same kind of information; what actually breaks systems is P99 concurrency during a burst, not the daily average.

Mutation is its own workload, not a footnote on the read path. A knowledge base absorbing a few thousand immutable documents a day has almost nothing in common, architecturally, with an agent-memory service that is continuously rewriting short-lived observations, expiring records, and changing authorization metadata while serving live queries against the same data.

Filter selectivity needs to be described as a distribution, not an average. A query scoped to one tenant might narrow the eligible corpus to a few hundred vectors. Another might span nearly the whole collection. Those two cases often need different execution plans on the same index — a problem this chapter returns to directly in the section on filtered search.

Tenant count means little without tenant skew, and geography adds a further dimension. Ten evenly-sized tenants partition cleanly. A thousand tenants where a handful account for most of the traffic create hot shards and noisy-neighbor risk regardless of how clean the partitioning logic is. Regional access requirements and data residency constraints can determine data placement before index performance even enters the conversation.

Finally, the envelope needs recovery objectives — and these are two different numbers, not one. **Recovery time objective (RTO)** is how long the system can stay degraded or unavailable. **Recovery point objective (RPO)** is how much acknowledged data, if any, the system is allowed to lose. A vector index that can be fully rebuilt from source documents can tolerate a much looser RPO than an episodic-memory store whose last hour of observations exists nowhere else.

Skipping this exercise does not make these constraints disappear. It just means you find them individually, in production, usually during an incident.

### Searchable visibility latency

Chapter 2 defined Mean Time to Index (MTTI) — the delay between a source document changing and its new embedding being computed. This chapter needs a related but distinct number, because MTTI stops the moment the vector exists. It says nothing about whether that vector is actually findable yet.

**Searchable visibility latency** is the time from the database accepting a write to the point at which an eligible query is guaranteed to observe it, under the system's stated consistency model.

Acceptance, durability, replication, and searchability are four separate states, not one event. A write can be safely appended to a durable log while still waiting to enter a mutable search segment. It can be searchable on the primary node while replicas lag behind. It can appear in unfiltered search while a separately-updated metadata index has not yet made it eligible for a scoped query. A single "write succeeded" response can be quietly concealing any of these gaps unless the contract states exactly what that acknowledgment actually promises.

The closest everyday analogy is a ledger entry that has posted but has not yet propagated to every downstream view — the transaction is real and safely recorded, but not every observer can see the new balance in the same instant. For an agent, this gap is not academic. An agent that writes a memory, immediately searches for it, and gets nothing back has no way to distinguish "that write failed" from "that write hasn't propagated yet" — and it may conclude the action never happened and repeat it, duplicating a tool call or recreating a task that already exists.

Measure this as a percentile, and measure it per workload class. Interactive agent memory usually needs a stronger read-your-writes guarantee than a bulk document backfill does. Authorization revocations arguably need a _tighter_ visibility SLO than ordinary content insertions, because stale access state is a different risk profile than a slightly stale document. And be specific about _whose_ visibility you're guaranteeing — a write might be visible within the writer's own session immediately through a transactional overlay, visible in-region within a second, and only eventually visible cross-region. "Searchable within two seconds" is an incomplete claim until you name the query path and consistency boundary it applies to.

### Three independent forms of correctness

Fast retrieval is not the same claim as correct retrieval. Production correctness for a vector database has three independent components, and a system can satisfy two of them perfectly while silently failing the third — and every dashboard will still look green.

**Retrieval correctness** is finding the right neighbors — the familiar ANN concern. But it has to be measured across the full distributed path, not just inside an isolated index benchmark. Routing, filtering, quantization, replica selection, and cross-shard merging can each quietly erode recall even when the local index performs exactly as advertised — the subject of the next section.

**State correctness** is whether the evidence returned reflects the current logical state of the world, not a snapshot from before the last update. A result can be semantically perfect and still wrong, because it represents a document that was superseded ten minutes ago and the index has not caught up yet. State correctness is a function of searchable visibility, versioning, and how quickly deletions actually take effect.

**Authorization correctness** is whether every result returned is something the authenticated caller is actually permitted to see. This has to be enforced by trusted infrastructure — the agent, the prompt, and the reranker are never authorization boundaries. That said, this does not mean every implementation must exclude ineligible content before ANN traversal even begins. In graph-based indexes, a node the caller cannot see may still be structurally necessary as a waypoint the traversal passes through to reach a node they can see. The requirement is narrower and non-negotiable: no unauthorized content, score, identifier, or other observable signal may cross the retrieval boundary and reach the caller. What happens inside the trusted boundary during traversal is an implementation detail; what crosses it is not.

These three are independent, and production systems fail in every combination. A system can retrieve the single most relevant vector in the corpus and hand back a stale version of it. It can return a current, perfectly relevant document to the wrong tenant. It can enforce authorization flawlessly while quietly routing the query to only part of the corpus and omitting the best evidence entirely. All three have to hold at once for the system to be correct — satisfying two out of three is not partial credit, it is a production incident waiting for someone to notice.

### Correctness during degradation

The contract also has to define behavior during failure, not just during a healthy steady state — this is where architectures that look sound on paper become ambiguous in an incident.

When a shard is unreachable, the system has options: fail the query outright, retry against another replica, return a partial candidate set, or serve from a cache. None of these is universally correct — each trades availability, freshness, and recall differently. What is never acceptable is the fifth option nobody chose on purpose: returning a partial result while presenting it to the caller as complete.

An agent that receives incomplete evidence needs to be able to tell "no relevant document exists" apart from "the retrieval system could not reach part of the corpus." Those two situations should lead to very different next actions, and a response with no provenance attached collapses them into the same thing. The response needs enough metadata to say whether every intended shard participated, which index version served the query, whether any replica involved was stale, and whether the result was produced under degraded conditions at all.

The same logic applies on the write side. If the durable log is healthy but indexing has fallen behind, the system can keep accepting writes while its searchable visibility SLO quietly breaches. If compaction is behind schedule, it can preserve availability while index quality or storage keeps degrading underneath it. The contract needs to define where the line sits between healthy, degraded, and "the system should refuse new work rather than keep violating its own correctness guarantees."

### Writing the contract as an engineering artifact

The production contract needs to be explicit enough to drive design reviews, capacity planning, failure testing, and incident response — not a shared assumption everyone on the team happens to hold in their head the same way, until the day they don't.

A useful contract does not say new memories become searchable "quickly." It states a percentile visibility objective and the conditions it applies under. It does not say retrieval is "high recall." It names the evaluation set, the Recall@k target, and what degraded-state expectations are. It does not say tenants are "isolated." It names the trusted enforcement point, the physical sharing model, and the blast radius if that enforcement ever fails.

Everything that follows in this chapter — the distributed read path, index selection, data placement, the write path, deletions, filtered search, multi-tenancy, and observability — is the engineering that makes a specific, written answer to these questions stay true while the corpus keeps changing and pieces of the infrastructure keep failing.

---

## The production contract: correctness before performance

_PLACEHOLDER — workload envelope, searchable visibility latency, retrieval/state/authorization correctness as three separate obligations._

---

## The distributed read path and the retrieval error budget

_PLACEHOLDER — trace one query through identity resolution, embedding, routing, replica selection, metadata planning, ANN traversal, refinement, cross-shard merge, reranking. Introduce the retrieval error budget concept._

---

## ANN index selection under real workload constraints

_PLACEHOLDER — HNSW vs. IVF vs. DiskANN as workload-driven tradeoffs (memory, recall, build time, concurrent writes), quantization as part of the error budget._

---

## Data placement, routing, replication, and topology evolution

_PLACEHOLDER — sharding strategies (tenant/hash/semantic), hot-shard mitigation, replica placement vs. sharding as distinct concerns, online rebalancing, partial-result semantics under failure. Hospital-routing analogy._

---

## The write path and searchable visibility

_PLACEHOLDER — write states (accepted, durable, embedded, replicated, indexed, compacted), acknowledgment contract, agent read-your-writes problem, WAL, mutable/immutable segments, compute-storage separation, backpressure._

---

## Updates, deletions, index aging, and non-model migrations

_PLACEHOLDER — tombstones, compaction, index aging as a quality problem not just a storage problem, Ghost Vectors research (properly calibrated), logical invisibility vs. physical erasure, brief subsection on non-model migrations (index params, quantization scheme, schema) with cross-reference to Chapter 2 for embedding-model migrations._

---

## Filtered search as adaptive query planning

_PLACEHOLDER — pre-filter vs. post-filter vs. iterative filtering as selectivity-driven execution strategies, authorization predicates as trusted-infrastructure-only, graph navigation vs. result eligibility distinction._

---

## Multi-tenant isolation, authorization, and noisy-neighbor control

_PLACEHOLDER — opens with: the LLM/agent/reranker is never an authorization boundary. Shared index vs. tenant-aware shards vs. dedicated indexes vs. dedicated deployments. Tiered tenancy. Quotas and admission control._

---

## Semantic observability, failure testing, and operational readiness

_PLACEHOLDER — infrastructure telemetry vs. semantic telemetry (mirrors Ch2's vital-signs framing). Production verification checklist. Closes by returning to the Section 1 contract._

---

_Next: [Chapter 4 — Chunking, Context Construction, and Document Pipelines](https://claude.ai/agentic/04-chunking-context)_