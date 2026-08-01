---
id: 03-vector-database
title: "Vector Database Architecture, Scaling, and Real-Time Indexing"
---
# Vector Database Architecture, Scaling, and Real-Time Indexing

<div class="chapter-summary"> The previous chapter treated the vector index as a black box — the application submitted a query, got back a ranked candidate set, and moved on. In production, that black box is part of the agent's state and correctness boundary. It may hold enterprise knowledge, episodic memory, prior tool results, workflow observations, and evidence scoped to specific users, tenants, and permissions. A stale or wrongly-scoped result does not just weaken an answer — it can cause an agent to repeat an action, act on revoked access, ignore work that already completed, or build a multi-step plan on a state that stopped being true minutes ago. This chapter is the architecture that keeps retrieval correct while the corpus changes, traffic goes uneven, and infrastructure fails. </div>

---

## The production contract: correctness before performance

Most teams start a vector database design by comparing [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/), [IVF](https://milvus.io/ai-quick-reference/how-do-inverted-file-ivf-indexes-work-in-vector-databases-and-what-role-do-clustering-centroids-play-in-the-search-process), and disk-oriented indexes. That is usually too early. An index is a mechanism, not a requirement, and choosing the mechanism before defining what the system must guarantee is the most common reason architectures that work cleanly in a proof of concept become unpredictable under real load.

A structural engineer does not pick steel, concrete, or timber and then figure out what the building needs to withstand. The design starts with occupancy, wind, seismic zone, and fire code — the material choice falls out of those constraints almost mechanically. Vector database architecture works the same way. HNSW, IVF, [DiskANN](https://milvus.io/blog/diskann-explained.md), sharding, and replication are materials. The production contract is the load spec.

The contract has to define more than expected scale. It has to define what the system means by a successful write, a complete query, an authorized result, an acceptable stale read, and a recoverable failure. Without those definitions, every component can behave exactly as its local design intends while the system as a whole violates what the application assumed was true.

### The workload envelope

The workload envelope describes the range of conditions the retrieval system must keep its guarantees under — sized to the planning horizon, not to the smaller, cleaner system that exists on launch day.

The term is borrowed deliberately from aviation. An aircraft is not certified because it can hit a particular top speed. It has to remain controllable across every combination of speed, altitude, weight, and turbulence it might encounter — a configuration that is safe at one corner of that envelope can be dangerous at another. Vector systems behave the same way. Corpus size, concurrency, mutation rate, filter selectivity, and tenant skew cannot be sized independently, because production will hand you every combination of them at once, usually at the worst possible time.

To stay concrete, picture two systems. The first serves a governed archive of legal and regulatory filings — steady inflow, mostly immutable, every record reconstructible from an authoritative source. The second serves memory for a production agent — continuous writes, permissions that can change mid-session, memories that range from minutes-relevant to months-relevant. At the level of "store embeddings, run nearest-neighbor search," they look identical. Their operating envelopes are close to opposites. Neither profile below is a fixed law of either domain — a regulatory change can make the archive intensely hot overnight, and an agent's durable preferences can stay warm for months. Treat them as illustrative pulls in different directions, not permanent labels.

**Logical live set, physical population, and deployed footprint — three numbers, not one.** The application sees the logical live set: records that should currently be retrievable. The index sees something larger: the physical population, which adds tombstoned nodes, superseded versions, and segments mid-compaction. The cluster is larger still — the deployed footprint multiplies that physical population across every replica, region, and snapshot. Think of a warehouse. The sales system reports a million items available — that's the logical live set. Returned stock, damaged pallets, and inventory awaiting write-off still occupy shelf space and slow down picking — that's the physical population. Operating three regional warehouses holding copies of all of it is the deployed footprint. In the archive, the first two numbers stay close — little gets deleted. In agent memory, they diverge fast, because expired observations sit as tombstones until compaction reclaims them. The _gap_ between logical and physical is a production signal on its own: a widening gap means compaction is falling behind, well before anything else tells you that.

**The real memory bill, not the encoding estimate.** `vector_count × dimensions × bytes_per_element` prices the vector array. It says nothing about graph edges, metadata indexes, allocator overhead, write buffers, and compaction headroom — which routinely outweigh the vectors themselves, and which replication then multiplies again.

**Temperature, described as a distribution, not a label.** Some vectors get hit constantly; most don't. This governs whether a cache does anything useful and whether a disk-oriented index stays inside its latency budget. The archive might run cold and even — until active litigation makes one narrow slice intensely hot. Agent memory usually runs hot on recent writes, but durable facts and user preferences can stay warm indefinitely. What the contract needs isn't "hot" or "cold" as an adjective — it's what fraction of traffic concentrates in the hottest slice, and how fast that slice moves. Average QPS hides the same failure: a service comfortably averaging 1,000 QPS can fall over during a five-second burst to 5,000, and one query touching a single shard costs nothing like one that fans out across dozens of partitions.

**Mutation as its own workload.** A library catalogue can be reorganized on a slow, periodic schedule and nobody notices. An airport departure board is only useful if it reflects right now — a five-minute lag changes its meaning entirely. The archive behaves like the catalogue: optimize for stable structure, infrequent writes. Agent memory behaves like the departure board: it is being rewritten, expired, and re-permissioned continuously, by the same system serving live queries against the same data. Insert rate alone doesn't capture this — replacing a vector, flipping one access-control field, and bulk-loading historical data each stress a different part of the system, and the envelope needs to name each one.

**Filter selectivity as a range, not an average.** The same physical index might see a query spanning the entire corpus and, a moment later, one scoped to a single user's last hour of activity — a swing from hundreds of millions of eligible vectors to a few dozen. What the system needs isn't the average selectivity; it's how often each end of that range shows up, because each end favors a different execution plan — the subject of its own section later in this chapter.

**Tenant skew and geography.** Tenant count alone says nothing. Ten evenly-sized tenants partition predictably by almost any method. A thousand tenants where a handful account for most of the data create hot shards and noisy-neighbor pressure no matter how correct the partitioning logic is. Data residency and regional latency requirements can decide physical placement before index performance is even part of the conversation.

**RTO and RPO — and precisely whose data they cover.** Recovery time objective is how long the system can stay degraded. Recovery point objective is how much acknowledged data can be lost. The archive's _index_ can often tolerate a loose RPO, because it's a derived representation — if source documents and ingestion checkpoints are durable elsewhere, the index rebuilds. That does not mean the underlying legal data tolerates loss; it means the vector subsystem specifically is recoverable. Get this scope wrong and you'll relax a durability guarantee on the wrong layer. Agent memory has less room here — if the last hour of observations exists only in the memory service, losing it silently erases part of what the agent believes has happened, and it may repeat completed work or reason from a gap it doesn't know is there.

The envelope isn't a spreadsheet of maximum values to hit. It's the territory inside which retrieval, state, and authorization correctness all have to keep holding. A system tuned for the archive will misbehave against agent-memory traffic. A system built for continuous agent memory will impose needless complexity and cost on a mostly-immutable archive. Filling this out honestly — before an index choice makes the decision for you by default — is what the rest of this chapter assumes you've done.

| Dimension                                                   | What to specify                                        |
| ----------------------------------------------------------- | ------------------------------------------------------ |
| Logical live set → physical population → deployed footprint | how far apart, and why                                 |
| Memory driver                                               | vectors, or graph + metadata + replicas                |
| Hottest slice of traffic                                    | what % of queries, how fast it shifts                  |
| Mutation types                                              | insert / replace / expire / re-permission — separately |
| Filter selectivity range                                    | narrowest and broadest cases seen, not the average     |
| Tenant skew                                                 | top-N tenants' share of data and traffic               |
| RTO / RPO, scoped                                           | which layer specifically — index, or source of truth   |

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