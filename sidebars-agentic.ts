import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  agenticSidebar: [
    { type: 'doc', id: 'index', label: 'Overview & Download' },

    { type: 'html', value: '<div class="sidebar-part-label">Part I · The Production Landscape</div>' },
    { type: 'doc', id: 'why-rag-agentic-hard', label: '→ Why RAG and Agentic Systems Break at Scale' },
    { type: 'doc', id: '01-llm-gateway',        label: '1 · LLM Gateway and Multi-Provider Routing' },

    { type: 'html', value: '<div class="sidebar-part-label">Part II · High-Throughput Retrieval</div>' },
    { type: 'doc', id: '02-embedding-models',    label: '2 · Embedding Models and Retrieval Quality' },
    { type: 'doc', id: '03-vector-database',     label: '3 · Vector Database Architecture, Scaling, and Real-Time Indexing' },
    { type: 'doc', id: '04-chunking-context',    label: '4 · Chunking, Context Construction, and Document Pipelines' },
    { type: 'doc', id: '05-context-engineering', label: '5 · Context Engineering: Budget, Assembly, and Governance' },
    { type: 'doc', id: '06-hybrid-search',       label: '6 · Hybrid Search, Query Routing, and Semantic Caching' },

    { type: 'html', value: '<div class="sidebar-part-label">Part III · RAG System Design</div>' },
    { type: 'doc', id: '07-rag-architectures',   label: '7 · RAG Architectures: Production Failure Modes and Design Patterns' },
    { type: 'doc', id: '08-rag-evaluation',      label: '8 · RAG Evaluation: Metrics That Survive Production' },
    { type: 'doc', id: '09-rag-cost-security',   label: '9 · RAG Cost, Latency, and Security' },

    { type: 'html', value: '<div class="sidebar-part-label">Part IV · Agentic System Design</div>' },
    { type: 'doc', id: '10-agentic-rag',         label: '10 · Agentic RAG: When Retrieval and Agency Interleave' },
    { type: 'doc', id: '11-agent-loops',         label: '11 · Agent Loop Design: State Machines, Re-planning, and Failure Isolation' },
    { type: 'doc', id: '12-tool-mcp',            label: '12 · Tool Design, MCP, and the Agentic Protocol Layer' },
    { type: 'doc', id: '13-memory-state',        label: '13 · Memory, State, and Context Across Agent Turns' },
    { type: 'doc', id: '14-multi-agent',         label: '14 · Multi-Agent Orchestration and Failure Isolation' },
    { type: 'doc', id: '15-latency-cost-security', label: '15 · Latency Budgets, Cost Control, and Agentic Security' },

    { type: 'html', value: '<div class="sidebar-part-label">Part V · Reliability and Governance</div>' },
    { type: 'doc', id: '16-observability',       label: '16 · DAG-Based Observability for Non-Deterministic Systems' },
    { type: 'doc', id: '17-testing-cicd',        label: '17 · Deterministic Testing and Agentic CI/CD' },
    { type: 'doc', id: '18-continuous-improvement', label: '18 · Continuous Improvement: Feedback Loops and Online Learning' },
    { type: 'doc', id: '19-guardrails-sizing',   label: '19 · Production Guardrails, Compliance, and Sizing' },
  ],
};

export default sidebars;