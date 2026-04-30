import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  agenticSidebar: [
    { type: 'doc', id: 'index', label: 'Overview & Download' },

    { type: 'html', value: '<div class="sidebar-part-label">Part I · Why These Systems Are Hard</div>' },
    { type: 'doc', id: 'why-rag-agentic-hard',    label: '→ Why RAG and Agentic Systems Break at Scale' },
    { type: 'doc', id: '01-first-principles',      label: '1 · First Principles: Retrieval, Generation, and Agency' },

    { type: 'html', value: '<div class="sidebar-part-label">Part II · Retrieval Infrastructure</div>' },
    { type: 'doc', id: '02-embedding-models',      label: '2 · Embedding Models and Retrieval Quality' },
    { type: 'doc', id: '03-vector-database',       label: '3 · Vector Database Architecture and Scaling' },
    { type: 'doc', id: '04-chunking-context',      label: '4 · Chunking, Context Construction, and Document Pipelines' },
    { type: 'doc', id: '05-hybrid-search',         label: '5 · Hybrid Search, Query Routing, and Query Understanding' },

    { type: 'html', value: '<div class="sidebar-part-label">Part III · RAG System Design</div>' },
    { type: 'doc', id: '06-rag-architectures',     label: '6 · RAG Architectures: From Naive to Production-Grade' },
    { type: 'doc', id: '07-rag-evaluation',        label: '7 · RAG Evaluation: Metrics That Survive Production' },
    { type: 'doc', id: '08-rag-cost-latency',      label: '8 · RAG Cost and Latency Optimization' },
    { type: 'doc', id: '09-rag-security',          label: '9 · RAG Security, Access Control, and Data Governance' },

    { type: 'html', value: '<div class="sidebar-part-label">Part IV · Agentic System Design</div>' },
    { type: 'doc', id: '10-agent-fundamentals',    label: '10 · Agent Fundamentals: The Production Engineering View' },
    { type: 'doc', id: '11-tool-design',           label: '11 · Tool Design, API Contracts, and Action Reliability' },
    { type: 'doc', id: '12-memory-state',          label: '12 · Memory, State, and Context Across Agent Turns' },
    { type: 'doc', id: '13-multi-agent',           label: '13 · Multi-Agent Orchestration and Failure Isolation' },
    { type: 'doc', id: '14-latency-cost',          label: '14 · Latency Budgets and Cost Control for Agentic Systems' },
    { type: 'doc', id: '15-agentic-security',      label: '15 · Agentic Security: Prompt Injection, Privilege, and Trust Boundaries' },

    { type: 'html', value: '<div class="sidebar-part-label">Part V · Operating at Scale</div>' },
    { type: 'doc', id: '16-observability',         label: '16 · Observability for Non-Deterministic Systems' },
    { type: 'doc', id: '17-continuous-improvement',label: '17 · Continuous Improvement: Feedback Loops and Online Learning' },
    { type: 'doc', id: '18-guardrails',            label: '18 · Production Guardrails, Content Safety, and Compliance' },
    { type: 'doc', id: '19-sizing',                label: '19 · Sizing RAG and Agentic Infrastructure' },
  ],
};

export default sidebars;
