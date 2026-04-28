import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  sizingSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: '← Back to Field Guides',
    },
    {
      type: 'html',
      value: '<div class="sidebar-guide-title">Sizing LLM Inference Systems at Scale</div>',
    },
    {
      type: 'doc',
      id: 'preface',
      label: 'Preface',
    },
    {
      type: 'category',
      label: 'Why This Matters',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'why-inference-sizing',
          label: 'Inference as Capital Allocation',
        },
      ],
    },
    {
      type: 'category',
      label: 'Part I — Foundations',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: '01-workload-characterization',
          label: '1. Workload Characterization',
        },
        {
          type: 'doc',
          id: '02-gpu-memory-sizing',
          label: '2. GPU Memory Sizing',
        },
        {
          type: 'doc',
          id: '03-roofline-model',
          label: '3. The Roofline Model',
        },
      ],
    },
    {
      type: 'category',
      label: 'Part II — Optimization',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: '04-quantization',
          label: '4. Quantization',
        },
        {
          type: 'doc',
          id: '05-parallelism-strategy',
          label: '5. Parallelism Strategy',
        },
        {
          type: 'doc',
          id: '06-batching-strategy',
          label: '6. Batching Strategy',
        },
        {
          type: 'doc',
          id: '07-kv-cache-optimization',
          label: '7. KV Cache Optimization',
        },
      ],
    },
    {
      type: 'category',
      label: 'Part III — Production',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: '08-latency-throughput-curve',
          label: '8. Latency-Throughput Curve',
        },
        {
          type: 'doc',
          id: '09-sizing-algorithm',
          label: '9. Sizing Algorithm & Monitoring',
        },
      ],
    },
    {
      type: 'html',
      value: '<div class="sidebar-divider"></div>',
    },
    {
      type: 'doc',
      id: 'first-principles',
      label: 'First Principles, Last',
    },
  ],
};

export default sidebars;