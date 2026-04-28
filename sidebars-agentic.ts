import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  agenticSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: '← Back to Field Guides',
    },
    {
      type: 'html',
      value: '<div class="sidebar-guide-title">Agentic Systems in Production</div>',
    },
    {
      type: 'category',
      label: 'Coming Soon',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'index',
          label: 'Overview & Roadmap',
        },
      ],
    },
  ],
};

export default sidebars;