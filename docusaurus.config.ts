import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import type * as Plugin from '@docusaurus/types/src/plugin';

const config: Config = {
  title: 'Vinay Jayanna',
  tagline: 'LLM Inference · ML Infrastructure · Distributed Systems',
  favicon: 'img/favicon.ico',

  url: 'https://vinayj.com',
  baseUrl: '/',

  organizationName: 'vinay-jayanna',
  projectName: 'vinayj-site',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'sizing',
          sidebarPath: './sidebars.ts',
          showLastUpdateTime: true,
          showLastUpdateAuthor: false,
          breadcrumbs: true,
          editUrl: undefined,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.8,
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'agentic',
        path: 'agentic',
        routeBasePath: 'agentic',
        sidebarPath: './sidebars-agentic.ts',
        showLastUpdateTime: true,
        showLastUpdateAuthor: false,
        breadcrumbs: true,
        editUrl: undefined,
      } satisfies Plugin.PluginOptions,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',

    metadata: [
      {
        name: 'description',
        content:
          'Field guides on LLM inference systems and production ML infrastructure by Vinay Jayanna — Staff ML Engineer, AWS SageMaker founding team.',
      },
      {
        name: 'keywords',
        content:
          'LLM inference, GPU sizing, KV cache, quantization, parallelism, ML infrastructure, vLLM, TensorRT-LLM',
      },
    ],

    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Vinay Jayanna',
      hideOnScroll: false,
      logo: {
        alt: 'Vinay Jayanna',
        src: 'img/logo.svg',
      },
      items: [
        {
          label: 'Field Guides',
          position: 'left',
          items: [
            {
              label: 'Sizing LLM Inference Systems',
              to: '/sizing',
            },
            {
              label: 'Agentic Systems in Production',
              to: '/agentic',
              // Remove this line when guide 2 is ready:
              className: 'navbar-item-coming-soon',
            },
          ],
        },
        {
          to: '/about',
          label: 'About',
          position: 'left',
        },
        {
          href: 'https://github.com/vinay-jayanna',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://linkedin.com/in/vinayjayanna',
          label: 'LinkedIn',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Field Guides',
          items: [
            {
              label: 'Sizing LLM Inference Systems at Scale',
              to: '/sizing',
            },
            {
              label: 'Agentic Systems in Production',
              to: '/agentic',
            },
          ],
        },
        {
          title: 'Writing',
          items: [
            {
              label: 'LinkedIn Articles',
              href: 'https://linkedin.com/in/vinayjayanna',
            },
            {
              label: 'Substack',
              href: 'https://substack.com/@vinayjayanna',
            },
          ],
        },
        {
          title: 'Connect',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/vinay-jayanna',
            },
            {
              label: 'LinkedIn',
              href: 'https://linkedin.com/in/vinayjayanna',
            },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Vinay Jayanna. All rights reserved.`,
    },

    prism: {
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: [
        'bash',
        'python',
        'yaml',
        'json',
        'typescript',
        'docker',
      ],
    },

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: false,
      },
    },

    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;