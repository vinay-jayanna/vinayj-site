import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import type * as Plugin from "@docusaurus/types/src/plugin";

const config: Config = {
  title: "Vinay Jayanna",
  tagline: "LLM Inference · ML Infrastructure · Distributed Systems",
  favicon: "img/site/favicon.svg",

  url: "https://vinayj.com",
  baseUrl: "/",

  organizationName: "vinay-jayanna",
  projectName: "vinayj-site",
  deploymentBranch: "gh-pages",
  trailingSlash: false,
  onBrokenLinks: "ignore",
  onBrokenMarkdownLinks: "ignore",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          path: "content/sizing",
          routeBasePath: "sizing",
          sidebarPath: "./sidebars-sizing.ts",
          showLastUpdateTime: true,
          showLastUpdateAuthor: false,
          breadcrumbs: true,
          editUrl: undefined,
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
        sitemap: {
          changefreq: "weekly",
          priority: 0.8,
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "agentic",
        path: "content/agentic",
        routeBasePath: "agentic",
        sidebarPath: "./sidebars-agentic.ts",
        showLastUpdateTime: true,
        showLastUpdateAuthor: false,
        breadcrumbs: true,
        editUrl: undefined,
      } satisfies Plugin.PluginOptions,
    ],
  ],

  themeConfig: {
    image: "img/site/docusaurus-social-card.jpg",

    metadata: [
      {
        name: "description",
        content:
          "Field guides on LLM inference, production RAG, and agentic systems by Vinay Jayanna — Staff ML Engineer, AWS SageMaker founding team.",
      },
      {
        name: "keywords",
        content:
          "LLM inference, GPU sizing, KV cache, quantization, parallelism, ML infrastructure, vLLM, TensorRT-LLM, RAG, agentic systems, production AI, vector database, observability, reliability",
      },
      { property: "og:type",         content: "website" },
      { property: "og:site_name",    content: "Vinay Jayanna" },
      { property: "og:image",        content: "https://vinayj.com/img/covers/sizing/cover-social-sizing.jpg" },
      { property: "og:image:width",  content: "1200" },
      { property: "og:image:height", content: "630" },
      { name: "twitter:card",        content: "summary_large_image" },
      { name: "twitter:site",        content: "@vinayjayanna" },
      { name: "twitter:image",       content: "https://vinayj.com/img/covers/sizing/cover-social-sizing.jpg" },
    ],

    colorMode: {
      defaultMode: "light",
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: "Vinay Jayanna",
      hideOnScroll: false,
      logo: {
        alt: "Vinay Jayanna",
        src: "img/site/logo.svg",
      },
      items: [
        {
          label: "Field Guides",
          position: "left",
          items: [
            {
              label: "Sizing LLM Inference for Production",
              to: "/sizing",
            },
            {
              label: "Agentic Systems in Production",
              to: "/agentic",
            },
          ],
        },
        {
          to: "/about",
          label: "About",
          position: "left",
        },
        {
          href: "https://github.com/vinay-jayanna",
          label: "GitHub",
          position: "right",
        },
        {
          href: "https://linkedin.com/in/vinayjayanna",
          label: "LinkedIn",
          position: "right",
        },
      ],
    },

    footer: {
      style: "dark",
      links: [
        {
          title: "Field Guides",
          items: [
            {
              label: "Sizing LLM Inference for Production",
              to: "/sizing",
            },
            {
              label: "Agentic Systems in Production",
              to: "/agentic",
            },
          ],
        },
        {
          title: "Writing",
          items: [
            {
              label: "LinkedIn Articles",
              href: "https://www.linkedin.com/in/vinayjayanna/recent-activity/articles/",
            },
            {
              label: "Substack",
              href: "https://substack.com/@vinayjayanna",
            },
          ],
        },
        {
          title: "Connect",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/vinay-jayanna",
            },
            {
              label: "LinkedIn",
              href: "https://linkedin.com/in/vinayjayanna",
            },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Vinay Jayanna. All rights reserved.`,
    },

    prism: {
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ["bash", "python", "yaml", "json", "typescript", "docker"],
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
