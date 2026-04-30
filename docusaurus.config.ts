import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import type * as Plugin from "@docusaurus/types/src/plugin";

const config: Config = {
  title: "Vinay Jayanna",
  tagline: "LLM Inference · ML Infrastructure · Distributed Systems",
  favicon: "img/site/favicon.svg",          // ← was: img/favicon.svg

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

  // Mermaid diagram support (free, built into Docusaurus)
  markdown: {
    mermaid: true,
  },
  themes: ["@docusaurus/theme-mermaid"],

  presets: [
    [
      "classic",
      {
        docs: {
          path: "content/sizing",            // ← was: docs
          routeBasePath: "sizing",
          sidebarPath: "./sidebars-sizing.ts", // ← was: sidebars.ts
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
        path: "content/agentic",            // ← was: agentic
        routeBasePath: "agentic",
        sidebarPath: "./sidebars-agentic.ts",
        showLastUpdateTime: true,
        showLastUpdateAuthor: false,
        breadcrumbs: true,
        editUrl: undefined,
      } satisfies Plugin.PluginOptions,
    ],
    // Future field guides follow the same pattern:
    // [
    //   "@docusaurus/plugin-content-docs",
    //   {
    //     id: "platform",
    //     path: "content/platform",
    //     routeBasePath: "platform",
    //     sidebarPath: "./sidebars-platform.ts",
    //     ...
    //   }
    // ],
  ],

  themeConfig: {
    image: "img/site/docusaurus-social-card.jpg",

    mermaid: {
      theme: { light: "neutral", dark: "dark" },
    },

    metadata: [
      {
        name: "description",
        content:
          "Field guides on LLM inference, production RAG, and agentic systems by Vinay Jayanna — Staff ML Engineer, AWS SageMaker founding team.",
      },
      {
        name: "keywords",
        content:
          "LLM inference, GPU sizing, KV cache, quantization, parallelism, ML infrastructure, vLLM, TensorRT-LLM, RAG, agentic systems, production AI, vector database",
      },
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
        src: "img/site/logo.svg",            // ← was: img/logo.svg
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
              label: "Production RAG and Agentic Systems",
              to: "/agentic",
            },
            // Add future guides here
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
              label: "Production RAG and Agentic Systems",
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
