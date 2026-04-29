import React from 'react';
import type { ReactNode } from 'react';

import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import styles from "./index.module.css";

interface GuideCardProps {
  tag: string;
  title: string;
  description: string;
  pages: string;
  chapters: string;
  audience: string;
  to: string;
  status: "live" | "coming-soon";
}

function GuideCard({
  tag,
  title,
  description,
  pages,
  chapters,
  audience,
  to,
  status,
}: GuideCardProps): ReactNode {
  const isLive = status === "live";

  const cardContent = (
    <>
      <div className="guide-card__tag">{tag}</div>
      <div className="guide-card__title">{title}</div>
      <div className="guide-card__description">{description}</div>
      <div className="guide-card__meta">
        <span>📄 {pages} pages</span>
        <span>◎ {chapters} chapters</span>
        <span>👤 {audience}</span>
        {!isLive && (
          <span style={{ marginLeft: "auto" }}>
            <span className={styles.comingSoonBadge}>In Progress</span>
          </span>
        )}
      </div>
      {isLive && <div className={styles.cardCta}>Read the guide →</div>}
    </>
  );

  if (!isLive) {
    return (
      <div className={`guide-card ${styles.guideCardDimmed}`}>
        {cardContent}
      </div>
    );
  }

  return (
    <Link to={to} className="guide-card">
      {cardContent}
    </Link>
  );
}

function Hero(): ReactNode {
  return (
    <div className={styles.hero}>
      <div className={styles.heroInner}>

        <div className={styles.heroText}>
          <p className={styles.heroEyebrow}>LLM Inference · GPU Systems · Production AI</p>
          <h1 className={styles.heroTitle}>
            LLM Inference &<br />
            ML Infrastructure
          </h1>
          <p className={styles.heroSubtitle}>
            Technical field guides written from production experience. Each guide is a comprehensive decision framework
            you can apply to production systems.
          </p>
          <div className={styles.heroLinks}>
            <Link
              to="https://github.com/vinay-jayanna"
              className={styles.heroLinkSecondary}
            >
              GitHub
            </Link>
            <Link
              to="https://linkedin.com/in/vinayjayanna"
              className={styles.heroLinkSecondary}
            >
              LinkedIn
            </Link>
            <Link to="/about" className={styles.heroLinkSecondary}>
              About
            </Link>
          </div>
        </div>

        <div className={styles.heroBio}>
          <p className={styles.heroBioLabel}>About the author</p>
          <p>
            Currently a Staff ML Engineer leading LLM inference optimization
            for one of the most consequential AI systems in the world -
            reaching hundreds of millions of users. Before that, spent nearly
            a decade at <strong>AWS</strong> building and scaling core inference
            infrastructure for <strong>SageMaker</strong> from its earliest
            days. Founded <strong>Vipas.AI</strong>, an AI inference marketplace
            that reached 25K daily visitors and received a VC term sheet.
            Earlier career spans building large-scale distributed systems and
            cloud infrastructure at Ericsson, Pegasystems, and global
            enterprises. Holder of a USPTO-pending patent in dynamic
            hierarchical storage and GPU optimization for LLM serving.
          </p>
        </div>

      </div>
    </div>
  );
}

function GuidesSection(): ReactNode {
  return (
    <section className={styles.guidesSection}>
      <div className={styles.sectionInner}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Field Guides</h2>
          <p className={styles.sectionSubtitle}>
            Each guide is a complete treatment of a production ML systems topic
            - not a survey, not a tutorial, but a decision framework you can
            apply directly to real deployments.
          </p>
        </div>
        <div className={styles.guidesGrid}>
          <GuideCard
            tag="Field Guide · v1.1"
            title="Sizing LLM Inference for Production"
            description="Inference is where AI infrastructure spend compounds indefinitely with usage growth. Most production LLM fleets are paying 2–3× what they need to - not from hardware limits, but from sizing decisions made without a disciplined framework. This guide covers the complete decision sequence: workload characterization, GPU memory sizing, parallelism strategy, KV cache optimization, and fleet sizing from the latency-throughput curve."
            pages="107"
            chapters="9"
            audience="Staff / Principal ML Engineers"
            to="/sizing"
            status="live"
          />
          <GuideCard
            tag="Field Guide · In Progress"
            title="Agentic Systems in Production"
            description="Production architecture for multi-agent LLM systems: orchestration patterns, tool reliability, memory and state management, latency budgets, failure modes, observability, and cost control. The guide that bridges research prototypes and production deployments."
            pages="100+"
            chapters="10"
            audience="Staff / Principal ML Engineers"
            to="/agentic"
            status="coming-soon"
          />
        </div>
      </div>
    </section>
  );
}

function WritingSection(): ReactNode {
  const articles = [
    {
      title: "The Great LLM Inference Showdown: TensorRT-LLM vs vLLM",
      meta: "LinkedIn · 2024",
      to: "https://www.linkedin.com/pulse/great-llm-inference-showdown-tensorrt-llm-vs-vllm-vinay-jayanna-9o9pc",
    },
    {
      title: "The Challenge of Production LLM Serving: A Ray Serve Perspective",
      meta: "LinkedIn · 2024",
      to: "https://www.linkedin.com/pulse/challenge-production-llm-serving-ray-serve-vinay-jayanna-08syc",
    },
    {
      title: "Multi-Tenant LLM Inference: Bridging Research and Reality",
      meta: "LinkedIn · 2024",
      to: "https://www.linkedin.com/pulse/multi-tenant-llm-inference-bridging-research-reality-vinay-jayanna-lpmtc",
    },
    {
      title: "KV Cache: The Hidden Optimization Behind Real-Time AI",
      meta: "LinkedIn · 2024",
      to: "https://www.linkedin.com/pulse/kv-cache-hidden-optimization-behind-real-time-ai-vinay-jayanna-cvfec",
    },
    {
      title: "The Infrastructure No One Talks About: How Vector Search Makes Gen AI Work",
      meta: "LinkedIn · 2024",
      to: "https://www.linkedin.com/pulse/infrastructure-one-talks-how-vector-search-makes-gen-ai-vinay-jayanna-ka2bc",
    },
  ];

  return (
    <section className={styles.writingSection}>
      <div className={styles.sectionInner}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Writing</h2>
          <p className={styles.sectionSubtitle}>
            Technical articles on LLM inference, ML infrastructure, and production AI systems.
          </p>
        </div>
        <div className={styles.writingGrid}>
          {articles.map((a) => (
            <Link key={a.to} to={a.to} className={styles.writingCard}>
              <div className={styles.writingCardTitle}>{a.title}</div>
              <div className={styles.writingCardMeta}>{a.meta}</div>
            </Link>
          ))}
        </div>
        <div style={{ marginTop: '1.5rem' }}>
          <Link
            to="https://www.linkedin.com/in/vinayjayanna/recent-activity/articles/"
            style={{ fontFamily: 'Inter,system-ui,sans-serif', fontSize: '13.5px', color: 'var(--ifm-color-primary)', textDecoration: 'none', fontWeight: 600 }}
          >
            All articles on LinkedIn →
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Technical field guides on LLM inference systems and production ML infrastructure by Vinay Jayanna - Staff ML Engineer."
    >
      <Hero />
      <main>
        <GuidesSection />
        <WritingSection />
      </main>
    </Layout>
  );
}