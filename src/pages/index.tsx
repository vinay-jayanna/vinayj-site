import React from "react";
import type { ReactNode } from "react";

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
          <p className={styles.heroEyebrow}>
            LLM Inference · GPU Systems · Production AI
          </p>
          <h1 className={styles.heroTitle}>
            LLM Inference &<br />
            ML Infrastructure
          </h1>
          <p className={styles.heroSubtitle}>
            Rigorous field guides for designing, sizing, and operating
            large-scale AI systems. Each guide turns complex infrastructure
            decisions into practical engineering frameworks.
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
            <strong>Vinay Jayanna</strong> is a Staff/Principal Machine Learning
            Engineer specializing in LLM inference optimization, GPU capacity
            planning, and production AI infrastructure. He currently leads
            inference optimization and GenAI platform architecture for
            large-scale AI systems reaching hundreds of millions of users.
            Previously, he helped build and scale core inference infrastructure
            for <strong>Amazon SageMaker</strong>, founded{" "}
            <strong>Vipas.AI</strong>, and is the inventor on a pending U.S.
            patent covering model caching, hierarchical storage, and GPU
            optimization for LLM serving.
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
            Comprehensive decision frameworks for real production
            systems—covering the architecture, trade-offs, calculations, and
            operating decisions that short tutorials leave out.
          </p>
        </div>
        <div className={styles.guidesGrid}>
          <GuideCard
            tag="Field Guide · v1.1"
            title="Sizing LLM Inference for Production"
            description="Inference is where AI infrastructure cost compounds with usage growth. Many production LLM fleets remain substantially overprovisioned—not because of fundamental hardware limits, but because capacity decisions are made without a disciplined sizing framework. This guide covers the complete decision sequence: workload characterization, GPU memory sizing, parallelism strategy, KV cache optimization, operating point selection, and fleet sizing from the latency-throughput curve."
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
      title: "Multi-Tenant LLM Inference: Bridging Research and Reality",
      meta: "LinkedIn · 2024",
      to: "https://www.linkedin.com/pulse/multi-tenant-llm-inference-bridging-research-reality-vinay-jayanna-lpmtc",
    },
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
      title: "KV Cache: The Hidden Optimization Behind Real-Time AI",
      meta: "LinkedIn · 2024",
      to: "https://www.linkedin.com/pulse/kv-cache-hidden-optimization-behind-real-time-ai-vinay-jayanna-cvfec",
    },
    {
      title:
        "The Infrastructure No One Talks About: How Vector Search Makes Gen AI Work",
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
            Technical articles on LLM inference, ML infrastructure, and
            production AI systems.
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
        <div style={{ marginTop: "1.5rem" }}>
          <Link
            to="https://www.linkedin.com/in/vinayjayanna/recent-activity/articles/"
            style={{
              fontFamily: "Inter,system-ui,sans-serif",
              fontSize: "13.5px",
              color: "var(--ifm-color-primary)",
              textDecoration: "none",
              fontWeight: 600,
            }}
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
      description="Technical field guides on LLM inference, GPU capacity planning, and production AI infrastructure by Vinay Jayanna, Staff/Principal ML Engineer."
    >
      <Hero />
      <main>
        <GuidesSection />
        <WritingSection />
      </main>
    </Layout>
  );
}
