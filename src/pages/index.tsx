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
          <p className={styles.heroEyebrow}>Staff ML Engineer · Field Guides</p>
          <h1 className={styles.heroTitle}>
            LLM Inference &<br />
            ML Infrastructure
          </h1>
          <p className={styles.heroSubtitle}>
            Long-form technical field guides for Staff and Principal engineers
            building production AI systems. Written from first principles,
            grounded in real deployment constraints.
          </p>
          <div className={styles.heroBio}>
            <p>
              Led engineering on <strong>AWS SageMaker</strong> from the 2017
              launch through eight years of scale — driving AI inference
              infrastructure for thousands of enterprise customers. Founded{" "}
              <strong>Vipas.AI</strong>, an AI inference marketplace that
              reached 25K daily visitors and received a VC term sheet. Currently
              leading LLM inference optimization and GenAI platform engineering
              at scale. Holder of a USPTO-pending patent in dynamic hierarchical
              storage and GPU optimization for LLM serving.
            </p>
          </div>
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
        <div className={styles.heroStats}>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>107</span>
            <span className={styles.statLabel}>pages, guide 1</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>9</span>
            <span className={styles.statLabel}>chapters</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>13</span>
            <span className={styles.statLabel}>step sizing algorithm</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>17+</span>
            <span className={styles.statLabel}>years ML infrastructure</span>
          </div>
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
            — not a survey, not a tutorial, but a decision framework you can
            apply directly to real deployments.
          </p>
        </div>
        <div className={styles.guidesGrid}>
          <GuideCard
            tag="Field Guide · v1.0"
            title="Sizing LLM Inference Systems at Scale"
            description="A complete framework for GPU capacity planning: workload characterization, memory budgeting, roofline analysis, quantization, parallelism, batching, KV cache optimization, and a 13-step sizing algorithm. Written for engineers who need to produce a GPU count and cost estimate they can stand behind."
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
  return (
    <section className={styles.writingSection}>
      <div className={styles.sectionInner}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Writing</h2>
          <p className={styles.sectionSubtitle}>
            Shorter technical pieces on LinkedIn and Substack.
          </p>
        </div>
        <div className={styles.writingGrid}>
          <Link
            to="https://linkedin.com/in/vinayjayanna"
            className={styles.writingCard}
          >
            <div className={styles.writingCardTitle}>
              TensorRT-LLM vs. vLLM: A Production Comparison
            </div>
            <div className={styles.writingCardMeta}>LinkedIn · 2024</div>
          </Link>
          <Link
            to="https://linkedin.com/in/vinayjayanna"
            className={styles.writingCard}
          >
            <div className={styles.writingCardTitle}>
              Ray Serve for Production LLM Serving
            </div>
            <div className={styles.writingCardMeta}>LinkedIn · 2024</div>
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
      description="Field guides on LLM inference systems and production ML infrastructure by Vinay Jayanna."
    >
      <Hero />
      <main>
        <GuidesSection />
        <WritingSection />
      </main>
    </Layout>
  );
}
