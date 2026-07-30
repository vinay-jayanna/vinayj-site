import React from "react";
import type { ReactNode } from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import styles from "./about.module.css";

function Section({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}): ReactNode {
  return (
    <div className={styles.section}>
      <h2 className={styles.sectionTitle}>{title}</h2>
      {children}
    </div>
  );
}

function TimelineItem({
  period,
  role,
  org,
  description,
  tags,
}: {
  period: string;
  role: string;
  org: string;
  description: string;
  tags?: string[];
}): ReactNode {
  return (
    <div className={styles.timelineItem}>
      <div className={styles.timelineMeta}>
        <span className={styles.timelinePeriod}>{period}</span>
      </div>
      <div className={styles.timelineContent}>
        <div className={styles.timelineRole}>{role}</div>
        <div className={styles.timelineOrg}>{org}</div>
        <p className={styles.timelineDesc}>{description}</p>
        {tags && (
          <div className={styles.timelineTags}>
            {tags.map((tag) => (
              <span key={tag} className={styles.tag}>
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function About(): ReactNode {
  return (
    <Layout
      title="About - Vinay Jayanna"
      description="Staff/Principal ML Engineer specializing in LLM inference optimization, GPU capacity planning, and production AI infrastructure. Helped build and scale AWS SageMaker. Founded Vipas.AI."
    >
      <div className={styles.page}>
        <div className={styles.inner}>
          <div className={styles.header}>
            <div className={styles.headerText}>
              <p className={styles.headerEyebrow}>About</p>
              <h1 className={styles.headerTitle}>Vinay Jayanna</h1>
              <p className={styles.headerSubtitle}>
                Staff/Principal ML Engineer specializing in LLM inference
                optimization, GPU capacity planning, and the architectural
                decisions that determine whether production AI systems are fast,
                reliable, and economically sustainable at scale. Currently
                leading inference optimization and GenAI platform engineering
                for large-scale AI systems reaching hundreds of millions of
                users. I write field guides that turn complex production
                engineering problems into rigorous, practical frameworks.
              </p>
              <div className={styles.headerLinks}>
                <Link
                  to="https://github.com/vinay-jayanna"
                  className={styles.headerLink}
                >
                  GitHub →
                </Link>
                <Link
                  to="https://linkedin.com/in/vinayjayanna"
                  className={styles.headerLink}
                >
                  LinkedIn →
                </Link>
              </div>
            </div>
          </div>

          <div className={styles.body}>
            <Section title="Experience">
              <div className={styles.timeline}>
                <TimelineItem
                  period="2025 - Present"
                  role="Staff/Principal ML Engineer"
                  org="Apple"
                  description="Leading LLM inference optimization and GenAI platform architecture for large-scale production AI systems reaching hundreds of millions of users. Focus areas include GPU memory architecture, KV cache optimization, serving framework evaluation across vLLM and TensorRT-LLM, parallelism strategies for frontier-scale models, latency-throughput operating point selection, capacity and cost modeling, and production observability for LLM serving systems."
                  tags={[
                    "LLM Inference",
                    "GenAI Platform",
                    "GPU Optimization",
                    "Capacity Planning",
                    "vLLM",
                    "TensorRT-LLM",
                    "Ray Serve",
                  ]}
                />
                <TimelineItem
                  period="2024 - 2025"
                  role="Founder"
                  org="Vipas.AI - AI Inference Marketplace"
                  description="Founded and built an AI inference marketplace enabling model creators to host and monetize industry-specific AI models with IP protection, monitoring, and pay-per-prediction APIs. Grew to 25K daily visitors within 90 days. Received a VC term sheet. Selected into NVIDIA Inception, AWS Activate, and Google Cloud Scale programs."
                  tags={[
                    "Founder",
                    "LLM Serving",
                    "Inference Marketplace",
                    "Product",
                    "Go-to-Market",
                  ]}
                />
                <TimelineItem
                  period="~8 years"
                  role="AI Platform and Inference Engineering Leader"
                  org="AWS SageMaker - Amazon Web Services"
                  description="Built core inference infrastructure for AWS SageMaker from its earliest days - scaling it to serve production ML workloads for enterprises across finance, healthcare, and telecom. Drove the full ML platform stack: AI inference, training infrastructure, MLOps infra, and platform abstractions that helped shape managed ML infrastructure at scale."
                  tags={[
                    "AWS SageMaker",
                    "ML Platform",
                    "AI Inference",
                    "Distributed Systems",
                    "Enterprise Scale",
                  ]}
                />
                <TimelineItem
                  period="Earlier career"
                  role="Engineering Leader"
                  org="Ericsson · Pegasystems · Global Enterprises"
                  description="Built large-scale distributed systems and cloud infrastructure across global enterprises. Designed and delivered systems operating at significant scale - from real-time network infrastructure at Ericsson to enterprise platform engineering at Pegasystems. This work established the distributed systems foundations in my career that underpin everything that followed in AI infrastructure."
                  tags={[
                    "Distributed Systems",
                    "Cloud Infrastructure",
                    "Enterprise Engineering",
                    "Global Scale",
                  ]}
                />
              </div>
            </Section>

            <Section title="Technical Focus">
              <div className={styles.focusGrid}>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>
                    LLM Inference Optimization
                  </div>
                  <p className={styles.focusDesc}>
                    GPU memory architecture, KV cache design, quantization
                    strategies (FP8, INT4, GPTQ, AWQ), parallelism (tensor,
                    pipeline, data, expert), continuous and disaggregated
                    batching, speculative decoding. Operating point selection
                    from latency-throughput curves.
                  </p>
                </div>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>
                    GPU Capacity Planning & Cost
                  </div>
                  <p className={styles.focusDesc}>
                    Workload characterization, roofline analysis, fleet sizing
                    from first principles, TCO modeling, utilization trap
                    avoidance, heterogeneous fleet composition. Turning
                    benchmark results into GPU counts and cost estimates you can
                    defend.
                  </p>
                </div>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>
                    GenAI Platform Engineering
                  </div>
                  <p className={styles.focusDesc}>
                    LLM serving framework selection and evaluation, multi-LoRA
                    serving architectures, production guardrails, rate limiting
                    strategies, SLO-aware scheduling, and the operational
                    engineering that keeps GenAI systems reliable at scale.
                  </p>
                </div>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>Agentic AI Systems</div>
                  <p className={styles.focusDesc}>
                    Multi-agent orchestration patterns, tool reliability, memory
                    and state management across agent turns, latency budgets for
                    compound AI systems, failure mode analysis, and
                    observability for non-deterministic LLM pipelines.
                  </p>
                </div>
              </div>
            </Section>

            <Section title="Patents & Publications">
              <div className={styles.pubList}>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>USPTO Patent - Pending</div>
                  <div className={styles.pubTitle}>
                    Dynamic Hierarchical Storage and GPU Optimization for LLM
                    Serving
                  </div>
                  <p className={styles.pubDesc}>
                    A system and method for dynamic tiered memory management
                    across GPU VRAM, CPU DRAM, and NVMe storage for large
                    language model inference workloads, with adaptive scheduling
                    based on request priority and latency constraints.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>
                    Field Guide - v1.1 · 107 pages
                  </div>
                  <div className={styles.pubTitle}>
                    Sizing LLM Inference for Production
                  </div>
                  <p className={styles.pubDesc}>
                    A complete decision framework for GPU capacity planning:
                    workload characterization, memory budgeting, roofline
                    analysis, quantization, parallelism, batching, KV cache
                    optimization, and a 13-step sizing algorithm. Written for
                    Staff and Principal ML Engineers.{" "}
                    <Link to="/sizing">Read the guide →</Link>
                  </p>
                </div>
              </div>
            </Section>

            <Section title="Selected Writing">
              <div className={styles.pubList}>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>
                    Technical Comparison · LinkedIn
                  </div>
                  <div className={styles.pubTitle}>
                    <Link to="https://www.linkedin.com/pulse/great-llm-inference-showdown-tensorrt-llm-vs-vllm-vinay-jayanna-9o9pc">
                      The Great LLM Inference Showdown: TensorRT-LLM vs vLLM
                    </Link>
                  </div>
                  <p className={styles.pubDesc}>
                    A framework-level comparison of the two dominant LLM serving
                    stacks - throughput characteristics, memory efficiency, ease
                    of deployment, and the workload profiles where each excels
                    in production.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>
                    Production Guide · LinkedIn
                  </div>
                  <div className={styles.pubTitle}>
                    <Link to="https://www.linkedin.com/pulse/challenge-production-llm-serving-ray-serve-vinay-jayanna-08syc">
                      The Challenge of Production LLM Serving: A Ray Serve
                      Perspective
                    </Link>
                  </div>
                  <p className={styles.pubDesc}>
                    Architecture patterns for deploying LLMs on Ray Serve at
                    scale - autoscaling configuration, batching strategy,
                    multi-model routing, and operational lessons from production
                    deployments.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>Deep Dive · LinkedIn</div>
                  <div className={styles.pubTitle}>
                    <Link to="https://www.linkedin.com/pulse/multi-tenant-llm-inference-bridging-research-reality-vinay-jayanna-lpmtc">
                      Multi-Tenant LLM Inference: Bridging Research and Reality
                    </Link>
                  </div>
                  <p className={styles.pubDesc}>
                    How multi-LoRA serving, prefix caching, and cache-aware
                    routing combine to make multi-tenant inference economically
                    viable - and where the production engineering challenges
                    diverge from the research framing.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>
                    Technical Explainer · LinkedIn
                  </div>
                  <div className={styles.pubTitle}>
                    <Link to="https://www.linkedin.com/pulse/kv-cache-hidden-optimization-behind-real-time-ai-vinay-jayanna-cvfec">
                      KV Cache: The Hidden Optimization Behind Real-Time AI
                    </Link>
                  </div>
                  <p className={styles.pubDesc}>
                    Why KV cache is the most under-used lever in production LLM
                    serving - and how PagedAttention, prefix caching, and
                    quantization combine to turn it into a concurrency
                    multiplier.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>
                    Infrastructure · LinkedIn
                  </div>
                  <div className={styles.pubTitle}>
                    <Link to="https://www.linkedin.com/pulse/infrastructure-one-talks-how-vector-search-makes-gen-ai-vinay-jayanna-ka2bc">
                      The Infrastructure No One Talks About: How Vector Search
                      Makes Gen AI Work
                    </Link>
                  </div>
                  <p className={styles.pubDesc}>
                    The retrieval infrastructure underneath RAG systems - why
                    vector search is a production engineering problem, not just
                    an algorithm selection.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>All writing</div>
                  <div className={styles.pubTitle}>
                    <Link to="https://www.linkedin.com/in/vinayjayanna/recent-activity/articles/">
                      View all articles on LinkedIn →
                    </Link>
                  </div>
                  <p className={styles.pubDesc}>
                    Additional writing on GenAI platform engineering, vector
                    search infrastructure, MLOps, and production AI strategy.
                  </p>
                </div>
              </div>
            </Section>

            <Section title="Contact">
              <p className={styles.contactText}>
                Reachable on{" "}
                <Link to="https://linkedin.com/in/vinayjayanna">LinkedIn</Link>{" "}
                and <Link to="https://github.com/vinay-jayanna">GitHub</Link>.
                For substantive technical discussions - inference sizing, GenAI
                platform architecture, or field guide feedback - LinkedIn DMs
                work best.
              </p>
            </Section>
          </div>
        </div>
      </div>
    </Layout>
  );
}
