import React from 'react';
import type { ReactNode } from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import styles from './about.module.css';

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
      title="About — Vinay Jayanna"
      description="Staff ML Engineer specializing in LLM inference optimization and production AI systems. Led engineering on AWS SageMaker. Founded Vipas.AI.">
      <div className={styles.page}>
        <div className={styles.inner}>

          {/* Header */}
          <div className={styles.header}>
            <div className={styles.headerText}>
              <p className={styles.headerEyebrow}>About</p>
              <h1 className={styles.headerTitle}>Vinay Jayanna</h1>
              <p className={styles.headerSubtitle}>
                Staff ML Engineer with 17+ years building production AI systems.
                I specialize in LLM inference optimization, GPU capacity planning,
                and the systems decisions that determine whether AI deployments
                are cost-efficient at scale. I write field guides to make
                rigorous engineering frameworks accessible to the engineers
                who need them.
              </p>
              <div className={styles.headerLinks}>
                <Link
                  to="https://github.com/vinay-jayanna"
                  className={styles.headerLink}>
                  GitHub →
                </Link>
                <Link
                  to="https://linkedin.com/in/vinayjayanna"
                  className={styles.headerLink}>
                  LinkedIn →
                </Link>
              </div>
            </div>
          </div>

          <div className={styles.body}>

            {/* Experience */}
            <Section title="Experience">
              <div className={styles.timeline}>
                <TimelineItem
                  period="2025 — Present"
                  role="Staff ML Engineer"
                  org="Large-scale Generative AI Platform"
                  description="Leading LLM inference optimization and GenAI platform engineering at scale. Focus areas include GPU memory architecture, KV cache optimization, serving framework evaluation (vLLM, TensorRT-LLM), parallelism strategy for frontier-class models, latency-throughput operating point selection, and production monitoring for LLM serving systems."
                  tags={[
                    'LLM Inference',
                    'GenAI Platform',
                    'GPU Optimization',
                    'vLLM',
                    'TensorRT-LLM',
                    'Ray Serve',
                  ]}
                />
                <TimelineItem
                  period="2024 — 2025"
                  role="Founder & CEO"
                  org="Vipas.AI — AI Inference Marketplace"
                  description="Founded and led an AI inference marketplace enabling model creators to host and monetize industry-specific AI models with IP protection, monitoring, and pay-per-prediction APIs. Grew to 25K daily active visitors. Received VC term sheet. Built end-to-end across engineering, product, and go-to-market."
                  tags={[
                    'Founder',
                    'LLM Serving',
                    'Inference Marketplace',
                    'Product',
                    'Go-to-Market',
                  ]}
                />
                <TimelineItem
                  period="~8 years"
                  role="Engineering Leader"
                  org="AWS SageMaker — Amazon Web Services"
                  description="Led engineering on AWS SageMaker from the 2017 launch through eight years of scale. Drove AI inference infrastructure serving production ML workloads across thousands of enterprise customers. Worked across the full ML platform stack — model hosting, training infrastructure, and the core abstractions that became the industry standard for managed ML. The experience that shaped how I think about production AI systems at scale."
                  tags={[
                    'AWS SageMaker',
                    'ML Platform',
                    'AI Inference',
                    'Distributed Systems',
                    'Enterprise Scale',
                  ]}
                />
              </div>
            </Section>

            {/* Focus Areas */}
            <Section title="Technical Focus">
              <div className={styles.focusGrid}>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>LLM Inference Optimization</div>
                  <p className={styles.focusDesc}>
                    GPU memory architecture, KV cache design, quantization
                    strategies (FP8, INT4, GPTQ, AWQ), parallelism
                    (tensor, pipeline, data, expert), continuous and
                    disaggregated batching, speculative decoding.
                    Operating point selection from latency-throughput curves.
                  </p>
                </div>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>GPU Capacity Planning & Cost</div>
                  <p className={styles.focusDesc}>
                    Workload characterization, roofline analysis, fleet
                    sizing from first principles, TCO modeling, utilization
                    trap avoidance, heterogeneous fleet composition.
                    Turning benchmark results into GPU counts and cost
                    estimates you can defend.
                  </p>
                </div>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>GenAI Platform Engineering</div>
                  <p className={styles.focusDesc}>
                    LLM serving framework selection and evaluation,
                    multi-LoRA serving architectures, production guardrails,
                    rate limiting strategies, SLA-aware scheduling,
                    and the operational engineering that keeps
                    GenAI systems reliable at scale.
                  </p>
                </div>
                <div className={styles.focusCard}>
                  <div className={styles.focusTitle}>Agentic AI Systems</div>
                  <p className={styles.focusDesc}>
                    Multi-agent orchestration patterns, tool reliability,
                    memory and state management across agent turns,
                    latency budgets for compound AI systems, failure
                    mode analysis, and observability for
                    non-deterministic LLM pipelines.
                  </p>
                </div>
              </div>
            </Section>

            {/* Patents & Publications */}
            <Section title="Patents & Publications">
              <div className={styles.pubList}>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>USPTO Patent — Pending</div>
                  <div className={styles.pubTitle}>
                    Dynamic Hierarchical Storage and GPU Optimization for LLM Serving
                  </div>
                  <p className={styles.pubDesc}>
                    A system and method for dynamic tiered memory management
                    across GPU VRAM, CPU DRAM, and NVMe storage for large
                    language model inference workloads, with adaptive
                    scheduling based on request priority and latency
                    constraints.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>Field Guide — v1.0 · 107 pages</div>
                  <div className={styles.pubTitle}>
                    Sizing LLM Inference Systems at Scale
                  </div>
                  <p className={styles.pubDesc}>
                    A complete decision framework for GPU capacity planning:
                    workload characterization, memory budgeting, roofline
                    analysis, quantization, parallelism, batching, KV cache
                    optimization, and a 13-step sizing algorithm. Written
                    for Staff and Principal ML Engineers.{' '}
                    <Link to="/sizing">Read the guide →</Link>
                  </p>
                </div>
              </div>
            </Section>

            {/* Selected Writing */}
            <Section title="Selected Writing">
              <div className={styles.pubList}>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>Technical Comparison · LinkedIn</div>
                  <div className={styles.pubTitle}>
                    TensorRT-LLM vs. vLLM: A Production Comparison
                  </div>
                  <p className={styles.pubDesc}>
                    A framework-level comparison of the two dominant LLM
                    serving stacks — throughput characteristics, memory
                    efficiency, ease of deployment, and the workload
                    profiles where each excels in production.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>Production Guide · LinkedIn</div>
                  <div className={styles.pubTitle}>
                    Ray Serve for Production LLM Serving
                  </div>
                  <p className={styles.pubDesc}>
                    Architecture patterns for deploying LLMs on Ray Serve
                    at scale — autoscaling configuration, batching strategy,
                    multi-model routing, and operational lessons from
                    production deployments.
                  </p>
                </div>
                <div className={styles.pubItem}>
                  <div className={styles.pubType}>More on LinkedIn</div>
                  <div className={styles.pubTitle}>
                    <Link to="https://linkedin.com/in/vinayjayanna">
                      View all articles on LinkedIn →
                    </Link>
                  </div>
                  <p className={styles.pubDesc}>
                    Additional writing on GenAI platform engineering,
                    MLOps, LLM fine-tuning, and production AI strategy.
                  </p>
                </div>
              </div>
            </Section>

            {/* Contact */}
            <Section title="Contact">
              <p className={styles.contactText}>
                I'm reachable on{' '}
                <Link to="https://linkedin.com/in/vinayjayanna">LinkedIn</Link>{' '}
                and{' '}
                <Link to="https://github.com/vinay-jayanna">GitHub</Link>.
                For substantive technical discussions — inference sizing,
                GenAI platform architecture, or field guide feedback —
                LinkedIn DMs work best.
              </p>
            </Section>

          </div>
        </div>
      </div>
    </Layout>
  );
}
