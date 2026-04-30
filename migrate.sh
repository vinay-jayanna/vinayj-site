#!/bin/bash
# ============================================================
# vinayj-site structural migration
# Run from: /Users/vinayjayanna/Desktop/pers/vinayj-site
# ============================================================
set -e

echo ""
echo "============================================================"
echo " vinayj-site structural migration"
echo "============================================================"

# STEP 1: Create new folder structure
echo ""
echo "→ Step 1: Creating new folder structure..."
mkdir -p content/sizing
mkdir -p content/agentic
mkdir -p static/img/sizing
mkdir -p static/img/agentic
mkdir -p static/img/covers/sizing
mkdir -p static/img/covers/agentic
mkdir -p static/img/site
mkdir -p static/pdf
echo "  ✓ Folders created"

# STEP 2: Move FG1 content: docs/ → content/sizing/
echo ""
echo "→ Step 2: Moving docs/ → content/sizing/..."
mv docs/index.md                           content/sizing/index.md
mv docs/preface.md                         content/sizing/preface.md
mv docs/why-inference-sizing.md            content/sizing/why-inference-sizing.md
mv docs/01-workload-characterization.md    content/sizing/01-workload-characterization.md
mv docs/02-gpu-memory-sizing.md            content/sizing/02-gpu-memory-sizing.md
mv docs/03-roofline-model.md               content/sizing/03-roofline-model.md
mv docs/04-quantization.md                 content/sizing/04-quantization.md
mv docs/05-parallelism-strategy.md         content/sizing/05-parallelism-strategy.md
mv docs/06-batching-strategy.md            content/sizing/06-batching-strategy.md
mv docs/07-kv-cache-optimization.md        content/sizing/07-kv-cache-optimization.md
mv docs/08-latency-throughput-curve.md     content/sizing/08-latency-throughput-curve.md
mv docs/09-sizing-algorithm.md             content/sizing/09-sizing-algorithm.md
mv docs/first-principles.md               content/sizing/first-principles.md
rmdir docs
echo "  ✓ Done. docs/ removed."

# STEP 3: Move FG2 placeholder: agentic/ → content/agentic/
echo ""
echo "→ Step 3: Moving agentic/ → content/agentic/..."
mv agentic/index.md content/agentic/index.md
rmdir agentic
echo "  ✓ Done. agentic/ removed."

# STEP 4: Move FG1 figures: static/img/figures/ → static/img/sizing/
echo ""
echo "→ Step 4: Moving static/img/figures/ → static/img/sizing/..."
mv static/img/figures/fig-1-1-request-lifecycle-ttft-itl.png        static/img/sizing/
mv static/img/figures/fig-1-2-four-workload-archetypes.png           static/img/sizing/
mv static/img/figures/fig-1-3-enterprise-traffic-24hrs.png           static/img/sizing/
mv static/img/figures/fig-2-1-gpu-memory-breakdown-fp16-fp8.png      static/img/sizing/
mv static/img/figures/fig-2-2-kv-cache-memory-vs-concurrency.png     static/img/sizing/
mv static/img/figures/fig-2-3-standard-vs-flashattention.png         static/img/sizing/
mv static/img/figures/fig-2-5-multi-lora-serving-memory.png          static/img/sizing/
mv static/img/figures/fig-3-1-roofline-model-h100.png                static/img/sizing/
mv static/img/figures/fig-3-2-prefill-matrix-multiply.png            static/img/sizing/
mv static/img/figures/fig-3-3-2-prefill-decode-operating-points.png  static/img/sizing/
mv static/img/figures/fig-3-3-decode-matrix-vector.png               static/img/sizing/
mv static/img/figures/fig-3-4-decode-throughput-vs-batch-size.png    static/img/sizing/
mv static/img/figures/fig-4-1-model-weight-memory-by-precision.png   static/img/sizing/
mv static/img/figures/fig-5-1-tensor-parallelism-tp4.png             static/img/sizing/
mv static/img/figures/fig-5-2-pipeline-parallelism-bubble.png        static/img/sizing/
mv static/img/figures/fig-5-4-expert-parallelism-moe-routing.png     static/img/sizing/
mv static/img/figures/fig-5-6-nvlink-vs-pcie-bandwidth.png           static/img/sizing/
mv static/img/figures/fig-6-2-four-scheduler-policies.png            static/img/sizing/
mv static/img/figures/fig-6-3-static-vs-continuous-batching.png      static/img/sizing/
mv static/img/figures/fig-6-4-chunked-prefill-itl-stability.png      static/img/sizing/
mv static/img/figures/fig-6-5-disaggregated-prefill-decode.png       static/img/sizing/
mv static/img/figures/fig-6-6-speculative-decoding-mechanism.png     static/img/sizing/
mv static/img/figures/fig-7-1-pagedattention-block-allocation.png    static/img/sizing/
mv static/img/figures/fig-7-3-cache-aware-routing.png                static/img/sizing/
mv static/img/figures/fig-7-4-kv-cache-memory-pressure-cascade.png   static/img/sizing/
mv static/img/figures/fig-8-1-latency-throughput-curve-knee.png      static/img/sizing/
mv static/img/figures/fig-8-2-slo-constrained-operating-point.png    static/img/sizing/
rmdir static/img/figures
echo "  ✓ Done. static/img/figures/ removed."

# STEP 5: Move FG1 covers → static/img/covers/sizing/
echo ""
echo "→ Step 5: Moving covers → static/img/covers/sizing/..."
mv static/img/covers/cover-front.png     static/img/covers/sizing/
mv static/img/covers/cover-contents.png  static/img/covers/sizing/
mv static/img/covers/cover-author.png    static/img/covers/sizing/
echo "  ✓ Done."

# STEP 6: Move site-level images → static/img/site/
echo ""
echo "→ Step 6: Moving site images → static/img/site/..."
mv static/img/favicon.ico                static/img/site/
mv static/img/favicon.svg                static/img/site/
mv static/img/logo.svg                   static/img/site/
mv static/img/docusaurus-social-card.jpg static/img/site/
# Remove unused Docusaurus placeholder images
rm -f static/img/undraw_docusaurus_mountain.svg
rm -f static/img/undraw_docusaurus_react.svg
rm -f static/img/undraw_docusaurus_tree.svg
rm -f static/img/docusaurus.png
echo "  ✓ Done."

# STEP 7: Fix /img/figures/ → /img/sizing/ in all FG1 markdown
echo ""
echo "→ Step 7: Updating image paths in content/sizing/*.md..."
find content/sizing -name "*.md" -exec sed -i '' 's|/img/figures/|/img/sizing/|g' {} \;
echo "  ✓ Done."

# STEP 8: Rename sidebars.ts → sidebars-sizing.ts
echo ""
echo "→ Step 8: Renaming sidebars.ts → sidebars-sizing.ts..."
mv sidebars.ts sidebars-sizing.ts
echo "  ✓ Done."

echo ""
echo "============================================================"
echo " Migration complete. Now:"
echo ""
echo "  1. Copy docusaurus.config.ts (provided) → project root"
echo "  2. Copy sidebars-agentic.ts (provided) → project root"
echo "  3. Copy agentic-index.md (provided) → content/agentic/index.md"
echo ""
echo "  Then run:"
echo "    yarn add @docusaurus/theme-mermaid"
echo "    yarn start"
echo ""
echo "  Verify localhost:3000/sizing looks identical to before."
echo "  Verify localhost:3000/agentic shows the new landing page."
echo "============================================================"
