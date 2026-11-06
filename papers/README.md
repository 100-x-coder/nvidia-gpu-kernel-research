# Extended Research Papers & Experimental Implementations

This directory contains research papers, novel extensions, and empirical verification benchmarks developed for high-performance GPU kernels, targeting NVIDIA Hopper (SM90) and Blackwell (SM100/SM120) architectures.

## Index of Papers

| Paper Title | Author | Base Work | Status | Formats & Artifacts |
|---|---|---|---|---|
| **Ada-CISA**: Adaptive Outlier-Aware Chunk-Indexed Sparse Attention with Dynamic Dispersion Gating | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Dabbah et al. (CISA, [arXiv:2611.03200v1](https://arxiv.org/abs/2611.03200v1)) | Seeded CPU simulation; analytical traffic model; no GPU benchmark | [PDF](./ada_cisa/ada_cisa.pdf) • [LaTeX](./ada_cisa/ada_cisa.tex) • [Markdown](./ada_cisa/ADA_CISA_RESEARCH_PAPER.md) • [Code attachment](./ada_cisa/code_attachment.zip) • [Script](./ada_cisa/benchmarks/run_benchmark.py) • [Results](./ada_cisa/results/benchmark_results.json) |
| **Causal Prefix Microscaling**: Scale-Dependency Study for Low-Precision Attention | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Velikonivtsev et al. (FLAMIA, [arXiv:2611.03344v1](https://arxiv.org/abs/2611.03344)) | CPU reference study; no GPU benchmark | [PDF](./causal_scale_attention/causal_prefix_microscaling.pdf) • [LaTeX](./causal_scale_attention/causal_prefix_microscaling.tex) • [Markdown](./causal_scale_attention/CAUSAL_PREFIX_MICROSCALING.md) • [Code attachment](./causal_scale_attention/code_attachment.zip) • [Script](./causal_scale_attention/scale_leakage_experiment.py) • [Results](./causal_scale_attention/results.json) |
| **Noise-Aware Kernel Promotion**: Paired Statistical Gate for Self-Evolving Kernels | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Chen et al. (ForgeKernel, [arXiv:2611.04998](https://arxiv.org/abs/2611.04998)) | Synthetic CPU Monte Carlo study; no GPU measurements | [ACL LaTeX](./noise_aware_promotion/noise_aware_promotion.tex) • [PDF](./noise_aware_promotion/noise_aware_kernel_promotion.pdf) • [Markdown](./noise_aware_promotion/NOISE_AWARE_KERNEL_PROMOTION.md) • [Code attachment](./noise_aware_promotion/code_attachment.zip) • [Experiment](./noise_aware_promotion/noise_aware_promotion_experiment.py) • [Results](./noise_aware_promotion/results.json) |

The Noise-Aware Promotion LaTeX, Markdown, HTML, and PDF files reflect the same research draft. The LaTeX and PDF use the official ACL style files; the HTML remains a supplementary rendering. The local research index mentions Bevy (Yu et al., [arXiv:2611.04794](https://arxiv.org/abs/2611.04794)) as related work on request scheduling for offloaded MoE inference; it is separate from statistical kernel promotion.

| **Drift-Aware Memory-Bounded Batching (DAMB)**: Online Footprint Feedback for Irregular GPU Pipelines | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Accordi et al. (muDock, [arXiv:2611.06207v1](https://arxiv.org/abs/2611.06207)) | CPU simulation; no GPU benchmark; prototype does not exercise estimate adaptation | [ACL LaTeX](./feedback_batching/feedback_batching.tex) • [PDF](./feedback_batching/drift_aware_batching.pdf) • [Markdown](./feedback_batching/DRIFT_AWARE_BATCHING.md) • [Code attachment](./feedback_batching/code_attachment.zip) • [Experiment](./feedback_batching/adaptive_batching_experiment.py) • [Results](./feedback_batching/results.json) |

### Scope of reported findings

- The Ada-CISA accuracy figures are seeded CPU reference simulation results with 50–60 trials per condition, not model-level retrieval benchmarks.
- The 1M-token read count is an analytical traffic-model estimate; the 10.47× latency number is a heuristic model output, not GPU timing.
- The formal proof suite checks selected algebraic implications. It does not prove empirical rates or hardware behavior. See [Lean-checked claims](./formal/FORMAL_VERIFICATION.md).
