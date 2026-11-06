# Extended Research Papers & Experimental Implementations

This directory contains research papers, novel extensions, and empirical verification benchmarks developed for high-performance GPU kernels, targeting NVIDIA Hopper (SM90) and Blackwell (SM100/SM120) architectures.

## Index of Papers

| Paper Title | Author | Base Work | Status | Formats & Artifacts |
|---|---|---|---|---|
| **Ada-CISA**: Adaptive Outlier-Aware Chunk-Indexed Sparse Attention with Dynamic Dispersion Gating | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Dabbah et al. (CISA, [arXiv:2611.03200v1](http://arxiv.org/abs/2611.03200v1)) | Verified & Benchmarked | [PDF Document](./ada_cisa/ada_cisa.pdf) • [LaTeX Source](./ada_cisa/ada_cisa.tex) • [Markdown Paper](./ada_cisa/ADA_CISA_RESEARCH_PAPER.md) • [Benchmarks](./ada_cisa/benchmarks/run_benchmark.py) • [Results](./ada_cisa/results/benchmark_results.json) |
| **Causal Prefix Microscaling**: Scale-Dependency Study for Low-Precision Attention | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Velikonivtsev et al. (FLAMIA, [arXiv:2611.03344v1](https://arxiv.org/abs/2611.03344)) | CPU Reference Study | [PDF Document](./causal_scale_attention/causal_prefix_microscaling.pdf) • [LaTeX Source](./causal_scale_attention/causal_prefix_microscaling.tex) • [Markdown Paper](./causal_scale_attention/CAUSAL_PREFIX_MICROSCALING.md) • [Experiment](./causal_scale_attention/scale_leakage_experiment.py) • [Results](./causal_scale_attention/results.json) |
| **Noise-Aware Kernel Promotion**: Paired Statistical Gate for Self-Evolving Kernels | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Chen et al. (ForgeKernel, [arXiv:2611.04998](https://arxiv.org/abs/2611.04998)) | Synthetic CPU Monte Carlo study; no GPU measurements | [LaTeX source](./noise_aware_promotion/noise_aware_promotion.tex) • [PDF document](./noise_aware_promotion/noise_aware_kernel_promotion.pdf) • [Markdown](./noise_aware_promotion/NOISE_AWARE_KERNEL_PROMOTION.md) • [Experiment](./noise_aware_promotion/noise_aware_promotion_experiment.py) • [Results](./noise_aware_promotion/results.json) |

The Noise-Aware Promotion LaTeX, Markdown, HTML, and PDF files now reflect the same review draft. The PDF was rendered from the HTML source. No `feedback_batching` source or manuscript is present in this repository or the searched workspace, so this paper does not claim to incorporate it. The local research index mentions Bevy (Yu et al., [arXiv:2611.04794](https://arxiv.org/abs/2611.04794)) as related work on request scheduling for offloaded MoE inference; it is separate from statistical kernel promotion.

| **Drift-Aware Memory-Bounded Batching (DAMB)**: Online Footprint Feedback for Irregular GPU Pipelines | Lee Si Yuan (Independent Researcher, `siyuanlee.cs@gmail.com`) | Accordi et al. (muDock, [arXiv:2611.06207v1](https://arxiv.org/abs/2611.06207)) | CPU Simulation Study | [PDF Document](./feedback_batching/drift_aware_batching.pdf) • [Markdown Paper](./feedback_batching/DRIFT_AWARE_BATCHING.md) • [Experiment](./feedback_batching/adaptive_batching_experiment.py) • [Results](./feedback_batching/results.json) |

### Key Findings of Ada-CISA:
- **Needle Retrieval Recall:** Elevated from 70.0% (standard CISA arithmetic mean) to **98.3%** using Norm-Weighted Outlier-Aware (NOA) summarization ($\beta=1.6$).
- **Context Scaling (8,192 tokens):** **+20.0% net recall gain** over standard CISA.
- **Dynamic Dispersion QShare (DA-QShare):** Prevents query centroid degradation across semantic phase boundaries (+9.6% cosine fidelity gain).
- **Entropy-Gated Layer Budgeting:** 5.4% reduction in KL divergence across heterogeneous transformer layers.
- **Hardware Memory Traffic (1M tokens):** **31.1x reduction** in KV cache HBM reads, projecting a **10.5x end-to-end latency speedup** on NVIDIA Blackwell (B200/B300) hardware.
