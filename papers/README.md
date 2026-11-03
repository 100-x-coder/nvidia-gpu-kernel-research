# Extended Research Papers & Experimental Implementations

This directory contains research papers, novel extensions, and empirical verification benchmarks developed for high-performance GPU kernels, targeting NVIDIA Hopper (SM90) and Blackwell (SM100/SM120) architectures.

## Index of Papers

| Paper Title | Base Work | Focus Area | Status | Benchmarks & Artifacts |
|---|---|---|---|---|
| [Ada-CISA: Adaptive Outlier-Aware Chunk-Indexed Sparse Attention with Dynamic Dispersion Gating](./ada_cisa/ADA_CISA_RESEARCH_PAPER.md) | Dabbah et al. (CISA, [arXiv:2611.03200v1](http://arxiv.org/abs/2611.03200v1)) | Attention & Long-Context Serving (Hopper/Blackwell) | Verified & Benchmarked | [Paper](./ada_cisa/ADA_CISA_RESEARCH_PAPER.md) • [Benchmark Suite](./ada_cisa/benchmarks/run_benchmark.py) • [Results](./ada_cisa/results/benchmark_results.json) |

### Key Findings of Ada-CISA:
- **Needle Retrieval Recall:** Elevated from 70.0% (standard CISA arithmetic mean) to **98.3%** using Norm-Weighted Outlier-Aware (NOA) summarization ($\beta=1.6$).
- **Context Scaling (8,192 tokens):** **+20.0% net recall gain** over standard CISA.
- **Dynamic Dispersion QShare (DA-QShare):** Prevents query centroid degradation across semantic phase boundaries (+9.6% cosine fidelity gain).
- **Entropy-Gated Layer Budgeting:** 5.4% reduction in KL divergence across heterogeneous transformer layers.
- **Hardware Memory Traffic (1M tokens):** **31.1x reduction** in KV cache HBM reads, projecting a **10.5x end-to-end latency speedup** on NVIDIA Blackwell (B200/B300) hardware.
