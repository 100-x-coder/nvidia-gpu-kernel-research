# NVIDIA GPU Kernel Research

A curated map of research and open-source implementations for high-performance kernels on NVIDIA GPUs.

**Research snapshot:** 2026-11-03. Focus: AI/HPC kernels on NVIDIA GPUs, especially Hopper (SM90) and Blackwell (SM100/SM120). “State of the art” is workload-, shape-, precision-, and GPU-dependent; this index highlights active directions and representative public work, not one universal winner.

## Current research directions

1. **Exploit the hardware pipeline.** Hopper and Blackwell expose asynchronous data movement and matrix operations (TMA, WGMMA, Tensor Memory and newer Tensor Core paths). Strong kernels overlap memory movement, math, and reductions through warp/warpgroup specialization and carefully staged pipelines.
2. **Fuse work around the expensive operation.** Attention, MoE, normalization, quantization, and serving kernels combine operations to reduce global-memory traffic, launch overhead, and intermediate tensors. Persistent scheduling and shape-aware dispatch matter as much as the inner math.
3. **Treat precision as an algorithmic choice.** FP8 and block-scaled low-precision formats (including Blackwell FP4/NVFP4 paths) can raise throughput, but scaling, accumulation, error, and reproducibility need explicit evaluation.
4. **Move the abstraction boundary.** CUDA/CuTe/CUTLASS offer control; Triton, TileLang, and ThunderKittens try to make optimized tiled kernels more accessible. The practical frontier is getting high-level productivity while exposing enough control of layouts, synchronization, pipelines, and architecture-specific instructions.
5. **Improve the optimization loop.** Autotuning and code-generation agents are increasingly evaluated on real kernels. Useful evaluation must include correctness, numerical behavior, robust timing, representative shapes, and proximity to hardware limits—not just speedup on a toy baseline.

## Papers by category

### Attention and sequence kernels

| Work | Contribution | Code / resources |
|---|---|---|
| [Mass-Adaptive History Promotion (MAHP)](./papers/mass_adaptive_attention/mass_adaptive_history_promotion.pdf) | Lee Si Yuan, Independent Researcher. CPU reference study of query-conditioned full-dimension promotion for mixed-dimensional attention; synthetic only, no video-model or GPU benchmark. | [PDF](./papers/mass_adaptive_attention/mass_adaptive_history_promotion.pdf) • [ACL LaTeX](./papers/mass_adaptive_attention/paper.tex) • [Markdown](./papers/mass_adaptive_attention/MASS_ADAPTIVE_HISTORY_PROMOTION.md) • [Experiment and results](./papers/mass_adaptive_attention/) |
| [Ada-CISA (2026)](./papers/ada_cisa/ada_cisa.pdf) | Adaptive outlier-aware chunk-indexed sparse attention with dispersion gating by Lee Si Yuan; elevates needle recall from 70% to 98.3% and cuts 1M-token HBM reads by 31.1x. | [PDF Paper](./papers/ada_cisa/ada_cisa.pdf) • [LaTeX](./papers/ada_cisa/ada_cisa.tex) • [Markdown](./papers/ada_cisa/ADA_CISA_RESEARCH_PAPER.md) • [Benchmarks](./papers/README.md) |
| [FlashAttention-3 (2024)](https://arxiv.org/abs/2407.08608) | Hopper-specific asynchronous execution, warp specialization, interleaved matmul/softmax, and FP8 techniques. | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (active CuTeDSL FA4 implementation; repository also contains FA2/FA3) |
| [FlashAttention-4 (2026)](https://arxiv.org/abs/2603.05451) | Co-designs the attention algorithm and kernel pipeline for asymmetric hardware scaling; targets newer GPU generations. | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| [TileLang: Bridge Programmability and Performance in Modern Neural Kernels (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76fb92288bf90360c527efb0d1c2aba6-Abstract-Conference.html) | Evaluates a composable tiled DSL across neural-kernel workloads, including attention. | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) |

### Kernel programming models and abstractions

| Work | Contribution | Code / resources |
|---|---|---|
| [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (MAPL 2019)](https://dl.acm.org/doi/10.1145/3315508.3329973) | Foundational tiled programming model and compiler; still a core way to write custom ML kernels. | [triton-lang/triton](https://github.com/triton-lang/triton), [documentation](https://triton-lang.org/) |
| [ThunderKittens: Simple, Fast, and Adorable Kernels (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/05dc08730e32441edff52b0fa6caab5f-Abstract-Conference.html) | C++ tile abstractions at warp, thread-block, and grid levels; focuses on compact high-performance AI kernels. | [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) |
| [TileLang: A Composable Tiled Programming Model for AI Systems (2025)](https://arxiv.org/abs/2504.17577) | Pythonic tiled DSL built on TVM/compiler infrastructure, aimed at productivity without giving up low-level kernel control. | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) |
| [CUTLASS and CuTe](https://github.com/NVIDIA/cutlass) | Production-oriented CUDA C++ templates and layout algebra; includes CuTeDSL and architecture-specific Hopper/Blackwell examples. | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass), [CuTeDSL examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) |

### GEMM, Tensor Cores, and low precision

| Work / resource | Why it matters | Code |
|---|---|---|
| [CUTLASS Blackwell examples](https://github.com/NVIDIA/cutlass/tree/main/examples) | Reference implementations for SM100/SM120 dense, grouped, sparse, convolution, narrow/block-scaled precision, and FMHA; useful for studying TMA/Tensor Core schedules. | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) |
| [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608) | Demonstrates that asynchronous Tensor Core scheduling and low precision can substantially improve attention throughput on H100. | [flash-attention](https://github.com/Dao-AILab/flash-attention) |
| [SOL-ExecBench paper (2026)](https://arxiv.org/abs/2603.19173) | 235 production-derived CUDA optimization tasks on Blackwell, scored by distance to estimated hardware limits. | [NVIDIA benchmark and submissions](https://research.nvidia.com/benchmarks/sol-execbench/collection/1) |

### MoE, inference, and fused model operators

| Work / resource | Why it matters | Code |
|---|---|---|
| [SOL-ExecBench](https://arxiv.org/abs/2603.19173) | Covers attention, MoE, normalization, embedding, projections, fused blocks, and other model operators drawn from real workloads. | [NVIDIA SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/) |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Kernel library for LLM serving, including attention, paged KV cache, and quantization paths. | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |
| [TileLang examples](https://github.com/tile-ai/tilelang/tree/main/examples) | Includes GEMM, quantization, FlashAttention, linear attention, DeepSeek MLA and other model kernels. | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) |

### Kernel generation, optimization, and evaluation

| Work / resource | Focus | Code / benchmark |
|---|---|---|
| [SOL-ExecBench (2026)](https://arxiv.org/abs/2603.19173) | Execution-based, production-model kernel problems evaluated against hardware ceilings on NVIDIA Blackwell. | [NVIDIA SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/) |
| [KernelArc (2026)](https://arxiv.org/abs/2608.17071) | Multi-agent exploration for GPU kernel optimization on H100 and B200; includes custom kernels and library/configuration choices. | [Paper](https://arxiv.org/abs/2608.17071) |
| [FastKernels](https://fastkernels.github.io/) | Studies kernel-generation evaluation under production-like execution and scheduling constraints. | [Project site](https://fastkernels.github.io/) |
| [GPU MODE Kernel Leaderboards](https://github.com/gpu-mode/reference-kernels) | Community tasks and ranked submissions across operator families and NVIDIA GPUs. | [Reference kernels](https://github.com/gpu-mode/reference-kernels), [leaderboards](https://www.gpumode.com/) |
| [nvbench](https://github.com/NVIDIA/nvbench) | CUDA microbenchmarking library for measuring device kernel execution. | [NVIDIA/nvbench](https://github.com/NVIDIA/nvbench) |

## Code map

| Layer | Projects | Use it for |
|---|---|---|
| Vendor primitives and templates | [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), [CUTLASS/CuTe](https://github.com/NVIDIA/cutlass), [CUB](https://github.com/NVIDIA/cccl/tree/main/cub) | Maximum control, architecture-specific kernels, GEMM and collectives |
| Kernel DSLs | [Triton](https://github.com/triton-lang/triton), [TileLang](https://github.com/tile-ai/tilelang), [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | Custom fused operators and research iteration at varying abstraction levels |
| Workload libraries | [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Attention, inference, quantization, and model-level kernels |
| Profiling / evaluation | [Nsight Compute](https://developer.nvidia.com/nsight-compute), [Nsight Systems](https://developer.nvidia.com/nsight-systems), [nvbench](https://github.com/NVIDIA/nvbench), [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/) | Diagnose bottlenecks, measure kernels, compare with realistic workloads and hardware limits |

## Practical reading path

1. Learn the execution and memory model with CUDA, then profile a simple kernel with Nsight Compute.
2. Implement the same tiled GEMM or reduction in Triton; study layouts, occupancy, memory coalescing, and launch overhead.
3. Read FlashAttention-3 and ThunderKittens to understand Hopper pipelines and warp specialization.
4. Study CUTLASS/CuTe Blackwell examples for current architecture-specific GEMM and low-precision techniques; compare with TileLang if you want a higher-level interface.
5. Use SOL-ExecBench or GPU MODE tasks to practice on fixed workloads with correctness checks and repeatable performance measurements.

## Keeping this index current

This is a curated research map, not an exhaustive bibliography. When adding an entry, include a paper or technical report, its code if public, the GPU architecture evaluated, and the benchmark/workload. Mark preprints and project claims as such, and revisit the links and project status periodically. “Latest” status in fast-moving repositories can change after this snapshot.

