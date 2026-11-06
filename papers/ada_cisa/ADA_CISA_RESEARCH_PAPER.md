# Ada-CISA: Adaptive Outlier-Aware Chunk-Indexed Sparse Attention with Dynamic Dispersion Gating for Long-Context Serving on NVIDIA GPUs

**Author:** Lee Si Yuan (*Independent Researcher*, \texttt{siyuanlee.cs@gmail.com})  
**Date:** November 2026  
**Repository:** [nvidia-gpu-kernel-research](https://github.com/100-x-coder/nvidia-gpu-kernel-research)  
**Evaluation Mode:** Algorithmic reference simulation and analytical roofline memory traffic modeling  
**Proposed target architectures (not benchmarked):** NVIDIA Hopper and Blackwell
**Publication PDF:** [`papers/ada_cisa/ada_cisa.pdf`](./ada_cisa.pdf)  
**LaTeX Source:** [`papers/ada_cisa/ada_cisa.tex`](./ada_cisa.tex)  
**Benchmark Suite:** [`papers/ada_cisa/benchmarks/run_benchmark.py`](./benchmarks/run_benchmark.py)  

---

## Abstract

Serving long-context Large Language Models (LLMs) with context windows exceeding $10^5$ to $10^6$ tokens incurs severe memory bandwidth bottlenecks due to repeated Key-Value (KV) cache retrieval across autoregressive generation. Recently, *Chunk-Indexed Sparse Attention* (CISA; Dabbah et al., Nov 2026, arXiv:2611.03200v1) demonstrated that pre-trained dense attention models can be retrofitted without fine-tuning by partitioning the KV cache into fixed chunks ($C=16$) and scoring arithmetic-mean representatives with an uncalibrated or KL-matched indexing query.

We examine three hypotheses about fixed chunk summaries, shared queries, and fixed budgets in generated CPU cases:
1. **The Arithmetic Mean Dilemma:** In our synthetic reference experiment, a salient key contributes only $1/C$ of a uniform chunk mean. We test whether norm weighting changes target-chunk selection on generated needle queries; the measured rates apply only to this setup.
2. **Query-sharing sensitivity:** Static query grouping ($G=16$) shares one averaged query across positions. Divergent queries may make that average less representative; the effect depends on the input and is tested in synthetic scenarios below.
3. **Fixed-budget tradeoff:** A fixed budget does not adapt to score entropy. The simulation below tests one entropy-gated alternative using synthetic queries; it does not establish layer-wise entropy behavior in deployed models.

We study **Ada-CISA** (Adaptive Chunk-Indexed Sparse Attention) as three algorithmic heuristics in a CPU reference simulation. A hardware implementation is future work:
- **Norm-Weighted Outlier-Aware (NOA) Key Summaries:** An exponentiated norm-weighted reduction operator. Constructing one chunk representative costs $O(Cd)$ arithmetic for chunk size $C$ and dimension $d$; its persistent representative storage matches the baseline's one vector per chunk.
- **Dispersion-Aware Dynamic QShare (DA-QShare):** An online directional coherence gate that dynamically partitions query blocks when cosine dispersion exceeds a threshold $\tau_{\text{disp}}$, preventing centroid drift across semantic shifts.
- **Entropy-Gated Dynamic Budgeting (Ada-Budget):** A heuristic score-entropy gate that changes the chunk budget. In the reported eight-layer simulation it fetched 192 chunks versus 128 for the fixed policy and reduced average KL by 5.4%; this example improves that metric by spending more modeled attention work.
- **Proposed TMA pipeline:** An unimplemented design sketch for future GPU evaluation; no zero-copy integration is provided here.

In the seeded CPU simulation, needle-chunk selection rates were **70.0% at $\beta=0$ and 98.3% at $\beta=1.6$** (60 trials each), and the difference at 8,192 tokens was **+20.0 percentage points** (50 trials). An analytical traffic model estimates **31.1$\times$ fewer reads** at 1M tokens; its **$10.5\times$ latency figure is a model projection**, not a GPU measurement.

---

## 1. Introduction & Background

### 1.1 The Long-Context Serving Crisis
At million-token context lengths, repeatedly reading the KV cache can make attention memory-bandwidth-bound during decoding.

For a batch of $B=32$ requests with sequence length $N=1,048,576$ tokens, a standard 8-head Grouped Query Attention (GQA) model with head dimension $d=128$ in BF16 stores:
$$\text{KV Memory} = 2 \times N \times H_{\text{KV}} \times d \times 2\text{ bytes} \approx 4.0\text{ GiB per sequence}$$
Evaluating dense attention over 1M tokens for every decoding step requires saturating high-bandwidth memory (HBM3e), imposing a throughput ceiling regardless of GPU Tensor Core compute capacity.

### 1.2 Baseline: Chunk-Indexed Sparse Attention (CISA)
Dabbah et al. (arXiv:2611.03200v1) introduced CISA as a post-training, gradient-free sparse attention retrofit. CISA operates under three core principles:
1. **Page-Aligned Chunking:** The sequence history is partitioned into chunks $\mathcal{C}_j$ of size $C=16$, aligning directly with vLLM physical page block allocations.
2. **Mean-Key Representative Indexing:** Each historical chunk is summarized by the arithmetic mean of its key vectors:
   $$r_j = \frac{1}{C}\sum_{u \in \mathcal{C}_j} k_u$$
3. **QShare:** During prefill, adjacent query tokens are partitioned into static blocks of size $G$ (e.g. $G=16$), sharing an averaged index query:
   $$\bar{q}^I = \frac{1}{G}\sum_{t \in \mathcal{G}} q_t^I$$

CISA reports GPU and end-to-end results in its paper. The extension here does not test its method on real model distributions; our CPU simulations explore selected synthetic cases only.

---

## 2. Shortcomings of Baseline CISA

### 2.1 The Arithmetic Mean Dilemma
For a chunk with one designated salient key and $C-1$ background keys, the arithmetic mean scales the salient key contribution by $1/C$. The CPU experiment measures target-chunk selection on generated needle inputs; it does not assert that such outlier patterns are common in trained models.

When using the uniform arithmetic mean:
$$r_j = \frac{1}{C}\sum_{u=1}^C k_u = \frac{1}{C} k_{\text{salient}} + \frac{C-1}{C} \bar{k}_{\text{background}}$$
The inner product score with a query targeting the salient token becomes:
$$\langle q, r_j \rangle = \frac{1}{C} \langle q, k_{\text{salient}} \rangle + \frac{C-1}{C} \langle q, \bar{k}_{\text{background}} \rangle$$
For $C=16$, the salient key contribution to the arithmetic mean is scaled by $1/16$. In the seeded CPU experiment, the baseline selected the target chunk in 70.0% of 60 trials. This does not establish a general failure rate or prove the cause of each miss.

### 2.2 Query Centroid Drift in Static QShare
CISA's QShare mechanism can average queries across a configured group of positions. The following equation illustrates that shared-query construction for group size $G=16$:
$$\bar{q}^I = \frac{1}{G}\sum_{t=1}^G q_t^I$$
When inputs contain divergent queries, the shared average may be a poor representative for some positions. Our synthetic query experiment shows one scenario where splitting helped and another where it reduced cosine similarity; it does not establish a general effect for language-model activations.

### 2.3 Layer-Homogeneous Budget Rigidity
For a fixed budget setting, the policy does not change the chunk count in response to score entropy. Our CPU simulation compares such a fixed budget with a simple entropy gate for eight synthetic queries. It does not model transformer layers or substantiate claims about early- or deep-layer attention distributions.

---

## 3. Ada-CISA: Architecture and Mathematical Formulation

We evaluate three algorithmic heuristics in a CPU reference simulation and outline a possible hardware design. The hardware path is not implemented or measured.

```
       [ Input KV Tokens (C=16) ]
                  |
        +---------+---------+
        |                   |
  [ Standard KV ]   [ Key Norm Extraction ]
        |                   |
        |             ||k_u||_2, mu_C
        |                   |
        |          [ NOA Weighting: w_u = exp(beta * ||k_u|| / mu_C) ]
        |                   |
        |       [ NOA Representative: r_j = sum(w_u * k_u) ]
        |                   |
        +---------> [ Representative Cache (1/(2C) HBM) ]
                            |
                   (Asynchronous TMA)
                            |
                 [ Query Indexer Engine ]
                 /          |          \
     [ Dynamic Dispersion ] | [ Entropy-Gated Budget ]
         (DA-QShare)        |       (Ada-Budget)
                 \          |          /
             [ Top-K Selected Physical Page IDs ]
                            |
              [ Zero-Copy Paged TMA Gather ]
                            |
             [ Hopper/Blackwell WGMMA GEMM ]
```

### 3.1 Norm-Weighted Outlier-Aware (NOA) Summaries
Instead of uniform arithmetic averaging, we formulate the chunk summary as an exponentiated norm-weighted convex combination:

$$r_j^{\text{NOA}} = \sum_{u \in \mathcal{C}_j} w_u k_u$$

where the weighting factor $w_u$ is defined as:

$$w_u = \frac{\exp\left( \beta \cdot \frac{\|k_u\|_2}{\mu_{\mathcal{C}_j}} \right)}{\sum_{v \in \mathcal{C}_j} \exp\left( \beta \cdot \frac{\|k_v\|_2}{\mu_{\mathcal{C}_j}} \right)}, \quad \mu_{\mathcal{C}_j} = \frac{1}{|\mathcal{C}_j|} \sum_{v \in \mathcal{C}_j} \|k_v\|_2$$

- $\beta \ge 0$ is the **salience temperature**.
- When $\beta = 0$, $w_u = 1/C$, recovering standard CISA uniform averaging.
- As $\beta > 0$, tokens possessing high $L_2$ activation norms receive exponential priority, ensuring that sharp entity markers and needle tokens dominate the representative.
- **Computational Cost:** Computing all chunk norms, exponent weights, and the weighted sum costs $O(Cd)$ arithmetic per chunk. Persistent storage is one $d$-element representative, the same size as baseline CISA. The Python simulation clips exponents at 20 and uses 1.0 when mean norm is below $10^{-8}$.

### 3.2 Dispersion-Aware Dynamic QShare (DA-QShare)
Rather than forcing static blocks of size $G$, DA-QShare evaluates online directional coherence using the cosine metric:

$$\rho(q_a, q_b) = \frac{\langle q_a, q_b \rangle}{\|q_a\|_2 \|q_b\|_2}$$

During prefill, given an active group seed $q_{\text{seed}}$ and incoming query $q_t$:
$$\mathcal{G}_{\text{active}} \leftarrow 
\begin{cases} 
\mathcal{G}_{\text{active}} \cup \{q_t\}, & \text{if } \rho(q_{\text{seed}}, q_t) \ge \tau_{\text{disp}} \text{ and } |\mathcal{G}| < G_{\max} \\
\text{Finalize } \mathcal{G}_{\text{active}} \text{ and start new partition } \{q_t\}, & \text{otherwise}
\end{cases}$$

This online rule groups a query with the current partition only when its cosine similarity to the partition seed meets the chosen threshold. The rule is a heuristic; it does not guarantee pairwise similarity among all queries in a partition or prevent centroid deterioration in every workload.

### 3.3 Entropy-Gated Dynamic Budget Allocation (Ada-Budget)
Ada-CISA dynamically adapts the chunk budget $k_{\text{alloc}}$ to the distribution of chunk scores. Given the raw representative scores $s_j = \frac{\langle q^I, r_j^{\text{NOA}} \rangle}{\sqrt{d}}$ over the eligible set $\mathcal{E}_t$:

$$p_j = \frac{\exp(s_j)}{\sum_{m \in \mathcal{E}_t} \exp(s_m)}$$

We compute the normalized Shannon entropy:

$$\hat{\mathcal{H}} = \frac{-\sum_{j \in \mathcal{E}_t} p_j \ln p_j}{\ln |\mathcal{E}_t|}$$

The allocated chunk count is gated according to:

$$k_{\text{alloc}} = \begin{cases}
\max\left(k_{\min}, \lfloor k_{\text{base}} \times 0.5 \rfloor\right), & \text{if } \hat{\mathcal{H}} < \theta_{\text{sharp}} \quad (\text{Focal attention}) \\
\min\left(|\mathcal{E}_t|, \lfloor k_{\text{base}} \times 1.5 \rfloor\right), & \text{if } \hat{\mathcal{H}} > \theta_{\text{diffuse}} \quad (\text{Dispersed attention}) \\
k_{\text{base}}, & \text{otherwise}
\end{cases}$$

This heuristic uses fewer chunks for low-entropy scores and more for high-entropy scores. In the reported eight-layer simulation it fetched 192 chunks versus 128 for fixed budgeting while reducing mean KL by 5.4%; that accuracy result costs more modeled attention work.

---

## 4. Proposed Hardware Design Sketch (Not Implemented)

### 4.1 TMA Paged Gather & Warp Specialization (Proposed)
A future Hopper/Blackwell implementation might use the Tensor Memory Accelerator for asynchronous transfers from global to shared memory, as described in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). This proposed integration has not been implemented or validated in this repository.

1. **Producer Warp (TMA Dispatch):** Computes top-$k$ chunk physical page indices. Once identified, the producer issues asynchronous TMA chunk copy descriptors directly into circular shared memory multibuffers.
2. **Consumer Warpgroups (WGMMA):** Compute fused matrix multiplications ($Q K^T$ and $P V$) using Warpgroup-level Matrix Multiply and Accumulate instructions while the next chunk is prefetched via TMA.
3. **Page integration:** A zero-copy path would depend on matching chunk and runtime page layouts and on supported TMA descriptors. This compatibility has not been demonstrated here.

---

## 5. Empirical Verification and Experimental Results

We ran a small CPU simulation and three unit tests. Retrieval rates and KL values are generated-input observations; the traffic and latency values are model outputs, not GPU benchmarks.

### 5.1 Dense Equivalence Verification
In one seeded CPU reference case with a full chunk budget, the maximum output difference from dense attention was:
$$\max |o^{\text{dense}} - o^{\text{Ada-CISA}}| = 0.00 \times 10^0$$

The test asserts a difference below $10^{-5}$. This single CPU test does not establish bitwise or all-input equivalence.

### 5.2 Ablation Study: Norm-Weighting Temperature ($\beta$)
We measured target-chunk selection rate at $k=4$ over a 2,048-token synthetic context (60 trials per temperature):

| Parameter Configuration | Formulation | Needle-Chunk Selection @ $k=4$ | Difference (percentage points) |
|---|---|---|---|
| $\beta = 0.0$ | Standard CISA (Arithmetic Mean) | **70.0%** | Baseline |
| $\beta = 0.4$ | Mild Norm Weighting | **81.7%** | $+11.7$ |
| $\beta = 0.8$ | Balanced Norm Weighting | **90.0%** | $+20.0$ |
| $\beta = 1.2$ | Higher Norm Weighting | **93.3%** | $+23.3$ |
| $\beta = 1.6$ | Stronger Norm Weighting | **98.3%** | **$+28.3$** |

> **Simulation result:** In this seeded 60-trial CPU experiment, the observed needle-chunk selection rate was 70.0% at $\beta=0$ and 98.3% at $\beta=1.6$. It does not establish a general retrieval rate; the weighting adds compute while retaining one representative vector per chunk.

### 5.3 Context Scaling: Needle-in-a-Haystack (1K to 8K Tokens)
We scaled sequence length $N$ from 1,024 to 8,192 tokens with fixed budget $k=6$ chunks:

| Context Length ($N$) | Number of Chunks | CISA Selection @ $k=6$ | Ada-CISA Selection @ $k=6$ | Difference (pp) |
|---|---|---|---|---|
| **1,024** | 64 | 92.0% | **98.0%** | $+6.0\%$ |
| **2,048** | 128 | 84.0% | **98.0%** | $+14.0\%$ |
| **4,096** | 256 | 74.0% | **86.0%** | $+12.0\%$ |
| **8,192** | 512 | 72.0% | **92.0%** | **$+20.0\%$** |

At 8,192 tokens in this 50-trial CPU setting, the target-chunk selection rate was 72.0% for the mean baseline and 92.0% for norm weighting, a 20 percentage-point difference.

### 5.4 Query Dispersion: Handling Semantic Phase Boundaries
Evaluating query grouping across $G=16$ positions under different levels of semantic divergence:

| Prompt Scenario | Dispersion Description | Static QShare CosSim | DA-QShare CosSim | Dynamic Subgroups |
|---|---|---|---|---|
| **Coherent Prompt** | Single continuous semantic flow | 0.4109 | **0.4109** | 1 (Unified) |
| **Moderate Divergence** | Gradual semantic drift | **0.3832** | 0.3716 | 3 |
| **Topic/Phase Boundary** | Abrupt syntax/task switch | 0.3467 | **0.3801** | 2 (Split) |

In the one synthetic topic-boundary scenario, the static method scored 0.3467 cosine similarity and the gated method scored 0.3801, with two partitions. The gradual-drift scenario instead scored lower with the gated method (0.3716 vs. 0.3832), so this heuristic did not improve every tested scenario.

### 5.5 Layer-Heterogeneous Entropy Budgeting
In a simulated 8-layer model with variable query scales (one synthetic query per layer):
- **Fixed-Budget CISA ($k=16$):** Average KL Divergence $= 13.3066$ (Total chunks fetched $= 128$).
- **Ada-Budget (Entropy-Gated):** Average KL Divergence $= 12.5901$ (Total chunks fetched $= 192$).
- This synthetic example reduces KL by 5.4% while fetching 50% more chunks; it does not establish a general quality improvement.

### 5.6 Hardware HBM Traffic & Latency Model (Hopper / Blackwell)
The analytical model assumes 8-head GQA in BF16 ($d=128$, $C=16$) and reports read-volume estimates up to 1,048,576 tokens. The hardware label identifies the intended configuration, not a measured device run:

| Context Tokens ($N$) | Dense HBM Read (MiB) | CISA HBM Read (MiB) | Ada-CISA HBM Read (MiB) | Memory Read Reduction | Projected Speedup |
|---|---|---|---|---|---|
| **4,096** | 16.00 MiB | 4.69 MiB | **4.00 MiB** | 4.0$\times$ | 2.39$\times$ |
| **16,384** | 64.00 MiB | 6.19 MiB | **5.50 MiB** | 11.6$\times$ | 5.15$\times$ |
| **65,536** | 256.00 MiB | 12.19 MiB | **11.50 MiB** | 22.3$\times$ | 8.22$\times$ |
| **131,072** | 512.00 MiB | 20.19 MiB | **19.50 MiB** | 26.3$\times$ | 9.25$\times$ |
| **524,288** | 2,048.00 MiB | 68.19 MiB | **67.50 MiB** | 30.3$\times$ | 10.27$\times$ |
| **1,048,576** | 4,096.00 MiB | 132.19 MiB | **131.50 MiB** | **31.1$\times$** | **10.47$\times$** |

At context length 1,048,576 tokens, under the traffic model's assumptions:
- Dense attention requires reading **4,096 MiB** of HBM memory per query step.
- Modeled Ada-CISA reads **131.50 MiB** (including index representative overhead), a **$31.1\times$ reduction**. The separate **$10.47\times$ latency estimate** is produced by a heuristic formula in the script and has not been validated on hardware.

---

## 6. Unimplemented Design Sketch for Triton and CuTe

To deploy Ada-CISA in high-performance serving engines (vLLM / TensorRT-LLM):

1. **Chunk Ingestion Hook (`append_paged_kv`):**
   ```python
   # In Triton or CUDA C++ during KV cache append:
   norm_k = tl.sqrt(tl.sum(k * k, axis=-1))
   w_u = tl.exp(beta * (norm_k / mean_norm))
   r_j = tl.sum(w_u[:, None] * k, axis=0) / tl.sum(w_u)
   ```
2. **Chunk Selection Kernel:**
   Execute a lightweight fused GEMV kernel scoring $q^I$ against $r_j^{\text{NOA}}$ in SRAM, utilizing a warp-level bitonic top-$k$ sort to write page IDs into the attention dispatch table.
3. **Pipelined FlashAttention-3 / CuTe Kernel:**
   Pass the selected page IDs to a TMA-enabled custom attention kernel that streams physical cache blocks directly into tensor core accumulators.

---

## 7. Conclusion

We describe Ada-CISA as a set of algorithmic heuristics evaluated in CPU reference simulations. The experiments do not establish robustness across real models or deployment ease. NOA weighting improves the selected synthetic retrieval cases, while query gating and entropy allocation have mixed tradeoffs; a hardware implementation remains future work.

Our CPU simulations report a 98.3% needle-chunk selection rate in one 60-trial setting. The analytical traffic model estimates 31.1× fewer reads at one million tokens. The script projects a 10.47× latency ratio using an explicit heuristic; neither latency nor GPU performance was measured.

---

## References

1. Dabbah, M., Golan, I., Tropp, O., et al. (2026). *CISA: Retrofitting Dense Attention Models with Chunk-Indexed Sparse Attention*. arXiv:2611.03200v1.
2. Dao, T., Haziza, D., Massa, F., & Gu, A. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*. arXiv:2407.08608.
3. Zadouri, T., et al. (2026). *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*. arXiv:2603.05451.
4. Kwon, W., Li, Z., Zhuang, S., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
5. DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report*. arXiv:2412.19437.
6. Velikonivtsev, F., Koltakov, M., & Kuznedelev, D. (2026). *FLAMIA: FLAsh MIcroscaling Attention for Fast and Causally Safe Training on Blackwell*. arXiv:2611.03344v1.
## Code and formal check

The [code attachment](./code_attachment.zip) includes the CPU reference simulation, tests, and output. Lean checks the uniform-mean coefficient decomposition and that normalized NOA weights become uniform at $\beta=0$; it does not prove the simulation's target-selection rates or hardware model. See [the formal verification notes](../formal/FORMAL_VERIFICATION.md).
