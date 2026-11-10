# Commit Now or Keep the Boundary Adjustable? Minimax-Regret Planning for Decode Disaggregation

**Lee Si Yuan**

Independent Researcher

`siyuanlee.cs@gmail.com`

## Abstract

Attention/feed-forward network disaggregation (AFD) can improve resource matching, but its extra communication and pool fragmentation make its benefit depend on future model and hardware conditions. Extending Hadidi's comparison of prefill/decode (P/D) and P/D+AFD, we study a narrower planning question: when a deployment decision must be made before those conditions are known, how should a fixed boundary be selected? We define scenario regret against the best boundary after a scenario is revealed and select the fixed choice with minimum worst-case regret. A deterministic, seeded CPU sensitivity study samples 20,000 dimensionless scenarios over declared resource, fabric, and role-matching ranges. In this deliberately uncalibrated surrogate, fixed P/D has lower mean utility regret (0.00234) than fixed AFD (0.2659), and minimax regret selects P/D; a scenario-aware oracle improves mean surrogate utility by 0.00234 over the best fixed decision. These values are properties of the chosen equations and ranges, not hardware measurements. We provide a finite-action regret guarantee and executable assertions for its ordering properties. The study offers a reproducible decision-analysis template, not a deployment recommendation.

## 1 Introduction

Hadidi (2026) compares P/D with P/D+AFD across model directions, accelerator classes, fabrics, and deployment choices. The paper reports that AFD does not have a robust deployment-level advantage across its main comparison, and that re-optimizing mappings can reduce losses from early hardware commitment. We extend this work by making uncertainty over future operating conditions the explicit decision object.

The 2026-11-08 daily index had already stored the title and abstract. Before this extension, the source arXiv identifier did not appear in the repository paper index or commit history. We reviewed the full v1 text for this study.

The question here is not which topology has the highest expected score under one chosen probability distribution. In early hardware planning, that distribution is usually not known well enough to justify a precise expected-value claim. Instead, for a declared finite scenario set, we ask: which fixed topology has the smallest worst-case regret relative to the topology that would have been chosen with perfect scenario information?

Our contribution is a simple regret criterion and an executable sensitivity harness. The experiment intentionally uses dimensionless surrogates, not calibrated service times, measured accelerator specifications, or an implementation of SiliconForge. It is useful for checking decision logic and exposing sensitivity to assumptions. It cannot establish a real P/D-versus-AFD winner.

## 2 Decision rule

Let \(D=\{\mathrm{PD},\mathrm{AFD}\}\) be the available fixed topology decisions and \(S\) a finite set of possible operating scenarios. Let \(u(d,s)\) be a utility to maximize, here a synthetic capacity-per-power surrogate. For scenario \(s\), define the oracle utility \(u^*(s)=\max_{d\in D}u(d,s)\). The regret of a fixed decision is

\[
 r(d,s)=u^*(s)-u(d,s), \qquad R_{\max}(d)=\max_{s\in S}r(d,s).
\]

We choose \(d_{\mathrm{MMR}}\in\arg\min_{d\in D}R_{\max}(d)\). This is a decision rule over the declared scenarios; it does not imply that all scenarios are equally likely. The scenario-aware oracle is an upper bound on achievable utility when the scenario is known before the decision and switching is free. The difference between oracle and best fixed mean utility measures the value of scenario-specific choice under the experiment's sampling distribution, not the value of a practical controller with switching, migration, or reconfiguration costs.

### Proposition 1 (finite-action minimax-regret guarantee)

For nonempty finite \(D,S\) and real-valued utility \(u\), the selected decision satisfies \(R_{\max}(d_{\mathrm{MMR}})\le R_{\max}(d)\) for every \(d\in D\). For every scenario, \(r(d,s)\ge0\); consequently, the scenario-aware oracle's mean utility is at least the mean utility of every fixed decision under any probability distribution on \(S\).

**Proof.** The first inequality follows directly because \(d_{\mathrm{MMR}}\) is an element of the finite set \(D\) attaining its minimum worst-case regret. For the second claim, \(u^*(s)\) is the maximum over \(D\), so \(u^*(s)\ge u(d,s)\) for all \(d,s\). Taking expectations preserves the pointwise inequality. \(\square\)

This proposition is an optimization identity, not a hardware-performance theorem. The expected-utility inequality is Lean-checked in `../formal/RobustDecodeClaims.lean`; the program checks regret and oracle ordering over sampled scenarios. Neither check validates the surrogate or replaces the decision assumptions.

## 3 Synthetic sensitivity model

The experiment uses Python's standard library only. A fixed pseudorandom seed (20261110) generates 20,000 independent scenarios. KV, FFN, and prefill pressure are sampled log-uniformly from [0.35, 3.0]; a normalized memory-to-compute ratio from [0.4, 2.5]; fabric ratio from [0.08, 2.0]; pooling loss uniformly from [0, 0.25]; and two independent role-specific placement factors log-uniformly from [0.65, 1.8]. The ranges are chosen for sensitivity exploration and are not estimated from a hardware catalog.

For a scenario, normalized prefill capacity is \(P=1/p\), and co-located decode capacity is \(C=\min(m/k,1/f)\), where \(p,k,f,m\) denote prefill pressure, KV pressure, FFN pressure, and the memory-to-compute ratio. P/D utility is \(u_{PD}=\min(P,C)\). The AFD surrogate matches attention and FFN roles independently: \(A=(m/k)a\), \(F=(1/f)g\), with placement factors \(a,g\). An exchange efficiency \(e=b/(b+0.35)\), based on fabric ratio \(b\), and a pooling factor \(1-q\), based on pooling loss \(q\), give \(u_{AFD}=\min(P,A,F)e(1-q)\). These equations omit queueing, SLOs, parallelism, real link collectives, memory capacity, and topology-specific costs. Both values are dimensionless surrogates.

The script computes mean utility, mean and 95th-percentile regret, worst regret, the minimax-regret fixed action, and oracle-versus-fixed mean utility. It also asserts nonnegative regret, minimax ordering, and the oracle ordering. Regenerate with `python3 experiment.py`; the output is `results.json`.

## 4 Results

For the fixed seed and ranges above, P/D mean surrogate utility is 0.6073 and AFD is 0.3437. The corresponding mean regrets are 0.00234 and 0.2659, and the minimax-regret choice is P/D. P/D has zero regret in 96.4% of sampled scenarios (including ties); AFD has zero regret in 14.9%. The oracle mean exceeds the best fixed mean by 0.00234. The worst sampled regrets are 0.558 for P/D and 2.148 for AFD.

The skewed results reflect the selected surrogate: AFD pays an explicit exchange penalty and pooling loss, while its only upside is role-specific capacity matching. Changing these ranges or equations can change the ranking. No confidence interval is reported because this is a sensitivity construction rather than a sample from a target population. AFD's nonzero win fraction confirms that the model permits role-specific placement to overcome its penalties in some draws; it does not estimate how often that occurs in production.

## 5 Limitations and next experiment

The extension does not reproduce the source paper's calibration, service queue, hardware catalog, model workload, or event-driven cross-check. It does not run GPU kernels, measure watts, or verify SLO goodput. The current machine has no visible NVIDIA device interface, and this code is CPU-only by design. The reported numerical values should therefore not be used to pick hardware.

A useful next test is to replace the synthetic utility function with measurements from a single serving engine while holding model, request trace, power budget, precision, and SLO fixed. The scenario set should be preregistered from actual uncertainty ranges; fixed P/D and AFD should each receive the same optimization budget; and the measured outputs should include p99 TTFT, p99 TPOT, goodput per watt, and the cost of changing mappings. Until such data are available, this paper contributes a transparent decision criterion and reproducible toy experiment only.

## References

Ramyad Hadidi. 2026. *Should You Split Decode? Disaggregation Across Future Models and Accelerators*. arXiv:2611.08299v1. https://arxiv.org/abs/2611.08299.
