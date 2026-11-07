#!/usr/bin/env python3
"""CPU reference study of adaptive contiguous decay grouping."""
import json, math, random
from pathlib import Path

SEED = 20261107
D, V, GROUP, STEPS = 256, 32, 64, 48


def approx_gate(gates, groups_per_tile):
    """Optimal contiguous scalar-gate segmentation per physical tile (DP SSE)."""
    out = [0.0] * len(gates)
    max_error = 0.0
    for lo in range(0, len(gates), GROUP):
        tile = gates[lo:lo + GROUP]
        n = len(tile)
        # Segment SSE from prefix sums; segmentation keeps channel order intact.
        p = [0.0]
        q = [0.0]
        for x in tile:
            p.append(p[-1] + x)
            q.append(q[-1] + x*x)
        cost = [[0.0]*(n+1) for _ in range(n+1)]
        for i in range(n):
            for j in range(i+1,n+1):
                total = p[j]-p[i]
                cost[i][j] = q[j]-q[i]-total*total/(j-i)
        kmax = min(groups_per_tile, n)
        dp = [[float('inf')]*(n+1) for _ in range(kmax+1)]
        prev = [[-1]*(n+1) for _ in range(kmax+1)]
        dp[0][0] = 0.0
        for k in range(1,kmax+1):
            for j in range(k,n+1):
                for i in range(k-1,j):
                    value = dp[k-1][i] + cost[i][j]
                    if value < dp[k][j]: dp[k][j], prev[k][j] = value, i
        bounds=[]; j=n
        for k in range(kmax,0,-1):
            i=prev[k][j]; bounds.append((i,j)); j=i
        for i,j in reversed(bounds):
            mean=(p[j]-p[i])/(j-i)
            for c in range(i,j):
                out[lo+c]=mean
                max_error=max(max_error,abs(tile[c]-mean))
    return out, max_error


def norm_fro(m):
    return math.sqrt(sum(x*x for row in m for x in row))


def run():
    rng = random.Random(SEED)
    # Four synthetic regimes: tightly clustered gates; one outlier/tile;
    # smoothly varying gates; and adversarial bimodal gates.
    regimes = {}
    for regime in ('clustered', 'one_outlier', 'smooth', 'bimodal'):
        gates_by_step = []
        for t in range(STEPS):
            g = []
            for i in range(D):
                tile_i, pos = i // GROUP, i % GROUP
                if regime == 'clustered': x = .90 + rng.uniform(-.03, .03)
                elif regime == 'one_outlier': x = .90 + rng.uniform(-.015, .015) if pos else .35 + .01*math.sin(t+tile_i)
                elif regime == 'smooth': x = .35 + .55 * (pos/(GROUP-1)) + .01*math.sin(t*.2+tile_i)
                else: x = .30 if pos < GROUP//2 else .98
                g.append(max(.05, min(1.0, x)))
            gates_by_step.append(g)

        # Common initial state and rank-one writes make the only difference the gate approximation.
        init = [[rng.uniform(-.1,.1) for _ in range(V)] for _ in range(D)]
        keys = [[rng.uniform(-.1,.1) for _ in range(D)] for _ in range(STEPS)]
        vals = [[rng.uniform(-.1,.1) for _ in range(V)] for _ in range(STEPS)]
        outputs = {}
        for budget in (1, 2, 4, 8, 16):
            state = [row[:] for row in init]
            max_gate_err = 0.0
            max_bound_ratio = 0.0
            for t, full_gate in enumerate(gates_by_step):
                gate, err = approx_gate(full_gate, budget)
                max_gate_err = max(max_gate_err, err)
                pre = state
                state = [[gate[i] * pre[i][j] + keys[t][i] * vals[t][j] for j in range(V)] for i in range(D)]
                # One-step perturbation obeys ||(F-Fhat)S||_F <= max_i |f_i-fhat_i| ||S||_F.
                bound = err * norm_fro(pre)
                # Reconstruct full-gate next state and check the local bound numerically.
                full = [[full_gate[i] * pre[i][j] + keys[t][i] * vals[t][j] for j in range(V)] for i in range(D)]
                diff = norm_fro([[state[i][j]-full[i][j] for j in range(V)] for i in range(D)])
                assert diff <= bound + 1e-10, (regime, budget, diff, bound)
                if bound > 0: max_bound_ratio = max(max_bound_ratio, diff/bound)
            outputs[str(budget)] = {'max_gate_error': max_gate_err, 'max_local_bound_ratio': max_bound_ratio}
            # Re-run recurrence and compare final state to exact gate baseline.
            approx_state = [row[:] for row in init]
            exact_state = [row[:] for row in init]
            for t, full_gate in enumerate(gates_by_step):
                gate, _ = approx_gate(full_gate, budget)
                approx_state = [[gate[i]*approx_state[i][j]+keys[t][i]*vals[t][j] for j in range(V)] for i in range(D)]
                exact_state = [[full_gate[i]*exact_state[i][j]+keys[t][i]*vals[t][j] for j in range(V)] for i in range(D)]
            outputs[str(budget)]['final_state_frobenius_error'] = norm_fro([[approx_state[i][j]-exact_state[i][j] for j in range(V)] for i in range(D)])
        errors = [outputs[str(k)]['final_state_frobenius_error'] for k in (1, 2, 4, 8, 16)]
        assert all(a + 1e-10 >= b for a, b in zip(errors, errors[1:])), (regime, errors)
        regimes[regime] = outputs

    result = {'study':'adaptive contiguous sub-tile scalar decay grouping gates',
              'seed':SEED, 'shape':{'key_channels':D,'value_channels':V,'tile_channels':GROUP,'steps':STEPS},
              'contiguous_gate_groups_per_64_channel_tile':[1,2,4,8,16], 'results':regimes,
              'limitations':'Deterministic CPU synthetic recurrence only; not a model-quality, TPU/GPU kernel, throughput, or training study.'}
    out = Path(__file__).with_name('results.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': run()
