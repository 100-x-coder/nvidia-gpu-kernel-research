#!/usr/bin/env python3
"""
Unit and Regression Test Suite for Ada-CISA:
Verifies algorithmic invariants, causal isolation, boundary safety,
and representative numerical stability.
"""

import unittest
import math
import random
from run_benchmark import AttentionSimulator, generate_synthetic_data, cosine_similarity, kl_divergence

class TestAdaCISAKernel(unittest.TestCase):
    def setUp(self):
        self.sim = AttentionSimulator(dim=32, chunk_size=8, recent_window=2)
        random.seed(42)

    def test_causality_and_bounds(self):
        """Verify that attended tokens never exceed causal current position"""
        keys, values, _ = generate_synthetic_data(seq_len=128, dim=32, seed=1)
        q = [random.gauss(0, 1) for _ in range(32)]
        
        for curr_pos in [15, 31, 63, 100, 127]:
            curr_chunk = curr_pos // self.sim.chunk_size
            reps = self.sim.build_ada_cisa_representatives(keys)
            selected = self.sim.select_chunks(q, reps, len(reps), k_chunks=2, current_chunk_idx=curr_chunk)
            
            # Check eligible chunks strictly precede current_chunk - recent_window
            eligible_limit = max(0, curr_chunk - self.sim.recent_window)
            for c in selected:
                self.assertLess(c, eligible_limit, f"Selected chunk {c} violates causal window {eligible_limit}")
                
            out, probs, num_tokens = self.sim.run_sparse_attention(q, keys, values, selected, curr_pos)
            self.assertEqual(len(out), 32)
            self.assertEqual(len(probs), curr_pos + 1)
            # Ensure probability sums to ~1.0
            self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_outlier_amplification(self):
        """Verify that NOA representatives preserve high-norm salient directions"""
        chunk_keys = [[random.gauss(0, 0.2) for _ in range(16)] for _ in range(15)]
        # Add 1 salient outlier key
        outlier = [2.0] * 16
        chunk_keys.append(outlier)
        
        sim = AttentionSimulator(dim=16, chunk_size=16)
        cisa_reps = sim.build_cisa_representatives(chunk_keys)
        ada_reps = sim.build_ada_cisa_representatives(chunk_keys, beta=2.0)
        
        # Check alignment with outlier vector
        cisa_align = cosine_similarity(cisa_reps[0], outlier)
        ada_align = cosine_similarity(ada_reps[0], outlier)
        
        self.assertGreater(ada_align, cisa_align, "Ada-CISA NOA representative must show higher alignment with salient outlier")
        self.assertGreater(ada_align, 0.85, "Ada-CISA should maintain > 0.85 alignment with salient outlier")

    def test_entropy_gated_budget_monotonicity(self):
        """Verify that sharper distributions yield tighter budget, diffuse yield higher"""
        reps = [[random.gauss(0, 1) for _ in range(16)] for _ in range(20)]
        sim = AttentionSimulator(dim=16, chunk_size=8, recent_window=1)
        
        # Diffuse query: all near zero
        q_diffuse = [0.001] * 16
        _, k_diffuse = sim.select_chunks_entropy_gated(q_diffuse, reps, len(reps), base_k=6, current_chunk_idx=15)
        
        # Peaky query: highly aligned with rep[3]
        q_peaky = [x * 10.0 for x in reps[3]]
        _, k_peaky = sim.select_chunks_entropy_gated(q_peaky, reps, len(reps), base_k=6, current_chunk_idx=15)
        
        self.assertLessEqual(k_peaky, k_diffuse, "Sharp query must receive less than or equal chunk budget compared to diffuse query")

if __name__ == "__main__":
    unittest.main()
