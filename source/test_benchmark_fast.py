"""Check the scale extension and recovery-time metric semantics."""
import unittest

from benchmark_fast import quality, scaled_instance
from recovery_benchmark import analytical_counterexample, list_schedule, make_instance


class FastBenchmarkTests(unittest.TestCase):
    def test_scale_extension_reproduces_original_small_inputs(self):
        for seed in range(6):
            for shared in [True, False]:
                self.assertEqual(make_instance(seed, shared),
                                 scaled_instance(seed, 3 + seed % 3, shared))

    def test_recovery_metrics_follow_activations_not_first_task_start(self):
        instance = analytical_counterexample()
        result = quality(instance, list_schedule(instance, "shared_successor_list"))
        self.assertEqual(result["weighted_unmet_demand_units"], 38)
        self.assertEqual(result["first_flow_recovery_ticks"], 2)
        self.assertEqual(result["restore50_ticks"], 2)
        self.assertEqual(result["restore90_ticks"], 3)
        self.assertEqual(result["makespan_ticks"], 3)

    def test_unchanged_demand_is_already_served(self):
        instance = make_instance(4, unchanged=True)
        result = quality(instance, list_schedule(instance, "shared_successor_list"))
        self.assertEqual(result["weighted_unmet_demand_units"], 0)
        self.assertEqual(result["restore90_ticks"], 0)
        self.assertGreater(result["makespan_ticks"], 0)


if __name__ == "__main__":
    unittest.main()

