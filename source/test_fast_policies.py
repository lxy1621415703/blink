"""Semantic checks for fixed cheap policies; uses development instances only."""
from __future__ import annotations

import dataclasses
import unittest

from fast_engine import run_list
from fast_policies import FAST_POLICIES, safe_portfolio
from recovery_benchmark import (analytical_counterexample, make_instance, makespan,
                                validate_schedule, weighted_loss)


class FastPolicyTests(unittest.TestCase):
    def test_shared_preparation_analytical_example(self):
        instance = analytical_counterexample()
        for name, policy in FAST_POLICIES.items():
            with self.subTest(policy=name):
                schedule = policy(instance)
                self.assertEqual(weighted_loss(instance, schedule), 38)
                self.assertEqual(makespan(schedule), 3)
                self.assertFalse(any(validate_schedule(instance, schedule).values()))

    def test_portfolio_joint_dominance_and_feasibility(self):
        for seed in range(12):
            for shared in [True, False]:
                instance = make_instance(seed, shared)
                baseline = run_list(instance, "shared_successor_list")
                chosen = safe_portfolio(instance)
                self.assertFalse(any(validate_schedule(instance, chosen).values()))
                self.assertLessEqual(makespan(chosen), makespan(baseline))
                self.assertLessEqual(weighted_loss(instance, chosen),
                                     weighted_loss(instance, baseline) + 1e-8)

    def test_task_storage_order_does_not_change_the_plan(self):
        instance = make_instance(7)
        reverse = dataclasses.replace(instance, tasks=tuple(reversed(instance.tasks)))
        for name, policy in FAST_POLICIES.items():
            with self.subTest(policy=name):
                self.assertEqual(policy(instance), policy(reverse))

    def test_no_demand_increase_preserves_zero_service_deficit(self):
        instance = make_instance(6, unchanged=True)
        self.assertTrue(all(f.weight > 0 and f.old_service == f.demand for f in instance.flows))
        for name, policy in FAST_POLICIES.items():
            with self.subTest(policy=name):
                schedule = policy(instance)
                self.assertFalse(any(validate_schedule(instance, schedule).values()))
                self.assertEqual(weighted_loss(instance, schedule), 0)


if __name__ == "__main__":
    unittest.main()

