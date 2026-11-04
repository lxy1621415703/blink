"""Focused semantic checks for the Phase B fluid DAG experiment."""
import dataclasses
import unittest

from recovery_benchmark import (analytical_counterexample, exact_schedule, list_schedule,
                                make_instance, rollout_schedule, validate_schedule,
                                weighted_loss)


class RecoveryBenchmarkTests(unittest.TestCase):
    def test_analytical_shared_state_example(self):
        instance = analytical_counterexample()
        priority = list_schedule(instance, "priority_list")
        shared = list_schedule(instance, "shared_successor_list")
        rollout = rollout_schedule(instance, "loss")
        self.assertEqual(weighted_loss(instance, priority), 52)
        self.assertEqual(weighted_loss(instance, shared), 38)
        self.assertEqual(weighted_loss(instance, rollout), 38)
        exact, info = exact_schedule(instance, "loss", seconds=5)
        self.assertTrue(info["optimal"])
        self.assertEqual(weighted_loss(instance, exact), 38)

    def test_invalid_schedule_detects_resources_and_dependencies(self):
        instance = analytical_counterexample()
        invalid = {"A": (0, 2), "E": (0, 1), "B": (0, 1), "C": (0, 1)}
        violations = validate_schedule(instance, invalid)
        self.assertGreater(violations["resource_violations"], 0)
        self.assertEqual(violations["precedence_violations"], 2)

    def test_generated_models_and_no_demand_change_control(self):
        for shared in [True, False]:
            instance = make_instance(7, shared, unchanged=True)
            changed = make_instance(7, shared)
            self.assertTrue(all(f.old_service > 0 for f in instance.flows))
            self.assertEqual([f.old_service for f in instance.flows],
                             [f.old_service for f in changed.flows])
            self.assertTrue(all(f.demand == f.old_service for f in instance.flows))
            self.assertEqual(instance.tasks, changed.tasks)
            for policy in ["priority_list", "shared_successor_list"]:
                schedule = list_schedule(instance, policy)
                self.assertFalse(any(validate_schedule(instance, schedule).values()))
                self.assertEqual(weighted_loss(instance, schedule), 0)
            self.assertEqual(weighted_loss(instance, rollout_schedule(instance)), 0)

    def test_shared_unshared_have_same_work_and_final_plan(self):
        first, second = make_instance(8, True), make_instance(8, False)
        self.assertEqual(first.flows, second.flows)
        def work(instance):
            result = {}
            for task in instance.tasks:
                for resource, amount in task.resources:
                    result[resource] = result.get(resource, 0) + task.duration * amount
            return result
        self.assertEqual(work(first), work(second))
        self.assertEqual(first.capacities, second.capacities)

    def test_rollout_does_not_worsen_its_completion_baseline(self):
        for seed in range(6):
            instance = make_instance(seed)
            shared = list_schedule(instance, "shared_successor_list")
            rollout = rollout_schedule(instance, "loss")
            self.assertFalse(any(validate_schedule(instance, rollout).values()))
            self.assertLessEqual(weighted_loss(instance, rollout), weighted_loss(instance, shared) + 1e-8)


if __name__ == "__main__":
    unittest.main()

