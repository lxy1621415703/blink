"""Metric, pairing and protocol tests for executable-closure experiments."""
import json
from pathlib import Path
import unittest

from benchmark_closure import (
    PRIMARY_METRICS, compare_rows, fingerprint, quality_metrics,
    rotated, selected_policies, state_metrics,
)
from recovery_benchmark import Flow, Instance, Task


PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "closure_protocol.json").read_text(encoding="utf-8"))


def row(seed, policy, value):
    return {"split": "holdout", "profile": "mixed", "n_flows": 8,
            "seed": seed, "policy": policy,
            **{metric: value for metric in PRIMARY_METRICS}}


class ClosureBenchmarkTests(unittest.TestCase):
    def test_business_times_exclude_background_work(self):
        tasks = (Task("a", 1, ()), Task("b", 1, ()), Task("background", 12, ()))
        flows = (Flow("f1", 1, 2, 1, "a", 2), Flow("f2", 9, 2, 1, "b", 2))
        instance = Instance("fixture", tasks, {}, flows)
        values = quality_metrics(instance, {"a": (0, 1), "b": (9, 10), "background": (0, 12)})
        self.assertEqual(values["weighted_deficit_unit_ticks"], 91)
        self.assertEqual(values["first_activation_ticks"], 1)
        self.assertEqual(values["weighted_t90_ticks"], 10)
        self.assertEqual(values["qos_all_ticks"], 10)
        self.assertEqual(values["work_makespan_ticks"], 12)

    def test_zero_deficit_has_zero_business_times(self):
        instance = Instance("unchanged", (Task("background", 7, ()),), {}, ())
        values = quality_metrics(instance, {"background": (0, 7)})
        for name in ("weighted_deficit_unit_ticks", "first_activation_ticks",
                     "weighted_t90_ticks", "qos_all_ticks"):
            self.assertEqual(values[name], 0)
        self.assertEqual(values["work_makespan_ticks"], 7)

    def test_rollout_is_predeclared_small_subset_only(self):
        self.assertIn("closure_rollout", selected_policies(PROTOCOL, "holdout", 4000, 8))
        self.assertNotIn("closure_rollout", selected_policies(PROTOCOL, "holdout", 4003, 8))
        self.assertNotIn("closure_rollout", selected_policies(PROTOCOL, "scale", 5000, 20))
        self.assertEqual(PROTOCOL["scale_profiles"], ["mixed"])
        self.assertFalse(PROTOCOL["controls"]["holdout_used_for_tuning"])

    def test_rollout_pairing_does_not_use_unmatched_full_set_mean(self):
        rows = [row(4000, "closure", 10), row(4001, "closure", 1000),
                row(4000, "closure_rollout", 8)]
        comparisons = compare_rows(rows, PROTOCOL)
        result = next(item for item in comparisons if item["metric"] == "weighted_deficit_unit_ticks")
        self.assertEqual(result["n_paired_seeds"], 1)
        self.assertEqual(result["mean_baseline"], 10)
        self.assertEqual(result["mean_candidate"], 8)
        self.assertEqual(result["wins_ties_losses"], [1, 0, 0])
        self.assertEqual(result["seed_level_differences"], [2])

    def test_missing_state_evidence_does_not_become_zero(self):
        with self.assertRaises(ValueError):
            state_metrics({"state_bytes": 0})

    def test_rotation_is_complete_and_fingerprint_ignores_dict_order(self):
        names = ["full_domain", "diff_full_state", "closure"]
        self.assertEqual(rotated(names, 1), names[1:] + names[:1])
        self.assertEqual(set(rotated(names, 12)), set(names))
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))

    def test_development_model_negative_control(self):
        from executable_closure import build_plan, make_scenario
        from fast_engine import run_list
        from validate_closure import validate_plan
        scenario = make_scenario(3000, n_flows=8, profile="unchanged")
        for mode in ("full_domain", "diff_full_state", "closure"):
            plan = build_plan(scenario, mode)
            schedule = run_list(plan.instance)
            self.assertFalse(any(validate_plan(scenario, plan, schedule).values()))
            self.assertEqual(plan.metadata["activation_operations"], 0)
            self.assertEqual(quality_metrics(plan.instance, schedule)["weighted_deficit_unit_ticks"], 0)


if __name__ == "__main__":
    unittest.main()
