"""Independent raw-state replay, adversarial plan mutations and control cases."""

from dataclasses import replace
import unittest

from executable_closure import (EdgeSpec, FlowSpec, MODES, OwnerSpec, Plan, RuleSpec,
                                Scenario, StateSnapshot, StateSpec, build_plan, make_scenario)
from fast_engine import run_list
from recovery_benchmark import weighted_loss
from validate_closure import reconstruct_forwarding, validate_plan


def manual_scenario(*, coupled=False, hot=False, unchanged=False, detour=False):
    """One flow; state work 3 vs 7 ticks, 50 vs 150 logical bytes.

    The old path provides 4 of 10 rate units even after its controller dies.
    It is therefore a 2*(10-4)=12 weighted-deficit-unit flow, not a blackout.
    """
    nodes = ("s", "m", "e", "unused", "n")
    old_route = ("s", "m", "e")
    new_route = ("s", "n", "m", "e") if detour else old_route
    policy = 1 if unchanged else 2
    flow = FlowSpec("f", old_route, new_route, 4 if unchanged else 10,
                    4, 2, 1, policy)
    states = (
        StateSpec("base", 2, (), 20, 1),
        StateSpec("need", 2, ("base", "extra") if coupled else ("base",), 30, 2),
        StateSpec("extra", 2, (), 100, 4),
        StateSpec("boot", 3, (), 10, 1, True),
    )
    owners = tuple(OwnerSpec(switch, "failed", 1, "b", 2, "boot", 1)
                   for switch in nodes)
    existing = (RuleSpec("f", "s", "m", 1, (), installed_by_epoch=1),
                RuleSpec("f", "m", "e", 0, (), installed_by_epoch=1))
    wanted = tuple(RuleSpec("f", source, destination, policy, ("need",))
                   for source, destination in zip(new_route, new_route[1:]))
    warm = tuple(StateSnapshot("b", state.name, state.version) for state in states
                 if hot or state.name == "boot")
    capacities = {"replica_transfer": 1, "control:b": 1, "ingress_writer": 1,
                  **{f"write:{switch}": 1 for switch in nodes}}
    return Scenario(0, "manual", nodes,
                    (EdgeSpec("s", "m", 10), EdgeSpec("m", "e", 10),
                     EdgeSpec("s", "n", 10), EdgeSpec("n", "m", 10)),
                    (flow,), existing, wanted, owners, states, warm,
                    tuple(StateSnapshot("r", state.name, state.version) for state in states),
                    ("failed",), ("b", "r"),
                    {"b": ("base", "need", "extra")}, capacities, 2, 1)


def drop_action(plan, name):
    """A well-formed mutant DAG: erase metadata and repair all graph references."""
    tasks = tuple(replace(task, predecessors=tuple(p for p in task.predecessors if p != name))
                  for task in plan.instance.tasks if task.name != name)
    ops = {task.name: {**plan.operations[task.name], "predecessors": list(task.predecessors)}
           for task in tasks}
    return Plan(replace(plan.instance, tasks=tasks), {"mode": plan.metadata["mode"]}, ops)


def mutate_predecessors(plan, name, predecessors):
    tasks = tuple(replace(task, predecessors=tuple(predecessors)) if task.name == name else task
                  for task in plan.instance.tasks)
    ops = {key: dict(value) for key, value in plan.operations.items()}
    ops[name]["predecessors"] = list(predecessors)
    return replace(plan, instance=replace(plan.instance, tasks=tasks), operations=ops)


class ClosureSemanticValidationTests(unittest.TestCase):
    def assert_clean(self, scenario, plan):
        schedule = run_list(plan.instance)
        checks = validate_plan(scenario, plan, schedule)
        self.assertFalse(any(checks.values()), checks)
        return schedule

    def test_generated_development_profiles_replay_without_violations(self):
        for seed in (3000, 3001, 3002):
            for profile in ("mixed", "fully_coupled", "hot_standby", "no_overlap", "unchanged"):
                scenario = make_scenario(seed, 4, profile)
                for mode in MODES:
                    with self.subTest(seed=seed, profile=profile, mode=mode):
                        self.assert_clean(scenario, build_plan(scenario, mode))

    def test_manual_state_work_and_forwarding_have_hand_computable_values(self):
        scenario = manual_scenario()
        full, closure = build_plan(scenario, "diff_full_state"), build_plan(scenario, "closure")
        self.assertEqual(full.metadata["state_bytes"], 150)
        self.assertEqual(closure.metadata["state_bytes"], 50)
        self.assertEqual(sum(t.duration for t in full.instance.tasks if t.kind == "state"), 7)
        self.assertEqual(sum(t.duration for t in closure.instance.tasks if t.kind == "state"), 3)
        schedule = self.assert_clean(scenario, closure)
        self.assertEqual(schedule["activate:f"][1], 5)
        self.assertEqual(weighted_loss(closure.instance, schedule), 12 * 5)
        self.assertEqual(reconstruct_forwarding(scenario, scenario.existing_rules, target=False)["f"],
                         ("s", "m", "e"))
        self.assertNotIn("install:f:m", closure.operations)
        # The retained rule was written at epoch 1, while new writes require 2.
        self.assertEqual(scenario.existing_rules[1].installed_by_epoch, 1)

    def test_missing_state_is_detected_even_after_mutant_dag_is_repaired(self):
        scenario = manual_scenario()
        mutant = drop_action(build_plan(scenario, "closure"), "state:b:need")
        checks = validate_plan(scenario, mutant, run_list(mutant.instance))
        self.assertGreater(checks["state_version_violations"], 0)
        self.assertGreater(checks["scope_violations"], 0)

    def test_expired_hot_state_cannot_authorize_a_write(self):
        scenario = manual_scenario(hot=True)
        plan = build_plan(scenario, "closure")
        stale = replace(scenario, warm_state=tuple(
            replace(snapshot, version=1) if snapshot.state == "need" else snapshot
            for snapshot in scenario.warm_state))
        checks = validate_plan(stale, plan, run_list(plan.instance))
        self.assertGreater(checks["state_version_violations"], 0)
        self.assertGreater(checks["scope_violations"], 0)

    def test_rule_writes_require_restored_owner_and_correct_epoch(self):
        scenario = manual_scenario()
        plan = build_plan(scenario, "closure")
        mutant = drop_action(plan, "owner:s")
        checks = validate_plan(scenario, mutant, run_list(mutant.instance))
        self.assertGreater(checks["owner_epoch_violations"], 0)
        operations = {name: dict(op) for name, op in plan.operations.items()}
        operations["activate:f"]["owner_epoch"] = 1
        stale_epoch = replace(plan, operations=operations)
        checks = validate_plan(scenario, stale_epoch, run_list(stale_epoch.instance))
        self.assertGreater(checks["owner_epoch_violations"], 0)

    def test_ingress_waits_for_every_missing_downstream_rule(self):
        scenario = manual_scenario(detour=True)
        plan = build_plan(scenario, "closure")
        activation = next(t for t in plan.instance.tasks if t.name == "activate:f")
        mutant = mutate_predecessors(plan, "activate:f",
                                     [p for p in activation.predecessors if p != "install:f:n"])
        checks = validate_plan(scenario, mutant, run_list(mutant.instance))
        self.assertGreater(checks["precedence_violations"], 0)
        self.assertGreater(checks["activation_violations"], 0)

    def test_same_next_hop_with_old_policy_is_not_safe_target_reuse(self):
        scenario = manual_scenario()
        plan = build_plan(scenario, "closure")
        stale = replace(scenario, existing_rules=tuple(
            replace(rule, policy_version=1) if rule.switch == "m" else rule
            for rule in scenario.existing_rules))
        checks = validate_plan(stale, plan, run_list(plan.instance))
        self.assertGreater(checks["scope_violations"], 0)
        self.assertGreater(checks["forwarding_violations"], 0)

    def test_in_place_target_install_cannot_destroy_retained_old_path(self):
        scenario = manual_scenario()
        scenario = replace(scenario, existing_rules=tuple(
            replace(rule, policy_version=1) if rule.switch == "m" else rule
            for rule in scenario.existing_rules))
        plan = build_plan(scenario, "closure")
        operations = {name: dict(op) for name, op in plan.operations.items()}
        operations["install:f:m"]["staging"] = False
        mutant = replace(plan, operations=operations)
        checks = validate_plan(scenario, mutant, run_list(mutant.instance))
        self.assertGreater(checks["old_forwarding_violations"], 0)

    def test_scope_gate_detects_omitted_full_state_background_work(self):
        scenario = manual_scenario()
        mutant = drop_action(build_plan(scenario, "diff_full_state"), "state:b:extra")
        checks = validate_plan(scenario, mutant, run_list(mutant.instance))
        self.assertGreater(checks["scope_violations"], 0)

    def test_unrelated_background_state_cannot_be_a_global_ingress_barrier(self):
        scenario = manual_scenario()
        plan = build_plan(scenario, "diff_full_state")
        activation = next(t for t in plan.instance.tasks if t.name == "activate:f")
        mutant = mutate_predecessors(plan, "activate:f",
                                     (*activation.predecessors, "state:b:extra"))
        checks = validate_plan(scenario, mutant, run_list(mutant.instance))
        self.assertGreater(checks["unnecessary_barrier_violations"], 0)

    def test_no_transfer_source_and_false_transfer_size_are_detected(self):
        scenario = manual_scenario()
        plan = build_plan(scenario, "closure")
        broken = replace(scenario, surviving_replica_versions=tuple(
            snapshot for snapshot in scenario.surviving_replica_versions if snapshot.state != "need"))
        checks = validate_plan(broken, plan, run_list(plan.instance))
        self.assertGreater(checks["state_source_violations"], 0)
        operations = {name: dict(op) for name, op in plan.operations.items()}
        operations["state:b:need"]["size_bytes"] = 0
        checks = validate_plan(scenario, replace(plan, operations=operations), run_list(plan.instance))
        self.assertGreater(checks["metadata_violations"], 0)

    def test_fully_coupled_hot_and_unchanged_controls(self):
        coupled = manual_scenario(coupled=True)
        plans = [build_plan(coupled, mode) for mode in MODES]
        self.assertEqual({plan.metadata["state_bytes"] for plan in plans}, {150})
        hot = manual_scenario(hot=True)
        for mode in MODES:
            plan = build_plan(hot, mode)
            self.assertEqual(plan.metadata["state_objects"], 0)
            self.assertEqual(plan.metadata["state_bytes"], 0)
            self.assert_clean(hot, plan)
        unchanged = manual_scenario(unchanged=True)
        for mode in MODES:
            plan = build_plan(unchanged, mode)
            self.assertEqual(plan.instance.tasks, ())
            self.assert_clean(unchanged, plan)

    def test_unwritable_unused_switch_does_not_invalidate_retained_forwarding(self):
        scenario = manual_scenario()
        scenario = replace(scenario, owners=tuple(
            replace(owner, backup_controller="not-live") if owner.switch == "unused" else owner
            for owner in scenario.owners))
        self.assert_clean(scenario, build_plan(scenario, "closure"))
        with self.assertRaises(ValueError):
            build_plan(scenario, "full_domain")


if __name__ == "__main__":
    unittest.main()
