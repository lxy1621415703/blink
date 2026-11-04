"""Operation-scope and version semantics; development seeds only."""
from __future__ import annotations

import dataclasses
import unittest

from executable_closure import MODES, PROFILES, RuleSpec, build_plan, make_scenario
from fast_engine import run_list
from recovery_benchmark import validate_schedule, weighted_loss


def ancestors(instance, name):
    tasks = {t.name: t for t in instance.tasks}
    result = set()
    def visit(current):
        for pred in tasks[current].predecessors:
            if pred not in result:
                result.add(pred)
                visit(pred)
    visit(name)
    return result


class ExecutableClosureTests(unittest.TestCase):
    def test_all_profiles_schedule_and_real_edges(self):
        for profile in PROFILES:
            scenario = make_scenario(3000, 4, profile)
            edge = {(e.source, e.target): e.capacity for e in scenario.edges}
            for flow in scenario.flows:
                self.assertGreater(flow.old_service, 0)
                for route, rate in [(flow.old_route, flow.old_service),
                                    (flow.new_route, flow.demand)]:
                    self.assertEqual(len(route), len(set(route)))
                    self.assertTrue(all(edge[a, b] >= rate for a, b in zip(route, route[1:])))
            for mode in MODES:
                plan = build_plan(scenario, mode)
                schedule = run_list(plan.instance, "shared_successor_list")
                self.assertFalse(any(validate_schedule(plan.instance, schedule).values()))
                self.assertEqual(len(plan.operations), len(plan.instance.tasks))
                if profile == "unchanged":
                    self.assertEqual(plan.instance.tasks, ())
                    self.assertEqual(plan.instance.flows, ())
                    self.assertEqual(weighted_loss(plan.instance, schedule), 0)

    def test_stale_matching_rule_requires_target_write(self):
        scenario = make_scenario(3001, 3)
        rule = next(r for r in scenario.target_rules if r.flow == "f0" and r.switch == "f0:b")
        retained = tuple(r for r in scenario.existing_rules
                         if (r.flow, r.switch) != (rule.flow, rule.switch))
        # The construction-time application read set can change without
        # invalidating a semantically correct materialized forwarding rule.
        valid = dataclasses.replace(rule, policy_version=0, installed_by_epoch=1,
                                    required_states=())
        reusable = dataclasses.replace(scenario, existing_rules=retained + (valid,))
        good = build_plan(reusable, "closure")
        self.assertNotIn("install:f0:f0:b", good.operations)
        stale = dataclasses.replace(rule, policy_version=rule.policy_version - 1)
        expired = dataclasses.replace(scenario, existing_rules=retained + (stale,))
        rewritten = build_plan(expired, "closure")
        self.assertIn("install:f0:f0:b", rewritten.operations)
        self.assertTrue(rewritten.operations["install:f0:f0:b"]["staging"])

    def test_warm_parent_does_not_hide_missing_dependencies(self):
        scenario = make_scenario(3002, 3)
        # Make all policy leaf objects warm but none of their dependencies.
        states = {s.name: s for s in scenario.states}
        warm = tuple(snapshot for snapshot in scenario.surviving_replica_versions
                     if snapshot.state.endswith(":policy"))
        snapshots = tuple(dataclasses.replace(s, controller=target)
                          for target in ("backup0", "backup1", "live0") for s in warm)
        scenario = dataclasses.replace(scenario, warm_state=snapshots)
        plan = build_plan(scenario, "closure")
        activation = plan.operations["activate:f0"]
        controller = activation["controller"]
        self.assertNotIn(f"state:{controller}:app:f0:policy", plan.operations)
        deps = ancestors(plan.instance, "activate:f0")
        for state_name in ("app:topology", "app:reservations"):
            self.assertIn(f"state:{controller}:{state_name}", deps)

    def test_hot_standby_never_rehydrates_valid_states(self):
        scenario = make_scenario(3003, 4, "hot_standby")
        for mode in MODES:
            plan = build_plan(scenario, mode)
            self.assertEqual(plan.metadata["state_objects"], 0)
            self.assertEqual(plan.metadata["state_bytes"], 0)
            self.assertFalse(any(op["kind"] == "state" for op in plan.operations.values()))

    def test_fully_coupled_state_scope_matches_diff_full_state(self):
        scenario = make_scenario(3004, 4, "fully_coupled")
        closure = build_plan(scenario, "closure")
        full = build_plan(scenario, "diff_full_state")
        self.assertEqual({name for name, op in closure.operations.items() if op["kind"] == "state"},
                         {name for name, op in full.operations.items() if op["kind"] == "state"})
        self.assertEqual(closure.metadata["participating_closure_state_fraction"], 1)

    def test_unrelated_background_work_is_not_an_activation_barrier(self):
        scenario = dataclasses.replace(make_scenario(3005, 4), warm_state=())
        plan = build_plan(scenario, "full_domain")
        required_by_any_flow = set().union(*(ancestors(plan.instance, f"activate:f{i}")
                                            for i in range(4)))
        background = {name for name, op in plan.operations.items()
                      if op["kind"] == "state" and op["state"].endswith(":statistics")}
        self.assertTrue(background)
        self.assertTrue(background.isdisjoint(required_by_any_flow))
        for name, op in plan.operations.items():
            if op["kind"] == "activation":
                for install, info in plan.operations.items():
                    if info["kind"] == "install" and info["flow"] == op["flow"]:
                        self.assertIn(install, ancestors(plan.instance, name))

    def test_full_domain_includes_app_state_for_background_owner_target(self):
        scenario = dataclasses.replace(make_scenario(3006, 1), warm_state=())
        owners = tuple(dataclasses.replace(o, current_controller="live0", backup_controller="live0",
                                           target_epoch=1, bootstrap_state="bootstrap:live0")
                       if o.switch != "f0:c" else
                       dataclasses.replace(o, current_controller="failed0", backup_controller="backup1",
                                           target_epoch=2, bootstrap_state="bootstrap:backup1")
                       for o in scenario.owners)
        scenario = dataclasses.replace(scenario, owners=owners)
        full, diff = (build_plan(scenario, mode) for mode in ("full_domain", "diff_full_state"))
        self.assertEqual(full.metadata["participating_targets"], ["live0"])
        self.assertIn("state:backup1:app:topology", full.operations)
        self.assertNotIn("state:backup1:app:topology", diff.operations)
        self.assertEqual(full.metadata["total_application_state_objects"],
                         diff.metadata["total_application_state_objects"])

    def test_absent_surviving_source_and_illegal_backup_are_rejected(self):
        scenario = dataclasses.replace(make_scenario(3007, 3), warm_state=(), surviving_replica_versions=())
        with self.assertRaisesRegex(ValueError, "No surviving replica"):
            build_plan(scenario, "closure")
        scenario = make_scenario(3007, 3)
        owners = tuple(dataclasses.replace(o, current_controller="failed0", backup_controller="dead_backup",
                                           target_epoch=2) if o.switch == "f0:entry" else o
                       for o in scenario.owners)
        with self.assertRaisesRegex(ValueError, "No surviving legal target"):
            build_plan(dataclasses.replace(scenario, owners=owners), "closure")

    def test_unrelated_unrecoverable_owner_does_not_block_partial_scope(self):
        scenario = make_scenario(3008, 1)
        owners = tuple(dataclasses.replace(o, current_controller="failed0",
                                           backup_controller="dead_backup", target_epoch=2)
                       if o.switch == "f0:c" else o for o in scenario.owners)
        scenario = dataclasses.replace(scenario, owners=owners)
        for mode in ("closure", "diff_full_state"):
            plan = build_plan(scenario, mode)
            schedule = run_list(plan.instance, "shared_successor_list")
            self.assertFalse(any(validate_schedule(plan.instance, schedule).values()))
        with self.assertRaisesRegex(ValueError, "No surviving legal target"):
            build_plan(scenario, "full_domain")


if __name__ == "__main__":
    unittest.main()
