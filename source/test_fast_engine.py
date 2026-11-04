"""Schedule-equivalence and feasibility checks for the incremental engine."""

import dataclasses
import unittest

from fast_engine import (EngineState, compile_instance, complete_state, run_compiled,
                         run_list, run_rollout)
from recovery_benchmark import (Flow, Instance, Task, analytical_counterexample,
                                list_schedule, make_instance, rollout_schedule,
                                validate_schedule, weighted_loss)


class FastEngineTests(unittest.TestCase):
    def test_original_thirty_seed_schedules_are_identical(self):
        for seed in range(30):
            for shared, unchanged in ((True, False), (False, False), (True, True)):
                instance = make_instance(seed, shared, unchanged)
                for policy in ("priority_list", "shared_successor_list"):
                    with self.subTest(seed=seed, shared=shared, unchanged=unchanged, policy=policy):
                        self.assertEqual(run_list(instance, policy), list_schedule(instance, policy))
                for objective in ("loss", "makespan"):
                    with self.subTest(seed=seed, shared=shared, unchanged=unchanged, objective=objective):
                        fast = run_rollout(instance, objective)
                        self.assertEqual(fast, rollout_schedule(instance, objective))
                        self.assertFalse(any(validate_schedule(instance, fast).values()))

    def test_analytical_losses_and_tie_breaks(self):
        instance = analytical_counterexample()
        priority = run_list(instance, "priority_list")
        shared = run_list(instance, "shared_successor_list")
        self.assertEqual(weighted_loss(instance, priority), 52)
        self.assertEqual(weighted_loss(instance, shared), 38)
        self.assertEqual(priority, list_schedule(instance, "priority_list"))
        self.assertEqual(shared, list_schedule(instance, "shared_successor_list"))
        for objective in ("loss", "makespan"):
            self.assertEqual(run_rollout(instance, objective), rollout_schedule(instance, objective))

    def test_simultaneous_events_and_multiple_predecessors(self):
        tasks = (
            Task("A", 2, (("r0", 1),)), Task("B", 2, (("r1", 1),)),
            Task("C", 3, (("r0", 1), ("r1", 1)), ("A", "B")),
            Task("D", 1, (("r2", 1),), ("A",)),
        )
        instance = Instance("joint", tasks, {"r0": 1, "r1": 1, "r2": 1},
                            (Flow("F", 3, 2, 1, "C", 2), Flow("G", 2, 3, 1, "D", 3)))
        for policy in ("priority_list", "shared_successor_list"):
            self.assertEqual(run_list(instance, policy), list_schedule(instance, policy))
        state = EngineState(compile_instance(instance))
        state.start(0)
        state.start(1)
        self.assertTrue(state.advance())
        self.assertEqual(state.now, 2)
        self.assertEqual(set(state.available()), {2, 3})
        self.assertEqual(state.usage, [0, 0, 0])

    def test_clones_do_not_mutate_parent_or_other_strategy(self):
        instance = make_instance(4)
        compiled = compile_instance(instance)
        first = EngineState(compiled)
        second = first.clone()
        second.start(second.available()[0])
        self.assertEqual(len(first.start_order), 0)
        self.assertEqual(first.usage, [0] * len(compiled.capacities))
        complete_state(second)
        self.assertEqual(first.to_schedule(), {})
        self.assertIsNot(compiled, compile_instance(instance))
        for policy in ("priority_list", "shared_successor_list"):
            self.assertEqual(run_compiled(compiled, policy), list_schedule(instance, policy))

    def test_larger_instance_resource_and_dependency_feasibility(self):
        tasks, flows = [], []
        capacities = {"transfer": 2, "writer": 2}
        for controller in range(4):
            capacities[f"ctrl:{controller}"] = 3
        for group in range(20):
            controller = group % 4
            state_name = f"state:{group}"
            tasks.append(Task(state_name, 1 + group % 4,
                              (("transfer", 1), (f"ctrl:{controller}", 1))))
            for offset in range(4):
                index = 4 * group + offset
                owner, install, activate = f"owner:{index}", f"install:{index}", f"activate:{index}"
                capacities[f"switch:{index}"] = 1
                tasks.extend((
                    Task(owner, 1 + index % 2, ((f"ctrl:{controller}", 1),), (state_name,)),
                    Task(install, 1 + index % 4,
                         ((f"ctrl:{controller}", 1), (f"switch:{index}", 1)), (owner,)),
                    Task(activate, 1, (("writer", 1),), (install,)),
                ))
                flows.append(Flow(f"flow:{index}", 1 + index % 10, 100, 40, activate, 100))
        instance = Instance("larger", tuple(tasks), capacities, tuple(flows))
        self.assertEqual(len(instance.tasks), 260)
        for policy in ("priority_list", "shared_successor_list"):
            schedule = run_list(instance, policy)
            self.assertFalse(any(validate_schedule(instance, schedule).values()))
            self.assertEqual(schedule, list_schedule(instance, policy))

    def test_invalid_dag_is_rejected_at_compilation(self):
        instance = analytical_counterexample()
        cyclic = dataclasses.replace(instance, tasks=(Task("A", 1, (("S", 1),), ("A",)),))
        with self.assertRaises(ValueError):
            compile_instance(cyclic)


if __name__ == "__main__":
    unittest.main()

