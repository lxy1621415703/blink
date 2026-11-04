"""Phase B: scheduling mechanisms in a reserved-path fluid recovery subscenario.

This is a new, self-contained simulation, NOT a reproduction of Ares, QUEST,
Dionysus or Migration Scheduling. A controller crash preserves the old data
path. A post-crash demand increase creates a constant, explicitly specified
service deficit; activating a pre-reserved feasible new path removes it. No
shared data-link congestion, packet drops, TCP, real controller protocol or
unknown future demand is simulated. Scheduling tasks and their durations are
known equally to every policy. See summary.json for full configuration.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

VERSION = "phase-b-fluid-dag-1.1"


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    duration: int
    resources: tuple[tuple[str, int], ...]
    predecessors: tuple[str, ...] = ()
    kind: str = "generic"


@dataclasses.dataclass(frozen=True)
class Flow:
    name: str
    weight: float
    demand: float
    old_service: float
    activation: str
    final_reserved_rate: float

    @property
    def deficit(self) -> float:
        return max(0.0, self.demand - self.old_service)


@dataclasses.dataclass(frozen=True)
class Instance:
    name: str
    tasks: tuple[Task, ...]
    capacities: dict[str, int]
    flows: tuple[Flow, ...]
    seed: int = 0
    shared: bool = True
    loss_scale: float = 1.0  # Synthetic rate units * uncalibrated simulation ticks.

    @property
    def horizon(self) -> int:
        return sum(t.duration for t in self.tasks)


Schedule = dict[str, tuple[int, int]]


def descendants(instance: Instance) -> dict[str, tuple[Flow, ...]]:
    children: dict[str, list[str]] = defaultdict(list)
    for task in instance.tasks:
        for pred in task.predecessors:
            children[pred].append(task.name)
    activations = {f.activation: f for f in instance.flows}
    result: dict[str, tuple[Flow, ...]] = {}

    def visit(name: str, seen: set[str]) -> None:
        if name in seen:
            return
        seen.add(name)
        for child in children[name]:
            visit(child, seen)

    for task in instance.tasks:
        nodes: set[str] = set()
        visit(task.name, nodes)
        result[task.name] = tuple(activations[n] for n in sorted(nodes)
                                  if n in activations)
    return result


def ready_tasks(instance: Instance, schedule: Schedule, now: int) -> list[Task]:
    usage: dict[str, int] = defaultdict(int)
    by_name = {t.name: t for t in instance.tasks}
    for name, (start, end) in schedule.items():
        if start <= now < end:
            for resource, amount in by_name[name].resources:
                usage[resource] += amount
    result = []
    for task in instance.tasks:
        if task.name in schedule:
            continue
        if not all(p in schedule and schedule[p][1] <= now
                   for p in task.predecessors):
            continue
        if all(usage[r] + amount <= instance.capacities[r]
               for r, amount in task.resources):
            result.append(task)
    return result


def weighted_loss(instance: Instance, schedule: Schedule) -> float:
    return sum(f.weight * f.deficit * min(schedule[f.activation][1], instance.horizon)
               * instance.loss_scale for f in instance.flows)


def makespan(schedule: Schedule) -> int:
    return max((end for _, end in schedule.values()), default=0)


def _baseline_key(task: Task, downstream: dict[str, tuple[Flow, ...]],
                  policy: str) -> tuple:
    flows = downstream[task.name]
    if policy == "priority_list":
        return (-max((f.weight for f in flows), default=0.0),
                task.duration, task.name)
    benefit = sum(f.weight * f.deficit for f in flows)
    return (-benefit / task.duration, task.duration, task.name)


def _complete(instance: Instance, partial: Schedule, now: int,
              policy: str, downstream: dict[str, tuple[Flow, ...]]) -> Schedule:
    schedule = dict(partial)
    while len(schedule) < len(instance.tasks):
        available = ready_tasks(instance, schedule, now)
        if available:
            task = min(available, key=lambda t: _baseline_key(t, downstream, policy))
            schedule[task.name] = (now, now + task.duration)
            continue
        next_events = [end for _, end in schedule.values() if end > now]
        if not next_events:
            raise ValueError("No feasible ready task: cyclic DAG or invalid capacities")
        now = min(next_events)
    return schedule


def list_schedule(instance: Instance, policy: str = "shared_successor_list") -> Schedule:
    return _complete(instance, {}, 0, policy, descendants(instance))


def rollout_schedule(instance: Instance, objective: str = "loss") -> Schedule:
    """Commit one ready action; complete each alternative with the SAME baseline.

    Completion policy is shared_successor_list for both rollout objectives.
    Decisions are work-conserving: deliberate idling is not a candidate action.
    """
    downstream = descendants(instance)
    schedule: Schedule = {}
    now = 0
    while len(schedule) < len(instance.tasks):
        available = ready_tasks(instance, schedule, now)
        if not available:
            ends = [end for _, end in schedule.values() if end > now]
            if not ends:
                raise ValueError("Invalid DAG/resources")
            now = min(ends)
            continue
        alternatives = []
        for task in available:
            partial = dict(schedule)
            partial[task.name] = (now, now + task.duration)
            completed = _complete(instance, partial, now, "shared_successor_list", downstream)
            if objective == "loss":
                value = (weighted_loss(instance, completed), makespan(completed), task.name)
            elif objective == "makespan":
                value = (makespan(completed), weighted_loss(instance, completed), task.name)
            else:
                raise ValueError(objective)
            alternatives.append((value, task))
        task = min(alternatives, key=lambda pair: pair[0])[1]
        schedule[task.name] = (now, now + task.duration)
    return schedule


def validate_schedule(instance: Instance, schedule: Schedule) -> dict[str, int]:
    violations = {"structure_violations": 0, "precedence_violations": 0,
                  "resource_violations": 0, "final_rate_violations": 0}
    names = {t.name for t in instance.tasks}
    violations["structure_violations"] += len(names.symmetric_difference(schedule))
    by_name = {t.name: t for t in instance.tasks}
    for name, (start, end) in schedule.items():
        if name not in by_name:
            continue
        task = by_name[name]
        if start < 0 or end - start != task.duration:
            violations["structure_violations"] += 1
        for pred in task.predecessors:
            if pred not in schedule or schedule[pred][1] > start:
                violations["precedence_violations"] += 1
    for now in sorted({x for interval in schedule.values() for x in interval}):
        usage: dict[str, int] = defaultdict(int)
        for name, (start, end) in schedule.items():
            if name in by_name and start <= now < end:
                for r, amount in by_name[name].resources:
                    usage[r] += amount
        for resource, amount in usage.items():
            if amount > instance.capacities.get(resource, 0):
                violations["resource_violations"] += 1
    for flow in instance.flows:
        if flow.final_reserved_rate < flow.demand or flow.old_service < 0:
            violations["final_rate_violations"] += 1
    return violations


def make_instance(seed: int, shared: bool = True, unchanged: bool = False) -> Instance:
    rng = np.random.default_rng(seed)
    n_flows = 3 + seed % 3
    assignments = [int(x) for x in rng.integers(0, 2, n_flows)]
    # Ensure at least one shared state group and both target controllers exist.
    assignments[0:3] = [0, 0, 1]
    capacities = {"replica_transfer": 1, "ctrl:0": 2, "ctrl:1": 2,
                  "ingress_writer": 1}
    groups = {c: [i for i, target in enumerate(assignments) if c == target]
              for c in range(2)}
    tasks: list[Task] = []
    state_pred: dict[int, str] = {}
    for ctrl, members in groups.items():
        unit = int(rng.integers(1, 4))
        # Shared/unshared use IDENTICAL aggregate preparation work/resources.
        # Splitting the preparation is a structural control, not free extra work.
        if shared:
            name = f"state:{ctrl}"
            tasks.append(Task(name, unit * len(members),
                              (("replica_transfer", 1), (f"ctrl:{ctrl}", 1)),
                              kind="surviving_replica_state"))
            for i in members:
                state_pred[i] = name
        else:
            for i in members:
                name = f"state:{ctrl}:flow:{i}"
                tasks.append(Task(name, unit,
                                  (("replica_transfer", 1), (f"ctrl:{ctrl}", 1)),
                                  kind="surviving_replica_state"))
                state_pred[i] = name
    flows = []
    for i, ctrl in enumerate(assignments):
        capacities[f"switch:{i}"] = 1
        owner, install, activation = f"owner:{i}", f"install:{i}", f"activate:{i}"
        tasks.extend([
            Task(owner, int(rng.integers(1, 3)), ((f"ctrl:{ctrl}", 1),),
                 (state_pred[i],), "ownership"),
            Task(install, int(rng.integers(1, 5)),
                 ((f"ctrl:{ctrl}", 1), (f"switch:{i}", 1)), (owner,), "rules"),
            Task(activation, 1, (("ingress_writer", 1),), (install,), "activation"),
        ])
        old = float(rng.integers(100, 501))
        demand = old + float(rng.integers(50, 401))
        weight = float(rng.choice([1, 2, 5, 10]))
        flows.append(Flow(f"flow:{i}", weight, old if unchanged else demand,
                          old, activation, demand))
    scenario = "negative_no_demand_change" if unchanged else ("shared_state" if shared else "unshared_equal_work")
    return Instance(scenario, tuple(tasks), capacities, tuple(flows), seed, shared)


def analytical_counterexample() -> Instance:
    """Collapsed task-graph unit example: priority loss 52, shared-first loss 38."""
    tasks = (Task("A", 2, (("S", 1),)), Task("E", 1, (("S", 1),)),
             Task("B", 1, (("I_B", 1),), ("E",)),
             Task("C", 1, (("I_C", 1),), ("E",)))
    flows = tuple(Flow(name, weight, 2, 1, name, 2)
                  for name, weight in [("A", 6), ("B", 5), ("C", 5)])
    return Instance("analytical_counterexample", tasks, {"S": 1, "I_B": 1, "I_C": 1},
                    flows, loss_scale=1)


def exact_schedule(instance: Instance, objective: str, seconds: float = 8) -> tuple[Schedule | None, dict]:
    """Integer-time, non-preemptive resource-constrained scheduling MILP.

    Unlike the heuristics, MILP can deliberately idle. Its incumbent is only
    called optimal when HiGHS reports status 0; status/gap are always retained.
    """
    horizon = instance.horizon
    options = {task.name: list(range(horizon - task.duration + 1)) for task in instance.tasks}
    columns: dict[tuple[str, int], int] = {}
    for task in instance.tasks:
        for start in options[task.name]:
            columns[(task.name, start)] = len(columns)
    n_start = len(columns)
    n_vars = n_start + (objective == "makespan")
    cost = np.zeros(n_vars)
    flow_cost = defaultdict(float)
    for flow in instance.flows:
        flow_cost[flow.activation] += flow.weight * flow.deficit * instance.loss_scale
    for task in instance.tasks:
        for start in options[task.name]:
            if objective == "loss":
                cost[columns[task.name, start]] = flow_cost[task.name] * (start + task.duration)
    if objective == "makespan":
        cost[-1] = 1
    elif objective != "loss":
        raise ValueError(objective)

    row_ids, col_ids, values = [], [], []
    lower, upper = [], []

    def add(coeffs: dict[int, float], lo: float, hi: float) -> None:
        row = len(lower)
        for col, value in coeffs.items():
            if value:
                row_ids.append(row); col_ids.append(col); values.append(value)
        lower.append(lo); upper.append(hi)

    by_name = {t.name: t for t in instance.tasks}
    for task in instance.tasks:
        add({columns[task.name, s]: 1 for s in options[task.name]}, 1, 1)
        for pred in task.predecessors:
            coeffs = {columns[task.name, s]: float(s) for s in options[task.name]}
            coeffs.update({columns[pred, s]: -float(s) for s in options[pred]})
            add(coeffs, by_name[pred].duration, math.inf)
        if objective == "makespan":
            coeffs = {columns[task.name, s]: float(s + task.duration) for s in options[task.name]}
            coeffs[n_start] = -1
            add(coeffs, -math.inf, 0)
    for resource, capacity in instance.capacities.items():
        for slot in range(horizon):
            coeffs = {}
            for task in instance.tasks:
                amount = dict(task.resources).get(resource, 0)
                if amount:
                    for start in range(max(0, slot - task.duration + 1),
                                       min(slot, horizon - task.duration) + 1):
                        coeffs[columns[task.name, start]] = amount
            if coeffs:
                add(coeffs, -math.inf, capacity)
    matrix = coo_matrix((values, (row_ids, col_ids)), shape=(len(lower), n_vars)).tocsc()
    integrality = np.ones(n_vars)
    bounds_upper = np.ones(n_vars)
    if objective == "makespan":
        integrality[-1] = 0
        bounds_upper[-1] = horizon
    result = milp(cost, integrality=integrality,
                  bounds=Bounds(np.zeros(n_vars), bounds_upper),
                  constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                  options={"time_limit": seconds, "mip_rel_gap": 0.0})
    info = {"solver_status": int(result.status), "solver_message": str(result.message),
            "optimal": bool(result.status == 0),
            "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None}
    if result.x is None:
        return None, info
    schedule = {}
    for task in instance.tasks:
        selected = [s for s in options[task.name] if result.x[columns[task.name, s]] > 0.5]
        if len(selected) != 1:
            return None, {**info, "optimal": False, "extraction_error": True}
        schedule[task.name] = (selected[0], selected[0] + task.duration)
    return schedule, info


POLICIES: dict[str, Callable[[Instance], Schedule]] = {
    "priority_list": lambda i: list_schedule(i, "priority_list"),
    "shared_successor_list": lambda i: list_schedule(i, "shared_successor_list"),
    "makespan_rollout": lambda i: rollout_schedule(i, "makespan"),
    "loss_rollout": lambda i: rollout_schedule(i, "loss"),
}


def _result(instance: Instance, name: str, schedule: Schedule, runtime: float, info: dict | None = None) -> dict:
    checks = validate_schedule(instance, schedule)
    if any(checks.values()):
        raise RuntimeError(f"Invalid {instance.name}/{instance.seed}/{name}: {checks}")
    return {"version": VERSION, "seed": instance.seed, "scenario": instance.name,
            "policy": name, "n_flows": len(instance.flows), "n_tasks": len(instance.tasks),
            "total_actual_deficit_rate_units": sum(f.deficit for f in instance.flows),
            "total_demand_rate_units": sum(f.demand for f in instance.flows),
            "total_old_service_rate_units": sum(f.old_service for f in instance.flows),
            "weighted_unmet_demand_units": weighted_loss(instance, schedule),
            "makespan_ticks": makespan(schedule), "compute_ms": runtime * 1000,
            **checks, "solver_status": (info or {}).get("solver_status", ""),
            "solver_optimal": (info or {}).get("optimal", ""),
            "mip_gap": (info or {}).get("mip_gap", "")}


def run_benchmark(output: Path, seeds: int = 24, exact_seeds: int = 2,
                  milp_seconds: float = 8) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows, instances, solver_records, schedule_records = [], [], [], []
    for seed in range(seeds):
        variants = [make_instance(seed, True), make_instance(seed, False),
                    make_instance(seed, True, unchanged=True)]
        for instance in variants:
            instance_record = dataclasses.asdict(instance)
            instance_record["flows"] = [{**dataclasses.asdict(f), "actual_deficit_rate_units": f.deficit}
                                        for f in instance.flows]
            instances.append(instance_record)
            for policy, executor in POLICIES.items():
                before = time.perf_counter()
                schedule = executor(instance)
                rows.append(_result(instance, policy, schedule, time.perf_counter() - before))
                schedule_records.append({"seed": seed, "scenario": instance.name,
                                         "policy": policy, "schedule": schedule})
            # Exact references only on small, preselected instances, no cherry-picking by outcome.
            if seed < exact_seeds and len(instance.flows) == 3 and "negative" not in instance.name:
                for objective in ["makespan", "loss"]:
                    before = time.perf_counter()
                    schedule, info = exact_schedule(instance, objective, milp_seconds)
                    runtime = time.perf_counter() - before
                    name = f"milp_{objective}_reference"
                    solver_records.append({"seed": seed, "scenario": instance.name,
                                           "policy": name, **info})
                    if schedule is not None:
                        rows.append(_result(instance, name, schedule, runtime, info))
                        schedule_records.append({"seed": seed, "scenario": instance.name,
                                                 "policy": name, "schedule": schedule})
        print(f"completed paired seed {seed + 1}/{seeds}", flush=True)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    summary.update({
        "version": VERSION,
        "execution": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                      "platform": platform.platform()},
        "configuration": {"paired_seeds": list(range(seeds)), "exact_seed_selection":
                          "seed < exact_seeds AND n_flows == 3; both shared/unshared; no negative reference",
                          "exact_seeds": exact_seeds, "milp_time_limit_seconds": milp_seconds,
                          "rate_unit": "synthetic rate units",
                          "time_unit": "uncalibrated simulation ticks",
                          "loss_unit": "priority-weighted rate-unit*tick",
                          "flow_count": "3 + seed % 3", "known_task_durations": True,
                          "work_conserving_rollout": True,
                          "rollout_completion_policy": "shared_successor_list",
                          "final_plan_fixed_across_policies": True,
                          "old_data_path_preserved": True,
                          "shared_unshared_equal_total_preparation_work": True},
        "scope": ["Self-implemented mechanism simulation; no published algorithm reproduced.",
                  "Constant post-failure demand; each original path provides old_service > 0.",
                  "New paths are pre-reserved and independently feasible, so each activation removes its own deficit.",
                  "No competing data links, queues, TCP, packet measurements or real SDN ownership protocol.",
                  "State, ownership, rule and activation actions are simulated duration/resource tasks.",
                  "Task durations are uncalibrated ticks, not measured device installation or recovery latencies.",
                  "All policies know the same durations and demands; exact references can additionally idle.",
                  "The unshared equal-work control changes DAG granularity, not just an algorithm flag.",
                  "Only zero-gap solver references are global optima; time-limited incumbents are labelled."],
        "solver_records": solver_records,
        "instances": instances,
        "schedules": schedule_records,
    })
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                   allow_nan=False), encoding="utf-8")
    return summary


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["policy"])].append(row)
    aggregates = []
    for (scenario, policy), values in sorted(grouped.items()):
        aggregates.append({"scenario": scenario, "policy": policy, "n": len(values),
                           "mean_weighted_unmet_demand_units": statistics.mean(v["weighted_unmet_demand_units"] for v in values),
                           "mean_makespan_ticks": statistics.mean(v["makespan_ticks"] for v in values),
                           "mean_compute_ms": statistics.mean(v["compute_ms"] for v in values)})
    paired = []
    for scenario in sorted({r["scenario"] for r in rows if "negative" not in r["scenario"]}):
        candidate = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == "loss_rollout"}
        for baseline in ["priority_list", "shared_successor_list", "makespan_rollout"]:
            reference = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == baseline}
            keys = sorted(candidate.keys() & reference.keys())
            differences = np.array([reference[s]["weighted_unmet_demand_units"] - candidate[s]["weighted_unmet_demand_units"] for s in keys])
            rng = np.random.default_rng(20261104)
            bootstrap = rng.choice(differences, size=(2000, len(keys)), replace=True).mean(axis=1)
            base_mean = np.mean([reference[s]["weighted_unmet_demand_units"] for s in keys])
            paired.append({"scenario": scenario, "baseline": baseline, "n": len(keys),
                           "mean_paired_reduction_units": float(differences.mean()),
                           "relative_reduction_of_means_pct": float(100 * differences.mean() / base_mean),
                           "bootstrap_95pct_ci_reduction_units": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
                           "wins_ties_losses": [int((differences > 1e-8).sum()), int((abs(differences) <= 1e-8).sum()), int((differences < -1e-8).sum())]})
    exact_gaps = []
    for row in rows:
        if row["policy"] == "milp_loss_reference" and row["solver_optimal"] is True:
            candidate = next(r for r in rows if r["seed"] == row["seed"] and r["scenario"] == row["scenario"] and r["policy"] == "loss_rollout")
            exact_gaps.append({"seed": row["seed"], "scenario": row["scenario"],
                               "optimal_loss_units": row["weighted_unmet_demand_units"],
                               "rollout_loss_units": candidate["weighted_unmet_demand_units"],
                               "gap_pct": 100 * (candidate["weighted_unmet_demand_units"] / row["weighted_unmet_demand_units"] - 1)})
    violation_columns = [k for k in rows[0] if k.endswith("_violations")]
    return {"aggregates": aggregates, "paired_loss_comparisons": paired,
            "exact_loss_gaps": exact_gaps,
            "total_violations": {k: sum(r[k] for r in rows) for k in violation_columns},
            "all_negative_controls_zero": all(r["weighted_unmet_demand_units"] == 0 for r in rows if "negative" in r["scenario"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--exact-seeds", type=int, default=2)
    parser.add_argument("--milp-seconds", type=float, default=8)
    args = parser.parse_args()
    if args.seeds < 1 or args.exact_seeds < 0 or args.milp_seconds <= 0:
        parser.error("seeds/time must be positive; exact-seeds must be nonnegative")
    result = run_benchmark(args.output, args.seeds, args.exact_seeds, args.milp_seconds)
    print(json.dumps({k: result[k] for k in ["paired_loss_comparisons", "exact_loss_gaps",
                                           "total_violations", "all_negative_controls_zero"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

