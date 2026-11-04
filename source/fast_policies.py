"""Fixed, inexpensive DAG scheduling candidates for the Phase B fluid model.

Policy definitions were fixed before evaluating holdout seeds 1000..1029.
All candidates use identical tasks, resources, demands and known durations.
There is no MILP, packet-level claim, or per-action full-schedule rollout here.
The portfolio's feasibility/dominance filter applies to simulated task time;
its additional real compute time MUST be reported separately.
"""
from __future__ import annotations

import heapq
from typing import Callable

from fast_engine import compile_instance, run_compiled
from recovery_benchmark import Instance, Schedule, makespan, weighted_loss

POLICY_SET_VERSION = "fast-list-portfolio-v1.0"

POLICY_DESCRIPTIONS = {
    "critical_path_list": (
        "Descending longest remaining task-DAG path; tie-break by downstream "
        "weighted service-deficit urgency, then duration and task name."
    ),
    "weighted_critical_list": (
        "Descending sum over descendant flows of weight*actual_deficit divided "
        "by the longest remaining task path to that flow's activation; "
        "tie-break by longest overall remaining path, duration and name."
    ),
    "safe_portfolio": (
        "Compile once, construct exactly three plans: shared-successor baseline, "
        "critical-path list, weighted-critical list. Retain plans whose simulated "
        "makespan is no greater than baseline, then minimize weighted loss, "
        "makespan and fixed baseline-first tie-break. Wall-clock planning cost "
        "is not covered by this dominance guarantee."
    ),
}


def _topological_order(compiled) -> tuple[int, ...]:
    """Stable Kahn traversal, independent of input task storage order."""
    indegree = [0] * len(compiled.tasks)
    for children in compiled.successors:
        for child in children:
            indegree[child] += 1
    available = [(task.name, i) for i, task in enumerate(compiled.tasks)
                 if indegree[i] == 0]
    heapq.heapify(available)
    result = []
    while available:
        _, current = heapq.heappop(available)
        result.append(current)
        for child in compiled.successors[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(available, (compiled.tasks[child].name, child))
    if len(result) != len(compiled.tasks):
        raise ValueError("Task graph contains a cycle")
    return tuple(result)


def _priority_keys(compiled) -> tuple[tuple[tuple, ...], tuple[tuple, ...]]:
    """Two static rankings; O(number_of_flows * (V+E)) preprocessing.

    Path lengths include the ready action itself. Distinct flows sharing an
    activation remain distinct contributions. Deficit, rather than full demand,
    is the rate that can actually be improved under the fluid subscenario.
    """
    order = _topological_order(compiled)
    durations = [task.duration for task in compiled.tasks]
    longest = [0] * len(durations)
    for current in reversed(order):
        longest[current] = durations[current] + max(
            (longest[child] for child in compiled.successors[current]), default=0)
    urgency = [0.0] * len(durations)
    for flow in compiled.instance.flows:
        benefit = flow.weight * flow.deficit
        if benefit == 0:
            continue
        activation = compiled.task_index[flow.activation]
        distance = [0] * len(durations)
        for current in reversed(order):
            if current == activation:
                distance[current] = durations[current]
            else:
                child_distance = max((distance[child] for child in compiled.successors[current]),
                                     default=0)
                if child_distance:
                    distance[current] = durations[current] + child_distance
            if distance[current]:
                urgency[current] += benefit / distance[current]
    critical = tuple((-longest[i], -urgency[i], task.duration, task.name)
                     for i, task in enumerate(compiled.tasks))
    weighted = tuple((-urgency[i], -longest[i], task.duration, task.name)
                     for i, task in enumerate(compiled.tasks))
    return critical, weighted


def critical_path_list(instance: Instance) -> Schedule:
    compiled = compile_instance(instance)
    critical, _ = _priority_keys(compiled)
    return run_compiled(compiled, keys=critical)


def weighted_critical_list(instance: Instance) -> Schedule:
    compiled = compile_instance(instance)
    _, weighted = _priority_keys(compiled)
    return run_compiled(compiled, keys=weighted)


def safe_portfolio(instance: Instance) -> Schedule:
    """Fixed three-plan Pareto filter, retaining the strong baseline itself."""
    compiled = compile_instance(instance)
    critical, weighted = _priority_keys(compiled)
    baseline = run_compiled(compiled, policy="shared_successor_list")
    plans = [baseline, run_compiled(compiled, keys=critical),
             run_compiled(compiled, keys=weighted)]
    baseline_span = makespan(baseline)
    eligible = [(plan, position) for position, plan in enumerate(plans)
                if makespan(plan) <= baseline_span]
    return min(eligible, key=lambda item: (weighted_loss(instance, item[0]),
                                           makespan(item[0]), item[1]))[0]


FAST_POLICIES: dict[str, Callable[[Instance], Schedule]] = {
    "critical_path_list": critical_path_list,
    "weighted_critical_list": weighted_critical_list,
    "safe_portfolio": safe_portfolio,
}

