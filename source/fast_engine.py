"""Incremental, event-driven execution of the Phase B scheduling policies.

The scheduling problem, policy scores and tie-breaks match recovery_benchmark.
Compilation is explicit and never cached: callers must include compile_instance
in the timed policy invocation. This module changes execution cost, not the
recovery model or its synthetic units.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from recovery_benchmark import Flow, Instance, Schedule, Task


@dataclass(frozen=True, slots=True)
class CompiledInstance:
    instance: Instance
    tasks: tuple[Task, ...]
    task_index: Mapping[str, int]
    durations: tuple[int, ...]
    resource_names: tuple[str, ...]
    resource_index: Mapping[str, int]
    capacities: tuple[int, ...]
    resources: tuple[tuple[tuple[int, int], ...], ...]
    predecessors: tuple[tuple[int, ...], ...]
    successors: tuple[tuple[int, ...], ...]
    topological_order: tuple[int, ...]
    downstream: tuple[tuple[Flow, ...], ...]
    priority_keys: tuple[tuple, ...]
    successor_keys: tuple[tuple, ...]
    loss_terms: tuple[tuple[int, float], ...]
    horizon: int


def compile_instance(instance: Instance) -> CompiledInstance:
    """Build indexes, resource vectors, DAG links and policy keys exactly once."""
    tasks = instance.tasks
    task_index = {task.name: i for i, task in enumerate(tasks)}
    if len(task_index) != len(tasks):
        raise ValueError("Task names must be unique")
    durations = tuple(int(task.duration) for task in tasks)
    if any(duration <= 0 or duration != task.duration
           for task, duration in zip(tasks, durations)):
        raise ValueError("Task durations must be positive integer simulation ticks")
    resource_names = tuple(instance.capacities)
    resource_index = {name: i for i, name in enumerate(resource_names)}
    capacities = tuple(int(instance.capacities[name]) for name in resource_names)
    if any(value < 0 or value != instance.capacities[name]
           for name, value in zip(resource_names, capacities)):
        raise ValueError("Resource capacities must be nonnegative integers")
    resources = []
    predecessors = []
    children: list[list[int]] = [[] for _ in tasks]
    for i, task in enumerate(tasks):
        resource_vector = []
        seen_resources = set()
        for name, amount in task.resources:
            if name in seen_resources:
                raise ValueError("A task must list each resource at most once")
            seen_resources.add(name)
            if name not in resource_index:
                raise ValueError(f"Unknown resource: {name}")
            index = resource_index[name]
            if int(amount) != amount or amount < 0 or amount > capacities[index]:
                raise ValueError(f"Task {task.name} exceeds resource capacity")
            resource_vector.append((index, int(amount)))
        resources.append(tuple(resource_vector))
        try:
            parents = tuple(dict.fromkeys(task_index[name] for name in task.predecessors))
        except KeyError as error:
            raise ValueError(f"Unknown predecessor: {error.args[0]}") from error
        predecessors.append(parents)
        for parent in parents:
            children[parent].append(i)

    pending = [len(parents) for parents in predecessors]
    roots = deque(i for i, count in enumerate(pending) if count == 0)
    topological = []
    while roots:
        current = roots.popleft()
        topological.append(current)
        for child in children[current]:
            pending[child] -= 1
            if pending[child] == 0:
                roots.append(child)
    if len(topological) != len(tasks):
        raise ValueError("Task graph contains a cycle")

    # Preserve the reference descendants() convention: the final Flow for a
    # duplicated activation is used by priority keys, while loss counts all Flows.
    activation_flows = {flow.activation: flow for flow in instance.flows}
    unknown = set(activation_flows) - task_index.keys()
    if unknown:
        raise ValueError(f"Flow activation tasks are missing: {sorted(unknown)}")
    activation_names = sorted(activation_flows)
    activation_bits = {name: 1 << i for i, name in enumerate(activation_names)}
    masks = [activation_bits.get(task.name, 0) for task in tasks]
    for current in reversed(topological):
        for child in children[current]:
            masks[current] |= masks[child]
    downstream = tuple(
        tuple(activation_flows[name] for i, name in enumerate(activation_names)
              if mask & (1 << i))
        for mask in masks
    )
    priority_keys = tuple(
        (-max((flow.weight for flow in flows), default=0.0), task.duration, task.name)
        for task, flows in zip(tasks, downstream)
    )
    successor_keys = tuple(
        (-sum(flow.weight * flow.deficit for flow in flows) / task.duration,
         task.duration, task.name)
        for task, flows in zip(tasks, downstream)
    )
    return CompiledInstance(
        instance, tasks, MappingProxyType(task_index), durations, resource_names,
        MappingProxyType(resource_index), capacities, tuple(resources),
        tuple(predecessors), tuple(tuple(values) for values in children),
        tuple(topological), downstream, priority_keys, successor_keys,
        tuple((task_index[flow.activation], flow.weight * flow.deficit)
              for flow in instance.flows), sum(durations),
    )


class EngineState:
    """Mutable scheduling frontier; immutable compiled data is shared by clones."""

    __slots__ = ("compiled", "now", "starts", "ends", "pending", "usage",
                 "running", "ready", "start_order", "recent_ready", "finish_time")

    def __init__(self, compiled: CompiledInstance):
        self.compiled = compiled
        self.now = 0
        self.starts = [-1] * len(compiled.tasks)
        self.ends = [-1] * len(compiled.tasks)
        self.pending = [len(values) for values in compiled.predecessors]
        self.usage = [0] * len(compiled.capacities)
        self.running: list[tuple[int, int]] = []
        self.ready = {i for i, count in enumerate(self.pending) if count == 0}
        self.start_order: list[int] = []
        self.recent_ready = tuple(sorted(self.ready))
        self.finish_time = 0

    @property
    def finished_scheduling(self) -> bool:
        return len(self.start_order) == len(self.compiled.tasks)

    def fits(self, task_id: int) -> bool:
        compiled = self.compiled
        return all(self.usage[index] + amount <= compiled.capacities[index]
                   for index, amount in compiled.resources[task_id])

    def available(self) -> tuple[int, ...]:
        # Original ready_tasks() returns candidates in instance order.
        return tuple(task_id for task_id in sorted(self.ready) if self.fits(task_id))

    def start(self, task_id: int) -> None:
        if task_id not in self.ready or not self.fits(task_id):
            raise ValueError("Cannot start a blocked or already scheduled task")
        end = self.now + self.compiled.durations[task_id]
        self.starts[task_id] = self.now
        self.ends[task_id] = end
        self.finish_time = max(self.finish_time, end)
        self.start_order.append(task_id)
        self.ready.remove(task_id)
        for index, amount in self.compiled.resources[task_id]:
            self.usage[index] += amount
        heapq.heappush(self.running, (end, task_id))

    def advance(self) -> bool:
        """Complete every event at the next timestamp before another decision."""
        if not self.running:
            self.recent_ready = ()
            return False
        self.now = self.running[0][0]
        newly_ready = []
        while self.running and self.running[0][0] == self.now:
            _, task_id = heapq.heappop(self.running)
            for index, amount in self.compiled.resources[task_id]:
                self.usage[index] -= amount
            for child in self.compiled.successors[task_id]:
                self.pending[child] -= 1
                if self.pending[child] == 0:
                    self.ready.add(child)
                    newly_ready.append(child)
        self.recent_ready = tuple(newly_ready)
        return True

    def clone(self) -> EngineState:
        other = object.__new__(EngineState)
        other.compiled = self.compiled
        other.now = self.now
        other.starts = self.starts.copy()
        other.ends = self.ends.copy()
        other.pending = self.pending.copy()
        other.usage = self.usage.copy()
        other.running = self.running.copy()
        other.ready = self.ready.copy()
        other.start_order = self.start_order.copy()
        other.recent_ready = self.recent_ready
        other.finish_time = self.finish_time
        return other

    def to_schedule(self) -> Schedule:
        return {self.compiled.tasks[i].name: (self.starts[i], self.ends[i])
                for i in self.start_order}

    def scores(self) -> tuple[float, int]:
        if not self.finished_scheduling:
            raise ValueError("Scores require a complete schedule")
        horizon = self.compiled.horizon
        scale = self.compiled.instance.loss_scale
        loss = sum(coefficient * min(self.ends[index], horizon) * scale
                   for index, coefficient in self.compiled.loss_terms)
        return loss, self.finish_time


def _keys(compiled: CompiledInstance, policy: str,
          keys: Sequence[tuple] | None) -> Sequence[tuple]:
    if keys is not None:
        if len(keys) != len(compiled.tasks):
            raise ValueError("There must be one priority key per task")
        return keys
    if policy == "priority_list":
        return compiled.priority_keys
    if policy == "shared_successor_list":
        return compiled.successor_keys
    raise ValueError(f"Unknown list policy: {policy}")


def complete_state(state: EngineState, keys: Sequence[tuple] | None = None) -> Schedule:
    """Mutate a frontier to a complete static-key schedule using a ready heap."""
    compiled = state.compiled
    keys = _keys(compiled, "shared_successor_list", keys)
    candidates = [(keys[i], compiled.tasks[i].name, i) for i in state.ready]
    heapq.heapify(candidates)
    blocked: list[int] = []
    while not state.finished_scheduling:
        chosen = None
        while candidates:
            _, _, task_id = heapq.heappop(candidates)
            if state.fits(task_id):
                chosen = task_id
                break
            blocked.append(task_id)
        if chosen is not None:
            state.start(chosen)
            continue
        if not state.advance():
            raise ValueError("No feasible ready task: invalid DAG/resources")
        # Only blocked candidates and newly unlocked successors need revisiting.
        for task_id in blocked:
            heapq.heappush(candidates, (keys[task_id], compiled.tasks[task_id].name, task_id))
        blocked.clear()
        for task_id in state.recent_ready:
            heapq.heappush(candidates, (keys[task_id], compiled.tasks[task_id].name, task_id))
    return state.to_schedule()


def run_compiled(compiled: CompiledInstance, policy: str = "shared_successor_list",
                 keys: Sequence[tuple] | None = None) -> Schedule:
    return complete_state(EngineState(compiled), _keys(compiled, policy, keys))


def run_list(instance: Instance, policy: str = "shared_successor_list") -> Schedule:
    """Public uncached policy entry: compilation is part of this invocation."""
    return run_compiled(compile_instance(instance), policy)


def run_with_chooser(compiled: CompiledInstance,
                     chooser: Callable[[tuple[int, ...], EngineState], int]) -> Schedule:
    state = EngineState(compiled)
    while not state.finished_scheduling:
        available = state.available()
        if available:
            chosen = chooser(available, state)
            if chosen not in available:
                raise ValueError("Chooser returned an unavailable task")
            state.start(chosen)
        elif not state.advance():
            raise ValueError("No feasible ready task: invalid DAG/resources")
    return state.to_schedule()


def run_rollout_compiled(compiled: CompiledInstance, objective: str = "loss") -> Schedule:
    """Same one-action rollout, completion policy and tie-breaks as the reference."""
    if objective not in ("loss", "makespan"):
        raise ValueError(objective)

    def choose(available: tuple[int, ...], state: EngineState) -> int:
        alternatives = []
        for task_id in available:
            candidate = state.clone()
            candidate.start(task_id)
            complete_state(candidate)
            loss, finish = candidate.scores()
            name = compiled.tasks[task_id].name
            value = (loss, finish, name) if objective == "loss" else (finish, loss, name)
            alternatives.append((value, task_id))
        return min(alternatives)[1]

    return run_with_chooser(compiled, choose)


def run_rollout(instance: Instance, objective: str = "loss") -> Schedule:
    """Uncached public rollout entry, with all compilation included."""
    return run_rollout_compiled(compile_instance(instance), objective)

