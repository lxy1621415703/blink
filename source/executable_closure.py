"""Executable-operation closure for a synthetic reserved-path SD-WAN model.

This module changes recovery scope, not the frozen scheduling policies. It is
not an implementation of any published system and does not claim a minimum
closure. Installed old data-plane rules continue forwarding after a fail-stop
controller crash. Times are uncalibrated simulation ticks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from recovery_benchmark import Flow, Instance, Task

MODEL_VERSION = "executable-closure-v1.0"
MODES = ("full_domain", "diff_full_state", "closure")
PROFILES = ("mixed", "fully_coupled", "hot_standby", "no_overlap", "unchanged")


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    capacity: float


@dataclass(frozen=True)
class StateSpec:
    name: str
    version: int
    dependencies: tuple[str, ...]
    size_bytes: int
    prepare_ticks: int
    bootstrap: bool = False


@dataclass(frozen=True)
class StateSnapshot:
    controller: str
    state: str
    version: int


@dataclass(frozen=True)
class OwnerSpec:
    switch: str
    current_controller: str
    current_epoch: int
    backup_controller: str
    target_epoch: int
    bootstrap_state: str
    fence_ticks: int = 1


@dataclass(frozen=True)
class RuleSpec:
    flow: str
    switch: str
    next_hop: str
    policy_version: int
    required_states: tuple[str, ...]
    valid: bool = True
    installed_by_epoch: int = 1


@dataclass(frozen=True)
class FlowSpec:
    name: str
    old_route: tuple[str, ...]
    new_route: tuple[str, ...]
    demand: float
    old_service: float
    weight: float
    old_policy_version: int
    target_policy_version: int


@dataclass(frozen=True)
class Scenario:
    seed: int
    profile: str
    nodes: tuple[str, ...]
    edges: tuple[EdgeSpec, ...]
    flows: tuple[FlowSpec, ...]
    existing_rules: tuple[RuleSpec, ...]
    target_rules: tuple[RuleSpec, ...]
    owners: tuple[OwnerSpec, ...]
    states: tuple[StateSpec, ...]
    warm_state: tuple[StateSnapshot, ...]
    surviving_replica_versions: tuple[StateSnapshot, ...]
    failed_controllers: tuple[str, ...]
    live_controllers: tuple[str, ...]
    application_state_by_target: dict[str, tuple[str, ...]]
    capacities: dict[str, int]
    rule_install_ticks: int = 2
    activation_ticks: int = 1


@dataclass(frozen=True)
class Plan:
    instance: Instance
    metadata: dict
    operations: dict[str, dict]


def make_scenario(seed: int, n_flows: int = 6, profile: str = "mixed") -> Scenario:
    if profile not in PROFILES or n_flows < 1:
        raise ValueError("Unknown profile or nonpositive flow count")
    rng = np.random.default_rng(seed)
    live = ("backup0", "backup1", "live0", "replica0")
    targets = live[:3]
    capacities = {"replica_transfer": 2, "ingress_writer": 2}
    capacities.update({f"control:{c}": 2 for c in targets})
    app_names = ["app:topology", "app:reservations"]
    app_names += [f"app:f{i}:{part}" for i in range(n_flows)
                  for part in ("policy", "acl", "statistics")]
    states = []
    for i, name in enumerate(app_names):
        if profile == "fully_coupled":
            deps = (app_names[i - 1],) if i else ()
        elif name == "app:topology":
            deps = ()
        elif name == "app:reservations":
            deps = ("app:topology",)
        else:
            deps = ("app:reservations",) if name.endswith(":policy") else ("app:topology",)
        size = int(rng.integers(1, 5)) * 1024
        states.append(StateSpec(name, 2 + i % 2, deps, size, max(1, math.ceil(size / 2048))))
    for target in targets:
        states.append(StateSpec(f"bootstrap:{target}", 3, (), 1024, 1, True))
    state_map = {s.name: s for s in states}
    warm = []
    for target in targets:
        for state in states:
            if state.bootstrap and state.name != f"bootstrap:{target}":
                continue
            draw = float(rng.random())
            if profile == "hot_standby" or draw < 0.25:
                warm.append(StateSnapshot(target, state.name, state.version))
            elif draw < 0.45:
                warm.append(StateSnapshot(target, state.name, state.version - 1))
    replicas = tuple(StateSnapshot("replica0", state.name, state.version) for state in states)
    nodes, edges, flows, owners, existing, required = [], {}, [], [], [], []
    for i in range(n_flows):
        prefix = f"f{i}"
        entry, a, x, b, c, y, d, end = [f"{prefix}:{part}" for part in
                                       ("entry", "a", "old", "b", "c", "new", "d", "exit")]
        local_nodes = (entry, a, x, b, c, y, d, end)
        nodes.extend(local_nodes)
        old_route = (entry, a, x, b, end)
        if profile == "unchanged":
            new_route = old_route
        elif profile == "no_overlap":
            new_route = (entry, c, y, d, end)
        else:
            new_route = ((entry, a, y, b, end), (entry, c, y, b, end),
                         (entry, a, y, d, end))[i % 3]
        old_rate = float(rng.integers(100, 401))
        demand = old_rate if profile == "unchanged" else old_rate + float(rng.integers(50, 251))
        flow = FlowSpec(prefix, old_route, new_route, demand, old_rate,
                        float(rng.choice([1, 2, 5, 10])), 1,
                        1 if profile == "unchanged" else 2)
        flows.append(flow)
        new_edges = set(zip(new_route, new_route[1:]))
        for source, target in zip(old_route, old_route[1:]):
            edges[source, target] = demand if (source, target) in new_edges else old_rate
        for source, target in new_edges:
            edges[source, target] = demand
        for switch in local_nodes:
            failed = bool(rng.random() < 0.7)
            target = f"backup{int(rng.integers(0, 2))}" if failed else "live0"
            owners.append(OwnerSpec(switch, "failed0" if failed else "live0", 1,
                                    target, 2 if failed else 1, f"bootstrap:{target}"))
            capacities[f"write:{switch}"] = 1
        old_next = dict(zip(old_route, old_route[1:]))
        new_next = dict(zip(new_route, new_route[1:]))
        def read_set_for(switch: str) -> tuple[str, ...]:
            if profile == "fully_coupled":
                return (app_names[-1],)
            return ((f"app:f{i}:policy",) if switch == entry
                    else (f"app:f{i}:policy", f"app:f{i}:acl"))
        for switch, next_hop in old_next.items():
            # 0 is an explicit context-independent entry; version 1 remains
            # correct for OLD traffic even if it cannot serve target version 2.
            stable = new_next.get(switch) == next_hop and rng.random() < 0.8
            existing.append(RuleSpec(prefix, switch, next_hop, 0 if stable else 1,
                                     read_set_for(switch)))
        for switch, next_hop in new_next.items():
            rule = RuleSpec(prefix, switch, next_hop, flow.target_policy_version, read_set_for(switch))
            required.append(rule)
            # A valid shadow rule can already exist. Its old writer epoch does
            # not invalidate forwarding; no gratuitous rewrite is generated.
            if switch != entry and rng.random() < 0.2:
                existing.append(rule)
        if i:
            # Unused backbone edges connect the synthetic topology components.
            edges[f"f{i-1}:exit", entry] = float(10000 * n_flows)
    return Scenario(seed, profile, tuple(nodes),
                    tuple(EdgeSpec(a, b, cap) for (a, b), cap in sorted(edges.items())),
                    tuple(flows), tuple(existing), tuple(required), tuple(owners), tuple(states),
                    tuple(warm), replicas, ("failed0",), live,
                    {target: tuple(app_names) for target in targets}, capacities)


def build_plan(scenario: Scenario, mode: str) -> Plan:
    if mode not in MODES:
        raise ValueError(f"Unknown recovery scope: {mode}")
    owners = {o.switch: o for o in scenario.owners}
    states = {s.name: s for s in scenario.states}
    flows = {f.name: f for f in scenario.flows}
    live = set(scenario.live_controllers)
    failed = set(scenario.failed_controllers)
    warm = {(s.controller, s.state, s.version) for s in scenario.warm_state}
    replicas = {(s.controller, s.state, s.version) for s in scenario.surviving_replica_versions}
    edges = {(e.source, e.target): e.capacity for e in scenario.edges}
    target_rules = {(r.flow, r.switch): r for r in scenario.target_rules}
    existing_by_key: dict[tuple[str, str], list[RuleSpec]] = {}
    for rule in scenario.existing_rules:
        existing_by_key.setdefault((rule.flow, rule.switch), []).append(rule)

    def state_closure(names: tuple[str, ...] | set[str]) -> set[str]:
        found: set[str] = set()
        visiting: set[str] = set()
        def visit(name: str) -> None:
            if name not in states:
                raise ValueError(f"Missing state manifest: {name}")
            if name in visiting:
                raise ValueError("Cyclic state manifest")
            if name in found:
                return
            visiting.add(name)
            for dep in states[name].dependencies:
                visit(dep)
            visiting.remove(name)
            found.add(name)
        for name in names:
            visit(name)
        return found

    def target_of(switch: str) -> str:
        if switch not in owners:
            raise ValueError(f"Missing owner: {switch}")
        owner = owners[switch]
        if owner.current_controller in live:
            target = owner.current_controller
            if owner.target_epoch != owner.current_epoch:
                raise ValueError("Live owner epoch must be preserved")
        elif owner.current_controller in failed:
            target = owner.backup_controller
            if owner.target_epoch <= owner.current_epoch:
                raise ValueError("Failed owner requires a newer fencing epoch")
        else:
            raise ValueError("Owner liveness is unknown")
        if target not in live:
            raise ValueError(f"No surviving legal target for {switch}")
        return target

    def compatible(rule: RuleSpec) -> bool:
        return any(r.valid and r.next_hop == rule.next_hop
                   and r.policy_version in (0, rule.policy_version)
                   for r in existing_by_key.get((rule.flow, rule.switch), ()))

    affected = [f for f in scenario.flows
                if f.new_route != f.old_route or f.target_policy_version != f.old_policy_version
                or f.demand > f.old_service]
    for flow in scenario.flows:
        for route in (flow.old_route, flow.new_route):
            if len(route) < 2 or len(set(route)) != len(route):
                raise ValueError("Non-simple or empty route")
            if any((a, b) not in edges for a, b in zip(route, route[1:])):
                raise ValueError("Route traverses nonexistent edge")
        if any(edges[a, b] < flow.demand for a, b in zip(flow.new_route, flow.new_route[1:])):
            raise ValueError("Target reserved path lacks capacity")
    needed_rules = []
    reused_rules = 0
    write_switches: set[str] = set()
    rule_requirements: dict[tuple[str, str], set[str]] = {}
    for flow in affected:
        entry = flow.new_route[0]
        for switch, next_hop in zip(flow.new_route, flow.new_route[1:]):
            rule = target_rules.get((flow.name, switch))
            if rule is None or rule.next_hop != next_hop or not rule.valid or rule.policy_version != flow.target_policy_version:
                raise ValueError("Invalid or absent target rule")
            if switch == entry or not compatible(rule):
                write_switches.add(switch)
                rule_requirements[flow.name, switch] = state_closure(rule.required_states)
                if switch != entry:
                    needed_rules.append(rule)
            else:
                reused_rules += 1
    participating_targets = {target_of(switch) for switch in write_switches}
    needed_owners = {s for s in write_switches if owners[s].current_controller in failed}
    full_scope_targets = set(participating_targets)
    if affected:
        # Metadata must not force an irrelevant switch to be recoverable.
        # The full-domain policy validates every selected owner below, whereas
        # difference/closure policies only need their actual write targets.
        full_scope_targets.update(o.backup_controller for o in scenario.owners
                                  if o.current_controller in failed and o.backup_controller in live)
    actual_required: set[tuple[str, str]] = set()
    for (flow, switch), names in rule_requirements.items():
        actual_required.update((target_of(switch), name) for name in names)
    for switch in needed_owners:
        actual_required.update((target_of(switch), name)
                               for name in state_closure((owners[switch].bootstrap_state,)))
    full_app = {(target, name) for target in full_scope_targets
                for name in scenario.application_state_by_target.get(target, ())}
    participating_app = {(target, name) for target in participating_targets
                         for name in scenario.application_state_by_target.get(target, ())}
    selected_owners = set(needed_owners)
    if mode == "full_domain" and affected:
        selected_owners.update(o.switch for o in scenario.owners if o.current_controller in failed)
    selected_states = set(actual_required)
    if mode in ("full_domain", "diff_full_state") and affected:
        for target, name in (full_app if mode == "full_domain" else participating_app):
            selected_states.update((target, dep) for dep in state_closure((name,)))
    for switch in selected_owners:
        selected_states.update((target_of(switch), dep)
                               for dep in state_closure((owners[switch].bootstrap_state,)))
    operations: dict[str, dict] = {}
    tasks = []
    cold = {(target, name) for target, name in selected_states
            if (target, name, states[name].version) not in warm}
    def state_name(target: str, name: str) -> str:
        return f"state:{target}:{name}"
    def state_dependencies(target: str, names: set[str]) -> tuple[str, ...]:
        return tuple(sorted(state_name(target, name) for name in names if (target, name) in cold))
    def add(task: Task, op: dict) -> None:
        tasks.append(task)
        operations[task.name] = {**op, "predecessors": list(task.predecessors)}
    for target, name in sorted(cold):
        state = states[name]
        if target not in live:
            raise ValueError("State target is not alive")
        sources = sorted(c for c, s, version in replicas
                         if c in live and s == name and version == state.version)
        if not sources:
            raise ValueError(f"No surviving replica of state {name} version {state.version}")
        deps = state_dependencies(target, state_closure(state.dependencies))
        add(Task(state_name(target, name), state.prepare_ticks,
                 (("replica_transfer", 1), (f"control:{target}", 1)), deps, "state"),
            {"kind": "state", "controller": target, "source": sources[0], "state": name,
             "version": state.version, "size_bytes": state.size_bytes,
             "required_states": list(state.dependencies)})
    for switch in sorted(selected_owners):
        owner = owners[switch]
        target = target_of(switch)
        deps = state_dependencies(target, state_closure((owner.bootstrap_state,)))
        add(Task(f"owner:{switch}", owner.fence_ticks,
                 ((f"control:{target}", 1), (f"write:{switch}", 1)), deps, "owner"),
            {"kind": "owner", "controller": target, "switch": switch,
             "owner_epoch": owner.target_epoch, "previous_epoch": owner.current_epoch,
             "bootstrap_state": owner.bootstrap_state,
             "required_states": [owner.bootstrap_state]})
    for rule in sorted(needed_rules, key=lambda r: (r.flow, r.switch)):
        target = target_of(rule.switch)
        deps = list(state_dependencies(target, rule_requirements[rule.flow, rule.switch]))
        if rule.switch in selected_owners:
            deps.append(f"owner:{rule.switch}")
        add(Task(f"install:{rule.flow}:{rule.switch}", scenario.rule_install_ticks,
                 ((f"control:{target}", 1), (f"write:{rule.switch}", 1)),
                 tuple(sorted(set(deps))), "install"),
            {"kind": "install", "controller": target, "flow": rule.flow,
             "switch": rule.switch, "next_hop": rule.next_hop,
             "policy_version": rule.policy_version, "owner_epoch": owners[rule.switch].target_epoch,
             "required_states": list(rule.required_states), "staging": True})
    output_flows = []
    for flow in affected:
        entry = flow.new_route[0]
        target = target_of(entry)
        rule = target_rules[flow.name, entry]
        deps = list(state_dependencies(target, rule_requirements[flow.name, entry]))
        if entry in selected_owners:
            deps.append(f"owner:{entry}")
        deps.extend(f"install:{r.flow}:{r.switch}" for r in needed_rules if r.flow == flow.name)
        task_name = f"activate:{flow.name}"
        add(Task(task_name, scenario.activation_ticks,
                 ((f"control:{target}", 1), (f"write:{entry}", 1), ("ingress_writer", 1)),
                 tuple(sorted(set(deps))), "activation"),
            {"kind": "activation", "controller": target, "flow": flow.name,
             "switch": entry, "next_hop": rule.next_hop, "policy_version": flow.target_policy_version,
             "owner_epoch": owners[entry].target_epoch, "required_states": list(rule.required_states),
             "staging": False, "switches_ingress_context": True})
        output_flows.append(Flow(flow.name, flow.weight, flow.demand, flow.old_service,
                                 task_name, flow.demand))
    for task in tasks:
        if task.duration <= 0 or any(r not in scenario.capacities or a > scenario.capacities[r]
                                     for r, a in task.resources):
            raise ValueError("Invalid action duration or resource capacity")
    app_required = {(target, name) for target, name in actual_required if not states[name].bootstrap}
    metadata = {
        "version": MODEL_VERSION, "seed": scenario.seed, "profile": scenario.profile, "mode": mode,
        "state_bytes": sum(states[name].size_bytes for _, name in cold),
        "state_objects": len(cold), "owner_operations": len(selected_owners),
        "rule_operations": len(needed_rules), "activation_operations": len(affected),
        "required_state_objects": len(actual_required),
        "required_application_state_objects": len(app_required),
        "total_application_state_objects": len(full_app),
        "closure_state_fraction": len(app_required) / len(full_app) if full_app else 0.0,
        "participating_application_state_objects": len(participating_app),
        "participating_closure_state_fraction": len(app_required) / len(participating_app) if participating_app else 0.0,
        "state_fraction_denominator": "common full-domain target/state manifest across all modes",
        "total_tasks": len(tasks), "reused_state_objects": len(selected_states) - len(cold),
        "reused_rules": reused_rules, "changed_rules": len(needed_rules),
        "participating_targets": sorted(participating_targets),
        "full_scope_targets": sorted(full_scope_targets),
        "state_prepare_semantics": "independent objects; no full-state barrier",
        "update_semantics": "versioned shadow target rules; old path retained until ingress switch; common semantically valid entries usable in both contexts",
        "capacity_assumption": "independent pre-reserved target paths; sufficient dual-context staging table space",
        "application_manifest": "supplied synthetic dependency graph; no automatic read-set inference",
        "size_unit": "synthetic logical payload bytes, not measured serialization size",
        "time_unit": "uncalibrated simulation ticks", "rate_unit": "synthetic rate units",
        "loss_unit": "priority-weighted rate-unit*tick", "route_mapping_fixed_across_modes": True,
    }
    instance = Instance(f"closure:{scenario.profile}:{mode}", tuple(tasks), dict(scenario.capacities),
                        tuple(output_flows), scenario.seed, True, 1.0)
    return Plan(instance, metadata, operations)
