"""Independent replay of recovery operations against a raw network scenario.

No planner closure/matching helper is imported. Requirements are reconstructed
from raw paths, state manifests, replica versions and ownership epochs. Rules
are staged in a separate target context; this is not a packet-level proof.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Iterable

from executable_closure import Plan, RuleSpec, Scenario
from recovery_benchmark import Schedule


CHECKS = (
    "structure_violations", "precedence_violations", "resource_violations",
    "raw_model_violations", "state_version_violations", "state_source_violations",
    "warm_state_reuse_violations", "owner_epoch_violations", "rule_diff_violations",
    "activation_violations", "forwarding_violations", "old_forwarding_violations",
    "data_capacity_violations", "unnecessary_barrier_violations", "metadata_violations",
    "scope_violations",
)


def _index_rules(rules):
    indexed = defaultdict(list)
    for rule in rules:
        indexed[rule.flow, rule.switch].append(rule)
    return indexed


def _trace(flow, rules, context: int, route: tuple[str, ...], requirements=None):
    """Follow actual next hops, rather than trusting a plan's list of tasks."""
    indexed = rules if isinstance(rules, Mapping) else _index_rules(rules)
    seen = set()
    walk = [route[0]]
    while walk[-1] != route[-1]:
        switch = walk[-1]
        if switch in seen:
            return None
        seen.add(switch)
        candidates = [rule for rule in indexed.get((flow.name, switch), ())
                      if rule.valid and rule.policy_version in (0, context)]
        specific = [r for r in candidates if r.policy_version == context]
        choices = {r.next_hop for r in specific or candidates}
        if len(choices) != 1:
            return None
        walk.append(next(iter(choices)))
        if len(walk) > len(indexed) + 2:
            return None
    return tuple(walk)


def reconstruct_forwarding(scenario: Scenario, rules: Iterable[RuleSpec],
                           target: bool = True) -> dict[str, tuple[str, ...] | None]:
    indexed = _index_rules(rules)
    requirements = {(r.flow, r.switch): r for r in scenario.target_rules} if target else None
    return {flow.name: _trace(flow, indexed,
                             flow.target_policy_version if target else flow.old_policy_version,
                             flow.new_route if target else flow.old_route, requirements)
            for flow in scenario.flows}


def validate_plan(scenario: Scenario, plan: Plan, schedule: Schedule) -> dict[str, int]:
    checks = dict.fromkeys(CHECKS, 0)

    def flag(key: str, count: int = 1):
        checks[key] += count

    states = {state.name: state for state in scenario.states}
    owners = {owner.switch: owner for owner in scenario.owners}
    flows = {flow.name: flow for flow in scenario.flows}
    target_rules = {(rule.flow, rule.switch): rule for rule in scenario.target_rules}
    tasks = {task.name: task for task in plan.instance.tasks}
    operations = plan.operations
    live = set(scenario.live_controllers) - set(scenario.failed_controllers)
    initial_cache = {(s.controller, s.state, s.version) for s in scenario.warm_state
                     if s.controller in live}
    replicas = {(s.controller, s.state, s.version) for s in scenario.surviving_replica_versions
                if s.controller in live}
    edges = {(edge.source, edge.target): edge.capacity for edge in scenario.edges}
    if len(tasks) != len(plan.instance.tasks):
        flag("structure_violations")
    if len(states) != len(scenario.states) or len(owners) != len(scenario.owners):
        flag("raw_model_violations")
    if len(flows) != len(scenario.flows) or len(target_rules) != len(scenario.target_rules):
        flag("raw_model_violations")
    if set(tasks) != set(operations) or set(tasks) != set(schedule):
        flag("structure_violations", len(set(tasks) ^ set(operations)) + len(set(tasks) ^ set(schedule)))
    if plan.instance.capacities != scenario.capacities or plan.instance.loss_scale != 1:
        flag("raw_model_violations")

    closure_cache = {}
    def closure(names):
        found, visiting = set(), set()
        def visit(name):
            if name in closure_cache:
                found.update(closure_cache[name])
                return
            if name in found:
                return
            if name not in states or name in visiting:
                raise ValueError("Missing/cyclic raw state dependency")
            visiting.add(name)
            for dep in states[name].dependencies:
                visit(dep)
            visiting.remove(name)
            found.add(name)
        for name in names:
            visit(name)
        # Cache only a single-root result; multi-root results must not be
        # mistaken for the individual dependency set of every member.
        if len(names) == 1:
            closure_cache[next(iter(names))] = set(found)
        return found

    def target_owner(switch):
        owner = owners.get(switch)
        if owner is None:
            raise ValueError("Missing raw owner")
        if owner.current_controller in live:
            if owner.current_epoch != owner.target_epoch:
                raise ValueError("Live owner epoch changed")
            return owner.current_controller, owner.current_epoch
        if owner.current_controller not in scenario.failed_controllers:
            raise ValueError("Unknown current owner liveness")
        if owner.backup_controller not in live or owner.target_epoch <= owner.current_epoch:
            raise ValueError("No surviving fenced owner")
        return owner.backup_controller, owner.target_epoch

    def initial_rule_matches(requirement):
        # required_states is a construction read-set. A materialized rule can
        # remain usable without hydrating that read-set or changing its writer epoch.
        return any(rule.valid and rule.flow == requirement.flow
                   and rule.switch == requirement.switch and rule.next_hop == requirement.next_hop
                   and rule.policy_version in (0, requirement.policy_version)
                   for rule in scenario.existing_rules)

    affected = {flow.name for flow in scenario.flows
                if flow.new_route != flow.old_route
                or flow.target_policy_version != flow.old_policy_version
                or flow.demand > flow.old_service}
    try:
        for state in scenario.states:
            closure((state.name,))
        for flow in scenario.flows:
            if flow.demand < 0 or flow.old_service < 0 or flow.weight < 0:
                raise ValueError("Invalid raw flow")
            for route in (flow.old_route, flow.new_route):
                if len(route) < 2 or len(set(route)) != len(route):
                    raise ValueError("Invalid raw path")
                if any((a, b) not in edges for a, b in zip(route, route[1:])):
                    raise ValueError("Nonexistent raw path edge")
            for switch, next_hop in zip(flow.new_route, flow.new_route[1:]):
                requirement = target_rules.get((flow.name, switch))
                if requirement is None or not requirement.valid or requirement.next_hop != next_hop:
                    raise ValueError("Raw target rule does not implement path hop")
                if requirement.policy_version != flow.target_policy_version:
                    raise ValueError("Raw target policy mismatch")
                closure(requirement.required_states)
    except (ValueError, KeyError):
        flag("raw_model_violations")
        return checks

    instance_flows = {flow.name: flow for flow in plan.instance.flows}
    if set(instance_flows) != affected:
        flag("raw_model_violations", len(set(instance_flows) ^ affected))
    for name in affected & instance_flows.keys():
        actual, raw = instance_flows[name], flows[name]
        if (actual.weight, actual.demand, actual.old_service, actual.activation) != (
                raw.weight, raw.demand, raw.old_service, f"activate:{name}"):
            flag("raw_model_violations")
        if actual.final_reserved_rate < raw.demand:
            flag("data_capacity_violations")

    def state_task(controller, state):
        return f"state:{controller}:{state}"

    def cold_dependencies(controller, names):
        return {state_task(controller, name) for name in closure(names)
                if (controller, name, states[name].version) not in initial_cache}

    def owner_dependencies(switch):
        return {f"owner:{switch}"} if owners[switch].current_controller not in live else set()

    # Independently reconstruct scope from raw path hops and manifests. This
    # catches a baseline that silently drops background state while relabelling
    # its remaining operations as a full-domain plan.
    expected_installs, expected_activations, write_switches = set(), set(), set()
    required_state_pairs = set()
    try:
        for name in affected:
            flow = flows[name]
            expected_activations.add(f"activate:{name}")
            for switch, _ in zip(flow.new_route, flow.new_route[1:]):
                rule = target_rules[name, switch]
                if switch == flow.new_route[0] or not initial_rule_matches(rule):
                    write_switches.add(switch)
                    controller, _ = target_owner(switch)
                    required_state_pairs.update((controller, state) for state in closure(rule.required_states))
                    if switch != flow.new_route[0]:
                        expected_installs.add(f"install:{name}:{switch}")
        selected_owners = {switch for switch in write_switches
                           if owners[switch].current_controller not in live}
        mode = plan.metadata.get("mode")
        if mode not in ("closure", "diff_full_state", "full_domain"):
            flag("scope_violations")
        if mode == "full_domain" and affected:
            selected_owners.update(owner.switch for owner in scenario.owners
                                   if owner.current_controller in scenario.failed_controllers)
        selected_state_pairs = set(required_state_pairs)
        participating = {target_owner(switch)[0] for switch in write_switches}
        full_state_targets = set(participating)
        for switch in selected_owners:
            controller, _ = target_owner(switch)
            selected_state_pairs.update((controller, state)
                                        for state in closure((owners[switch].bootstrap_state,)))
            if mode == "full_domain":
                full_state_targets.add(controller)
        if mode in ("full_domain", "diff_full_state") and affected:
            for controller in full_state_targets:
                selected_state_pairs.update((controller, state) for state in closure(
                    scenario.application_state_by_target.get(controller, ())))
        expected_states = {state_task(controller, state) for controller, state in selected_state_pairs
                           if (controller, state, states[state].version) not in initial_cache}
        expected_owners = {f"owner:{switch}" for switch in selected_owners}
        expected_scope = expected_states | expected_owners | expected_installs | expected_activations
        if expected_scope != set(operations):
            flag("scope_violations", len(expected_scope ^ set(operations)))
    except (ValueError, KeyError):
        flag("scope_violations")

    expected_dependencies = {}
    canonical_resources = {}
    valid_operations = set()
    for name, op in operations.items():
        kind = op.get("kind")
        controller = op.get("controller")
        switch = op.get("switch")
        deps = set()
        duration = None
        resource = {}
        try:
            if controller not in live:
                flag("owner_epoch_violations")
            if kind == "state":
                state = states[op["state"]]
                if name != state_task(controller, state.name):
                    flag("structure_violations")
                if op.get("version") != state.version or set(op.get("required_states", ())) != set(state.dependencies):
                    flag("state_version_violations")
                if (controller, state.name, state.version) in initial_cache:
                    flag("warm_state_reuse_violations")
                if (op.get("source"), state.name, state.version) not in replicas:
                    flag("state_source_violations")
                if op.get("size_bytes") != state.size_bytes:
                    flag("metadata_violations")
                deps = cold_dependencies(controller, state.dependencies)
                duration = state.prepare_ticks
                resource = {"replica_transfer": 1, f"control:{controller}": 1}
            elif kind == "owner":
                owner = owners[switch]
                expected_controller, expected_epoch = target_owner(switch)
                if name != f"owner:{switch}":
                    flag("structure_violations")
                if (controller, op.get("owner_epoch")) != (expected_controller, expected_epoch):
                    flag("owner_epoch_violations")
                if op.get("previous_epoch") != owner.current_epoch:
                    flag("owner_epoch_violations")
                if op.get("bootstrap_state") != owner.bootstrap_state:
                    flag("state_version_violations")
                if set(op.get("required_states", ())) != {owner.bootstrap_state}:
                    flag("state_version_violations")
                deps = cold_dependencies(controller, (owner.bootstrap_state,))
                duration = owner.fence_ticks
                resource = {f"control:{controller}": 1, f"write:{switch}": 1}
            elif kind in ("install", "activation"):
                flow = flows[op["flow"]]
                requirement = target_rules[flow.name, switch]
                if (controller, op.get("owner_epoch")) != target_owner(switch):
                    flag("owner_epoch_violations")
                if (op.get("next_hop"), op.get("policy_version")) != (
                        requirement.next_hop, requirement.policy_version):
                    flag("rule_diff_violations")
                if set(op.get("required_states", ())) != set(requirement.required_states):
                    flag("state_version_violations")
                deps = cold_dependencies(controller, requirement.required_states) | owner_dependencies(switch)
                resource = {f"control:{controller}": 1, f"write:{switch}": 1}
                if kind == "install":
                    if name != f"install:{flow.name}:{switch}" or switch == flow.new_route[0]:
                        flag("structure_violations")
                    if initial_rule_matches(requirement):
                        flag("rule_diff_violations")
                    if op.get("staging") is not True:
                        flag("old_forwarding_violations")
                    duration = scenario.rule_install_ticks
                else:
                    if name != f"activate:{flow.name}" or switch != flow.new_route[0] or flow.name not in affected:
                        flag("activation_violations")
                    if op.get("switches_ingress_context") is not True:
                        flag("activation_violations")
                    for downstream in flow.new_route[1:-1]:
                        wanted = target_rules[flow.name, downstream]
                        if not initial_rule_matches(wanted):
                            deps.add(f"install:{flow.name}:{downstream}")
                    duration = scenario.activation_ticks
                    resource["ingress_writer"] = 1
            else:
                flag("structure_violations")
                continue
            valid_operations.add(name)
        except (KeyError, ValueError, TypeError):
            flag("raw_model_violations")
            continue
        expected_dependencies[name] = deps
        canonical_resources[name] = resource
        task = tasks.get(name)
        if task is None:
            continue
        if task.kind != kind or task.duration != duration or task.duration <= 0:
            flag("structure_violations")
        if dict(task.resources) != resource or len(task.resources) != len(resource):
            flag("resource_violations")
        if set(task.predecessors) != set(op.get("predecessors", ())):
            flag("structure_violations")

    actual_ancestors, semantic_ancestors = {}, {}
    def ancestors(name, graph, memo, visiting=None):
        if name in memo:
            return memo[name]
        visiting = set() if visiting is None else visiting
        if name in visiting:
            raise ValueError("Cyclic action dependency")
        visiting.add(name)
        result = set()
        for pred in graph.get(name, ()):
            result.add(pred)
            result.update(ancestors(pred, graph, memo, visiting))
        visiting.remove(name)
        memo[name] = result
        return result
    task_graph = {name: task.predecessors for name, task in tasks.items()}
    try:
        for name in tasks:
            actual = ancestors(name, task_graph, actual_ancestors)
            semantic = ancestors(name, expected_dependencies, semantic_ancestors)
            missing = semantic - actual
            extra = actual - semantic
            if missing:
                flag("precedence_violations", len(missing))
            if extra:
                flag("unnecessary_barrier_violations", len(extra))
            if semantic - tasks.keys():
                flag("structure_violations", len(semantic - tasks.keys()))
    except ValueError:
        flag("precedence_violations")

    starts, ends = defaultdict(list), defaultdict(list)
    for name, interval in schedule.items():
        if name not in tasks or not isinstance(interval, (tuple, list)) or len(interval) != 2:
            flag("structure_violations")
            continue
        start, end = interval
        if start < 0 or end <= start or end - start != tasks[name].duration:
            flag("structure_violations")
            continue
        for pred in tasks[name].predecessors:
            if pred not in schedule or schedule[pred][1] > start:
                flag("precedence_violations")
        starts[start].append(name)
        ends[end].append(name)

    active_owner = {switch: (owner.current_controller, owner.current_epoch)
                    for switch, owner in owners.items() if owner.current_controller in live}
    cache = set(initial_cache)
    old_bank = _index_rules(scenario.existing_rules)
    target_bank = _index_rules(scenario.existing_rules)
    activated = set()
    usage = defaultdict(int)

    def state_ready(controller, names):
        return all((controller, name, states[name].version) in cache for name in closure(names))

    def write_ready(op):
        switch = op["switch"]
        return (active_owner.get(switch) == (op["controller"], op["owner_epoch"])
                and (op["controller"], op["owner_epoch"]) == target_owner(switch))

    def check_forwarding():
        load = defaultdict(float)
        for flow in scenario.flows:
            is_new = flow.name in activated
            bank = target_bank if is_new else old_bank
            context = flow.target_policy_version if is_new else flow.old_policy_version
            expected = flow.new_route if is_new else flow.old_route
            actual = _trace(flow, bank, context, expected, target_rules if is_new else None)
            if actual != expected:
                flag("forwarding_violations" if is_new else "old_forwarding_violations")
                continue
            rate = flow.demand if is_new else flow.old_service
            for edge in zip(actual, actual[1:]):
                load[edge] += rate
        for edge, rate in load.items():
            if edge not in edges or rate > edges[edge] + 1e-9:
                flag("data_capacity_violations")

    check_forwarding()
    for now in sorted(starts.keys() | ends.keys()):
        forwarding_changed = False
        for name in sorted(ends[now]):
            for resource, amount in canonical_resources.get(name, {}).items():
                usage[resource] -= amount
            if name not in valid_operations:
                continue
            op = operations[name]
            kind = op["kind"]
            if kind == "state":
                cache.add((op["controller"], op["state"], op["version"]))
            elif kind == "owner":
                active_owner[op["switch"]] = (op["controller"], op["owner_epoch"])
            else:
                if not write_ready(op):
                    flag("owner_epoch_violations")
                rule = RuleSpec(op["flow"], op["switch"], op["next_hop"],
                                op["policy_version"], tuple(op["required_states"]), True,
                                op["owner_epoch"])
                target_bank[rule.flow, rule.switch].append(rule)
                if kind == "install" and not op.get("staging"):
                    old_bank[rule.flow, rule.switch] = [rule]
                    forwarding_changed = True
                if kind == "activation":
                    activated.add(op["flow"])
                    forwarding_changed = True
                elif op.get("flow") in activated:
                    forwarding_changed = True
        if forwarding_changed:
            check_forwarding()
        for name in sorted(starts[now]):
            for resource, amount in canonical_resources.get(name, {}).items():
                usage[resource] += amount
            if name not in valid_operations:
                continue
            op = operations[name]
            kind = op["kind"]
            controller = op["controller"]
            if kind == "state":
                if not state_ready(controller, states[op["state"]].dependencies):
                    flag("state_version_violations")
            elif kind == "owner":
                if not state_ready(controller, (owners[op["switch"]].bootstrap_state,)):
                    flag("state_version_violations")
            else:
                requirement = target_rules[op["flow"], op["switch"]]
                if not write_ready(op):
                    flag("owner_epoch_violations")
                if not state_ready(controller, requirement.required_states):
                    flag("state_version_violations")
                if kind == "activation":
                    flow = flows[op["flow"]]
                    # Entry is written by activation itself. Every downstream
                    # target hop must already exist in the target context.
                    for switch, next_hop in zip(flow.new_route[1:-1], flow.new_route[2:]):
                        probe = replace(flow, new_route=(switch, *flow.new_route[flow.new_route.index(switch) + 1:]))
                        if _trace(probe, target_bank, flow.target_policy_version, probe.new_route,
                                  target_rules) != probe.new_route:
                            flag("activation_violations")
        for resource, amount in usage.items():
            if amount < 0 or amount > scenario.capacities.get(resource, 0):
                flag("resource_violations")
    if affected - activated:
        flag("activation_violations", len(affected - activated))
    check_forwarding()

    counts = {"state_bytes": sum(states[op["state"]].size_bytes for op in operations.values()
                                 if op.get("kind") == "state" and op.get("state") in states),
              "state_objects": sum(op.get("kind") == "state" for op in operations.values()),
              "owner_operations": sum(op.get("kind") == "owner" for op in operations.values()),
              "rule_operations": sum(op.get("kind") == "install" for op in operations.values()),
              "activation_operations": sum(op.get("kind") == "activation" for op in operations.values()),
              "total_tasks": len(tasks)}
    for key, value in counts.items():
        if key in plan.metadata and plan.metadata[key] != value:
            flag("metadata_violations")
    return checks
