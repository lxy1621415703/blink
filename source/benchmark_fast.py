"""Paired speed/quality evaluation; same inputs, all per-call compilation timed.

Development seeds are disclosed separately from untouched holdout seeds.
Synthetic recovery ticks are never added to measured planner milliseconds.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

from recovery_benchmark import (Flow, Instance, Task, list_schedule, make_instance,
                               makespan, rollout_schedule, validate_schedule, weighted_loss)
from fast_engine import run_list, run_rollout
from fast_policies import FAST_POLICIES, POLICY_SET_VERSION

ROOT = Path(__file__).resolve().parents[1]


def scaled_instance(seed: int, n_flows: int, shared: bool) -> Instance:
    """Extend only the number of groups in the existing two-controller model."""
    if n_flows < 3:
        raise ValueError("At least three flow groups are required")
    rng = np.random.default_rng(seed)
    assignments = [int(x) for x in rng.integers(0, 2, n_flows)]
    assignments[:3] = [0, 0, 1]
    capacities = {"replica_transfer": 1, "ctrl:0": 2, "ctrl:1": 2, "ingress_writer": 1}
    tasks, flows, state_pred = [], [], {}
    for ctrl in range(2):
        members = [i for i, target in enumerate(assignments) if target == ctrl]
        unit = int(rng.integers(1, 4))
        for i in members if not shared else [members[0]]:
            name = f"state:{ctrl}" if shared else f"state:{ctrl}:flow:{i}"
            duration = unit * len(members) if shared else unit
            tasks.append(Task(name, duration, (("replica_transfer", 1), (f"ctrl:{ctrl}", 1)),
                              kind="surviving_replica_state"))
            for member in members if shared else [i]:
                state_pred[member] = name
    for i, ctrl in enumerate(assignments):
        capacities[f"switch:{i}"] = 1
        owner, install, activation = f"owner:{i}", f"install:{i}", f"activate:{i}"
        tasks.extend([
            Task(owner, int(rng.integers(1, 3)), ((f"ctrl:{ctrl}", 1),),
                 (state_pred[i],), "ownership"),
            Task(install, int(rng.integers(1, 5)), ((f"ctrl:{ctrl}", 1), (f"switch:{i}", 1)),
                 (owner,), "rules"),
            Task(activation, 1, (("ingress_writer", 1),), (install,), "activation"),
        ])
        old = float(rng.integers(100, 501))
        demand = old + float(rng.integers(50, 401))
        weight = float(rng.choice([1, 2, 5, 10]))
        flows.append(Flow(f"flow:{i}", weight, demand, old, activation, demand))
    return Instance("shared_state" if shared else "unshared_equal_work", tuple(tasks),
                    capacities, tuple(flows), seed, shared)


def quality(instance: Instance, schedule: dict) -> dict:
    violations = validate_schedule(instance, schedule)
    if any(violations.values()):
        raise RuntimeError(f"Invalid schedule: {violations}")
    recovered = sorted((schedule[f.activation][1], f.weight * f.deficit)
                       for f in instance.flows if f.deficit > 0 and f.weight > 0)
    total = sum(benefit for _, benefit in recovered)
    thresholds = {}
    for fraction, name in [(0.5, "restore50_ticks"), (0.9, "restore90_ticks")]:
        accumulated = 0.0
        thresholds[name] = 0
        for tick, benefit in recovered:
            accumulated += benefit
            if accumulated >= fraction * total:
                thresholds[name] = tick
                break
    return {"weighted_unmet_demand_units": weighted_loss(instance, schedule),
            "makespan_ticks": makespan(schedule),
            "first_flow_recovery_ticks": recovered[0][0] if recovered else 0,
            **thresholds, **violations}


POLICIES = {
    "legacy_shared": lambda i: list_schedule(i, "shared_successor_list"),
    "legacy_loss_rollout": lambda i: rollout_schedule(i, "loss"),
    "fast_shared": lambda i: run_list(i, "shared_successor_list"),
    "fast_loss_rollout": lambda i: run_rollout(i, "loss"),
    **FAST_POLICIES,
}


def cases(protocol: dict, suite: str):
    for split in ["development", "holdout"]:
        if suite not in ("all", split):
            continue
        start, end = protocol[split + "_seeds"]
        for seed in range(start, end):
            for shared, unchanged in [(True, False), (False, False), (True, True)]:
                instance = make_instance(seed, shared, unchanged)
                yield split, instance
    if suite in ("all", "scale"):
        for size in protocol["scale_flow_counts"]:
            for seed in range(*protocol["scale_seeds"]):
                for shared in [True, False]:
                    yield "scale", scaled_instance(seed, size, shared)


def dump_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], protocol: dict) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["scenario"], row["size_group"], row["policy"])].append(row)
    aggregates = []
    metrics = ["median_compute_ms", "p95_compute_ms", "weighted_unmet_demand_units",
               "makespan_ticks", "first_flow_recovery_ticks", "restore50_ticks", "restore90_ticks"]
    for (split, scenario, size, policy), members in sorted(groups.items()):
        aggregates.append({"split": split, "scenario": scenario, "size_group": size,
                           "policy": policy, "n": len(members),
                           **{"mean_" + k: statistics.mean(r[k] for r in members) for k in metrics}})
    comparisons = []
    by_case = defaultdict(dict)
    for row in rows:
        by_case[(row["split"], row["scenario"], row["size_group"], row["seed"])][row["policy"]] = row
    for candidate, baseline in [("fast_shared", "legacy_shared"),
                                ("fast_loss_rollout", "legacy_loss_rollout"),
                                ("safe_portfolio", "legacy_loss_rollout"),
                                ("safe_portfolio", "fast_loss_rollout"),
                                ("safe_portfolio", "fast_shared"),
                                ("critical_path_list", "fast_shared"),
                                ("weighted_critical_list", "fast_shared")]:
        matched = defaultdict(list)
        for (split, scenario, size, _), mapping in by_case.items():
            if candidate in mapping and baseline in mapping:
                matched[(split, scenario, size)].append((mapping[candidate], mapping[baseline]))
        for (split, scenario, size), pairs in sorted(matched.items()):
            ctime = np.array([c["median_compute_ms"] for c, _ in pairs])
            btime = np.array([b["median_compute_ms"] for _, b in pairs])
            closs = np.array([c["weighted_unmet_demand_units"] for c, _ in pairs])
            bloss = np.array([b["weighted_unmet_demand_units"] for _, b in pairs])
            cspan = np.array([c["makespan_ticks"] for c, _ in pairs])
            bspan = np.array([b["makespan_ticks"] for _, b in pairs])
            differences = bloss - closs
            rng = np.random.default_rng(protocol["bootstrap_seed"])
            samples = rng.choice(differences, size=(protocol["bootstrap_resamples"], len(pairs)), replace=True)
            comparisons.append({
                "split": split, "scenario": scenario, "size_group": size,
                "candidate": candidate, "baseline": baseline, "n": len(pairs),
                "compute_speedup_ratio_of_means": float(btime.mean() / ctime.mean()),
                "compute_reduction_pct": float(100 * (1 - ctime.mean() / btime.mean())),
                "loss_reduction_pct": float(100 * differences.mean() / bloss.mean()) if bloss.mean() else None,
                "makespan_reduction_pct": float(100 * (1 - cspan.mean() / bspan.mean())),
                "loss_wins_ties_losses": [int((differences > 1e-8).sum()),
                                          int((abs(differences) <= 1e-8).sum()),
                                          int((differences < -1e-8).sum())],
                "loss_difference_bootstrap95_units": [float(x) for x in np.quantile(samples.mean(axis=1), [0.025, 0.975])],
                "worst_loss_increase_units": float(np.maximum(closs - bloss, 0).max()),
                "worst_makespan_increase_ticks": float(np.maximum(cspan - bspan, 0).max()),
            })
    return {"aggregates": aggregates, "paired_comparisons": comparisons}


def run(output: Path, suite: str, protocol: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows, raw_times, inputs, skips, schedules = [], [], [], [], []
    equality_checks = portfolio_checks = 0
    for split, instance in cases(protocol, suite):
        case_id = f"{split}:{instance.name}:{len(instance.flows)}:{instance.seed}"
        input_record = dataclasses.asdict(instance)
        fingerprint = hashlib.sha256(json.dumps(input_record, sort_keys=True).encode()).hexdigest()
        inputs.append({"case_id": case_id, "sha256": fingerprint, "instance": input_record})
        names = []
        for name in POLICIES:
            if "rollout" in name and len(instance.flows) > protocol["full_rollout_max_flows"]:
                skips.append({"case_id": case_id, "policy": name,
                              "status": "not_run_predeclared_scale_limit"})
            else:
                names.append(name)
        references, measured, first_call = {}, defaultdict(list), {}
        rounds = 1 + protocol["warmup_rounds"] + protocol["measurement_rounds"]
        for repeat in range(rounds):
            shift = (instance.seed + repeat) % len(names)
            for name in names[shift:] + names[:shift]:
                before = time.perf_counter_ns()
                schedule = POLICIES[name](instance)
                elapsed = (time.perf_counter_ns() - before) / 1e6
                if repeat == 0:
                    references[name] = schedule
                    first_call[name] = elapsed
                    quality(instance, schedule)
                elif schedule != references[name]:
                    raise RuntimeError(f"Nondeterministic schedule: {case_id}/{name}")
                if repeat > protocol["warmup_rounds"]:
                    measured[name].append(elapsed)
                    raw_times.append({"case_id": case_id, "policy": name,
                                      "measurement": repeat - protocol["warmup_rounds"] - 1,
                                      "plan_compute_ms": elapsed})
        for fast, legacy in [("fast_shared", "legacy_shared"), ("fast_loss_rollout", "legacy_loss_rollout")]:
            if fast in references and references[fast] != references[legacy]:
                raise RuntimeError(f"Changed semantics in {case_id}/{fast}")
            if fast in references:
                equality_checks += 1
        baseline_quality = quality(instance, references["fast_shared"])
        portfolio_quality = quality(instance, references["safe_portfolio"])
        if (portfolio_quality["makespan_ticks"] > baseline_quality["makespan_ticks"]
                or portfolio_quality["weighted_unmet_demand_units"] > baseline_quality["weighted_unmet_demand_units"] + 1e-8):
            raise RuntimeError("Portfolio baseline dominance violated")
        portfolio_checks += 1
        for name in names:
            values = measured[name]
            row = {"case_id": case_id, "split": split, "scenario": instance.name,
                   "size_group": len(instance.flows) if split == "scale" else "small",
                   "seed": instance.seed, "n_flows": len(instance.flows), "n_tasks": len(instance.tasks),
                   "policy": name, "instance_sha256": fingerprint,
                   "first_call_compute_ms": first_call[name],
                   "median_compute_ms": statistics.median(values),
                   "p95_compute_ms": float(np.quantile(values, 0.95)),
                   **quality(instance, references[name])}
            rows.append(row)
            schedules.append({"case_id": case_id, "policy": name, "schedule": references[name]})
        print(f"completed {case_id}", flush=True)
    dump_csv(output / "results.csv", rows)
    dump_csv(output / "raw_timings.csv", raw_times)
    for name, data in [("instances.json", inputs), ("schedules.json", schedules), ("skipped.json", skips)]:
        (output / name).write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    summary = {"version": protocol["version"], "policy_version": POLICY_SET_VERSION,
               "protocol": protocol, "suite": suite,
               "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                               "platform": platform.platform(), "harness_commit": os.environ.get("GITHUB_SHA")},
               "checks": {"equivalent_engine_schedules": equality_checks,
                          "portfolio_dominance_cases": portfolio_checks,
                          "total_violations": {k: sum(r[k] for r in rows) for k in rows[0] if k.endswith("_violations")},
                          "negative_controls_zero": all(r["weighted_unmet_demand_units"] == 0 for r in rows
                                                        if r["scenario"] == "negative_no_demand_change")},
               "skipped_count": len(skips), "case_count": len(inputs),
               **summarize(rows, protocol)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["all", "development", "holdout", "scale"], default="all")
    parser.add_argument("--protocol", type=Path, default=ROOT / "speed_protocol.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = run(args.output, args.suite, protocol)
    compact = {key: result[key] for key in ["version", "policy_version", "environment", "checks", "skipped_count", "case_count"]}
    compact["holdout_shared_aggregates"] = [x for x in result["aggregates"]
                                            if x["split"] == "holdout" and x["scenario"] == "shared_state"]
    compact["holdout_shared_comparisons"] = [x for x in result["paired_comparisons"]
                                             if x["split"] == "holdout" and x["scenario"] == "shared_state"]
    compact["scale_shared_aggregates"] = [x for x in result["aggregates"]
                                         if x["split"] == "scale" and x["scenario"] == "shared_state"]
    print("FAST_SUMMARY_JSON=" + json.dumps(compact, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

