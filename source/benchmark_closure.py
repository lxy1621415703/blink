"""Paired executable-closure scope experiment with uncached planning timings.

Raw old/new rules, state versions, backups and resources are common inputs.
Each mode builds its own genuinely required task set; independent raw-state
auditing is performed after timing. These are synthetic ticks, not packet or
controller testbed measurements, and are never added to measured milliseconds.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

from fast_engine import run_list, run_rollout
from recovery_benchmark import makespan, weighted_loss


ROOT = Path(__file__).resolve().parents[1]
VERSION = "executable-closure-benchmark-1.0"
MODES = ("full_domain", "diff_full_state", "closure")
PRIMARY_METRICS = (
    "weighted_deficit_unit_ticks", "first_activation_ticks", "weighted_t90_ticks",
    "qos_all_ticks", "work_makespan_ticks", "median_compute_ms", "p95_compute_ms",
    "state_bytes", "state_objects", "owner_operations", "rule_operations",
    "activation_operations", "closure_state_fraction",
    "participating_closure_state_fraction",
)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def plan_record(plan) -> dict:
    return {"instance": dataclasses.asdict(plan.instance),
            "metadata": plan.metadata, "operations": plan.operations}


def quality_metrics(instance, schedule: dict) -> dict:
    recovered = sorted((schedule[flow.activation][1], flow.weight * flow.deficit)
                       for flow in instance.flows if flow.deficit > 0 and flow.weight > 0)
    total = sum(benefit for _, benefit in recovered)
    accumulated = 0.0
    t90 = 0
    for tick, benefit in recovered:
        accumulated += benefit
        if accumulated >= 0.9 * total:
            t90 = tick
            break
    return {
        "weighted_deficit_unit_ticks": weighted_loss(instance, schedule),
        "first_activation_ticks": recovered[0][0] if recovered else 0,
        "weighted_t90_ticks": t90,
        "qos_all_ticks": max((tick for tick, _ in recovered), default=0),
        "work_makespan_ticks": makespan(schedule),
        "positive_deficit_flows": len(recovered),
        "task_count": len(instance.tasks),
    }


def selected_policies(protocol: dict, split: str, seed: int, size: int) -> list[str]:
    policies = list(protocol["modes"])
    reference = protocol["quality_reference"]
    if (split != "scale" and size <= reference["maximum_flows"]
            and seed in reference.get(split + "_seeds", [])):
        policies.append(reference["policy"])
    return policies


def rotated(names: list[str], offset: int) -> list[str]:
    offset %= len(names)
    return names[offset:] + names[:offset]


def execute_policy(scenario, policy: str):
    from executable_closure import build_plan
    mode = "closure" if policy == "closure_rollout" else policy
    plan = build_plan(scenario, mode)
    schedule = (run_rollout(plan.instance, "loss") if policy == "closure_rollout"
                else run_list(plan.instance, "shared_successor_list"))
    return plan, schedule


def cases(protocol: dict, suite: str):
    from executable_closure import make_scenario
    for split in ("development", "holdout"):
        if suite not in ("all", split):
            continue
        for seed in range(*protocol[split + "_seeds"]):
            for profile in protocol["profiles"]:
                yield split, seed, protocol["flow_count"], profile, make_scenario(
                    seed, n_flows=protocol["flow_count"], profile=profile)
    if suite in ("all", "scale"):
        for size in protocol["scale_flow_counts"]:
            for seed in range(*protocol["scale_seeds"]):
                for profile in protocol["scale_profiles"]:
                    yield "scale", seed, size, profile, make_scenario(
                        seed, n_flows=size, profile=profile)


def state_metrics(metadata: dict) -> dict:
    # These names form the model/harness interface; omissions must not silently
    # turn missing evidence into zero savings.
    names = ("state_bytes", "state_objects", "owner_operations", "rule_operations",
             "activation_operations", "required_state_objects",
             "required_application_state_objects", "total_application_state_objects",
             "closure_state_fraction", "participating_application_state_objects",
             "participating_closure_state_fraction")
    missing = [name for name in names if name not in metadata]
    if missing:
        raise ValueError(f"Plan metadata is missing mandatory fields: {missing}")
    return {name: metadata[name] for name in names}


def compare_rows(rows: list[dict], protocol: dict) -> list[dict]:
    by_case = defaultdict(dict)
    for row in rows:
        by_case[(row["split"], row["profile"], row["n_flows"], row["seed"])][row["policy"]] = row
    output = []
    for candidate, baseline in (("closure", "full_domain"),
                                ("closure", "diff_full_state"),
                                ("closure_rollout", "closure")):
        grouped = defaultdict(list)
        for (split, profile, size, _), mapping in by_case.items():
            if candidate in mapping and baseline in mapping:
                grouped[(split, profile, size)].append((mapping[candidate], mapping[baseline]))
        for (split, profile, size), pairs in sorted(grouped.items()):
            for metric in PRIMARY_METRICS:
                candidate_values = np.asarray([a[metric] for a, _ in pairs], dtype=float)
                baseline_values = np.asarray([b[metric] for _, b in pairs], dtype=float)
                differences = baseline_values - candidate_values
                rng = np.random.default_rng(protocol["bootstrap_seed"])
                samples = rng.choice(differences, size=(protocol["bootstrap_resamples"], len(pairs)), replace=True)
                baseline_mean = float(baseline_values.mean())
                output.append({
                    "split": split, "profile": profile, "n_flows": size,
                    "candidate": candidate, "baseline": baseline, "metric": metric,
                    "n_paired_seeds": len(pairs),
                    "mean_candidate": float(candidate_values.mean()),
                    "mean_baseline": baseline_mean,
                    "mean_difference_baseline_minus_candidate": float(differences.mean()),
                    "reduction_pct_of_means": (100 * float(differences.mean()) / baseline_mean
                                               if baseline_mean else None),
                    "paired_bootstrap95_difference": [float(x) for x in np.quantile(samples.mean(axis=1), [0.025, 0.975])],
                    "wins_ties_losses": [int((differences > 1e-8).sum()),
                                         int((abs(differences) <= 1e-8).sum()),
                                         int((differences < -1e-8).sum())],
                    "worst_candidate_increase": float(np.maximum(-differences, 0).max()),
                    "seed_level_differences": [float(value) for value in differences],
                })
    return output


def summarize(rows: list[dict], protocol: dict) -> dict:
    groups = defaultdict(list)
    violations = defaultdict(int)
    for row in rows:
        groups[(row["split"], row["profile"], row["n_flows"], row["policy"])].append(row)
        for name, count in row["violations"].items():
            violations[name] += count
    aggregates = []
    for (split, profile, size, policy), members in sorted(groups.items()):
        aggregates.append({"split": split, "profile": profile, "n_flows": size,
                           "policy": policy, "n_seeds": len(members),
                           **{"mean_" + name: statistics.mean(value[name] for value in members)
                              for name in PRIMARY_METRICS}})
    unchanged = [row for row in rows if row["profile"] == "unchanged"]
    return {
        "aggregates": aggregates, "paired_comparisons": compare_rows(rows, protocol),
        "total_violations": dict(violations),
        "unchanged_business_metrics_zero": all(
            row["weighted_deficit_unit_ticks"] == 0 and row["first_activation_ticks"] == 0
            and row["weighted_t90_ticks"] == 0 and row["qos_all_ticks"] == 0
            and row["activation_operations"] == 0 for row in unchanged),
        "unchanged_cases_checked": len(unchanged),
        "row_count": len(rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_text(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def run(output: Path, suite: str, protocol: dict) -> dict:
    from validate_closure import validate_plan
    output.mkdir(parents=True, exist_ok=True)
    rows, timings = [], []
    outputs = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("scenarios", "plans", "schedules")}
    try:
        for split, seed, size, profile, scenario in cases(protocol, suite):
            case_id = f"{split}:{profile}:{size}:{seed}"
            raw = dataclasses.asdict(scenario)
            scenario_hash = fingerprint(raw)
            outputs["scenarios"].write(json_text({"case_id": case_id, "sha256": scenario_hash,
                                                  "scenario": raw}) + "\n")
            policies = selected_policies(protocol, split, seed, size)
            canonical, measured, cold_times = {}, defaultdict(list), {}
            phases = [("cold", 1), ("warmup", protocol["timing"]["warmup_calls"]),
                      ("measured", protocol["timing"]["measured_repetitions"])]
            for phase, count in phases:
                for repetition in range(count):
                    for policy in rotated(policies, seed + repetition):
                        before = time.perf_counter_ns()
                        plan, schedule = execute_policy(scenario, policy)
                        elapsed = (time.perf_counter_ns() - before) / 1_000_000
                        signature = (fingerprint(plan_record(plan)), fingerprint(schedule))
                        if policy not in canonical:
                            checks = validate_plan(scenario, plan, schedule)
                            if any(checks.values()):
                                (output / "audit-failure.json").write_text(json_text({
                                    "case_id": case_id, "policy": policy,
                                    "scenario_sha256": scenario_hash, "violations": checks,
                                    "plan": plan_record(plan), "schedule": schedule,
                                }) + "\n", encoding="utf-8")
                                raise RuntimeError(f"Independent audit failed: {case_id}/{policy}: {checks}")
                            canonical[policy] = (signature, plan, schedule, checks)
                        elif signature != canonical[policy][0]:
                            raise RuntimeError(f"Non-deterministic repeated result: {case_id}/{policy}")
                        timings.append({"case_id": case_id, "policy": policy, "phase": phase,
                                        "repetition": repetition, "compute_ms": elapsed})
                        if phase == "measured":
                            measured[policy].append(elapsed)
                        elif phase == "cold":
                            cold_times[policy] = elapsed
            if fingerprint(dataclasses.asdict(scenario)) != scenario_hash:
                raise RuntimeError(f"Planner mutated the common raw scenario: {case_id}")
            for policy in policies:
                _, plan, schedule, checks = canonical[policy]
                row = {"version": VERSION, "case_id": case_id, "split": split,
                       "profile": profile, "n_flows": size, "seed": seed,
                       "policy": policy, "mode": "closure" if policy == "closure_rollout" else policy,
                       "scenario_sha256": scenario_hash,
                       **quality_metrics(plan.instance, schedule), **state_metrics(plan.metadata),
                       "cold_compute_ms": cold_times[policy],
                       "median_compute_ms": statistics.median(measured[policy]),
                       "p95_compute_ms": float(np.quantile(measured[policy], 0.95)),
                       "timing_repetitions": len(measured[policy]), "violations": checks}
                rows.append(row)
                outputs["plans"].write(json_text({"case_id": case_id, "policy": policy,
                                                  "plan": plan_record(plan)}) + "\n")
                outputs["schedules"].write(json_text({"case_id": case_id, "policy": policy,
                                                      "schedule": schedule}) + "\n")
            print(f"CLOSURE_CASE_COMPLETED={case_id}", flush=True)
    finally:
        for handle in outputs.values():
            handle.close()
        write_csv(output / "results.csv", rows)
        write_csv(output / "timings.csv", timings)
    summary = summarize(rows, protocol)
    if not summary["unchanged_business_metrics_zero"]:
        raise RuntimeError("Unchanged-demand negative control has nonzero business actions/loss/time")
    summary.update({"version": VERSION, "status": "passed", "suite": suite,
                    "protocol": protocol, "protocol_sha256": fingerprint(protocol),
                    "scope": protocol["scope"], "environment": {
                        "python": sys.version, "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
                        "threads": {name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                        "github_sha": os.environ.get("GITHUB_SHA"),
                    },
                    "metrics_note": "QoS times concern positive-deficit activations; work_makespan includes background work; milliseconds and uncalibrated ticks remain separate; cold_compute_ms is first per-instance call, not process cold start; rollout comparisons use only the predeclared three-seed subset; required_state_objects includes warm and bootstrap dependencies, while required_application_state_objects excludes bootstrap; closure_state_fraction uses the common full-domain application manifest and participating_closure_state_fraction uses only actual write targets; state_bytes/state_objects count actual cold objects and logical payload bytes, not measured serialization"})
    (output / "summary.json").write_text(json_text(summary) + "\n", encoding="utf-8")
    return summary


def log_summary(summary: dict) -> dict:
    if summary.get("status") != "passed":
        return summary
    metrics = {"weighted_deficit_unit_ticks", "qos_all_ticks", "state_bytes", "median_compute_ms"}
    return {key: value for key, value in summary.items() if key not in ("paired_comparisons", "protocol")} | {
        "paired_comparisons": [
            {key: value for key, value in comparison.items() if key != "seed_level_differences"}
            for comparison in summary["paired_comparisons"]
            if comparison["split"] == "holdout" and comparison["metric"] in metrics],
        "protocol_sha256": summary["protocol_sha256"],
        "full_record": "results/closure/summary.json and raw JSONL/CSV artifacts",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "closure")
    parser.add_argument("--protocol", type=Path, default=ROOT / "closure_protocol.json")
    parser.add_argument("--suite", choices=("all", "development", "holdout", "scale"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    (output / "protocol.json").write_text(json_text(protocol) + "\n", encoding="utf-8")
    code = 0
    try:
        summary = run(output, args.suite, protocol)
    except Exception as error:
        code = 1
        summary = {"version": VERSION, "status": "failed", "suite": args.suite,
                   "protocol_sha256": fingerprint(protocol),
                   "error": f"{type(error).__name__}: {error}"}
        detail = traceback.format_exc()
        (output / "runner.stderr.txt").write_text(detail, encoding="utf-8")
        (output / "summary.json").write_text(json_text(summary) + "\n", encoding="utf-8")
        print(detail, file=sys.stderr, flush=True)
    print("BEGIN_CLOSURE_JSON", flush=True)
    print(json_text(log_summary(summary)), flush=True)
    print("END_CLOSURE_JSON", flush=True)
    complete = json_text(summary)
    if len(complete.encode("utf-8")) <= 500_000:
        print("CLOSURE_FULL_SUMMARY_JSON=" + complete, flush=True)
    else:
        print("CLOSURE_FULL_SUMMARY_NOT_LOGGED_JSON=" + json_text({
            "reason": "summary exceeds the 500000-byte log budget",
            "summary_bytes": len(complete.encode("utf-8")),
            "full_summary_artifact": "results/closure/summary.json",
        }), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
