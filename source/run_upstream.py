"""Compare pinned author CLI configurations and a separately recorded one-line fix.

This phase reports the authors' VFF/CBF round counts on their four bundled
instances. It does not represent controller-failure or packet-level experiments.
The pristine checkout is never modified. A temporary copy receives only the
resilient controller-membership fix. Source is not an output artifact.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback


REPOSITORY = "https://github.com/nu-lab/Controller-Load-Migration-Scheduling.git"
COMMIT = "3f6cd3b6d3aca1ff38c3b985535d31bc7698fc05"
AUTHOR_SEED = 42
HASH_SEED = 0
INSTANCES = tuple(f"migrations{i}.txt" for i in range(4))
PATCH_FILE = "MigrationScheduling/utils.py"
PATCH_OLD = b"control_const.get_constraint_switches(False)"
PATCH_NEW = b"control_const.get_constraint_switches(resiliency)"
VARIANTS = (
    ("original_resilient", False, True),
    ("original_destination_only", False, False),
    ("patched_resilient", True, True),
)
METRICS = {
    "vector_first_fit_rounds": r"Number of Rounds for Vector First Fit:\s*(\d+)",
    "current_bottleneck_first_rounds": (
        r"Number of Rounds for Current Bottleneck First:\s*(\d+)"
    ),
}

# This subprocess imports the selected source tree but does not run heuristics
# or an optimizer. Expected membership is independently derived below.
MEMBERSHIP_PROBE = '''
import json
import sys
from MigrationScheduling.Model import Parser
from MigrationScheduling import utils

resiliency = sys.argv[1] == "true"
records = []
for filename in sys.argv[2:]:
    parser = Parser()
    parser.parse_migrations(filename)
    data = parser.to_data()
    constraints = utils.get_constraints_dict(data, resiliency)
    controllers = []
    for constraint in data.get_control_consts():
        name = constraint.get_controller()
        observed = constraints[name]
        controllers.append({
            "controller": name,
            "observed_switches": sorted(observed.get_switches()),
            "observed_load": observed.get_load(),
        })
    records.append({
        "instance": filename.rsplit("/", 1)[-1],
        "migrations": [{
            "switch": migration.get_switch(),
            "source": migration.get_src_controller(),
            "destination": migration.get_dst_controller(),
            "load": migration.get_load(),
        } for migration in data.get_migrations().values()],
        "controllers": sorted(controllers, key=lambda value: value["controller"]),
    })
print(json.dumps(records, sort_keys=True))
'''


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "scipy", "gurobipy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run_logged(command: list[str], cwd: Path, output: Path, stem: str,
               *, timeout: float, env: dict[str, str] | None = None) -> dict:
    """Keep both streams, including partial output from failed/timed-out commands."""
    started = time.perf_counter()
    record = {"returncode": None, "timed_out": False, "error": None}
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{stem}.stdout.txt").open("w", encoding="utf-8") as stdout, \
            (output / f"{stem}.stderr.txt").open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                timeout=timeout, check=False,
            )
            record["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            record["error"] = f"Timeout after {timeout:g} seconds"
        except OSError as error:
            record["error"] = f"{type(error).__name__}: {error}"
            stderr.write(record["error"] + "\n")
    record["wall_seconds"] = round(time.perf_counter() - started, 6)
    record["stdout_file"] = f"{stem}.stdout.txt"
    record["stderr_file"] = f"{stem}.stderr.txt"
    return record


def git_output(upstream: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(upstream), *arguments], text=True,
        encoding="utf-8", stderr=subprocess.STDOUT, timeout=30,
    ).strip()


def prepare_upstream(upstream: Path, output: Path) -> dict:
    checkout_logs = output / "checkout"
    stages = []
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        record = run_logged(
            ["git", "clone", "--no-checkout", REPOSITORY, str(upstream)],
            upstream.parent, checkout_logs, "clone", timeout=180,
        )
        stages.append({"stage": "clone", **record})
        write_json(checkout_logs / "commands.json", stages)
        if record["returncode"] != 0:
            raise RuntimeError("Upstream clone failed; see checkout/clone.stderr.txt")
        record = run_logged(
            ["git", "-C", str(upstream), "checkout", "--detach", COMMIT],
            upstream.parent, checkout_logs, "checkout", timeout=90,
        )
        stages.append({"stage": "checkout", **record})
        write_json(checkout_logs / "commands.json", stages)
        if record["returncode"] != 0:
            raise RuntimeError("Pinned checkout failed; see checkout/checkout.stderr.txt")
    elif not (upstream / ".git").exists():
        raise RuntimeError("Existing upstream directory is not a Git checkout; refusing to overwrite it")

    actual_commit = git_output(upstream, "rev-parse", "HEAD")
    if actual_commit != COMMIT:
        raise RuntimeError(f"Upstream commit mismatch: expected {COMMIT}, found {actual_commit}")
    origin = git_output(upstream, "remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git") != REPOSITORY.removesuffix(".git"):
        raise RuntimeError("Existing upstream origin does not match the selected public repository")
    tracked_changes = git_output(upstream, "status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise RuntimeError("Upstream has modified tracked files; refusing to report it as the pinned source")
    if not (upstream / "Scripts" / "solution" / "solve_single.py").is_file():
        raise RuntimeError("Pinned upstream CLI is missing")
    return {
        "repository": REPOSITORY,
        "expected_commit": COMMIT,
        "actual_commit": actual_commit,
        "tracked_source_modified": False,
        "upstream_source_vendored_in_results": False,
    }


def patch_controller_membership(original: bytes) -> tuple[bytes, dict]:
    """Apply exactly the documented fix, rejecting missing/ambiguous targets."""
    if original.count(PATCH_OLD) != 1 or PATCH_NEW in original:
        raise ValueError("Expected exactly one pristine controller-membership expression")
    patched = original.replace(PATCH_OLD, PATCH_NEW, 1)
    difference = "".join(difflib.unified_diff(
        original.decode("utf-8").splitlines(keepends=True),
        patched.decode("utf-8").splitlines(keepends=True),
        fromfile=f"original/{PATCH_FILE}", tofile=f"patched/{PATCH_FILE}",
    ))
    return patched, {
        "file": PATCH_FILE,
        "replacement_count": 1,
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "unified_diff": difference,
    }


def prepare_patched_copy(upstream: Path, patched: Path) -> dict:
    """Keep a separate runtime copy; verify the sole changed tracked file."""
    if patched == upstream or upstream in patched.parents or patched in upstream.parents:
        raise RuntimeError("Patched and pristine source locations must be separate")
    expected_bytes, patch = patch_controller_membership((upstream / PATCH_FILE).read_bytes())
    if not patched.exists():
        shutil.copytree(
            upstream, patched,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        (patched / PATCH_FILE).write_bytes(expected_bytes)
    tracked_files = [name for name in git_output(upstream, "ls-files", "-z").split("\0") if name]
    differences = []
    for name in tracked_files:
        source = upstream / name
        target = patched / name
        if not target.is_file():
            raise RuntimeError(f"Patched runtime copy is missing {name}")
        if source.read_bytes() != target.read_bytes():
            differences.append(name)
    if differences != [PATCH_FILE] or (patched / PATCH_FILE).read_bytes() != expected_bytes:
        raise RuntimeError(f"Patched copy has unexpected tracked differences: {differences}")
    patch["changed_tracked_files"] = differences
    patch["pristine_source_modified"] = False
    return patch


def validate_membership_records(records: list[dict], resiliency: bool) -> dict:
    """Derive expected source/destination membership directly from migrations."""
    instances = []
    for record in records:
        differences = []
        empty_controllers = []
        for controller in record["controllers"]:
            name = controller["controller"]
            expected_migrations = [
                migration for migration in record["migrations"]
                if migration["destination"] == name
                or (resiliency and migration["source"] == name)
            ]
            expected = {migration["switch"] for migration in expected_migrations}
            observed = set(controller["observed_switches"])
            expected_load = sum(migration["load"] for migration in expected_migrations)
            if not expected and not observed and abs(controller["observed_load"]) < 1e-8:
                empty_controllers.append(name)
            if expected != observed or abs(expected_load - controller["observed_load"]) > 1e-8:
                differences.append({
                    "controller": name,
                    "expected_switches": sorted(expected),
                    "observed_switches": sorted(observed),
                    "missing_switches": sorted(expected - observed),
                    "extra_switches": sorted(observed - expected),
                    "expected_load": expected_load,
                    "observed_load": controller["observed_load"],
                })
        instances.append({
            "instance": record["instance"],
            "consistent": not differences,
            "differences": differences,
            "empty_controllers": empty_controllers,
        })
    return {"resiliency": resiliency, "consistent": all(x["consistent"] for x in instances),
            "instances": instances}


def run_membership_validation(source: Path, names: list[str], output: Path,
                              env: dict[str, str], resiliency: bool,
                              timeout: float) -> dict:
    command = [sys.executable, "-c", MEMBERSHIP_PROBE,
               str(resiliency).lower(), *[f"instances/{name}" for name in names]]
    execution = run_logged(command, source, output, "membership", timeout=timeout, env=env)
    if execution["returncode"] != 0:
        raise RuntimeError(f"Membership probe failed; see {output.name}/membership.stderr.txt")
    records = json.loads((output / execution["stdout_file"]).read_text(encoding="utf-8"))
    result = {**validate_membership_records(records, resiliency), "execution": execution}
    write_json(output / "membership.json", result)
    return result


def is_known_source_membership_failure(record: dict, stderr: str, validation: dict) -> bool:
    """Only whitelist the demonstrated source-membership failure, not any error."""
    if (record["returncode"] in (None, 0) or record["timed_out"] or record["error"]
            or record.get("vector_first_fit_rounds") is None
            or record.get("current_bottleneck_first_rounds") is not None
            or validation["consistent"]
            or "remove_migration_from_controller" not in stderr
            or "remove_migration_from_constraints" not in stderr):
        return False
    missing_switches = {switch for difference in validation["differences"]
                        for switch in difference["missing_switches"]}
    source_controllers = {difference["controller"] for difference in validation["differences"]
                          if difference["missing_switches"]}
    switch_match = re.search(
        r"(?:MigrationScheduling\.exceptions\.)?SwitchNotFound: Switch (s\d+) not found\.", stderr)
    key_match = re.search(r"(?:^|\n)KeyError: ['\"](c\d+)['\"](?:\n|$)", stderr)
    return bool((switch_match and switch_match.group(1) in missing_switches)
                or (key_match and key_match.group(1) in source_controllers))


def is_known_empty_constraint_failure(record: dict, stderr: str, validation: dict) -> bool:
    """Recognize the independently evidenced zero-membership CBF failure."""
    return bool(
        record["returncode"] not in (None, 0) and not record["timed_out"]
        and not record["error"] and record.get("vector_first_fit_rounds") is not None
        and record.get("current_bottleneck_first_rounds") is None
        and validation["consistent"] and validation.get("empty_controllers")
        and "current_bottleneck_first" in stderr and "find_scheduling_round" in stderr
        and "AttributeError: 'NoneType' object has no attribute 'get_dst_controller'" in stderr
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "results" / "upstream")
    parser.add_argument("--upstream-dir", type=Path, default=root / "build" / "upstream")
    parser.add_argument("--instances", nargs="+", choices=INSTANCES, default=list(INSTANCES))
    parser.add_argument("--timeout", type=float, default=120,
                        help="Maximum seconds for each original CLI invocation")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    output = args.output.resolve()
    upstream = args.upstream_dir.resolve()
    patched_upstream = upstream.parent / "patched-upstream"
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "A_author_heuristics",
        "status": "running",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "scope": "Three explicit source/configuration variants; author bundled examples; VFF/CBF rounds only",
        "optimizer_requested": False,
        "gurobi_license_required_for_this_path": False,
        "author_rng_seed": AUTHOR_SEED,
        "python_hash_seed": HASH_SEED,
        "variants": [],
        "compatibility_environment": {
            "description": "Python 3.10 compatibility dependencies, not the authors' original environment",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "runs": [],
    }
    write_json(output / "summary.json", summary)
    code = 1
    try:
        provenance = prepare_upstream(upstream, output)
        provenance.update({
            "author_cli": "python Scripts/solution/solve_single.py <instance> <resiliency> false",
            "seed_source": "solve_single.py calls utils.initialize_seeds(specs.SEED_NUM), with SEED_NUM=42",
            "pristine_source_changes": [],
        })
        patch = prepare_patched_copy(upstream, patched_upstream)
        provenance["separate_patched_runtime_copy"] = patch
        write_json(output / "patch.json", patch)
        (output / "controller-membership.patch").write_text(patch["unified_diff"], encoding="utf-8")
        write_json(output / "provenance.json", provenance)
        env = os.environ.copy()
        env.update({
            "PYTHONHASHSEED": str(HASH_SEED),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        for variant, source_modified, resiliency in VARIANTS:
            source = patched_upstream if source_modified else upstream
            variant_output = output / variant
            variant_env = {**env, "PYTHONPATH": str(source)}
            validation = run_membership_validation(
                source, args.instances, variant_output, variant_env, resiliency, args.timeout)
            validation_by_name = {value["instance"]: value for value in validation["instances"]}
            variant_record = {
                "variant": variant, "source_modified": source_modified,
                "resiliency": resiliency, "status": "running",
                "membership_consistent": validation["consistent"],
                "membership_validation_file": f"{variant}/membership.json",
                "patch_file": "controller-membership.patch" if source_modified else None,
            }
            summary["variants"].append(variant_record)
            for name in args.instances:
                instance = source / "instances" / name
                if not instance.is_file():
                    raise RuntimeError(f"Bundled instance is missing: {name}")
                stem = instance.stem
                command = [sys.executable, "Scripts/solution/solve_single.py", f"instances/{name}",
                           str(resiliency).lower(), "false"]
                record = {
                    "variant": variant, "source_modified": source_modified,
                    "resiliency": resiliency, "instance": name,
                    "instance_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
                    "command": ["python", *command[1:]],
                    **run_logged(command, source, variant_output, stem,
                                 timeout=args.timeout, env=variant_env),
                }
                stdout = (variant_output / record["stdout_file"]).read_text(encoding="utf-8")
                stderr_text = (variant_output / record["stderr_file"]).read_text(encoding="utf-8")
                for metric, pattern in METRICS.items():
                    match = re.search(pattern, stdout)
                    record[metric] = int(match.group(1)) if match else None
                record["success"] = (record["returncode"] == 0 and all(
                    record.get(metric) is not None for metric in METRICS))
                record["status"] = "passed" if record["success"] else "failed"
                record["known_upstream_failure"] = (
                    (variant == "original_resilient" and is_known_source_membership_failure(
                        record, stderr_text, validation_by_name[name]))
                    or (variant == "original_destination_only" and is_known_empty_constraint_failure(
                        record, stderr_text, validation_by_name[name])))
                record["accepted_for_pipeline"] = (
                    (record["success"] and validation_by_name[name]["consistent"])
                    or record["known_upstream_failure"])
                record["stdout_file"] = f"{variant}/{record['stdout_file']}"
                record["stderr_file"] = f"{variant}/{record['stderr_file']}"
                summary["runs"].append(record)
                write_json(variant_output / f"{stem}.json", record)
                write_json(output / "summary.json", summary)
                print("AUTHOR_RESULT_JSON=" + json.dumps(record, sort_keys=True), flush=True)
                if not record["success"]:
                    print(f"AUTHOR_STDERR_BEGIN={variant}/{name}", flush=True)
                    print(stderr_text, flush=True)
                    print(f"AUTHOR_STDERR_END={variant}/{name}", flush=True)
            runs = [value for value in summary["runs"] if value["variant"] == variant]
            if all(value["success"] for value in runs) and validation["consistent"]:
                variant_record["status"] = "passed"
            elif all(value["accepted_for_pipeline"] for value in runs):
                variant_record["status"] = "completed_with_known_upstream_failures"
            else:
                variant_record["status"] = "failed"
        source_status = git_output(upstream, "status", "--porcelain", "--untracked-files=no")
        if source_status:
            raise RuntimeError("Upstream tracked source changed during author CLI execution")
        if prepare_patched_copy(upstream, patched_upstream) != patch:
            raise RuntimeError("Patched runtime source changed during execution")
        required_variants = [value for value in summary["variants"]
                             if value["source_modified"]]
        summary["pipeline_success"] = (
            all(value["status"] == "passed" for value in required_variants)
            and all(value["accepted_for_pipeline"] for value in summary["runs"]))
        known_failures = sum(value["known_upstream_failure"] for value in summary["runs"])
        summary["known_upstream_failure_count"] = known_failures
        summary["status"] = (
            ("passed_with_known_upstream_failure" if known_failures else "passed")
            if summary["pipeline_success"] else "failed")
        code = 0 if summary["pipeline_success"] else 1
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        (output / "runner.stderr.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        write_json(output / "summary.json", summary)
        print("AUTHOR_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

