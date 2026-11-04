"""Run the pinned authors' migration-scheduling CLI without editing its code.

This phase reports the authors' VFF/CBF round counts on their four bundled
instances. It does not represent controller-failure or packet-level experiments.
Only public source is cloned at runtime; upstream source is not an output artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback


REPOSITORY = "https://github.com/nu-lab/Controller-Load-Migration-Scheduling.git"
COMMIT = "3f6cd3b6d3aca1ff38c3b985535d31bc7698fc05"
AUTHOR_SEED = 42
HASH_SEED = 0
INSTANCES = tuple(f"migrations{i}.txt" for i in range(4))
METRICS = {
    "vector_first_fit_rounds": r"Number of Rounds for Vector First Fit:\s*(\d+)",
    "current_bottleneck_first_rounds": (
        r"Number of Rounds for Current Bottleneck First:\s*(\d+)"
    ),
}


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
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "A_author_heuristics",
        "status": "running",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "scope": "Unmodified author CLI; bundled examples; VFF and CBF round counts only",
        "optimizer_requested": False,
        "gurobi_license_required_for_this_path": False,
        "author_rng_seed": AUTHOR_SEED,
        "python_hash_seed": HASH_SEED,
        "resiliency_argument": True,
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
            "author_cli": "python Scripts/solution/solve_single.py <instance> true false",
            "seed_source": "solve_single.py calls utils.initialize_seeds(specs.SEED_NUM), with SEED_NUM=42",
            "source_changes": [],
        })
        write_json(output / "provenance.json", provenance)
        env = os.environ.copy()
        env.update({
            "PYTHONHASHSEED": str(HASH_SEED),
            "PYTHONPATH": str(upstream),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        for name in args.instances:
            instance = upstream / "instances" / name
            if not instance.is_file():
                raise RuntimeError(f"Bundled instance is missing: {name}")
            stem = instance.stem
            command = [sys.executable, "Scripts/solution/solve_single.py", f"instances/{name}", "true", "false"]
            record = {
                "instance": name,
                "instance_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
                "command": ["python", *command[1:]],
                **run_logged(command, upstream, output, stem, timeout=args.timeout, env=env),
            }
            stdout = (output / record["stdout_file"]).read_text(encoding="utf-8")
            for metric, pattern in METRICS.items():
                match = re.search(pattern, stdout)
                record[metric] = int(match.group(1)) if match else None
            record["success"] = (
                record["returncode"] == 0
                and all(record.get(metric) is not None for metric in METRICS)
            )
            summary["runs"].append(record)
            write_json(output / f"{stem}.json", record)
            write_json(output / "summary.json", summary)
            print("AUTHOR_RESULT_JSON=" + json.dumps(record, sort_keys=True), flush=True)
        source_status = git_output(upstream, "status", "--porcelain", "--untracked-files=no")
        if source_status:
            raise RuntimeError("Upstream tracked source changed during author CLI execution")
        summary["status"] = "passed" if all(run["success"] for run in summary["runs"]) else "failed"
        code = 0 if summary["status"] == "passed" else 1
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

