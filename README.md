# Controller recovery scheduling experiments

This repository runs two explicitly separate experiments on GitHub Actions.

## A. Public author-code reproduction

- Upstream: https://github.com/nu-lab/Controller-Load-Migration-Scheduling
- Revision: `3f6cd3b6d3aca1ff38c3b985535d31bc7698fc05`
- Paper: *Switch Migration Scheduling in Distributed SDN Controllers*, NetSoft 2022. This is a later work related to the two-page migration-scheduling poster in the supplied reference collection; the versions are not interchangeable.
- Run the upstream heuristic CLI on its own supplied instances, without editing its algorithm. Record the command, source revision, environment, exit status, and raw output.
- Install the Gurobi Python package because the upstream package imports it eagerly. The heuristic run disables the optimizer; it does not request a commercial solver license.
- The runner uses Python 3.10 because the unmodified upstream code calls random.sample on a set, an API removed in Python 3.11. Dependency versions are a compatibility environment, not an exact recreation of the authors' old requirements file.
- Upstream source is fetched at run time and is not copied into this repository. The repository does not supply a root license; public availability is not described as an open-source license grant.

These runs measure the original migration heuristic's scheduling outputs and runtime. They do not measure network packet loss or reproduce Ares/QUEST.

## B. New recovery-mechanism experiment

Question: with identical final recovery assignments, action dependencies, and resource budgets, can a loss-aware schedule reduce cumulative unmet demand compared with simpler schedules?

The benchmark models recovery as a task DAG. State comes from a surviving replica. Controller failure leaves existing forwarding intact. A flow's old path continues to provide its specified service rate until the replacement path is activated. State preparation, ownership acquisition, rule installation, and activation have explicit precedence and resource constraints. Shared preparation tasks execute once.

The first benchmark deliberately isolates the scheduling mechanism: demand and old-path service deficits are constant during the recovery window, and final paths are pre-reserved so activating one flow does not steal another flow's service. It is a discrete scheduling/fluid service-deficit experiment, not a packet simulator, Mininet testbed, or complete implementation of the proposed SD-WAN system.

The loss metric is the time integral of weighted unmet demand. Its simulation units are stated in the configuration. It must not be called measured packet loss, milliseconds, or production throughput without a corresponding packet-level experiment and time calibration.

The generator uses 30 paired seeds, 3-5 flow groups, two surviving controllers, and synthetic task durations. It includes shared-state, equal-work per-flow-state, and unchanged-demand negative-control scenarios. It does not use a real WAN traffic matrix. The same instances and observations are used for every scheduler. Include competitive shared-successor-aware scheduling, a completion-time reference where available, a loss-optimal reference for small instances, and a zero-deficit negative control. Validate precedence and resource feasibility independently of the objective.

## Running

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s source -p 'test_*.py' -v
python source/run_upstream.py --output results/upstream
python source/recovery_benchmark.py --output results/mechanism --seeds 30 --exact-seeds 10 --milp-seconds 8
```

The Actions workflow uploads raw results and diagnostics as an artifact. Inspect per-instance rows and feasibility checks before using aggregate results. A successful workflow is not by itself evidence of a scientific improvement.

## Evidence status

Execution status and measured outcomes are recorded in the workflow logs and result artifacts. No performance improvement is assumed in advance. This repository contains no private source PDFs or account credentials.

## Research references

- Ares: https://doi.org/10.1109/JSAC.2025.3528814
- QUEST: https://doi.org/10.1109/TON.2026.3673992
- Migration scheduling author implementation: https://github.com/nu-lab/Controller-Load-Migration-Scheduling
- Dynamic Scheduling of Network Updates: https://ratul.org/papers/tr2014-dionysus.pdf

