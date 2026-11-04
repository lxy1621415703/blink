# Controller recovery scheduling experiments

This repository separates author-code reproduction, scheduling experiments, and a new executable-recovery-closure model on GitHub Actions.

## A. Public author-code reproduction

- Upstream: https://github.com/nu-lab/Controller-Load-Migration-Scheduling
- Revision: `3f6cd3b6d3aca1ff38c3b985535d31bc7698fc05`
- Paper: *Switch Migration Scheduling in Distributed SDN Controllers*, NetSoft 2022. This is a later work related to the two-page migration-scheduling poster in the supplied reference collection; the versions are not interchangeable.
- Run the upstream heuristic CLI on its own supplied instances. Record the command, source revision, environment, exit status, and raw output for every variant.
- Install the Gurobi Python package because the upstream package imports it eagerly. The heuristic run disables the optimizer; it does not request a commercial solver license.
- The runner uses Python 3.10 because the unmodified upstream code calls random.sample on a set, an API removed in Python 3.11. Dependency versions are a compatibility environment, not an exact recreation of the authors' old requirements file.
- Upstream source is fetched at run time and is not copied into this repository. The repository does not supply a root license; public availability is not described as an open-source license grant.

These runs measure the original migration heuristic's scheduling outputs and runtime. They do not measure network packet loss or reproduce Ares/QUEST.

Three variants are recorded separately: the unmodified destination-only mode; the unmodified source-and-destination (`resiliency=true`) mode; and a separately labelled one-line bugfix of the latter. In the selected upstream revision, `get_constraint_dict_for_controller` calculates source-and-destination load but hardcodes `get_constraint_switches(False)`. The original resilient CBF path consequently raises `SwitchNotFound` while removing a source migration. The bugfix forwards the existing `resiliency` argument to that call in a separate temporary copy. The pristine checkout remains unchanged. Original failures remain in the result artifacts; a successful repaired run is not labelled an unmodified author-code success.

The unmodified destination-only mode also fails on `migrations1.txt`: an initially empty controller constraint remains in the CBF dictionary after real tasks finish, so it selects no migration and dereferences `None`. A separate membership/load check establishes that empty constraint before this known failure is accepted as an experimental outcome. Original failed rows remain failed. The workflow succeeds only when all rows are either successful or match independently evidenced upstream defects, and the repaired resilient configuration passes every supplied instance. Unexpected failures still fail the workflow.

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


## Latest verified cloud result

Run [41786900624](https://github.com/lxy1621415703/blink/actions/runs/41786900624) completed successfully on GitHub Actions at harness revision `1780e7da324049fbfb80de8bdeb3bbf7510c39e1`. The upstream evaluator preserved five original CLI failures, confirmed their independent defect evidence, and passed all four cases in the separately repaired resilient configuration. Failed original rows were not relabelled as successful.

In the Phase B shared-state scenario (30 paired seeds, 3-5 flow groups), mean weighted unmet demand was 50018.6 for priority-list, 48753.0333 for shared-successor-list, 50316.3 for makespan-rollout, and 48669.0333 for loss-rollout. The candidate reduced this metric by 0.1723% relative to the stronger shared-successor baseline while using about 11.9 times its wall-clock computation. All schedule validation counters were zero and all unchanged-demand negative controls had zero unmet demand. These are limited mechanism-simulation results, not a real-network improvement claim.

[Persistent machine-readable summary](reports/cloud-summary.json) · [Raw CSV, configurations, and diagnostics](https://github.com/lxy1621415703/blink/actions/runs/41786900624/artifacts/13300864894). The raw Actions artifact has seven-day retention; the summary remains in the repository.


## Fast scheduling experiment

The incremental engine and fixed lightweight policy set were evaluated with `speed_protocol.json`: development seeds 0-29, untouched holdout seeds 1000-1029, and separate 20/50/100-flow-group scale cases. Each call includes compilation and all portfolio work. Three warmup rounds and nine timed rounds rotate policy order; samples are summarized per instance before aggregation. Full rollout is not run above 20 groups, as specified before evaluation.

[Run 41788972979](https://github.com/lxy1621415703/blink/actions/runs/41788972979) succeeded: 210 cases, 400 exact engine schedule-equivalence checks, zero feasibility violations, and all negative controls zero. On the 30 shared-state holdout cases, mean per-instance median planner time was 2.62184 ms for legacy loss-rollout and 0.93867 ms for its equivalent incremental implementation (64.2% reduction, identical plans). The fixed three-plan portfolio used 0.30484 ms but increased unmet demand by 0.279% relative to loss-rollout; its mean completion time was 14.433 ticks versus 14.400 ticks. Relative to the fast shared-successor baseline, portfolio loss improved only 0.0072%, with no mean makespan improvement. The holdout did not reproduce the larger development-set benefit.

At 100 groups, the equivalent shared-successor implementation reduced planner time from 52.149 ms to 8.789 ms on five shared-state cases. These are task-DAG scale measurements, not a 100-node WAN or packet-level testbed. The current evidence supports implementation acceleration, not a claim of materially shorter real-network recovery.

[Persistent speed/quality summary](reports/fast-summary.json) · [Raw timings, instances and schedules](https://github.com/lxy1621415703/blink/actions/runs/41788972979/artifacts/13302542910).

## Executable recovery closure

The new question is which recovery operations a particular QoS repair actually requires. Earlier phases gave every scheduler the same fixed recovery DAG. This phase derives that DAG from old and target forwarding paths, installed rule identities, surviving state replicas, state-version dependencies, and switch ownership. It preserves the earlier code and results as separate experiments.

### Mechanism and objective

1. Keep the old forwarding path running after the controller crash. A specified demand increase creates a service deficit; the controller crash does not itself erase forwarding rules.
2. Compare each target path with installed rules. Reuse a rule only when its forwarding action and required semantic version are still valid. An ownership change alone does not invalidate a forwarding rule.
3. For each necessary write, require a live authorized controller, the current fenced ownership epoch, and the transitive application-state dependencies of that write. Reuse valid hot state and fetch missing or stale state from a surviving replica. Shared prerequisites execute once per target controller.
4. Stage changed downstream rules in a target-version context before the ingress activates that context. Preserve the old context while it carries traffic. Common unchanged entries are assumed usable in both contexts; staging space is assumed sufficient. Each flow waits for its own prerequisites; the full-domain comparison has no artificial all-domain ownership barrier.
5. Schedule the selected operations with the same event-driven resource-constrained scheduler used by the scope comparisons. Background restoration of unrelated programmability can proceed separately after the current QoS repair.

For the flows with a positive service deficit, let `Delta_f` be the required writes and `Anc(Delta_f)` their state, ownership, and installation dependencies. The selected foreground work is the union of these dependency closures after removing already valid prerequisites. This is the least predecessor-closed set for the supplied dependency graph; it is not a claim of globally optimal path, backup-controller, or application-state placement.

The scheduling objective is `sum_f weight_f * max(demand_f - old_service_f, 0) * activation_completion_f`. This is cumulative weighted unmet demand in synthetic rate-unit ticks. The implemented list heuristic approximates this objective. A small-instance rollout on the same closure is a separate quality reference. Target paths and backup-controller assignments are fixed and identical across comparisons.

State dependencies are supplied by the synthetic application manifest. They are not automatically inferred from a real controller. An application with a global routing-state dependency can require the entire state bundle; the fully coupled control represents this case. A deployment would need to establish these dependencies from the actual application's state accesses and consistency contract before omitting any state.

### Sources and transferred ideas

- [Ares](https://doi.org/10.1109/JSAC.2025.3528814) and [QUEST](https://doi.org/10.1109/TON.2026.3673992) motivate relating controller recovery to the actual traffic/QoS repair. Their full algorithms are not implemented here.
- [Ravana, SOSR 2015](https://nkatta.github.io/papers/ravana15.pdf) motivates considering controller and switch state together. This model represents version/ownership prerequisites; it does not implement or claim Ravana's transactional exactly-once protocol.
- [ez-Segway, SOSR 2017](https://sands.kaust.edu.sa/papers/ez-segway.sosr17.pdf) motivates deriving update work from the existing forwarding configuration and requiring a safe update order. This experiment uses pre-reserved target capacity and an ingress-last abstraction, rather than ez-Segway's distributed switch protocol.
- [OnlineTE](https://www.microsoft.com/en-us/research/publication/near-optimal-online-traffic-engineering/) and [SSDO](https://www.usenix.org/conference/nsdi26/presentation/mao) motivate retaining useful existing work and avoiding repeated full optimization. This phase reuses valid rules/state and the incremental scheduler. Continuous TE convergence guarantees do not transfer to this discrete model; a live anytime replanner is not implemented.

### Fair scope comparisons

All variants receive the same raw scenario, valid hot state, surviving replicas, old/target routes, fixed backup mappings, and resource budgets. They all reuse matching rules. These are newly implemented mechanism ablations, not executions of the cited papers.

- `full_domain`: recover ownership for the failed controller's whole domain and prepare the full application state at the designated targets, while allowing an individual flow to activate once its actual dependencies are ready.
- `diff_full_state`: recover only the ownership needed by changed rules, but also prepare the full application state at participating targets.
- `closure`: recover only the ownership and transitive state dependencies needed by the current rule changes and ingress activation.
- `closure_rollout`: use the identical closure with the more expensive loss-rollout scheduler on small instances, separating scope selection from scheduling quality.

### Evidence and units

`closure_protocol.json` specifies fresh development, holdout, and scale seeds before evaluation. The benchmark records full build-and-schedule computation time, required state objects and logical transfer bytes, ownership and rule operations, first useful activation, weighted T90, time until all deficient flows recover, and completion of all scheduled work. Background work completion is distinct from business recovery. Independent validation reconstructs the target forwarding and checks state versions, ownership, and activation readiness from the raw scenario.

The controls include fully coupled state, full hot standby, no old/new route overlap, and unchanged demand. Full hot standby must have zero state-transfer work; fully coupled state can remove state-slicing savings; unchanged demand has zero business deficit and requires no foreground QoS repair. No improvement is assumed for these controls.

This is an executable mechanism simulation on synthetic topologies and paths, with explicitly reserved final capacity. Durations remain uncalibrated simulation ticks; planner measurements are wall-clock milliseconds. The two cannot be added to report end-to-end milliseconds. State bytes and rule counts are modeled operations, not observed transfer traffic or FlowMod measurements. No actual controller failover, quorum/fencing protocol, packet queues, in-flight packet consistency, or shared-link congestion is executed.
