# Module 15 — High-Level Design & Scaling

**Files covered:** cross-cutting. Numbers come from `README.md` §1–2, `docs/IMPLEMENTATION_PLAN.md` §3 and §8, and `configs/base.yaml`; every derivation from them is shown. No new source files.

**Where we are:** Module 14 was the inside view (classes, contracts, states). This module is the outside view: what runs where, how long a run takes and where that time goes, what one machine can sustain, what breaks first as the work grows, and which parts would survive a 1000× scale-up unchanged.

**The one-line thesis:** This is an **evidence-limited system, not a compute-limited one**. On the README's own numbers, evaluation takes about as long as training, a fine-tuning run sees **well under one epoch** of its data, and the first thing that breaks as you scale is the **statistical validity of comparisons** (shared test sets, real seeds), long before GPU, disk or the index.

---

## 15.1 Prerequisites — the absolute basics

- **Deployment topology.** Which processes run on which machines, and what they share (files, databases, GPUs).
- **Latency budget.** A breakdown of where wall-clock time goes in one request, here one experiment run, so you know what to optimize.
- **Throughput / capacity.** Work completed per unit time (steps per second, runs per day), and the ceiling set by the scarcest resource.
- **Epoch.** One full pass over the training data. *Fraction of an epoch* = `steps × batch_size / training_frames`, assuming per-frame sampling. For SmolVLA each sampled frame supervises an *action chunk* of future actions, so per-frame "coverage" understates supervision somewhat. The engine's exact sampling unit is unverified.
- **Bottleneck analysis.** Ordering resources by which one saturates first as load grows. The order tells you what kind of system you have.
- **Scaling ladder.** For each order-of-magnitude increase, the single change without which the system fails.
- **Statistical power.** The probability that an experiment detects a real effect of a given size. It falls with small test sets, few seeds and noisy metrics.
- **Content-addressed cache.** Results stored under a hash of their inputs, so identical requests are served from cache.
- **Multi-tenancy.** Several users or teams sharing one system with isolation (namespacing, quotas, access control).
- **Quality vs infrastructure metrics.** *Quality* metrics say whether the results are right (test-set overlap, confidence intervals, baseline ratios). *Infrastructure* metrics say whether the system is healthy (GPU utilization, run failures, disk).

---

## 15.2 Deployment topology, as it actually is

```
 ┌──────────────────────────── one laptop: RTX 4060 Laptop GPU (8 GB) ──────────────────────────────┐
 │  Linux VM, ~10 GB RAM  (IMPLEMENTATION_PLAN §3: host-visible GPU via nvidia-smi, driver 592.82,  │
 │                          CUDA 13.1; "host virtual machine" — likely WSL2, inferred)             │
 │                                                                                                  │
 │  <parent>/                                                                                       │
 │   ├── episodic-data-engine/   (rde; editable)        ┌── ../.venv      light: no torch           │
 │   ├── vla-finetune/           (this repo; editable)  └── ../.venv-vla  torch(CUDA) + lerobot 0.4.4│
 │   │     ├── runs/<date>/<name>/seed_<n>/   (checkpoints, metrics, reports)    git-ignored        │
 │   │     └── experiments/*.jsonl            (cross-run index)                  git-ignored        │
 │   └── rde data dir (settings.paths): catalog DB + Parquet frames + video files  (engine-owned)    │
 │                                                                                                  │
 │  ~/.cache/huggingface  ← lerobot/smolvla_base weights (downloaded once; network on first run)    │
 │                                                                                                  │
 │  process model: ONE `mtrain` process at a time, single GPU, no daemon, no server, no queue      │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
          external: Hugging Face Hub (base checkpoint, source datasets) — first run only
```

**Decision: single-machine, single-process, file-based.** It's forced by C1 (one laptop GPU) and C9 (one or two researchers, short runs) from the Module 13 capstone. There's nothing to coordinate, so there's nothing to deploy. *Cost:* everything in Module 14's concurrency table becomes a problem the moment a second process appears.

---

## 15.3 The lifecycle of one run, with a latency budget

The README gives **training time** and **per-frame inference time**. I derived evaluation time from those. Lines marked *est.* are estimates I can't measure here.

**SO-100 `baseline_full`** (1,000 steps, batch 4, 1,908 test frames, about 1,900 val frames, inference about 48 ms/frame at batch 16):

| Stage | Time | Derivation |
|---|---:|---|
| env + git + config (torch import, CUDA init) | ~2–5 s | *est.* |
| intervention: engine build of `training/vN` | seconds–tens of s | *est.*, engine-side, unverified |
| split check + profile (12 histogram scans: 6 action + 6 state) | seconds | *est.* (DuckDB on inline columns only) |
| **training (incl. 10 × 32-frame validations, 10 checkpoint saves)** | **324 s** | README Table 2 |
| of which validation | ~15 s | 10 × 32 frames × ~48 ms |
| baselines (NN index on ~15k train frames, score 1,908) | seconds | *est.* (CPU, state-only) |
| SmolVLA test `overall` | ~92 s | 1,908 × 48 ms |
| `task_0` slice (single task ⇒ full duplicate, Module 7) | ~92 s | same frames re-inferred |
| three length slices (partition the test set) | ~92 s | same frames re-inferred |
| temporal re-read, feedback, reports, figures | seconds | *est.* |
| **total** | **~10–11 min** | |
| **evaluation share** | **~45%** (~275 s of ~610 s) | |

**ALOHA `aloha_coffee_baseline`** (1,000 steps, 4,400 test frames, about 86 ms/frame, `slices: [per_length_bucket]`):

| Stage | Time | Derivation |
|---|---:|---|
| training | **576 s** | README Table 2 |
| test `overall` | ~378 s | 4,400 × 86 ms |
| length slice: all 4 test episodes are 1,100 frames, so one bucket = a full duplicate | ~378 s | Module 7 tie collapse |
| **total** | **~22 min** | |
| **evaluation share** | **~57%**, more than training | |

**What the budget says:**
1. **Evaluation, not training, dominates wall-clock.** About 45% on SO-100 and about 57% on ALOHA. **Half of that evaluation time is pure redundancy:** the duplicate `task_0` on single-task SO-100, and the all-in-one-bucket length slice on ALOHA. Predict-once slicing (Module 7) would cut total run time by about 25–30% with *zero* loss of information.
2. **These are lower bounds.** The README's per-frame figure "excludes data loading", and ALOHA's frames come from four AV1 video streams (the config comment says "AV1, 4 cameras"). Decode time is extra and unmeasured.
3. **Validation is cheap (about 5% of training time)** precisely because it's capped at 32 frames. That's the Module 6 trade: cheap, and too noisy to select well.

---

## 15.4 Capacity arithmetic: one laptop

**Training throughput (SO-100):** 1,000 steps / 324 s ≈ **3.1 steps/s**, or about 12 samples/s at batch 4 (an upper bound on step time, since 324 s includes validation and saves).

**How much of the data does one run see?**

| Dataset | Train episodes | ≈ Train frames | Samples seen (1,000 steps × 4) | ≈ Fraction of an epoch |
|---|---:|---:|---:|---:|
| SO-100 | 38 | 38 × ~400 ≈ 15,200 (episodes 326–475 frames) | 4,000 | **≈ 0.26** |
| SO-100, extended (+1,000) | 38 | ≈ 15,200 | 8,000 total | **≈ 0.53** |
| ALOHA | 32 | 32 × 1,100 ≈ 35,200 (test episodes are 1,100; assumed similar for train) | 4,000 | **≈ 0.11** |

**This is the most important number in this module.** The headline runs train on **a quarter of an epoch (SO-100) and a ninth of an epoch (ALOHA)**, assuming per-frame sampling and before counting action-chunk supervision. Read that next to the Module 7 result:
- On SO-100, doubling steps (0.26 → 0.53 epoch) cut test MSE by 18%. The model was still learning.
- On ALOHA, at about 0.11 epoch, SmolVLA is **2× worse than nearest-neighbour retrieval**, which by construction has "seen" 100% of the training frames.

So "SmolVLA loses to retrieval on ALOHA" is, plausibly, **"SmolVLA was undertrained relative to a baseline that memorizes everything."** That's a better-supported and more useful interpretation than "retrieval is just strong". It also suggests the cheapest next experiment: ALOHA at 5–10k steps, about 50–100 min on the same laptop by linear extrapolation. The caveat: if the engine samples action-chunk windows rather than frames, the effective coverage is higher, and I can't verify which.

**Memory headroom:** peak GPU 2.2–3.0 GB of 8 GB (allocator-reported, Module 6), and peak host 3.0–3.3 GB of about 10 GB. **The 8 GB GPU isn't the binding constraint at batch 4.** Batch 8 or 16 probably fits (unverified; activations scale roughly linearly with batch, and the allocator figure excludes the CUDA context). Batch size 4 is **conservative, not forced**, by the measured numbers. That revises the C1 row in the Module 13 capstone. The constraint motivated it, but the measurements show slack.

**Runs per day, serially:** about 10–11 min (SO-100) or about 22 min (ALOHA) per run → **about 130 SO-100 runs/day or about 65 ALOHA runs/day** at 100% utilization, so realistically a few dozen. **Concurrency:** 2–3 GB peak per training process means two would *fit* in 8 GB, but Module 14 showed two concurrent runs race on the run directory, catalog and index. Capacity is single-process by correctness, not by hardware.

**Disk per run** (*est.*): SmolVLA is publicly about 450M parameters, so a checkpoint is roughly 0.9 GB in 16-bit or 1.8 GB in 32-bit (unverified which). Kept per run: last 3 periodic, plus best if older, plus `final` (a duplicate of `step_1000`, Module 6), so **4–5 checkpoints ≈ 4–9 GB per run**. At a few dozen runs, that's **hundreds of GB**. Disk fills long before the GPU saturates.

---

## 15.5 Bottleneck analysis, in the order things break

Scale the *number of experiments* (the axis a research harness actually grows on):

| Order | Bottleneck | Breaks at | Symptom |
|---|---|---|---|
| **1** | **Statistical validity of comparisons** | the **first** comparison between two interventions | different test sets per intervention (Module 4), fake seeds (Module 11), no intervals (Module 7), feedback tuned on test (Module 8): conclusions are wrong *now*, at 3 runs |
| **2** | **Evaluation time** | ~10 runs | about half of each run is inference, half of that redundant |
| **3** | **Disk** | ~20–50 runs | 4–9 GB per run in checkpoints, with no retention policy beyond keep-last-3 |
| **4** | **Single-GPU serial execution** | ~50–100 runs per week | a queue of runs; no scheduler |
| **5** | **Process-level correctness** | the first time anyone parallelizes | run-directory TOCTOU, catalog version race, torn JSONL lines (Module 14) |
| **6** | **Index and catalog as files** | thousands of runs | full-scan queries, no locking, no status |
| **7** | **Data path** (video decode, loader) | 10–100× dataset size | decode-bound evaluation; the profile's per-dimension DuckDB scans grow |

**What the ordering says about the system:** the *first* failure isn't a resource at all. It's that the harness can't yet produce a trustworthy comparison between two experiments. The *second* is an algorithmic waste (redundant evaluation), and only then come the physical resources. That ordering identifies a **research-methodology system**: its scaling problem is evidence quality per GPU-hour, not GPU-hours. The engineering priority list follows directly: fix the comparison protocol before buying hardware.

---

## 15.6 The scaling ladder

Each step names **the one change without which that step fails**, plus what comes along with it.

**Today (1×): about 3 published runs, 1 researcher, 1 laptop GPU, 40–48 episodes.**

**10× (tens of runs, a real intervention study, about 500 episodes).**
- **The one thing: a frozen, shared test split plus predict-once evaluation.** Split the full pool once, apply interventions only to its train split (Module 4), score every run on the same episodes, and slice stored predictions instead of re-inferring (Module 7).
- With it: real seed propagation (Module 11), episode-bootstrap intervals from `test_outcomes.jsonl` (Module 7), feedback on validation (Module 8), and a checkpoint retention policy (keep best plus final, drop the `step_N`/`final` duplicate).
- Without it, ten runs produce ten numbers that can't be compared.

**100× (hundreds of runs, 2–5 researchers, a few GPUs).**
- **The one thing: an explicit run state machine plus a scheduler.** Write `status.json` atomically at every transition (Module 14 §14.7), and have one job queue that owns the GPUs and serializes catalog version allocation.
- With it: the index moves to SQLite or DuckDB (or MLflow as a sink, Module 3), with per-user namespaces for `experiment_name`; checkpoints move to object storage with retention; config gets `extra="forbid"` and fail-fast name validation (Module 11), because a typo now wastes someone else's queue slot; and there's CI with both repos and a pinned engine version (Module 1).
- Without it, concurrent runs corrupt the shared catalog and index.

**1000× (thousands of runs, tens of thousands of episodes, multi-GPU training).**
- **The one thing: the data path.** Pre-decoded or streaming frame storage and a parallel loader owned by the engine (the `num_workers` and `mixed_precision` fields that exist in the config today and are wired to nothing, Module 2), plus distributed training (DDP or FSDP) inside the engine's `fit`.
- With it: the engine catalog becomes a server database with transactional version allocation; content-addressed `training/vN` (identical builds reused, Module 4); sweeps become DAG jobs with stage caching (Module 11); profiling goes from per-dimension full scans to one pass with all dimensions; and real closed-loop evaluation (simulator farm), because at this scale offline MSE stops being an acceptable proxy for a decision.
- Without it, every run is decode-bound and a single GPU per run can't finish an epoch in useful time.

---

## 15.7 Caching strategy

| What | Cached today? | Opportunity | Key |
|---|---|---|---|
| Base checkpoint download | yes, by the Hugging Face cache (external) | — | Hub revision |
| Training dataset version | **no**: every run mints a new `training/vN` (Module 4) | content-addressed build reuse | hash(input version, subset, splits, seed, weights) |
| Baseline fits (NN index) | **no**: refit every run | cache per `training/vN` | training version plus baseline name |
| Test predictions | **no**: re-inferred per slice and per re-score | predict once, slice in memory; store vectors | checkpoint hash plus version plus split plus sampling seed |
| Decoded frames | engine-side, unknown | pre-decoded tensor cache | version plus camera plus resolution |
| Profile | **no**: recomputed per run on an immutable version | cache per version (immutable, so never stale) | dataset version |

**Principle:** because versions are immutable, **everything keyed by a version is cacheable forever.** That's an unusually strong position, and it's the payoff of Module 4's design. The system doesn't exploit it yet.

---

## 15.8 Multi-tenancy

**Today: single tenant, none of the mechanisms.**
- `experiment_name` is a global namespace in `runs/`, so two users' `baseline_full` collide (`FileExistsError`) or, worse, interleave under TOCTOU.
- The engine catalog is shared, with no access control (engine-side, unverified).
- The index has no `user` field. `environment.json` records `hostname` but not the user.
- `make_run_dir`'s no-overwrite guard is the only isolation.

**Minimum for multi-tenancy:** a `user` (or `team`) path segment in `runs/` and an index column; catalog writes through a service that serializes allocation; and per-user GPU quotas in the scheduler. **None of this is warranted before the 100× step**, and the C9 constraint (team of one or two) justifies its absence.

---

## 15.9 Observability: what to measure, quality first

**Today:** files only. `metrics/train.jsonl` (loss, LR, grad norm per step), `validation.jsonl`, `checkpoint_errors.jsonl`, `crash.json`, `environment.json`, peak memory and wall clock. There's **no logging** (Module 3), no live view, and no cross-run dashboard.

**What I'd add, in priority order. Quality before infrastructure:**

| # | Metric | Type | Why it's first |
|---|---|---|---|
| 1 | **test-set identity per run** (hash of test episode IDs) and **"comparable?" flag** in `compare` | quality | the #1 bottleneck; makes the moving test set visible |
| 2 | **SmolVLA / nearest-neighbour MSE ratio**, not just vs mean-action | quality | the honest baseline; mean-action ratios (2.6%) flatter |
| 3 | **bootstrap 95% CI over test episodes** for MSE | quality | 5 test episodes; every difference needs an interval |
| 4 | **inference-noise band** (score overall twice, report the spread) | quality | measured once by accident (up to 5%); should be deliberate |
| 5 | **val–test gap** for the selected checkpoint | quality | 13× today (Module 6); flags a broken selection signal |
| 6 | **epochs seen** = steps × batch / train frames | quality | explains under-training (0.11 epoch on ALOHA) |
| 7 | per-slice frame and episode counts | quality | stops "long episodes are easier" from 1–2 episodes |
| 8 | run status distribution (completed / crashed / killed) | infrastructure | needs the state machine |
| 9 | steps/s, eval frames/s, eval share of run time | infrastructure | evaluation dominates |
| 10 | GPU utilization (not just allocator peak), disk per run | infrastructure | headroom and retention |

**Why quality metrics come first:** this system's characteristic failure isn't a crash, which is loud and already mostly handled. It's a **plausible, wrong conclusion produced silently**: a comparison across different test sets, a variance from fake replicates, a recommendation that can never say "no action". Infrastructure dashboards would show a perfectly healthy system producing those. The metrics that would have caught every serious finding in this manual are quality metrics.

---

## 15.10 Cost model

**Formula per run:** `cost ≈ GPU_hours × GPU_price + disk_GB × storage_price × retention + researcher_minutes × hourly_rate`.

**On the laptop (today):** the marginal GPU cost is electricity, which is negligible. The real costs are **researcher time** and **disk**.

**Cloud-equivalent order of magnitude** (*assumption:* a single entry-level datacenter GPU at roughly $0.5–1 per hour; real prices vary by provider and region, and I haven't looked them up):
- SO-100 run, about 0.18 h → **about $0.10–0.20 per run**
- ALOHA run, about 0.37 h → **about $0.20–0.40 per run**
- A 10-point data-fraction sweep with 3 seeds = 30 runs → **about $5–10**
- Checkpoint storage at 4–9 GB per run quickly exceeds compute cost if kept indefinitely

**Where money is actually wasted, in order:**
1. **Wrong conclusions** (re-running studies whose comparisons were invalid). That's unbounded, and it's the real cost.
2. **Redundant evaluation**, about 25–30% of every run's GPU time (15.3).
3. **Duplicate and retained checkpoints** (`final` = `step_1000`; keep-3 regardless of value).
4. **Config typos found after training** (Module 11): one full run each.

**What survives cost-wise:** at these prices, compute is cheap enough that **adding seeds and longer training costs dollars**. The barrier to trustworthy results is protocol, not budget.

---

## 15.11 What survives unchanged: the good bones

These would carry through every rung of the ladder without redesign:

1. **Data identity by immutable version, with lineage** (Modules 2 and 4). It scales from a laptop to a catalog server unchanged, and it's what makes caching safe forever.
2. **Interventions expressed as engine transforms with recorded reasons** (Module 4). Only the *order* (split first) needs changing, not the mechanism.
3. **The fully resolved config persisted with every run** (Module 2). It becomes the job specification for a scheduler as-is.
4. **The refusal to fabricate as data** (`"unavailable"`, `verified: false`, `crash.json`, rollout records). That's an honesty contract that only gets more valuable as more people read the results.
5. **Validation-only checkpoint selection, structurally enforced** (Module 6). It's the right discipline; only the *sample* needs fixing.
6. **Per-frame outcomes with full provenance** (`test_outcomes.jsonl`, Module 11). That's the raw material for bootstraps, per-joint analysis and predict-once slicing.
7. **Plain-file artifacts readable without this code** (Module 3). DuckDB over the same JSONL is the 100× index with no format migration.
8. **Git identity of both repositories** (Module 3). Only the path resolution needs fixing.
9. **Dry-run-by-default config resolution from the real catalog** (Module 12). It's the human gate a multi-user system needs.
10. **Lazy, import-light CLI** (Module 12). It stays fast as the dependency stack grows.

**The pattern:** the **mechanisms** (identity, lineage, honesty, provenance) are the bones, and they scale. The **statistical protocol** (one test set, real seeds, intervals, val-driven curation) is what's missing, and it's cheap to add.

---

## 15.12 Honest gaps to own before an interviewer finds them

1. **Evaluation is about 45–57% of run time, half of it redundant** (derived from README numbers). *Consequence:* about 25–30% of GPU time per run is wasted.
2. **Headline runs see about 0.26 (SO-100) and about 0.11 (ALOHA) of an epoch** (per-frame assumption). *Consequence:* the ALOHA loss to retrieval is plausibly under-training, and the README's "retrieval is a strong baseline" framing under-states this.
3. **Batch 4 uses 2.2–3.0 of 8 GB.** *Consequence:* "constrained by the GPU" is softer than it sounds. Larger batches were plausibly available (unverified).
4. **Disk, not GPU, is the first physical limit** (4–9 GB per run, estimated). *Consequence:* there's no retention policy beyond keep-last-3, plus a duplicate `final`.
5. **No caching despite immutable versions.** *Consequence:* every run recomputes builds, baselines and profiles that could be reused forever.
6. **No observability beyond files, and no quality metrics** (test-set identity, NN ratio, intervals, val–test gap, epochs seen). *Consequence:* the system looks healthy while producing incomparable results.
7. **Single tenant by design** (global experiment names, shared catalog, no user field). *Consequence:* fine until a second researcher.
8. **All timing beyond training and per-frame inference is estimated**, since data loading and video decode aren't measured anywhere. *Consequence:* the latency budget's lower bounds are real, and its totals are approximations.

---

## The shape to carry into the Fixes & Roadmap module

```
 1 laptop · 8 GB GPU (2–3 GB used) · ~10 GB RAM · 1 process · files + engine catalog
 run (SO-100):  train 324 s  ▸  eval ~275 s (½ redundant)  ▸  total ~10–11 min  ▸  ~0.26 epoch
 run (ALOHA):   train 576 s  ▸  eval ~756 s (½ redundant)  ▸  total ~22 min     ▸  ~0.11 epoch
 breaks in order:  ① comparison validity ② eval time ③ disk ④ serial GPU ⑤ multi-process races
                   ⑥ file index/catalog ⑦ data path
 ladder:  10× frozen test split + predict-once   100× run state machine + scheduler
          1000× data path (decode/loader) + distributed fit
 bones that survive: version identity & lineage · resolved config · "unavailable" honesty ·
                     val-only selection · per-frame provenance · plain-file artifacts
```

---

## Drill — answer from memory, files closed

1. Give the latency budget of one SO-100 run and say where you'd cut first.
2. How much of an epoch does a 1,000-step run see on each dataset, and what does that change about how you read the ALOHA result?
3. *Why didn't you just* move this onto a cluster with a job scheduler and MLflow from the start?
4. *What breaks in production:* the team grows from one researcher to five sharing two GPUs. List what fails, in order.
5. *Interviewer framing:* "How would this system scale to a thousand times more data and experiments?"

---

# Appendix A — Model answers to the drill

**A1. SO-100 latency budget.**
"Training is 324 seconds for 1,000 steps, including ten 32-frame validations and ten checkpoint saves. Evaluation is the surprise. Test inference is about 48 milliseconds a frame over 1,908 frames, roughly 90 seconds per pass, and the default slices run three passes: overall, a per-task slice that's a full duplicate because SO-100 has one task, and three length slices that together cover the test set again. So about 275 seconds of evaluation, against 324 of training, around 45% of a ten-to-eleven-minute run. I'd cut there first. Predict once and slice the stored outcomes, and evaluation drops to about 90 seconds with no information lost. That's about a quarter of total run time, recovered with no hardware change. Those evaluation figures are lower bounds, since the per-frame number excludes data loading."

*Why that answer lands:* it derives the budget from real numbers, finds that evaluation, not training, is the big lever, and quantifies the fix. That's capacity reasoning, not guesswork.

**A2. Epochs seen.**
"At batch 4, 1,000 steps is 4,000 samples. SO-100's 38 training episodes are around 15,000 frames, so that's about a quarter of an epoch, and the extended run about half. ALOHA's 32 training episodes at around 1,100 frames are about 35,000 frames, so 1,000 steps is about a ninth of an epoch. That assumes per-frame sampling. If the engine samples action-chunk windows, coverage is higher, which I couldn't verify. It changes how I read ALOHA: SmolVLA losing two-to-one to nearest-neighbour retrieval is a model that's seen about 11% of the data, against a baseline that has memorized all of it. Plus, doubling training on SO-100 gave 18%. So the defensible reading is 'undertrained', not 'retrieval is just better'. And the cheapest next experiment is ALOHA at five to ten thousand steps, about an hour on the same laptop."

*Why that answer lands:* a simple calculation reframes the project's most uncomfortable result into a testable hypothesis with a costed next step, and it states the assumption.

**A3. Why not a cluster, scheduler and MLflow from day one?**
"Because none of the first bottlenecks are infrastructure. With one or two researchers and ten-minute runs, a scheduler and a tracking server would have added operational surface while the real problems were methodological: interventions scored on different test sets, seeds that didn't propagate, no intervals. Infrastructure doesn't fix any of those, and it would have produced the same incomparable numbers faster. The file-based design also kept the artifacts readable by DuckDB and pandas, so moving to MLflow or SQLite later is an export, not a migration. The trigger for a scheduler is the second concurrent user, because that's when the run-directory, catalog and index races I know about start firing."

*Why that answer lands:* it argues from the measured bottleneck order, shows the infrastructure wouldn't have helped the real problem, and names the precise trigger.

**A4. Five researchers, two GPUs.**
"First, results stay incomparable, the same as with one researcher but with more of them. Then name collisions: experiment names are a global namespace in `runs/`, so two people's `baseline_full` either collide on `FileExistsError` or, if launched at the same moment, race into one directory. Then the shared engine catalog: version allocation isn't atomic, so concurrent runs can claim the same training version. Then the JSONL index: unlocked appends of large model rows can interleave, and one torn line breaks every later read of the registry. Then the GPUs: with no scheduler, two runs can land on one card. Each needs 2–3 GB, so they'd fit, but they'd contend. Then disk fills, at four to nine gigabytes of checkpoints per run. The fixes, in that order: frozen test split and real seeds, per-user namespaces, a status file plus a job queue that also serializes catalog writes, a SQLite or MLflow index, and a checkpoint retention policy."

*Why that answer lands:* it's ordered by when each failure actually occurs, distinguishes correctness races from resource contention, and maps each to a fix.

**A5. "How would this scale to 1000×?"**
"In three steps, each with one non-negotiable change. At 10×, tens of runs and hundreds of episodes, the blocker is comparability, so: freeze one shared test split, apply interventions only to train, propagate real seeds, and predict once for evaluation. At 100×, multiple users and GPUs, it's correctness under concurrency, so: an explicit run state machine and a scheduler that owns GPUs and serializes catalog writes, with the index in SQLite or MLflow. At 1000×, the data path, so: pre-decoded or streaming frames, a parallel loader and distributed training in the engine, plus a transactional catalog service and real closed-loop evaluation, because at that scale offline MSE is too weak to make decisions on. What doesn't change at any step is the core: immutable versioned data with lineage, the resolved config as the job spec, per-frame outcomes with provenance, and the refusal to fabricate. Those are the bones."

*Why that answer lands:* each rung has exactly one named blocker and its fix, the ordering matches the bottleneck analysis, and it ends by naming what's already right. That shows you can tell the parts that need to change from the parts that don't.

---

# Module 15 Summary — the one page to revise

**What this module is.** The system seen from outside: one laptop, one process, files plus an engine catalog; about 10–11 minutes per SO-100 run and about 22 minutes per ALOHA run, with evaluation the larger or near-equal share; under one epoch of training per run; and a bottleneck order that begins with statistical validity, not hardware.

**Five claims to make cold**
1. Evaluation is about 45% (SO-100) and about 57% (ALOHA) of run time, and half of it is redundant slicing. Predict-once saves about 25–30% of each run.
2. 1,000 steps at batch 4 is about 0.26 epoch on SO-100 and about 0.11 on ALOHA. The ALOHA loss to retrieval is plausibly under-training.
3. Peak GPU is 2.2–3.0 GB of 8 GB, so batch 4 is conservative, not forced.
4. Bottleneck order: comparison validity → eval time → disk → serial GPU → process races → file index → data path. It's an evidence-limited system.
5. The ladder: 10× frozen test split plus predict-once; 100× state machine plus scheduler; 1000× data path plus distributed fit.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Single machine, single process, files | cluster + scheduler + tracking server | C1, C9; nothing to coordinate | every concurrency race latent |
| Batch 4 | larger batch | 8 GB caution | under 3 GB used; fewer samples per second |
| 1,000-step budget | epoch-based budget | fast iteration | 0.11–0.26 epoch; under-training |
| No caching | content-addressed caches | simplicity | recompute despite immutable versions |
| File-only observability | logging plus dashboards | no service to run | no quality metrics; no live view |
| No multi-tenancy | namespaces and quotas | team of one or two | collisions at user two |

**Volunteer unprompted:** evaluation dominates run time; the under-one-epoch arithmetic and the ALOHA reinterpretation; disk before GPU; immutable versions cached nowhere; quality metrics before infrastructure metrics.

**Numbers and names:** 324 / 385 / 576 s training; 47–50 ms/frame (SO-100) and 86 ms/frame (ALOHA) at eval batch 16; 1,908 and 4,400 test frames; about 3.1 steps/s; peak GPU 2.2–3.0 GB / 8 GB; host 3.0–3.3 GB / ~10 GB; SO-100 about 15,200 train frames (0.26 epoch), ALOHA about 35,200 (0.11 epoch); checkpoints about 0.9–1.8 GB each (estimated), 4–9 GB per run; cloud-equivalent about $0.10–0.40 per run (assumed $0.5–1/GPU-hour); lerobot 0.4.4, CUDA 13.1, driver 592.82.

**Sentences worth memorizing**
- "Evaluation, not training, dominates my run time, and half of it was redundant."
- "My ALOHA run saw about a ninth of an epoch; losing to a baseline that memorized everything says 'undertrained' before it says 'retrieval is better'."
- "The first thing that breaks when this scales isn't the GPU. It's whether two of my numbers are comparable."
- "Immutable versions mean everything keyed by a version is cacheable forever; I don't cache anything yet."
