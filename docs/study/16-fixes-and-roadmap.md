# Module 16 — Fixes & Roadmap

**Files covered:** the running gaps register (`docs/study/_gaps.md`, about 200 entries across Modules 1–15), consolidated. Each fix names the file or files to change.

**Where we are:** Fifteen modules each ended with "honest gaps". This module turns them into a plan. They're deduplicated, so a gap found in three modules appears once, and they're prioritized by **what they do to the project's conclusions**, not by how easy they are. The module ends with a one-day shortlist, a feature roadmap, and an anti-roadmap of tempting changes that would make the system worse.

**The one-line thesis:** About a dozen fixes matter more than everything else, and almost all of them are about **making two results comparable** (shared test split, real seeds, validation-driven decisions, strict config, non-destructive re-scoring), not about features or infrastructure. Most of them are hours of work, not weeks.

---

## 16.1 Prerequisites — the absolute basics

- **Priority levels used here.**
  - **P0:** the gap invalidates a result or a claim the project makes, *now*, at current scale. Fix it before reporting anything new.
  - **P1:** the gap causes wrong output, lost work or misleading artifacts under realistic use, or a doc states something false.
  - **P2:** hygiene, robustness, cost, or something that only bites at larger scale.
- **Effort estimates** assume someone who knows this codebase, including writing a test for the fix: **XS** < 30 min, **S** about 1–2 h, **M** about half a day, **L** 1–3 days. They're estimates, and engine-side work could change them.
- **Regression test.** A test written to fail before the fix and pass after it. Every P0/P1 below names one.
- **Anti-roadmap.** Changes that look like progress but would make the system worse at its current stage, each with the condition under which the answer flips.

---

## 16.2 P0: fix before reporting anything new

| # | Fix | Gaps consolidated | Files | Effort | Regression test |
|---|---|---|---|---|---|
| **P0-1** | **Freeze one test split; apply interventions only to the train split.** Build `training/v_base` once from the full pool. Each intervention then becomes a transform on `v_base`'s *train* episodes (filter or weight), inheriting val/test unchanged. | M4-1, M11-10, M8 (compare on same slice), M4-9 | `data/interventions.py`, `experiments/runner.py`, `experiments/data_fraction.py`; possibly an engine transform that preserves split attributes | **L** | assert `reduced` and `imbalanced` runs' test episode IDs == baseline's |
| **P0-2** | **Make `seed` a real seed.** Copy the config with `training.seed = seed`, re-stamp `experiment_id`, and loop over `config.seeds` from the CLI. Keep the split seed fixed. | S1, M11-1, M12-12 | `experiments/runner.py`, `cli.py` | **S** | two seeds → different `training.seed` in `config.resolved.yaml` and different `experiment_id` / `model_id` |
| **P0-3** | **Strict, fail-fast config.** `model_config = ConfigDict(extra="forbid")` on all six models; `Literal` for `training.model`, `selection_metric`, `slices`, `baseline_models` (or a stage-0 lookup against `MODEL_REGISTRY`); bound split fractions to [0, 1] with `val > 0`; require `checkpoint_every % validate_every == 0`. | M2-1, M2-2, M2-8, M2-14, M11-2, M6-5 | `config/schema.py`, `experiments/runner.py` (stage 0) | **S** | `learning_rat:` → error; `model: SmolVLA` → error; `baseline_models: [nearest_neighbour]` → error *before* `make_run_dir` |
| **P0-4** | **Close the imbalanced silent no-op.** Raise when `imbalanced` resolves to *every* episode, or when `variable` is set but `variable_value` is `None`. Coerce `task_id` values to `int` in the schema. | M2-4, M12-6, M4-5 | `data/interventions.py`, `config/schema.py`, `cli.py` | **XS** | `imbalanced, variable=episode_length_bucket, value=None` → `ValueError` |
| **P0-5** | **Run the feedback loop on validation, not test.** Do one full val evaluation of the selected checkpoint, and pass it to `build_feedback_artifact`. Test stays untouched until the final report. | M8-1 | `experiments/runner.py` | **S** | spy on `build_feedback_artifact`: its `result.split == "val"` |
| **P0-6** | **Never overwrite results in `mtrain evaluate` / `test`.** Write to `evaluation/rescore_<UTC-ts>_<split>/`, name files by split (`<split>_overall.json`), and refuse to write `test_overall.json`. | S7, M12-1, M12-2 | `cli.py` | **S** | after `mtrain evaluate --split val`, the original `test_overall.json` is byte-identical |
| **P0-7** | **A trustworthy selection signal.** Choose a fixed, seeded, stratified validation subset (frames spread across every val episode and across progress deciles) once per run, score every checkpoint on it, and average 2 stochastic samples per frame. | M6-1 | `training/trainer.py` | **M** | subset covers every val episode; identical subset across checkpoints |

**Why these seven:** each one, unfixed, produces a number that's wrong or a comparison that's invalid *today*. P0-1 is the only one larger than half a day, and it's the one the experiment design depends on.

---

## 16.3 P1: wrong output, lost work, or false documentation

### Correctness and data integrity

| # | Fix | Gaps | Files | Effort |
|---|---|---|---|---|
| P1-1 | **Predict once, slice stored outcomes.** `stratified_evaluate` groups `overall.outcomes` by episode. Measure inference noise deliberately (score overall twice with different sampling seeds). | M7-1, M15-1 | `evaluation/offline.py` | S |
| P1-2 | **Bootstrap 95% CI over test episodes** from `test_outcomes.jsonl`, reported in `test_overall.json`, the report and `compare`. | M7-3, M15-6 | `evaluation/offline.py`, `reports/*` | S |
| P1-3 | **Explicit run state machine.** Write `status.json` atomically at every transition; `make_run_dir --force` only from `failed`; mark `summary.json` complete. | M3-3, M11-3, M14 §14.7, S13 | `tracking/run_dir.py`, `experiments/runner.py`, `cli.py` | M |
| P1-4 | **Atomic writes and tolerant reads.** Write to temp then `os.replace` for JSON; a single `os.write` per JSONL line; `read_all` skips and counts bad lines. | M3-1, M3-2 | `tracking/run_dir.py`, `tracking/index.py` | S |
| P1-5 | **Crash coverage.** `except BaseException`, a SIGTERM handler that raises, and `status: running` written before `fit`. | M6-2 | `training/trainer.py`, `experiments/runner.py` | S |
| P1-6 | **Selection-fallback honesty.** Add a limitation when `best → final` fires; reject non-finite val metrics (`math.isfinite`). | M6-3, M6-4 | `training/trainer.py`, `experiments/runner.py` | XS |
| P1-7 | **Feedback thresholds that can say "no".** An absolute or baseline-relative gate on *whether* to act; concentration tested against the p-percentile base rate (binomial). | M8-2, M8-3 | `feedback/artifact.py` | S |
| P1-8 | **`health_check` cross-checks checkpoint ↔ contract** (action dim, camera keys and count, dataset id), and compares preprocessor stats with contract stats. | S8, M9-1, M9-2 | `deployment/loader.py` | S |
| P1-9 | **Coverage normalized by `log2(_HIST_BINS)`.** Keep evenness as a separately named metric. | M5-4 | `data/profile.py` | XS |
| P1-10 | **Split integrity `ok` = complete, non-empty, disjoint partition with sizes matching config ±1.** | M5-2 | `data/split_integrity.py` | XS |
| P1-11 | **One definition of length buckets**, computed once on the full pool with fixed cut points, used by both intervention and evaluation. | M4-4, S11 | `data/interventions.py`, `evaluation/offline.py` | S |
| P1-12 | **Engine provenance from `rde.__file__`**, not a guessed path; save `git diff HEAD` for dirty trees. | M3-5, M3-6 | `tracking/run_dir.py`, `experiments/runner.py` | XS |
| P1-13 | **The zero-data anchor = evaluate the base checkpoint with no fine-tuning** (its own path); fix or delete the misleading test. | S6, M4-2, M13-3 | `data/interventions.py`, `experiments/data_fraction.py`, tests | S |
| P1-14 | **Sweep fault isolation.** A per-fraction `try`, incremental results to `experiments/sweeps/*.jsonl`, nested subsets (prefixes of one permutation), and collision-free names. | M11-11, M11-12, M4-6 | `experiments/data_fraction.py`, `data/interventions.py` | S |

### Artifacts and reporting

| # | Fix | Gaps | Files | Effort |
|---|---|---|---|---|
| P1-15 | **Report correctness.** Fix the git key (`vla_finetune`); show SmolVLA MAE; include per-dimension MAE; `n/a` for `None`. | S2, M3-8, M10-1, M10-3, M10-4, M10-15 | `reports/experiment_report.py` | XS |
| P1-16 | **The limitations list carries every caveat**: rebalanced ("redistributes attention, cannot close a coverage gap"), selection fallback, epochs seen, test set differs from baseline (until P0-1). | M4-8, M10-12 | `experiments/runner.py` | XS |
| P1-17 | **`compare` that can't mislead.** Build the rich table by key (union of columns); add a test-set hash column with a "comparable?" flag; group by experiment and report mean ± sd and n over seeds. | M10-5, M10-6, M10-7, M15-6 | `reports/comparison.py`, `cli.py` | S |
| P1-18 | **Model artifact labels.** Store the `selection: {metric, value, split: "val", step}` that justified selection, separately from `reported: test metrics`. | M9-7 | `deployment/artifact.py` | XS |
| P1-19 | **`export` includes `git.json`, `environment.json`, reports and predictions**; drops the empty `tables/`. | M12-9 | `cli.py` | XS |
| P1-20 | **`build-intervention --apply` preserves comments** (ruamel round-trip, or a separate `*.intervention.yaml` merged by the loader); records `kind: imbalanced`. | M12-4, M12-5 | `cli.py`, `config/loader.py` | S |
| P1-21 | **Factor runner stage 4 into one function** used by both the runner and the CLI (`run_evaluation_stage(run_dir, config, view, checkpoint)`), reading the run's own resolved config. | M12-3, M14-3 | `experiments/runner.py`, `cli.py` | M |
| P1-22 | **Rebuild reports from disk.** `build_report_ctx(run_dir)` plus `mtrain report --run`. | M10-2 | `reports/experiment_report.py`, `cli.py` | S |

### Testing and supply chain

| # | Fix | Gaps | Files | Effort |
|---|---|---|---|---|
| P1-23 | **Cross-stage invariant tests**: val-only selection (spy), seed propagation, slices partition overall, artifacts agree, light venv doesn't import torch. | M13-1, M1-7 | `tests/` | M |
| P1-24 | **Tests for untested surfaces**: CLI (`typer.testing.CliRunner`), reports, `_prune_checkpoints`, `crash.json`, registry ordering, sweep. | M13-4, M10-16, M12-15, M6-10, M9-11 | `tests/` | M |
| P1-25 | **A fixture that exercises real logic**: varied episode lengths and ≥ 10 episodes, so length buckets and length slices actually run. | M13-2 | `tests/conftest.py` | XS |
| P1-26 | **CI, a pinned engine and a lockfile.** A two-repo checkout, `uv.lock`, an engine version range; light suite per push, smoke nightly on GPU. | M1-2, M1-3, M1-5, M13-5 | `pyproject.toml`, `.github/workflows/` | M |

### Documentation that states something false (fix the doc, or make it true)

| # | Doc claim | Reality | Action |
|---|---|---|---|
| D1 | `FAILURE_ANALYSIS.md` §4: "predictions are deterministic given the checkpoint and the frame" | stochastic (README §2.3) | correct; store predicted and target vectors (M8-5) |
| D2 | `FAILURE_ANALYSIS.md` §3: per-dimension per-episode via `episode_metrics`; "sorted by mse" | neither | correct, or implement (M8-6) |
| D3 | `FAILURE_ANALYSIS.md` §6: recommendation "names a concrete next step" | fixed template | correct, or implement attribute analysis (M8-4) |
| D4 | `EXPERIMENTS.md` §4: rebalanced limitation in every report | absent | P1-16 makes it true |
| D5 | `EXPERIMENTS.md` §6: "necessarily" different test splits; `dataset.json` records evaluated IDs | not necessary; it doesn't | P0-1 makes the first moot; correct the second (M4-9) |
| D6 | `EXPERIMENTS.md` §7: `compare` reports mean/std over seeds | not implemented | P1-17 makes it true |
| D7 | `METRICS.md` §2: `evaluate_baselines` computes ratios | it doesn't | correct (M7-7) |
| D8 | `METRICS.md` §6 and profile methodology: entropy "measures how evenly the observed range is used" | evenness among occupied bins | P1-9 makes it true |
| D9 | `API_REFERENCE.md` §4.5/§9.3: use `evaluate --split val` | writes val into `test_overall.json` | P0-6 makes it safe |
| D10 | `API_REFERENCE.md` §5.5: `success_tolerance` defines the proxy | never passed | wire it (one line, M7-12) |
| D11 | `DEPLOYMENT.md` §5: predict invokes `select_action` | `predict_action_chunk` | correct (M9-9) |
| D12 | `DEPLOYMENT.md` Table 2, registry docstring: metrics "justified selection" | test metrics | P1-18 |
| D13 | `loader.py` `DeploymentHealthError`: "any checkpoint/contract mismatch" | no comparison | P1-8 |
| D14 | `schema.py`: "enforced by `model_training.experiments.selection`" | no such module | point at `Trainer.on_step` (S3) |
| D15 | `schema.py`: identity fields "not user-editable" | editable | enforce with a before-validator (M2-5) |
| D16 | `interventions.py`: length buckets "same method as profile.py" | profile has none | correct (M4-4) |
| D17 | `experiment_report.py`: "assembled entirely from saved run artifacts" | in-memory ctx | P1-22 makes it true |
| D18 | `trainer.py` crash comment: catches SIGTERM | doesn't | P1-5 makes it true |
| D19 | `data_fraction.py` / API Table 8: "isolates the quantity effect" | different test sets per point | P0-1 makes it true |
| D20 | `configs/smolvla/so100_pickplace.yaml`: "point --base at it" | no `--base` flag | add the flag or delete the file (M2-9) |
| D21 | `scripts/compare.py`: `--runs a b c` | invalid | fix the usage text (M12-11) |
| D22 | README §2.2: "long SO-100 episodes are easier" | 1–2 episodes per slice | downgrade (M7-2) |
| D23 | README §2 point 3: "retrieval is a strong baseline" as the explanation | plausibly under-training (≈0.11 epoch) | add the epoch arithmetic (M15-2) |
| D24 | `plot_results.py`: "reproducible from disk" | selection markers hard-coded | read `model.json` (M10-8) |

| D25 | README Table 5: "A fraction of 0.0 is honoured as zero episodes" | the builder raises on zero episodes | correct, or implement P1-13 (M4-2) |
| D26 | README Table 5: imbalanced = "under-representation of one regime" | removes the regime entirely | correct the wording (M4-3) |
| D27 | README §3.3: "a process killed mid-write cannot leave a corrupt 'best' model" | verification is presence-only and can't run in a killed process | downgrade (M6-9) |
| D28 | README §2.4: "scattered ... points to a model or coverage limitation" as a finding | the only possible output at p90 with 4–5 episodes | reframe (M8-3) |

**Effort:** each doc correction is XS. Together they're about 2 hours, and they buy the most credibility per minute in this whole module.

---

## 16.4 P2: hygiene, robustness, cost, scale

Grouped, with files. Each is XS–S unless noted.

- **Config and identity:** a config hash separate from the run ID (M2-6); one home for defaults (M2-10); a discriminated-union intervention (M2-12); a single 0.5 default (M2-13); bounded weights (M2-15); dated resume path → a run reference (M2-11). Files: `config/schema.py`, `config/loader.py`, configs.
- **Tracking:** an atomic `mkdir(exist_ok=False)` (M3-4); `environment.json` adds driver, cuDNN, `transformers`, `pip freeze`, and renames compute capability (M3-7); delete `logs/` or tee stdout (M3-9); index rows gain `status` and `schema_version` (M3-10, M14-1); strict JSON types instead of `default=str` (M3-11); route all writes through `RunDir` (M3-12); default dirs not from `parents[3]` (M3-13).
- **Interventions:** validate explicit IDs and weights (M4-7); content-addressed builds to stop version proliferation (M4-10, M15-5, **M**); a clearer unsupported-variable error (M4-12); a stale docstring (M4-11).
- **Profile:** profile the train and test splits separately (M5-6); a real CV (M5-5); report out-of-range values and NULLs (M5-8); drop the duplicate metric (M5-9); a notes fix (M5-10); `to_dict` via `fields()` (M5-11).
- **Trainer:** resume checked against the dataset, verify before load (M6-6); optional optimizer-state save for long runs (M6-7); skip the duplicate `final` when aligned, verify `final`, move it inside `try` (M6-8); a content integrity check (M6-9); prune bookkeeping (M6-10); move the empty-train guard to the top (M6-13); report allocator vs reserved GPU memory (M6-12).
- **Evaluation:** rename per-episode "mse" (M7-5); drop `per_task` from `base.yaml` for single-task data (M7-6); a note when slices are skipped (M7-8); robust `frame_index` (M7-9); frame count in the error distribution (M7-11).
- **Feedback:** consume or drop `retraining_selection` (M8-8); configurable thresholds (M8-10); rename `affected_slice` (M8-9).
- **Deployment:** an observation-to-`Sample` adapter (roadmap); fix the latency breakdown (measured `total_ms`, sync, warm-up, percentiles) (M9-5); registry stages (M9-6, roadmap); identity fields (M9-8).
- **Reports and figures:** object-oriented matplotlib without the global backend (M10-10); guard plotting so it can't abort a completed run (M10-13); wire or delete the unused plot and matrix functions (M10-9); stop the dead `manifest.json` reads (M10-14).
- **Runner and CLI:** a GPU guard (M11-6); NN baseline and total time in `summary.json` (M11-7); relative `run_id` (M11-8); evaluate on `training.device` (M11-14); `crash.json`-aware `validate` and `analyze-failures` (M12-10); dedupe `_target_episodes` (M12-7); resolution-version check (M12-8); `inspect` default (M12-13); remove script wrappers in favour of `python -m` (M12-14); timestamp rollout records (M12-16).
- **Packaging:** a top-level `__init__.py` (M1-6); drop `allow-direct-references`, unused `pytest-cov` and dead `psutil` fallbacks (M1-8); promote the engine's private functions used by tests (M1-4); session-scoped read-only fixtures (M13-6).
- **Scale (defer until the trigger in Module 15):** a scheduler, a SQLite or MLflow index, multi-tenancy namespaces, a checkpoint retention policy and object store (M15-4, M15-7), measuring data loading and decode (M15-8).

---

## 16.5 The five-fix shortlist: about one day

Chosen for **maximum change in what the project can truthfully claim, per hour**. Each comes with its regression test.

| Order | Fix | Why this one | Time |
|---|---|---|---|
| 1 | **P0-3 strict, fail-fast config** (`extra="forbid"`, `Literal`s, stage-0 name checks, cadence alignment) | closes the whole class of "typo silently changes the run" and "typo found after training" | 1.5 h |
| 2 | **P0-2 real seed propagation** plus identity re-stamp plus a seed loop in the CLI | turns the documented multi-seed workflow from fake replicates into real ones; unlocks every variance claim | 1.5 h |
| 3 | **P0-6 non-destructive re-scoring** (versioned rescore directory, split-named files) | stops published numbers from being overwritten; removes the doc-endorsed val-as-test footgun | 1.5 h |
| 4 | **P1-1 predict-once slicing** | removes stochastic non-reconciliation *and* about 25–30% of each run's GPU time | 2 h |
| 5 | **P1-15 + P1-16 + D1–D28 doc and report truthfulness pass** (git key, SmolVLA MAE, per-dimension MAE, limitations, every false doc sentence) | the cheapest credibility in the project; every reader-facing claim becomes true | 1.5 h |

**Total: about 8 hours.** Deliberately *not* on the list despite being P0: **P0-1, the frozen test split** (1–3 days, may touch the engine), and **P0-7, the frozen validation subset** (half a day). They're the first two items of the roadmap below, not afternoon work.

**After this day you can say:** "Config errors fail before the GPU starts, seeds are real, results are never overwritten, slices reconcile with the overall number, and every statement in the docs and reports matches the code."

---

## 16.6 Feature roadmap

**Phase A — Validity (1–2 weeks).** Everything a reviewer needs before believing a comparison.
1. Frozen test split; interventions on train only (P0-1).
2. Frozen stratified validation subset (P0-7); feedback on validation (P0-5).
3. Bootstrap CIs and paired per-episode comparisons on the shared split (P1-2).
4. Three seeds per configuration, as standard.
5. One length-bucket definition (P1-11), and train/test-split coverage in the profile.

**Phase B — Experiments the data can answer (1–2 weeks of GPU-evenings).**
1. **ALOHA at 5k–10k steps** (≈0.5–1 epoch) to test the under-training hypothesis (Module 15). This is the highest-information experiment available.
2. A batch-size check (8/16) for throughput, given 2–3 GB peak of 8 GB.
3. A nested data-efficiency curve on the frozen split (25/50/75/100%, 3 seeds each) with the 0% point = the base checkpoint.
4. A rebalanced run on SO-100's length bucket, compared *paired* against the full run on the shared test split.
5. A held-out-session or held-out-condition split, if the data supports one, as the first real generalization test (Module 5's content-leakage point).

**Phase C — Reliability (about 1 week).** The run state machine (P1-3), atomic I/O (P1-4), crash coverage (P1-5), invariant and CLI tests (P1-23 to P1-25), CI plus lockfile plus pinned engine (P1-26), the factored evaluation stage (P1-21), and reports rebuildable from disk (P1-22).

**Phase D — Toward closed loop (open-ended).**
1. An observation adapter (raw camera frames, joint vector, instruction → model input) so `DeployedModel` takes robot data, not `Sample`s (Module 9).
2. An action-chunk execution loop with control-rate handling.
3. A real latency benchmark: single sample, warmed up, synchronized, p50/p95/p99.
4. A simulator task matching one embodiment, *if one exists*. I haven't checked which LeRobot simulation environments match these datasets. That gives the first non-"unavailable" rollout number.

**Phase E — Scale (only on the Module 15 triggers).** Scheduler plus GPU lock; SQLite, DuckDB or MLflow index; content-addressed builds; retention policy and object storage; per-user namespaces.

---

## 16.7 Anti-roadmap: tempting changes that would make it worse

| Tempting change | Why it's worse now | Flips when |
|---|---|---|
| **Adopt MLflow or W&B** | adds a service and a second source of truth while the real problem is comparability; dashboards would show incomparable runs side by side | a second concurrent researcher, or hundreds of runs |
| **Build a FastAPI inference server** | `predict` takes dataset `Sample`s; a server would expose a replay harness over HTTP | a robot controller in another process needs to call the model |
| **Replace JSONL with Postgres** | migration and operations cost for tens of rows; loses the "readable without our code" property | concurrent writers, or thousands of runs (and even then, DuckDB-over-JSONL first) |
| **Move config to Hydra** | the file-per-experiment record *is* the experiment log; Hydra's CLI overrides encourage unrecorded variants | hyperparameter sweeps become routine (and outputs auto-record the resolved config) |
| **Mock the engine so tests run without it** | the bugs live at the engine boundary; a mock encodes our assumptions (it would never reveal filter-then-split) | never for boundary tests; *yes* for pure logic (pruning, thresholds, rendering) |
| **Add more slices** (per joint, per phase, per object) | 5 test episodes; slices are already 1–2 episodes; more slices mean more noise read as findings | the test set reaches tens of episodes, with CIs per slice |
| **Report `success_rate` as task success, or estimate rollout success from offline MSE** | breaks the project's central honesty contract | never, without closed-loop evaluation |
| **Close the feedback loop automatically** (apply the recommendation and retrain unattended) | the loop runs on test, can't say "no action", and can't produce its defect branch, so automation would overfit the test set at machine speed | feedback runs on val with an absolute gate, *and* a fresh untouched test set exists |
| **Switch to a bigger or different model** | the comparison protocol can't yet tell whether *this* model is undertrained | after Phase A and the ALOHA longer-training experiment |
| **Replace the percentile threshold with an absolute one** | loses ranking that's robust to normalization | never replace; *add* an absolute gate alongside |
| **Delete the split-integrity check as tautological** | it's free, documents the invariant, and guards against a representation change | never delete; strengthen it (P1-10) |
| **A blanket `--force` to overwrite run directories** | destroys the write-once guarantee for good runs too | allow `--force` only from `status: failed` (P1-3) |
| **Rebalance by duplicating episodes on disk** | heavier, less traceable, and the same effect as weights | never; weights are the honest mechanism |
| **Always save optimizer state** | about 3× checkpoint size on a disk-bound laptop | multi-hour or preemptible runs that need exact resume |
| **Add a DAG framework** | 10–20-minute runs don't need caching or resume infrastructure | multi-hour runs or sweeps over ~10 runs; first try "skip a stage if its artifact is complete" |

---

## 16.8 Honest gaps to own before an interviewer finds them

1. **The P0 list is long for a project with three published runs.** *Consequence:* the README's results predate the protocol fixes, so present them as "pipeline validation runs", not as findings, until P0-1, P0-2 and P0-7 land.
2. **P0-1 may need engine work.** A transform that preserves split attributes might not exist in the engine. *Consequence:* the most important fix isn't fully within this repository's control, and its estimate is uncertain.
3. **Estimates are untested.** None of these fixes has been attempted, and the engine isn't available here. *Consequence:* treat efforts as planning numbers.
4. **Some README numbers would change after the fixes** (any result scored on a moving test set, any selection from 32 frames). *Consequence:* be ready to say which published numbers are robust (continued-training on the identical split) and which aren't.

---

## The shape to carry into the Interview Kit

```
 P0 (claims invalid now):  frozen test split · real seeds · strict/fail-fast config · imbalanced no-op
                           guard · feedback on val · non-destructive rescoring · frozen val subset
 P1 (wrong/lost/false):    predict-once · bootstrap CI · run state machine · atomic I/O · crash coverage
                           · health cross-check · coverage metric · compare by key + test-hash + seeds
                           · report truth · 28 doc corrections · invariant/CLI tests · CI + lockfile
 P2 (hygiene/scale):       ~60 items, XS–S each, scale items deferred to Module 15 triggers
 1-day shortlist:          config ▸ seeds ▸ rescoring ▸ predict-once ▸ truthfulness pass   (~8 h)
 Roadmap:                  A validity ▸ B experiments (ALOHA 5–10k steps first) ▸ C reliability
                           ▸ D closed loop ▸ E scale-on-trigger
```

---

## Drill — answer from memory, files closed

1. If you had one day, which five fixes would you make, and why that order?
2. Why is the frozen test split P0 but not on the one-day list?
3. *Why didn't you just* add MLflow? It would have given you experiment comparison for free.
4. *What breaks in production* if you automate the feedback loop today?
5. *Interviewer framing:* "If you had another month on this project, what would you do?"

---

# Appendix A — Model answers to the drill

**A1. The one-day list.**
"First, make the config strict and fail fast: forbid unknown keys, use literal types for the model, metric, slice and baseline names, and check all names before the run directory is created. That closes every 'typo silently changed the run' and 'typo found after training' bug in one change. Second, make `--seed` actually set the training seed and re-stamp the experiment ID, because today replicates share a seed and any variance I report is fake. Third, make re-scoring non-destructive, so `mtrain test` can't overwrite published numbers and `--split val` can't write into a file called test. Fourth, predict once and slice the stored outcomes. That makes slices add up to the overall number and saves about a quarter of each run's GPU time. Fifth, a truthfulness pass: fix the report bugs and correct the nearly thirty doc sentences that don't match the code. The order goes from silent-wrong-results to lost-results to waste to credibility, and each fix comes with a test that fails before it."

*Why that answer lands:* it's concrete, ordered by consequence with the reason stated, sized to a day, and every item has a test. That's what "prioritize" means.

**A2. Why the frozen split isn't in the day.**
"It's the most important fix. Without it, every intervention is scored on a different test set, so the quantity-versus-coverage comparisons the project exists for aren't like-for-like. But it isn't an afternoon change. It reverses the order of operations from 'filter, then split' to 'split once, then intervene on train', and that may need a transform in the data engine that preserves split assignments, which I can't confirm exists. That's one to three days and partly outside this repo. So it heads the roadmap, and until it lands I'd present the intervention results as unvalidated."

*Why that answer lands:* it separates importance from effort, explains the dependency honestly, and says how results should be qualified in the meantime.

**A3. Why not MLflow?**
"MLflow would have given me a comparison UI, but the problem wasn't displaying runs side by side. The runs weren't comparable: different test sets per intervention, fake seed replicates, no intervals. MLflow would have shown those incomparable numbers in a nicer table. My artifacts are plain files that DuckDB or pandas can read directly, so adding MLflow later as a sink is cheap. I'd add it when a second researcher starts running experiments concurrently or runs reach the hundreds. That's when a tracking server's locking and UI earn their operational cost."

*Why that answer lands:* it identifies the real problem the tool wouldn't solve, keeps the option open cheaply, and names the flip condition.

**A4. Automating the feedback loop.**
"It would overfit the test set at machine speed. The feedback artifact is computed on test outcomes, so each automatic iteration would choose the next data intervention by looking at test errors and judge it on the same test errors. It also can't say 'no action', because a 90th-percentile threshold always flags 10% of frames, so it would intervene forever. And its 'data defect' branch is unreachable at these test sizes, so it would always attribute failures to the model. Automation is the right end state, but only after feedback runs on a validation evaluation, there's an absolute or baseline-relative gate on whether to act, and there's a fresh test set the loop has never seen."

*Why that answer lands:* it combines three independent findings into one decisive argument and states exactly what has to be true before automation is safe.

**A5. "Another month?"**
"Week one: validity. Freeze one test split and apply interventions only to train, propagate real seeds, select checkpoints on a frozen stratified validation subset, run feedback on validation, and add bootstrap intervals. Weeks two and three: the experiments that change conclusions. First, ALOHA at five to ten thousand steps, because the current run saw about a ninth of an epoch, and I think 'SmolVLA loses to retrieval' is really 'SmolVLA is undertrained'. Then a nested data-efficiency curve with three seeds per point, and a rebalanced run compared paired against the full run on the shared split. Week four: reliability, meaning a run state machine, atomic writes, CI with both repositories and a lockfile, plus an observation adapter so the deployment path takes real robot inputs. I would *not* spend the month on MLflow, a server or a bigger model. None of those helps until two numbers from this system can be compared."

*Why that answer lands:* it's a sequenced plan where each week's purpose is stated, the headline experiment is justified by a calculation, and it explicitly rejects the tempting distractions. That shows judgement, not just a to-do list.

---

# Module 16 Summary — the one page to revise

**What this module is.** About 200 gap entries consolidated into 7 P0 fixes (comparability and truthfulness of results), 26 P1 fixes plus 28 doc corrections, and about 60 P2 items. Then a one-day shortlist (about 8 hours), a five-phase roadmap, and a fifteen-row anti-roadmap with flip conditions.

**Five claims to make cold**
1. The P0s are all about comparability: frozen test split, real seeds, strict config, no imbalanced no-op, val-driven feedback, non-destructive re-scoring, a frozen val subset.
2. One day buys: fail-fast config, real seeds, non-destructive re-scoring, predict-once slicing, a truthfulness pass.
3. The frozen test split is the most important fix and not a one-day fix. It may need engine work.
4. The first experiment after the fixes is ALOHA at 5–10k steps, testing under-training (≈0.11 epoch today).
5. Not now: MLflow, a server, Hydra, mocking the engine, more slices, automated feedback, a bigger model, each with its flip condition.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Prioritize by effect on conclusions | by ease or by module | wrong results matter most | P0s include a multi-day fix |
| Fix docs or make them true | leave docs as aspirations | credibility per minute | 28 small edits |
| Validity before experiments before reliability | infrastructure first | the bottleneck order (Module 15) | reliability gaps persist for a few weeks |
| Add an absolute gate beside the percentile | replace it | keep normalization-robust ranking | two thresholds to explain |
| Defer scale items to triggers | build now | C9, short runs | rework later, by design |

**Volunteer unprompted:** the README's three runs are pipeline validation until P0-1/2/7 land; P0-1 depends on the engine; continued training on an identical split is the robust result.

**Numbers and names:** 7 P0 · 26 P1 · 28 doc corrections · ~60 P2; shortlist about 8 h (1.5 + 1.5 + 1.5 + 2 + 1.5); P0-1 effort L (1–3 days); ALOHA target 5–10k steps ≈ 0.5–1 epoch.

**Sentences worth memorizing**
- "My P0 list is about one thing: making two numbers from this system comparable."
- "In one day I'd make config errors fail before the GPU starts, make seeds real, make re-scoring non-destructive, predict once, and make every doc sentence true."
- "Automating the feedback loop today would overfit the test set at machine speed."
- "Until the test split is frozen, the intervention results are pipeline validation, not findings."
