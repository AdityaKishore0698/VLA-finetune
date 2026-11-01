# Module 11 — Orchestration: `ExperimentRunner` and the data-fraction sweep

**Files covered:** `src/model_training/experiments/runner.py`, `src/model_training/experiments/data_fraction.py` (`experiments/__init__.py` was covered in Module 1)

**Where we are:** Modules 2–10 each described one stage. This module is the conductor that calls them in order and threads results between them: `ExperimentRunner.run`, plus a sweep driver that calls `run` once per data fraction. Most of the cross-module bugs found so far *happen* here. This is where a bad config meets an expensive GPU run, where seeds are or aren't applied, and where partial failure leaves its traces. So this module is mostly about **ordering and identity**.

**The one-line thesis:** The runner is a clear, linear *Transaction Script* that makes every stage leave a named artifact. But it validates late (config errors surface *after* training), it applies `seed` only to directory names (so a "multi-seed" study trains identical seeds under one shared `experiment_id`), and it isn't transactional. The sweep driver inherits all of that, and adds a curve whose points are scored on different test sets.

---

## 11.1 Prerequisites — the absolute basics

- **Orchestrator / pipeline driver.** Code whose job is ordering calls to other components and passing results along, with little logic of its own.
- **Transaction Script (Fowler).** One procedure that runs a whole business operation step by step. It's simple and readable, and it grows long. *Alternatives:* a pipeline of stage objects (Chain of Responsibility), or a DAG framework (Airflow, Prefect, Kedro) that adds caching, retries and resume.
- **Fail fast.** Validate everything you can *before* doing anything expensive, so errors surface in seconds, not after a GPU hour.
- **Transactional vs best-effort side effects.** A transaction commits all of its effects or none. This runner writes to four places (the run directory, the engine catalog, the JSONL index and engine failure records) with no coordination, so a failure midway leaves some writes done and others not.
- **Seed vs replicate.** A *replicate* is a run that differs only in its random seed. A study of seed variance needs the seed to reach every source of randomness (initialization, data order, dropout, sampling). A run whose "seed" changes only its folder name isn't a replicate.
- **Data-efficiency (learning) curve.** Test performance plotted against training-set size. To be valid, every point must be scored on the *same* test set, and ideally trained on *nested* subsets.

---

## 11.2 `ExperimentRunner.__init__` and module constants

```python
RDE_REPO = Path(os.environ.get("RDE_REPO_PATH", str(_default_repo_root() / "episodic-data-engine")))
MT_REPO = Path(os.environ.get("MODEL_TRAINING_REPO_PATH", str(Path(__file__).resolve().parents[3])))

class ExperimentRunner:
    def __init__(self, settings=None, catalog=None, runs_dir=None, index_dir=None):
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.paths.catalog_db)
        self.runs_dir = Path(runs_dir) if runs_dir else MT_REPO / "runs"
        self.index = ExperimentIndex(Path(index_dir) if index_dir else MT_REPO / "experiments")
```

- **Constructor injection with defaults.** Every collaborator can be passed in, and the tests do exactly that (`ExperimentRunner(settings=..., catalog=..., runs_dir=tmp_path/..., index_dir=...)`). That's *Dependency Injection* done simply, without a container, and it's the reason the runner is testable against a temporary catalog.
- **Construction has side effects.** `ExperimentIndex(...)` creates the `experiments/` directory (each `JsonlIndex.__init__` calls `mkdir`), so instantiating a runner touches disk.
- **The path constants are read at import time.** Setting `RDE_REPO_PATH` after `model_training.experiments` has been imported has no effect. Module 3 covered the provenance consequences of guessing these paths.

---

## 11.3 `run`, in order, with what can fail where

```python
def run(self, config: ExperimentConfig, seed: int | None = None) -> ExperimentRunResult:
    seed = seed if seed is not None else config.seeds[0]
    run_dir = make_run_dir(self.runs_dir, config.experiment_name, seed)
```

### Seed: the identity bug

`seed` is used in exactly two places: `make_run_dir(..., seed)`, which names the directory, and `summary["seed"]`. **It never reaches `config.training.seed`, the intervention seed or the split seed.** `Trainer` reads `tc.seed` (Module 6). So:

- `mtrain train --config X --seed 1` trains with `training.seed = 0`, like the default run, and writes to `seed_1/`.
- `docs/EXPERIMENTS.md` §7 says: "Running the same configuration across several seeds is a matter of calling `run` once per seed; `mtrain compare` can then report the mean and standard deviation". Following that literally yields **replicates that share a training seed**. They'd differ only through GPU nondeterminism and flow-matching sampling, so the "seed variance" you measured would **underestimate the real variance**. That's arguably worse than having no replicates, because it looks like evidence.
- **The replicates also share one `experiment_id`.** The config is loaded once, and `experiment_id` is stamped at load (Module 2). Calling `run(cfg, seed=s)` for several `s` reuses it. Downstream, `model_id = f"{tag}-{experiment_id}"` (Module 9) is identical across seeds, so `ModelRegistry.get(model_id)` returns only the *last* seed's model, and the index's `experiments.jsonl`, `datasets.jsonl` and `metrics.jsonl` rows can't be told apart by `experiment_id`.
- `config.seeds[1:]` is never read by anything (Module 2).

**The fix:** `cfg = config.model_copy(update={"training": config.training.model_copy(update={"seed": seed})})`, re-stamp the identity, and loop over `config.seeds` in the CLI. Whether the split seed should vary across replicates is a design choice. Usually it shouldn't, so the test set stays fixed.

### Stage 0: identity and environment, before anything can fail usefully

```python
environment = collect_environment()
git_info = collect_git_info(RDE_REPO, MT_REPO)
run_dir.write_json("environment.json", environment)
run_dir.write_json("git.json", git_info)
(run_dir.path / "config.resolved.yaml").write_text(yaml.safe_dump(config.resolved_dict(), sort_keys=False))

limitations: list[str] = []
if environment.get("gpu") is None:
    limitations.append("no GPU detected in this environment at run time -- training ran on CPU if it ran at all")
if environment.get("torch") is None:
    limitations.append("torch not installed in this environment -- this run could not perform real fine-tuning")
```

- **Writing provenance first is a defensible choice.** Even a crashed run records what it ran on. The cost (Module 3) is that the directory is now non-empty, so a same-day retry is blocked.
- **`sort_keys=False`** keeps the resolved YAML in schema field order, so it reads like the source config. That's a small, deliberate readability touch.
- **The `limitations` list starts here** and is threaded through to the report and model card. It's the right mechanism, but underused: it never receives the rebalanced caveat (Module 4), the `best → final` fallback (Module 6) or the shared-seed issue above.
- **CPU training isn't blocked.** With no GPU, a SmolVLA run proceeds on CPU (the limitation text admits this). `docs/IMPLEMENTATION_PLAN.md` records that CPU-only fine-tuning was "impractical". A `--require-gpu` guard would turn hours of wasted CPU time into an instant error.

### Stage 1: intervention, the first irreversible write

```python
intervention_result = build_intervention_dataset(
    config.dataset.dataset_id, config.dataset.version, config.intervention,
    config.splits, settings=self.settings, catalog=self.catalog,
)
run_dir.write_json("dataset.json", {...})
self.index.record_dataset(experiment_id=..., training_version=..., kind=..., episode_count=len(...), recorded_at=...)
```

- **This creates engine versions** (Module 4): `clean/vK` and `training/vM`. From here on, a failure orphans catalog versions.
- **`record_dataset` is written to the index *before* training**, so a later crash leaves an index row for a run that never finishes (Module 3 gap 10).
- **`episode_count = len(episode_ids)`** is the whole selected pool (train + val + test), not the training count. The column name invites misreading.

### Stage 2: open the view, check, profile

```python
training_version = intervention_result.training_version
engine = DataEngine(self.settings)
view = engine.get_dataset(config.dataset.dataset_id, training_version)
try:
    split_report = verify_split_integrity(view)  # raises on leakage
    run_dir.write_json("split_integrity.json", split_report.to_dict())
    profile = build_dataset_profile(config.dataset.dataset_id, training_version, settings=..., catalog=...)
    run_dir.write_json("dataset_profile.json", profile.to_dict())
    figures.plot_task_distribution(...); figures.plot_episode_length_distribution(...)
```

- **`try/finally: view.close()`** guarantees the view is released on every exit path, including exceptions. Without it, a crashed run would leak file handles (and presumably a DuckDB connection) for the life of the process. That matters in a sweep (11.5), which runs many experiments in one process. It's correct, and it's the most important resource-management line in the file.
- The profile opens its *own* second view internally (Module 5). Two views of one version are open at once, which is harmless and slightly wasteful.

### Stage 3: training

```python
train_ids = view.episode_ids(split="train")
episode_weights = {
    eid: float((view.episode(eid).attributes or {}).get("sampling_weight", 1.0))
    for eid in train_ids
}
episode_weights = episode_weights if any(w != 1.0 for w in episode_weights.values()) else None
```

- **Weights are read back from the *built* version**, not from `intervention_result.episode_weights`. That's a good choice: the trainer uses what the engine actually stored (the `sampling_weight` attribute), so a disagreement between what was requested and what was built can't silently change training. It's the same "trust the materialized artifact" stance as Module 5's split check. Here it's actually load-bearing.
- **Only train episodes' weights are passed**, which resolves Module 4's "weights recorded for val and test too" at the point that matters.
- **All-ones collapses to `None`**, so unweighted runs take the engine's unweighted sampling path (inferred).

```python
training_outcome = None
selected_checkpoint = None
if config.training.model == "smolvla":
    from ..training.trainer import Trainer
    trainer = Trainer(run_dir, config)
    training_outcome = trainer.train(view, view, episode_ids=train_ids, episode_weights=episode_weights)
    run_dir.write_json("training_outcome.json", training_outcome.to_dict())
    figures.plot_training_curves([json.loads(l) for l in (run_dir.path / "metrics" / "train.jsonl").read_text().splitlines()]
                                 if (run_dir.path / "metrics" / "train.jsonl").exists() else [], ...)
    ...
    selected_checkpoint = training_outcome.best_checkpoint
else:
    limitations.append(f"training.model={config.training.model!r} has no gradient-based fit; "
                       "this run only scored baselines, no fine-tuning occurred")
```

- **`trainer.train(view, view, ...)`.** The same view is passed as `train_view` and `val_view`. The trainer's two-view signature anticipates separate datasets for validation (for example, a held-out scene), and the runner never uses that.
- **The lazy `Trainer` import** is the import-light idea. Module 1 showed it saves nothing on the engine side, because `deployment.artifact` already imports `SmolVLAPolicy` at module top.
- **The `else` branch is Module 2's typo trap.** Any `model` string other than exactly `"smolvla"` lands here, and the run completes with exit 0.
- **Re-reading `train.jsonl` from disk for plotting** (instead of using `fit_result["history"]`, which the API doc says exists) is consistent with "figures from saved artifacts" (Module 10), and it costs one file read.

### Stage 4: baselines, then the model under test

```python
baseline_results = evaluate_baselines(view, config.evaluation.baseline_models, split=config.evaluation.split,
                                      batch_size=config.evaluation.batch_size, max_frames=config.evaluation.max_frames)
...
if selected_checkpoint:
    from rde.model_adapters.models import SmolVLAPolicy
    policy = SmolVLAPolicy.from_checkpoint(selected_checkpoint)
    adapter = VisionLanguageAdapter(view.contract)
    strat_results = stratified_evaluate(view, policy, adapter, base_split=config.evaluation.split, ...,
                                        slices=config.evaluation.slices)
    test_overall = strat_results.pop("overall")
```

**Validation happens after the expensive part.** Two config errors are detected only here, *after* fine-tuning has finished:
- an unknown baseline name raises `KeyError` inside `evaluate_baselines` (Module 7);
- an unknown slice name raises `ValueError` inside `stratified_evaluate` (Module 7).

A typo like `baseline_models: [nearest_neighbour]` (British spelling) costs a full training run (5–10 minutes on the README's hardware, or hours at scale) and then fails, leaving a run directory with checkpoints and no evaluation, and no `summary.json`. Both checks are pure name lookups that could run in stage 0 (or in the Pydantic schema with `Literal`). That's the clearest *fail-fast* violation in the codebase.

- **`config.evaluation.split` is used for the "test" evaluation.** It defaults to `"test"`. Setting it to `"val"` would make the headline "test metrics", the feedback artifact and the model card all describe validation, with every label still saying "test". It's a sharp edge, and arguably the hook for Module 8's fix: feedback should run on val *in addition to* test.
- **`from_checkpoint(selected_checkpoint)` is called without `device=`.** `training.device` is honoured for training but not for evaluation, which auto-selects. Minor.
- **The evaluation policy is never released.** After stratified evaluation, `policy` stays referenced until `run` returns. In a sweep (11.5), memory is reclaimed only when the next `run` frame replaces it. That's fine on one GPU, because the next run's trainer allocates after this frame is gone. It's worth an explicit `del` for symmetry with the trainer's careful release.

### Stage 5: artifacts, feedback, model card

```python
predictions_dir = run_dir.sub("predictions")
with (predictions_dir / "test_outcomes.jsonl").open("w") as f:
    for o in test_overall.outcomes:
        f.write(json.dumps({"run_id": str(run_dir.path), "experiment_id": ..., "dataset_id": ...,
                            "dataset_version": training_version, "frame_id": o.frame_id,
                            "episode_id": o.episode_id, "error": o.error, "confidence": o.confidence,
                            "per_dim_error": o.per_dim_error}, default=str) + "\n")
```

- **Per-frame outcomes with full provenance.** This file is quietly the most valuable artifact in the run. It's the only place the evaluated test episodes are recorded (Module 4), the input for the bootstrap Module 7 recommends, and the base for per-joint analysis. `FAILURE_ANALYSIS.md` §4 is right to call it the traceability anchor.
- **`run_id` is the absolute path of the run directory**, so it breaks if the runs directory is moved or copied to another machine. A relative path, or the `experiment_id` plus seed, would be portable.
- **Predicted and target action vectors aren't stored**, only the errors. Module 8 showed the stated justification ("deterministic, reproducible on demand") is false for SmolVLA.

```python
model_artifact = build_model_artifact(checkpoint_path=selected_checkpoint, contract=view.contract, ...,
                                      metrics=test_overall.metrics)
self.index.record_model(**model_artifact.to_dict())
feedback_artifact = build_feedback_artifact(config.experiment_id, model_artifact.model_id, ..., test_overall, ...)
(run_dir.sub("failures") / "feedback_artifact.json").write_text(json.dumps(feedback_artifact, ...))
run_dir.write_json("model_artifact.json", model_artifact.to_dict())
(run_dir.sub("reports") / "model_card.md").write_text(render_model_card(...))
```

- **`record_model` goes into the index *before* feedback runs.** A failure in feedback (an engine error inside `FeedbackLoop.detect`) leaves a registered model in `models.jsonl` with no `model_artifact.json` in its run directory, because that's written *after* feedback. The index then points at a run that has no artifact file. Writing the local artifact first and indexing last would keep the run directory authoritative.
- **Feedback runs on `test_overall`.** That's Module 8's central methodological issue, and this is the line to change.

### Stage 6: closing writes

```python
if test_overall is not None:
    run_dir.append_jsonl("metrics/test.jsonl", {"experiment_id": ..., **test_overall.metrics})
run_dir.append_jsonl("metrics/rollout.jsonl", {"experiment_id": ..., "status": "unavailable",
                                               "reason": "no simulator or physical robot connected in this environment"})
report_md = render_experiment_report(report_ctx)
(run_dir.sub("reports") / "experiment_report.md").write_text(report_md)
summary = {..., "test_mse": ..., "baseline_mean_action_mse": baseline_json.get("mean_action", {}).get("mse"),
           "steps_completed": ..., "wall_clock_s": ...}
run_dir.write_json("summary.json", summary)
self.index.record_experiment(run_dir=..., created_at=..., **summary)
if test_overall is not None:
    self.index.record_metric(experiment_id=..., run_dir=..., split=config.evaluation.split, **test_overall.metrics)
```

- **The rollout "unavailable" record is written in every run.** That's the refusal-to-fabricate principle made into data, so a downstream tool reading `rollout.jsonl` can't mistake absence for zero.
- **`summary.json` records the mean-action baseline but not nearest-neighbour**, which is the baseline that actually matters (Module 7's tie). So the index's `experiments.jsonl`, built from `summary`, can't show the project's key result without opening each run's `baselines.json`.
- **`wall_clock_s` is training only.** Total run time (including about 3× test inference from the slices, Module 7) isn't recorded anywhere.
- **`summary.json` is written last** and is the implicit completion marker (Module 3).
- **There's no top-level `try/except`.** Any exception after stage 1 propagates out of `run` with partial state in four stores. The trainer's `crash.json` covers only `fit` (Module 6).

### What `run` writes, and where, in order

| # | Write | Store | Irreversible? |
|---|---|---|---|
| 1 | run directory, env, git, config | run directory | blocks same-day retry |
| 2 | `clean/vK`, `training/vM` | **engine catalog** | **yes** (immutable versions) |
| 3 | `datasets.jsonl` row | index | append-only |
| 4 | checkpoints, metrics, outcome | run directory | — |
| 5 | `models.jsonl` row | index | append-only |
| 6 | failure records (`persist=True`) | **engine** | yes |
| 7 | artifacts, report, `summary.json` | run directory | — |
| 8 | `experiments.jsonl`, `metrics.jsonl` rows | index | append-only |

There's no coordinator across the three stores, and no status. Module 3's state-machine fix (`status: running → completed/failed`, written first and atomically replaced) is the minimal remedy.

---

## 11.4 Why a Transaction Script, and when it stops being right

**Decision: one linear `run` method of about 250 lines.**
- *Why:* every stage's inputs are visible in one place, the order is obvious, and the README's Figure 1 ("seven ExperimentRunner stages") maps directly onto the code. For a research harness with one user, that readability is worth a lot.
- *Alternative:* stage objects with a common interface (`Stage.run(ctx) -> ctx`), or a DAG tool (Kedro, Prefect, Airflow) with per-stage caching.
- *Cost:* no stage can be re-run alone. If evaluation fails, you can't resume from the trained checkpoint through `run`; you'd use `mtrain evaluate` (Module 12), which re-implements a *subset* of stage 4 with different settings. There's no caching (a re-run rebuilds versions and retrains), and the method mixes orchestration with artifact formatting (the inline `json.dumps` for predictions and evaluation files).
- *When it flips:* once you need **resume-from-stage** (multi-hour runs, preemptible GPUs) or **sweeps of more than a handful of runs**. Both want stage-level checkpointing and caching, which a DAG framework or a simple "skip stage if its output artifact exists and is complete" rule would give.

---

## 11.5 `run_data_fraction_sweep`

```python
"""... Fractions are not executed unless explicitly requested -- calling this function *does* run them
(the framework makes them possible; they do not run by default); the caller decides which fractions and how many."""

def run_data_fraction_sweep(base_config, fractions, runner=None) -> list[dict[str, Any]]:
    """Run ``base_config`` once per fraction, varying only the data quantity.

    Every other knob (model, hyperparameters, splits, seed) is held fixed, so
    the resulting curve isolates the quantity effect -- ..."""
    runner = runner or ExperimentRunner()
    results = []
    for frac in fractions:
        data = base_config.model_dump()
        data["experiment_name"] = f"{base_config.experiment_name}_frac{int(round(frac * 100))}"
        data["intervention"] = {"kind": "fraction", "fraction": frac, "seed": base_config.intervention.seed}
        data["experiment_id"] = None
        data["created_at"] = None
        cfg = ExperimentConfig.model_validate(data)
        outcome = runner.run(cfg)
        results.append({"fraction": frac, **outcome.summary})
    return results
```

- **The module docstring contradicts itself** ("not executed unless explicitly requested -- calling this function *does* run them ... they do not run by default"). What it means is that nothing calls this automatically. The prose reads like two edits merged. `docs/API_REFERENCE.md` Table 8 states the same idea cleanly.
- **Re-stamping identity** (`experiment_id = None`, `created_at = None`) is deliberate and correct: each fraction gets a fresh ID. This is the pattern Module 2 noted as relying on the "not user-editable" guard.
- **"Isolates the quantity effect" is false, because of Module 4.** Each fraction builds its own `training/vN` *from the filtered pool*, so **every point on the curve is scored on a different test set**, of a different size (at 25% of 48 episodes the test split is about 1 episode). The curve conflates "less training data" with "different, smaller test set". The fix is the same: split once, and subsample only the train split.
- **Non-nested subsets** (Module 4 gap 6): every fraction draws with the same seed independently, so the 25% subset isn't inside the 50% subset.
- **`fraction = 0.0` raises** (Module 4 gap 2), and **there's no `try` per fraction**. One failing fraction aborts the sweep, and `results` for the completed fractions is **lost from the return value**. Only the run directories survive. A sweep that took hours returns nothing. *Fix:* catch per fraction, record the failure, continue, and persist `results` incrementally (for example, to `experiments/sweeps/<name>.jsonl`).
- **Name collisions:** `int(round(frac * 100))` maps 0.125 and 0.12 both to `_frac12`, and Python rounds 0.125 × 100 = 12.5 to 12. The second fraction's `make_run_dir` raises `FileExistsError` the same day. Use `f"{frac:g}"` or permille.
- **The base intervention is discarded.** A sweep over a `rebalanced` or `imbalanced` base config silently becomes a sweep over plain random fractions. That's only acceptable if documented.
- **A single seed per fraction, run sequentially.** A curve from one seed per point at these test sizes is mostly noise (Module 7).
- **The output isn't wired to its plot.** `plot_data_fraction_curve(rows, metric, ...)` exists (Module 10). The rows here carry `test_mse` (from `summary`), so `metric="test_mse"` would work, but nothing calls it.
- **Untested and uncalled.** No test or CLI command references `run_data_fraction_sweep` (grep). Module 1's table has been updated accordingly. Its first real use would hit the 0.0 crash and the moving test set.

---

## 11.6 Honest gaps to own before an interviewer finds them

1. **`seed` only names the directory.** Replicates share `training.seed` and `experiment_id`. *Consequence:* a multi-seed study following `EXPERIMENTS.md` §7 produces fake replicates that underestimate variance, plus colliding `model_id`s (the registry returns only the last seed).
2. **Config errors are detected after training** (unknown baseline names and slice names raise in stage 4). *Consequence:* a typo costs a full GPU run and leaves an incomplete run directory.
3. **Not transactional across four stores** (run directory, engine catalog, index, engine failure records), with no status marker. *Consequence:* partial failures leave orphan versions, index rows without runs, and runs without artifacts.
4. **`record_model` is indexed before feedback and before `model_artifact.json` is written.** *Consequence:* a feedback failure leaves an indexed model whose run lacks the artifact file.
5. **`evaluation.split` silently relabels.** Setting it to `"val"` makes "test" reports describe validation. *Consequence:* a sharp edge on the project's central val/test discipline.
6. **No GPU guard.** SmolVLA training proceeds on CPU with only a limitation note. *Consequence:* hours lost to an impractical configuration.
7. **`summary.json` and `experiments.jsonl` omit the nearest-neighbour baseline**, and total run time isn't recorded. *Consequence:* the key comparison isn't queryable from the index.
8. **`run_id` in predictions is an absolute path.** *Consequence:* outcomes don't survive moving the runs directory.
9. **No stage-level resume or caching** (Transaction Script). *Consequence:* any failure after training means retraining, or using `mtrain evaluate`'s divergent re-implementation.
10. **The sweep scores each fraction on a different test set and uses non-nested subsets.** *Consequence:* the data-efficiency curve doesn't isolate quantity, contrary to its docstring.
11. **The sweep aborts on the first failure (0.0 always fails) and loses completed results.** *Consequence:* hours of sweep produce an exception and no rows.
12. **Sweep name collisions** (`_frac12` for 0.12 and 0.125) and **a discarded base intervention**. *Consequence:* a `FileExistsError` mid-sweep, and silently changed semantics.
13. **The sweep is untested and has no CLI entry point.** *Consequence:* a headline capability that has never run.
14. **`training.device` isn't used for evaluation, and the evaluation policy isn't released explicitly.** *Consequence:* minor inconsistency.
15. **The same view serves as train and val; the trainer's two-view seam is unused.** *Consequence:* none today. It's the natural place for a held-out-scene val set later.

---

## The shape to carry into Module 12

```
 ExperimentRunner.run(config, seed)                     seed → dir name + summary ONLY
  0  make_run_dir · environment.json · git.json · config.resolved.yaml · limitations[]
  1  build_intervention_dataset ──▶ ENGINE: clean/vK, training/vM      (irreversible)
     dataset.json · index.datasets
  2  view = get_dataset(training/vM)  ┐ try
     verify_split_integrity · profile │
  3  weights ← view attrs (train only)│
     Trainer.train(view, view) ───────│──▶ checkpoints, metrics, training_outcome.json
  4  evaluate_baselines (KeyError on bad name ← AFTER training)
     SmolVLA.from_checkpoint(best) → stratified_evaluate (ValueError on bad slice ← AFTER training)
  5  predictions/test_outcomes.jsonl · index.models (before feedback!)
     FeedbackLoop on TEST ──▶ ENGINE failure records · feedback_artifact.json · model_artifact.json · model_card.md
  6  metrics/test.jsonl · rollout.jsonl("unavailable") · experiment_report.md · summary.json (last)
     index.experiments · index.metrics          ┘ finally view.close()

 run_data_fraction_sweep(base, fractions): for f: cfg(kind=fraction, new id) → run()
     different test set per f · not nested · aborts on first failure · results lost · untested
```

---

## Drill — answer from memory, files closed

1. List, in order, every place `ExperimentRunner.run` writes, and say which writes are irreversible.
2. *Why didn't you just* use Airflow, Prefect or Kedro for the pipeline?
3. *What breaks in production:* you run `mtrain train --config baseline.yaml --seed 0`, then `--seed 1`, then `--seed 2`, and report mean ± std. What's wrong?
4. A teammate sets `baseline_models: [nearest_neighbour]`. Walk through what happens.
5. *Interviewer framing:* "How would you produce a trustworthy data-efficiency curve with this system?"

---

# Appendix A — Model answers to the drill

**A1. The write order.**
"First the run directory itself, with environment, git info and the fully resolved config. Then the intervention, which creates new dataset versions in the engine's catalog. That's irreversible, since versions are immutable, and a dataset row goes into the index. Then the training artifacts in the run directory: checkpoints, train and validation metrics, the training outcome. Then evaluation files and the per-frame test outcomes, and a model row in the index. Then the feedback step, which persists failure records into the engine, so that's a second irreversible write. Then the feedback artifact, model artifact, model card, report and `summary.json`, and finally the experiment and metric rows in the index. Three stores, with no coordination between them, so a failure halfway leaves some stores updated and others not. The minimal fix is a status file written first and flipped atomically at the end."

*Why that answer lands:* it's complete, it's in the right order, and it identifies the two engine writes as the irreversible ones. That's a sign you've thought about failure, not just the happy path.

**A2. Why not a DAG framework?**
"For one researcher running a handful of experiments, a single linear method is the most readable form. The stages, their order and their artifacts are all on one screen, and they map one-to-one onto the architecture diagram. A DAG tool would add a scheduler, a metadata store and a deployment surface, mostly to buy caching and retries I didn't need at five-minute runs. Where that stops being true is resume-from-stage and large sweeps. Today, if evaluation fails after training, I can't resume through the runner, I'd retrain. So the first step wouldn't be Airflow. It'd be making each stage skip itself if its output artifact already exists and is complete, which gives resume with no new infrastructure. I'd adopt Prefect or Kedro once runs got long or numerous enough that caching mattered."

*Why that answer lands:* it argues from scale, admits the concrete gap (no resume), and proposes an intermediate step before reaching for a framework. That's proportionate engineering.

**A3. Three seeds, mean ± std.**
"Those aren't replicates. The `--seed` value only names the run directory and goes into the summary; it never reaches `training.seed`, which stays at the config's zero. All three runs train with the same seed, so any differences come from GPU nondeterminism and the stochastic action sampler, and the standard deviation would understate real seed variance, which is worse than reporting nothing. They also share one experiment ID, because the config was loaded once, so their model IDs collide, and the registry would return only the last one. The fix is for the runner to copy the config with `training.seed` set, re-stamp the identity, and for the CLI to loop over the config's `seeds` list, keeping the split seed fixed so all replicates share a test set."

*Why that answer lands:* it explains why the numbers are *misleadingly* small, not just wrong, catches the identity collision, and specifies the fix including the split-seed subtlety.

**A4. `nearest_neighbour`.**
"Nothing checks baseline names until the evaluation stage. So the runner builds the dataset versions, runs the integrity check and profile, fine-tunes SmolVLA for the full budget, writes the checkpoints and training outcome, and only then calls `evaluate_baselines`, which raises `KeyError` because the registry name is `nearest_neighbor`. The run ends with checkpoints, no evaluation, no `summary.json`, orphaned dataset versions in the catalog, and a run directory that blocks a same-day retry. The same applies to an unknown slice name. Both are pure lookups that belong in config validation, as `Literal` types or a stage-zero check, so the error appears in under a second instead of after training."

*Why that answer lands:* it traces the exact sequence and its debris, then classifies it correctly as a fail-fast violation with a one-line fix.

**A5. "How would you produce a trustworthy data-efficiency curve with this system?"**
"Not with the sweep function as it stands. It filters episodes and then re-splits, so each fraction is scored on a different and shrinking test set: at 25% of 48 episodes, the test split is about one episode. The subsets aren't nested, it fails outright at 0%, and it loses completed results if any fraction fails. I'd change four things. Split the full dataset once and freeze val and test. Subsample only the train split, as prefixes of one seeded permutation, so the subsets are nested. Make the zero point 'evaluate the pretrained base with no fine-tuning'. And run at least three training seeds per point, with real seed propagation, reporting a bootstrap interval over test episodes. Then each point differs only in training data quantity, which is what the curve claims to show."

*Why that answer lands:* it starts with a frank "not as it stands" and the precise reasons, then gives a protocol that addresses every confound. That's what an interviewer probing experimental design wants to hear.

---

# Module 11 Summary — the one page to revise

**What this module is.** A linear Transaction Script that calls every stage in order (intervention, integrity, profile, train, baselines, stratified evaluation, artifact, feedback, reports, summary, index) with injectable collaborators and a `finally`-closed view. Plus a sweep driver that loops it over data fractions. The ordering and identity handling are where the project's cross-cutting bugs concentrate.

**Five claims to make cold**
1. One method, seven-plus stages, every stage leaves a named artifact. Collaborators are constructor-injected for tests.
2. `seed` names the directory only. `training.seed` is unchanged and replicates share `experiment_id`, so multi-seed studies are fake.
3. Baseline and slice names are validated after training. That's the clearest fail-fast violation.
4. Writes span the run directory, the engine catalog (irreversible), the index and engine failure records, with no transaction and no status.
5. The sweep's points use different test sets and non-nested subsets, abort on the first failure, and have never been run.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Transaction Script `run` | stage objects / DAG tool | readable, maps onto Figure 1 | no resume or caching; mixed concerns |
| Constructor DI with defaults | globals / container | testable against a temporary catalog | construction touches disk |
| Provenance written first | write at end | crashed runs keep provenance | blocks same-day retry |
| Weights read back from the built version | from the intervention request | train on what the engine stored | none |
| `try/finally` view close | rely on GC | no handle leaks across a sweep | none |
| Sweep = loop over `run` | dedicated sweep pipeline | reuse everything | inherits moving test set; no fault isolation |

**Volunteer unprompted:** fake seeds and the ID collision; late name validation; the four-store non-transactional writes; `record_model` before the artifact file; the sweep's test-set confound and lost results.

**Numbers and names:** stage order 0–6 above; `summary.json` keys `experiment_id, experiment_name, dataset_id, training_version, intervention, tags, seed, model, test_mse, baseline_mean_action_mse, steps_completed, wall_clock_s`; sweep names `{name}_frac{round(f×100)}`; env vars `RDE_REPO_PATH`, `MODEL_TRAINING_REPO_PATH` (read at import).

**Sentences worth memorizing**
- "My `--seed` flag renamed a folder; it didn't change a seed. Replicates that share a seed understate variance, which is worse than no replicates."
- "Every config error my runner can detect, it should detect before it touches the GPU. Two of them it detects after."
- "The runner writes to three stores without a transaction; the minimum fix is a status file written first and flipped atomically last."
- "A learning curve is only a learning curve if every point is scored on the same test set."
