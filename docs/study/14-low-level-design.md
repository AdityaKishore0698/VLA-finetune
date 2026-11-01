# Module 14 — Low-Level Design

**Files covered:** all of `src/model_training/` (cross-cutting). No new files; every claim here points back to a code module.

**Where we are:** Modules 1–13 walked the code one stage at a time. This module looks at the *same* code as a design: how it's layered, which patterns it actually uses (with exact locations), where SOLID holds or strains, which boundaries are typed and which are bare dicts, how a run moves through its states, how errors travel, and what's safe to share or repeat. It's the material for the "draw me the class and sequence diagrams" part of an interview.

**The one-line thesis:** The package has a clean, acyclic, strictly downward dependency graph with a single composition root. But almost every boundary *between* stages is an **untyped `dict[str, Any]`**, the only real state machine is implicit in which files exist, and the error model is "raise built-in exceptions and let them escape". So the structure is sound, and the contracts are weak.

---

## 14.1 Prerequisites — the absolute basics

- **Layering / dependency rule.** Code is arranged in layers, and each layer may import only layers *below* it. Cycles are forbidden. That's what lets you change a high layer without touching low ones.
- **Composition root.** The one place where concrete collaborators are created and wired together. Everything else receives its collaborators.
- **Design pattern.** A named, reusable solution shape (Gang of Four, Fowler's *Patterns of Enterprise Application Architecture*). Naming one is useful only if you can point to where it is and say what it buys.
- **SOLID.**
  - **S**ingle responsibility: one reason to change.
  - **O**pen/closed: extend without editing.
  - **L**iskov substitution: subtypes are usable wherever their base type is.
  - **I**nterface segregation: no client depends on methods it doesn't use.
  - **D**ependency inversion: depend on abstractions, not concretions.
- **Typed contract.** A boundary whose data shape is declared and checked (a Pydantic model, a dataclass with type hints checked by mypy, a `TypedDict`). `dict[str, Any]` is an *untyped* contract: the shape lives only in the code that reads it.
- **Sequence diagram.** The order of calls between components for one scenario.
- **State machine.** The set of states an entity can be in and the legal transitions between them.
- **Idempotent operation.** Doing it twice has the same effect as doing it once.
- **Thread safety.** Correct behaviour when called from several threads at once. **Process safety** is the same across processes, which matters for files and catalogs.

---

## 14.2 Layering and the dependency rule

Every internal import, gathered with grep:

```
                         ┌─────────┐
            Layer 4      │  cli    │  (lazy imports; composition for the human)
                         └────┬────┘
                         ┌────▼─────────────┐
            Layer 3      │  experiments     │  runner = COMPOSITION ROOT; data_fraction → runner
                         └─┬──┬──┬──┬──┬──┬─┘
          ┌────────────────┘  │  │  │  │  └──────────────────┐
     ┌────▼────┐   ┌──────────▼┐ │ ┌▼──────────┐   ┌────────▼───┐
L2   │training │   │   data    │ │ │evaluation │   │ deployment │   feedback   reports
     │→config, │   │ →config   │ │ │ (rde only)│   │ →tracking  │   (rde only) (no internal
     │ tracking│   └───────────┘ │ └───────────┘   │   .index   │               imports)
     └────┬────┘                 │                 └─────┬──────┘
     ┌────▼─────────────┐        │                       │
L1   │ config  tracking │◄───────┴───────────────────────┘
     └──────────────────┘   (config: pydantic+yaml only; tracking: stdlib only)
                 all layers ─────────────▶ rde (external engine)
```

**Verified properties:**
- **There are no cycles.** No module imports `experiments` or `cli` except the CLI itself.
- **The leaves are leaves.** `config` imports only `pydantic` and `yaml`; `tracking` imports only the standard library (plus optional `torch`, `lerobot` and `psutil` for observation).
- **`reports` imports nothing internal.** It takes plain dicts. That's maximal decoupling, which is also why its inputs are untyped (14.5).
- **`evaluation` and `feedback` import only `rde`.** They're pure adapters over engine services.
- **`deployment.registry` depends on `tracking.index`**, the one lateral, cross-feature reuse, and it's a reasonable one (shared JSONL mechanism).

**Where the rule bends:**
- **The CLI reaches below the façade.** It imports `model_training.config.schema.DatasetIntervention` and `model_training.data.interventions.resolve_episode_subset` / `compute_length_bucket` directly, and reimplements both `_target_episodes` and evaluation (Module 12). The CLI depends on internals instead of a service API.
- **Everything depends directly on `rde`, a concrete external package**, with no abstraction layer of our own. `rde` is effectively *layer 0 for every layer*. That's a deliberate choice (Module 1: the model layer "never owns the data"), with the cost that there's no seam for substituting the engine. Tests use the real engine (Module 13) partly because there's nothing to mock *against*.
- **Import-time side effects cross layers.** `reports.figures` sets the matplotlib backend globally (Module 10), and `deployment.artifact` imports the engine's model module eagerly (Module 1). Layers are separated by imports, but not by *effects*.

**Composition root:** `ExperimentRunner.__init__` / `run` builds `Catalog`, `DataEngine`, `ExperimentIndex`, `Trainer`, `Evaluator` (via helpers), `SmolVLAPolicy` and `VisionLanguageAdapter`. The CLI builds `ExperimentRunner()` with defaults. That's a good shape: one place to change wiring.

---

## 14.3 Catalogue of patterns actually present

| Pattern | Exact location | What it buys *here* | Alternative passed over |
|---|---|---|---|
| **Façade** (package level) | each `*/__init__.py` with `__all__` | stable import paths; file layout can change | deep imports everywhere |
| **Façade** (service level) | `ExperimentIndex` over 4 `JsonlIndex`es (`tracking/index.py`) | one object to record any run fact | four loose files |
| **Transaction Script** | `ExperimentRunner.run` (`experiments/runner.py`) | whole lifecycle readable in one method, matching README Fig. 1 | stage objects / DAG (Module 11) |
| **Composition Root + constructor DI with defaults** | `ExperimentRunner.__init__` | tests inject a temporary catalog and directories | globals or a DI container |
| **Template Method via callback (Hollywood principle / inversion of control)** | `Trainer.train` passes `on_step` into engine `fit` (`training/trainer.py`) | run policy injected into an engine-owned loop | own training loop; Lightning callbacks |
| **Closure-captured state** | `on_step` with `nonlocal peak_rss, last_step` and mutable `best` dict | per-run state without a class | a `TrainingState` object |
| **Adapter** | `evaluation/offline.py`, `feedback/artifact.py` wrap `Evaluator` / `FeedbackLoop` | engine results shaped into project artifacts | reimplement scoring or detection |
| **Value Object** | `InterventionResult`, `SplitIntegrityReport`, `DatasetProfile`, `TrainingOutcome`, `ModelArtifact`, `LatencyBreakdown` (all dataclasses) | named, typed results between stages | tuples / dicts |
| **Specification (validation rules on a model)** | `@model_validator` on `DatasetIntervention`, `SplitConfig`, `ExperimentConfig` | methodology rules as load-time errors | runtime checks scattered across stages |
| **Layered configuration** | `_deep_merge(base, override)` (`config/loader.py`) | shared defaults plus per-experiment deltas | Hydra / OmegaConf |
| **Append-only log / event-sourcing-lite** | `JsonlIndex.append`, `ModelRegistry.get` takes the last match | history preserved; latest wins | mutable DB rows |
| **Repository (read side only)** | `ModelRegistry.get/list/for_experiment` | query models by identity | raw file scans by callers |
| **Guard clause / fail loudly** | `make_run_dir`, `verify_split_integrity`, schema validators, `health_check` | refuse bad states early | best-effort continue |
| **Null Object / explicit "unavailable"** | `{"status": "unavailable", "reason": ...}` in profile, rollout, environment, figures that return `None` | absence is data, never fabricated | omit the field / fake a value |
| **Registry (lookup by name)** | engine `MODEL_REGISTRY` used by `evaluate_baselines` | baselines named in config | `if name == ...` chains |
| **Strategy (implicit)** | `selection_mode: min/max`, `split_mode: quantile/independent`, `intervention.kind` | behaviour chosen by config | — but implemented as `if/elif`, not polymorphism |
| **Builder-ish function** | `build_model_artifact`, `build_feedback_artifact`, `build_dataset_profile` | assemble a large record from many sources | constructors with many arguments |

**Patterns named in docs or comments that aren't really there:** a "registry" in the MLflow sense (no promotion, Module 9), a "flywheel" as a decision loop (Module 8), and a "matrix" (`build_model_dataset_matrix`, Module 10).

---

## 14.4 SOLID, and where each principle is strained

| Principle | Where it holds | Where it's strained |
|---|---|---|
| **S** | `split_integrity.py` (one check), `JsonlIndex`, `_deep_merge`, `figures._save` | `ExperimentRunner.run` (orchestration plus JSON formatting plus artifact naming plus limitation text, about 250 lines); `cli.evaluate` (re-scoring plus persistence plus formatting); `DatasetProfile` builder (stats, coverage, QC, splits, notes) |
| **O** | new baselines need only an engine `MODEL_REGISTRY` entry; new figures are new functions | new intervention kinds need edits in **three** places (schema `Literal` plus validator, `resolve_episode_subset`, `build_intervention_dataset` branch) plus the CLI; new slice types need an `elif` in `stratified_evaluate`; a new model family needs the runner's `if model == "smolvla"` changed |
| **L** | baselines and SmolVLA are substitutable behind the engine's `Model` / `ModelInputAdapter` for evaluation (the same `Evaluator.evaluate` call) | `Trainer` only accepts SmolVLA (it raises otherwise), so "model" isn't substitutable for *training*. `train_view` and `val_view` are typed as `DatasetView` but used as the same object |
| **I** | `reports` take only the dict fields they render | `ModelArtifact` carries 16 fields, and the registry and model card each use a subset. `RunDir` exposes `write_json` / `append_jsonl` / `sub`, but half the writers bypass it (Module 3) |
| **D** | runner collaborators injectable; `settings` and `catalog` passed down rather than fetched where possible | every module depends on **concrete** `rde` classes (`Catalog`, `DataEngine`, `SmolVLAPolicy`, `Evaluator`); there's no project-owned protocol, and `SmolVLAPolicy` is constructed directly inside `Trainer` and the runner, so it can't be injected |

**The honest summary:** SRP and DIP are the strained ones, and both for the same reason. The runner is the only seam, and it does too much, while the engine is depended on concretely everywhere.

---

## 14.5 Contract map: every boundary, and whether it's enforced

| Boundary | Type | Enforced? | Notes |
|---|---|---|---|
| YAML → `ExperimentConfig` | Pydantic models | **partly** | types and cross-field rules yes; **unknown keys silently ignored** (no `extra="forbid"`); free strings for `model`, `selection_metric`, `slices`, `baseline_models` (Module 2) |
| CLI options → functions | Typer annotations | yes, for type | semantic validity (does the version exist?) is checked only later |
| intervention → runner | `InterventionResult` dataclass | type hints only (no runtime check) | its `build_summary: dict[str, Any]` is the engine's untyped blob |
| runner → `dataset.json` | `dict` literal | **no** | shape exists only in the runner and its readers (CLI `evaluate`, comparison) |
| split check → runner | `SplitIntegrityReport` | hints only | `ok` semantics too weak (Module 5) |
| profile → report and figures | `DatasetProfile.to_dict()` → `dict[str, Any]` | **no** | consumers index `profile["episode_length"]["histogram"][i]["lo"]` etc. |
| engine → profile | explorer dicts (`durations["duration_s"]["max"]`, `dims[k]["per_dimension"]`) | **no** | undocumented engine shapes; a change means `KeyError` (Module 5) |
| engine `fit` → `on_step(record)` | `dict` with `step`, `loss`, `lr`, `grad_norm` | **no** | `record["step"]` and `r["loss"]` are hard key accesses (Modules 6, 10) |
| trainer → runner | `TrainingOutcome` dataclass | hints only | `checkpoints: list[dict]` is untyped |
| checkpoint directory → everything | 5 files plus `rde_checkpoint_meta.json` | **presence only** (`verify_checkpoint`) | no content or integrity check; metadata keys untyped |
| `EvaluationResult` → offline helpers | engine object (`.metrics` dict, `.outcomes` list of `FrameOutcome`) | engine-side | `metrics` keys untyped; `FrameOutcome.error` semantics (per-frame MAE) in docs only |
| offline helpers → JSON | `dict` | **no** | per-episode `"mse"` misnamed (Module 7) |
| feedback → artifact | `dict` | **no** | `report.summary()` and `artifacts` are engine blobs |
| artifact → index / registry | `ModelArtifact.to_dict()` | **no** on read | `ModelRegistry.get` returns `dict` |
| runner → report | `ctx: dict[str, Any]` with about 15 keys | **no** | the documented shape is "see `ExperimentRunner`" (docstring) |
| run directory → `compare` / CLI | JSON files | **no schema, no version** | missing files mean silent `None`s; partial files mean `JSONDecodeError` |
| checkpoint ↔ contract (deployment) | — | **no** | the cross-check is missing (Module 9) |

**Pattern:** the *edges* into the system (YAML, CLI) are typed. The *interior* stage-to-stage handoffs are value objects that immediately degrade to dicts, and everything *persisted* is schemaless JSON. A Pydantic model per persisted artifact (`DatasetRecord`, `ModelRecord`, `SummaryRecord`) with a `schema_version` field would type both the write and the read path, and it would have caught the `vla_finetune` / `model_training` key bug (Module 3) at once.

---

## 14.6 Sequence diagrams

### Happy path: `mtrain train --config baseline.yaml`

```
 User   cli.train   loader   Runner        tracking      interventions   rde(Catalog/Builder)  Trainer   rde fit   offline   feedback  reports  index
  │ train ─▶│         │         │              │               │                 │              │         │         │         │         │        │
  │         │─load───▶│ merge+validate          │               │                 │              │         │         │         │         │        │
  │         │◀─cfg────│         │              │               │                 │              │         │         │         │         │        │
  │         │──run(cfg,seed)───▶│─make_run_dir▶│ env,git,cfg   │                 │              │         │         │         │         │        │
  │         │         │         │──build_intervention_dataset─▶│─get_manifest/get_episodes───▶│         │         │         │         │        │
  │         │         │         │              │               │─(filter)─transform/run────────▶│ clean/vK │         │         │         │        │
  │         │         │         │              │               │─DatasetBuilder.build──────────▶│ training/vM        │         │         │        │
  │         │         │         │◀──────────InterventionResult─│                 │              │         │         │         │         │        │
  │         │         │         │─record_dataset───────────────────────────────────────────────────────────────────────────────────────────────▶│
  │         │         │         │─get_dataset(view) → verify_split_integrity → build_dataset_profile → figures                                 │
  │         │         │         │──────────────────────────train(view,view,ids,weights)─────────▶│─fit(on_step)─▶│ loop          │        │        │
  │         │         │         │              │               │                 │              │◀─on_step─│ val(32f)/ckpt/verify/prune         │
  │         │         │         │              │               │                 │              │ save final, free GPU │         │        │        │
  │         │         │         │◀────────────────────────TrainingOutcome(best)────│              │         │         │         │        │        │
  │         │         │         │─evaluate_baselines────────────────────────────────────────────────────────────▶│         │        │        │
  │         │         │         │─from_checkpoint(best) → stratified_evaluate (overall + slices)────────────────▶│         │        │        │
  │         │         │         │─build_model_artifact → record_model───────────────────────────────────────────────────────────────────────▶│
  │         │         │         │─build_feedback_artifact(TEST)───────────────────────────────────────────────────▶│ FeedbackLoop.detect(persist)   │
  │         │         │         │─render_model_card / render_experiment_report────────────────────────────────────────────────▶│        │
  │         │         │         │─summary.json → record_experiment / record_metric──────────────────────────────────────────────────────────▶│
  │         │         │         │─finally view.close()                                                                                         │
  │         │◀─result─│         │                                                                                                              │
  │◀─summary│         │
```

### The interesting failure path: a typo in `baseline_models`, discovered after training

```
 User   cli.train   Runner          rde Catalog/Builder     Trainer/fit       offline.evaluate_baselines
  │ train ─▶│──run──▶│ make_run_dir (env, git, config written)            │                     │
  │         │        │──build versions──────────▶│ clean/vK, training/vM  (IRREVERSIBLE)       │
  │         │        │ index.record_dataset (row written)                 │                     │
  │         │        │──train───────────────────────────────────────────▶│ 1000 steps, ckpts  │
  │         │        │◀──TrainingOutcome──────────────────────────────────│ (5–10 min of GPU)  │
  │         │        │──evaluate_baselines(["nearest_neighbour"])────────────────────────────────▶│
  │         │        │◀────────────────────────────────── KeyError("unknown baseline model ...")─│
  │         │        │ finally: view.close()                               (no summary.json,    │
  │         │◀─KeyError (traceback)                                          no crash.json,      │
  │◀─traceback                                                                no status)         │
  │ retry same day ─▶ make_run_dir → FileExistsError (directory non-empty)                        │
```

This one path exercises four separate gaps: late validation (Module 11), irreversible catalog writes before the risky step (Module 4), no status marker (Module 3) and the retry block (Module 3).

---

## 14.7 State machine of a run (implicit, reconstructed from which files exist)

There's no explicit `status` field. The states below are what you can *infer* from the run directory.

```
               make_run_dir
   (none) ───────────────────▶ CREATED  {environment.json, git.json, config.resolved.yaml}
                                  │ build_intervention_dataset ok
                                  ▼
                              DATA_READY {dataset.json} + index.datasets row + engine versions
                                  │ split check + profile ok
                                  ▼
                              PROFILED {split_integrity.json, dataset_profile.json}
                                  │ fit starts
                                  ▼
                              TRAINING {metrics/train.jsonl growing, checkpoints/step_N}
                       ┌──────────┼──────────────────────────────┐
     Exception in fit  │          │ fit returns                   │ SIGTERM / SIGKILL / Ctrl-C
                       ▼          ▼                               ▼
              CRASHED_TRAINING  TRAINED {checkpoints/final,     KILLED (indistinguishable from
              {crash.json}       training_outcome.json, model.json}  TRAINING: no marker)
                                  │ evaluation + feedback
                                  ▼
                              EVALUATED {evaluation/*.json, predictions/, model_artifact.json,
                                         failures/feedback_artifact.json}
                                  │ reports
                                  ▼
                              COMPLETED {reports/*.md, summary.json} + index.experiments row
                                  │ mtrain evaluate/test
                                  ▼
                              RESCORED  (test_overall.json replaced; rest stale — Module 12)

   Any exception outside fit (intervention, profile, baselines, feedback, plotting) ⇒ stuck in the
   last state reached, with no marker. Every non-(none) state blocks a same-day re-run.
```

**Missing transitions:** CRASHED/KILLED → RETRY (only possible by renaming or deleting), and RESCORED → consistent (no recompute of dependents). **The fix:** write `status.json = {"state": ..., "updated_at": ...}` atomically at each arrow, and let `make_run_dir` accept `--force` only from a terminal failure state.

---

## 14.8 Error taxonomy

| Error | Raised where | Caught where | What the user sees |
|---|---|---|---|
| `FileNotFoundError("no config file at …")` | `config/loader.py load_yaml` | nowhere | CLI traceback |
| `ValueError("must contain a YAML mapping")` | `loader.py` | nowhere | traceback |
| Pydantic `ValidationError` (wrapping validator `ValueError`s: sums, imbalanced or rebalanced target, custom IDs, fraction) | `config/schema.py` | nowhere | traceback with field paths, the clearest errors in the system |
| `FileExistsError("run directory already exists…")` | `tracking/run_dir.py make_run_dir` | nowhere | traceback; message explains the fix |
| `LookupError("unknown version")` | `data/interventions.py`, engine `get_dataset`, explorer | nowhere | traceback |
| `ValueError` (no episodes / zero resolved / empty weights / no automatic rule / no episodes to upweight) | `data/interventions.py` (6 sites) | nowhere | traceback, *after* run directory creation |
| `SplitLeakageError` | `data/split_integrity.py` | nowhere | traceback (practically unreachable, Module 5) |
| profile split lookup `Exception` | engine | `profile.py` (broad `except`) | **swallowed** into a note (sometimes a misleading one) |
| histogram SQL `Exception` | engine DuckDB | `profile.py` | **swallowed** into `{"reason": ...}` |
| git subprocess `Exception` | `run_dir._git_info` | same | **swallowed** into `{"error": ...}` in `git.json` |
| `ImportError` torch / lerobot / psutil | `run_dir`, `trainer` | same | **converted** to `None` plus a note (by design) |
| `ValueError` (trainer model guard, empty train IDs) | `training/trainer.py` | nowhere | traceback |
| `FileNotFoundError` / `ValueError` from `verify_checkpoint` | engine | `trainer.on_step` | **converted** to `verified: false` plus `checkpoint_errors.jsonl` |
| any `Exception` in `fit` (CUDA OOM, NaN if raised, `on_step` errors) | engine / trainer | `trainer.train` | `crash.json` written, then **re-raised** to a traceback |
| `KeyboardInterrupt`, SIGTERM, SIGKILL | OS / user | **nothing** | process ends; no marker |
| `KeyError("unknown baseline model")` | `evaluation/offline.py` | nowhere | traceback **after training** |
| `ValueError("unknown evaluation slice")` | `evaluation/offline.py` | nowhere | traceback **after training** |
| `KeyError` on engine dict shapes (`r["loss"]`, `durations[...]`) | profile, figures, cli | nowhere | traceback, possibly after training |
| `DeploymentHealthError` | `deployment/loader.py` | nowhere (only tests call it) | — |
| `RuntimeError("call load() before predict()")` | `deployment/loader.py` | nowhere | — |
| `JSONDecodeError` on torn or partial files | `JsonlIndex.read_all`, `load_run_summary`, CLI readers | nowhere | traceback with no hint of cause (Module 3) |
| `typer.Exit(1)` with a yellow or red message | `cli.py` (6 sites) | Typer | friendly one-line message |

**Shape of the error model:** only **two** custom exception types (`SplitLeakageError`, `DeploymentHealthError`). Everything else is a built-in, nothing is caught at the top level, and the CLI turns *some* missing-file conditions into friendly exits but lets every pipeline error surface as a raw traceback. **Swallowing is confined to observation code** (git, environment, profile extras), and that's the right place for it. **The conversion of errors into data** (`verified: false`, `"unavailable"`, `crash.json`) is the system's best error-handling idea, and it's applied inconsistently: there's no equivalent for failures outside `fit`.

---

## 14.9 Lifetime, sharing and thread safety

| Object | Created by | Lifetime | Shared across | Thread / process safety |
|---|---|---|---|---|
| `Settings` | `get_settings()` / test fixture `set_settings` | process (global singleton) | everything | global mutable state; tests rely on fixture order |
| `Catalog` | runner `__init__` (one per runner) | runner | interventions, profile, feedback | **engine-side, unverified**; process-level concurrency on the catalog file is unknown (version allocation race, Module 4) |
| `DataEngine` / `DatasetView` | runner `run`, profile, CLI | one `run`; closed in `finally` | trainer, evaluation, feedback (same view) | single-threaded use; two views of one version open at once (runner plus profile) |
| `ExperimentIndex` / `JsonlIndex` | runner `__init__` | runner | all runs in the process | appends are small writes; **not safe** for large rows across processes, **no locking** (Module 3) |
| `RunDir` | `make_run_dir` | one run | runner, trainer | single writer by convention; **TOCTOU** on creation |
| `SmolVLAPolicy` (train) | `Trainer.train` | `train()` call; explicitly freed | closure `on_step` | GPU-resident; freed before return (and not fully freed on the crash path) |
| `SmolVLAPolicy` (eval) | runner `run` | rest of `run` | evaluation, feedback | not explicitly freed |
| `Evaluator` | per helper call | call | — | stateless (inferred) |
| `console` (rich) | `cli.py` module level | process | all commands | fine |
| matplotlib backend | `figures.py` import | **process** | every importer | **global side effect** |
| `RDE_REPO` / `MT_REPO` | `runner.py` import | process | runner | frozen at import |

**No threads are created anywhere in the package.** Concurrency risk is entirely **multi-process**: two `mtrain` invocations sharing `runs/`, `experiments/*.jsonl` and the engine catalog.

---

## 14.10 Idempotency table

| Operation | Idempotent? | Second invocation does… |
|---|---|---|
| `load_experiment_config` | **no** | new `created_at`, so a new `experiment_id` |
| `mtrain train` (same day) | **no**, guarded | `FileExistsError` |
| `mtrain train` (next day) | **no** | new run directory **plus new engine versions** plus new index rows |
| `build_intervention_dataset` | **no** | mints new `clean/vK` and `training/vM` even for identical input |
| `verify_split_integrity`, `build_dataset_profile` | yes (apart from `generated_at`) | same report |
| `Trainer.train` | no (by construction) | new checkpoints; stochastic |
| `evaluate_baselines` | yes, for deterministic baselines | same metrics |
| `stratified_evaluate` (SmolVLA) | **no** | stochastic re-inference, numbers move up to 5% |
| `build_feedback_artifact` | **no** | duplicate engine failure records (`persist=True`) |
| `JsonlIndex.append` / `record_*` | **no** | duplicate rows |
| `ModelRegistry.get` | yes | the latest row |
| `mtrain evaluate` / `test` | **no** | overwrites with new stochastic numbers |
| `mtrain rollout` | **no** | appends another identical record |
| `mtrain build-intervention --apply` | **yes for the block**, **no for the file** | same block; comments already gone after the first run |
| `mtrain export` | yes (overwrite) | same copy; stale extra files remain |
| `mtrain compare` | yes | same table |

**Takeaway:** the read paths are idempotent and the write paths are not, and nothing *dedupes* (versions, index rows, failure records). Content-addressing builds (Module 4) and a run status (14.7) are the two changes that would make re-execution safe.

---

## 14.11 Patterns deliberately absent, and the trigger to add each

| Absent pattern | Why absent is fine today | Trigger to add it |
|---|---|---|
| Plugin / Strategy registry for interventions and slices | 6 kinds and 2 slices, edited in place | a third slice type, or a fourth new intervention kind (O/C strain) |
| Project-owned engine protocol (`typing.Protocol` for `Catalog`/`View`/`Evaluator`) | one engine, real-engine tests | a second backend, or the need for engine-free unit tests |
| Stage pipeline / DAG with caching | 5–10 minute runs | multi-hour or preemptible runs, sweeps over 10 runs |
| Unit of Work / transaction across stores | one user, rare failures | concurrent users, or any automation that retries |
| Schema-versioned persisted records | one code version produced every run | the first breaking change to an artifact shape |
| Observer / event bus for run progress | a terminal and files suffice | a dashboard or remote monitoring |
| Retry with backoff | local disk and GPU errors aren't transient | network-backed storage or Hub downloads in CI |
| Locking (`fcntl` / sqlite) on the index | single process | two concurrent `mtrain train` |
| Dependency-injection container | one composition root suffices | multiple entry points wiring differently |
| Factory for models | one trainable model | a second trainable model family |
| Server (FastAPI / gRPC) | no consumer process | a robot controller in another process |

---

## 14.12 Honest gaps to own before an interviewer finds them

1. **Interior contracts are untyped dicts**, and persisted artifacts have no schema or version. *Consequence:* key-mismatch bugs (`vla_finetune` / `model_training`) and engine shape changes surface as `KeyError` or silent `n/a`, never at write time.
2. **There's no explicit run state machine.** State is inferred from file existence, and killed, crashed and in-progress runs are indistinguishable. *Consequence:* retries are blocked, and compare and export can't tell complete from partial.
3. **The runner violates SRP** (orchestration plus formatting plus naming), and **the CLI duplicates runner internals.** *Consequence:* the `evaluate` drift (Module 12), and no reusable stage functions.
4. **Concrete dependence on `rde` everywhere, with no project-owned protocol.** *Consequence:* no seam for engine-free unit tests or a second backend (DIP strained).
5. **Open/closed is strained for interventions** (3–4 edit sites per new kind) and slices (an `elif` chain). *Consequence:* extension means editing core functions.
6. **Only two custom exceptions, no top-level handler, and late validation.** *Consequence:* expensive failures, and raw tracebacks for user errors.
7. **Write operations aren't idempotent, and nothing dedupes.** *Consequence:* re-execution multiplies versions, index rows and engine failure records.
8. **Process-level concurrency is unsafe** (TOCTOU, unlocked appends, non-atomic version allocation). *Consequence:* two concurrent runs can corrupt shared state.
9. **Import-time global side effects** (matplotlib backend, path constants). *Consequence:* importing the package changes process behaviour.

---

## The shape to carry into the High-Level Design module

```
  cli ─▶ experiments(runner = composition root) ─▶ {training, data, evaluation, feedback, deployment, reports}
                                                         │            ─▶ {config, tracking}
                                                         ▼
                                                  rde (concrete, unabstracted)
  Contracts:  YAML/CLI typed ──▶ value objects ──▶ dict[str, Any] ──▶ schemaless JSON on disk
  State:      CREATED → DATA_READY → PROFILED → TRAINING → (CRASHED | KILLED | TRAINED)
              → EVALUATED → COMPLETED → (RESCORED)        ← all implicit, inferred from files
  Errors:     raise built-ins · 2 custom types · convert-to-data in fit/observation only · no top handler
  Idempotent: reads yes · writes no · dedupe nowhere      Concurrency: single process assumed
```

---

## Drill — answer from memory, files closed

1. Draw the package dependency graph. Where's the composition root, and which edge breaks the layering?
2. Name four design patterns in this codebase, with exact locations and what each buys.
3. *Why didn't you just* define a typed `Protocol` for the data engine?
4. *What breaks in production:* two researchers launch `mtrain train` at the same moment on a shared machine. Walk through every shared resource.
5. *Interviewer framing:* "What's the weakest part of your low-level design, and how would you fix it?"

---

# Appendix A — Model answers to the drill

**A1. Dependency graph.**
"Four layers. At the bottom, `config` (Pydantic and YAML only) and `tracking` (standard library only). Above them, the stage packages: `data` and `training` depend on `config`, `training` and `deployment` also use `tracking`, and `evaluation` and `feedback` depend only on the engine. `reports` depends on nothing internal, because it takes plain dicts. Above that, `experiments`, whose runner is the composition root: it builds the catalog, engine, index, trainer and models, and calls every stage. The CLI sits on top. There are no cycles. The edge that breaks the layering is the CLI reaching past the façade into `data.interventions` internals and reimplementing the evaluation stage instead of calling a service function. And the engine sits underneath every layer as a concrete dependency, with no abstraction."

*Why that answer lands:* it's precise about direction and leaves, names the composition root, and identifies the real violation instead of claiming purity.

**A2. Four patterns.**
"Template Method via callback in `Trainer.train`: the engine's `fit` owns the loop, and I inject an `on_step` hook that validates, checkpoints, verifies and prunes. It lets run policy live in this repo while the step mechanics stay in the engine. Transaction Script in `ExperimentRunner.run`: one linear method for the whole lifecycle, readable and mapping directly onto the architecture figure, at the cost of no resume. Specification via Pydantic model validators on `DatasetIntervention`: the methodology rule 'don't presume the weak dimension' becomes a load-time error. And a Null-Object-like explicit 'unavailable' record, in the profile's visual diversity, the rollout log and the environment's missing torch: absence is data, so nothing downstream mistakes it for a zero."

*Why that answer lands:* each pattern has a file, a mechanism and a payoff specific to this system, which shows the patterns are real and not name-dropped.

**A3. Why not a `Protocol` for the engine?**
"Because there's one engine, and my tests deliberately use the real one, since the risky bugs live at that boundary, so a protocol would have had a single implementation and no test doubles. The cost showed up anyway. With no seam, pure logic like pruning, threshold arithmetic or report rendering can't easily be tested without spinning up a catalog, so a lot of it isn't tested. I'd introduce small protocols where a stage needs only a narrow slice of the engine: a `SplitSource` with `episode_ids(split)` for the integrity check, an `EpisodeCatalog` for intervention resolution. Then unit tests use fakes and the integration tests keep using the real engine. The trigger for a full abstraction would be a second backend."

*Why that answer lands:* it explains the original reasoning, admits the cost that materialized, and proposes a targeted, incremental fix with a clear trigger for going further.

**A4. Two concurrent launches.**
"Four shared resources. First, the run directory: if it's the same experiment and seed, `make_run_dir` checks for emptiness and then creates, so both can pass the check and write into one directory. Second, the engine catalog: both call `next_version` and then write, which isn't atomic, so both may be allocated the same `training/vN`. Whether the engine rejects the second write I couldn't verify. Third, the JSONL index: appends are unlocked, and model rows can exceed the write buffer, so lines can interleave, and one torn line breaks every later read. Fourth, the GPU: two SmolVLA trainings on an 8 GB card will likely OOM one of them, and that one gets a `crash.json`. No threads are involved; it's all multi-process. Fixes are an atomic `mkdir(exist_ok=False)`, a lock around version allocation and index appends, and a per-machine GPU lock or a queue."

*Why that answer lands:* it enumerates every shared resource, gives the precise race for each, separates verified from unverified, and proposes minimal fixes.

**A5. "Weakest part of your low-level design?"**
"The contracts between stages. The edges are typed (YAML goes through Pydantic, the CLI through Typer), but inside the pipeline every handoff degrades to `dict[str, Any]`, and everything on disk is schemaless JSON. That's why a one-word key mismatch made every report print 'n/a' for my own repo's commit, and why an engine change to a stats dict would show up as a `KeyError` after training. The fix is a Pydantic model per persisted artifact (dataset record, model record, summary, report context) with a schema version, used on both write and read. Second on the list is the implicit state machine: I'd add an atomic status file so crashed, killed and completed runs are distinguishable and retries are possible."

*Why that answer lands:* it picks a genuinely structural weakness, proves it with a real bug, and gives a fix that addresses the class of bugs, not the instance.

---

# Module 14 Summary — the one page to revise

**What this module is.** The design view of the package: an acyclic four-layer graph with one composition root (the runner), about a dozen real patterns at known locations, SRP and DIP strained at the runner and the engine boundary, typed edges with untyped interiors, an implicit file-existence state machine, a built-in-exception error model, single-process assumptions and non-idempotent writes.

**Five claims to make cold**
1. No cycles. `config` and `tracking` are leaves, `reports` imports nothing internal, and the runner is the composition root. The CLI is the one layering violation.
2. Patterns: Template Method via callback (Trainer), Transaction Script (runner), Specification (validators), Façade (index and packages), explicit "unavailable" Null Objects.
3. Contracts are typed at the edges and dicts inside; persisted JSON has no schema or version.
4. Run state is implicit in which files exist. Killed equals in-progress, and every state blocks retry.
5. Writes aren't idempotent and nothing dedupes. Concurrency is multi-process only, and unsafe.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Strict downward layering, `reports` dict-only | shared typed models | decoupling | untyped interiors |
| Runner as the only composition root | DI container | one wiring point | SRP-heavy runner |
| Concrete `rde` dependence | project protocols | one engine; real-engine tests | no seam for unit tests or a second backend |
| Built-in exceptions, no top handler | custom hierarchy | simplicity | tracebacks for user errors; late validation |
| Errors as data in `fit` and observation | raise everywhere | partial results survive | inconsistent outside `fit` |
| File-existence state | explicit status | nothing to maintain | ambiguous states; blocked retries |

**Volunteer unprompted:** the untyped contracts and the `n/a` key bug as proof; the implicit state machine; the CLI layering violation; non-idempotent writes; the multi-process races.

**Numbers and names:** 4 layers; 2 custom exceptions; 6 `raise typer.Exit` sites; 36 raise/except sites outside the CLI; 7 broad or `ImportError` excepts (all in observation or trainer code); 0 threads; 4 shared resources under concurrency (run directory, catalog, index, GPU).

**Sentences worth memorizing**
- "Typed at the edges, dicts in the middle, schemaless on disk. That's the weakest part of my design."
- "The runner is the composition root and the only seam; that's why it does too much."
- "My run state machine is implicit in which files exist, so a killed run looks like a running one."
- "Reads are idempotent, writes aren't, and nothing dedupes."
