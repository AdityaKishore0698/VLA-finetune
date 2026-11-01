# Module 2 — Configuration as the unit of reproducibility

**Files covered:** `src/model_training/config/schema.py`, `src/model_training/config/loader.py`, `configs/base.yaml`, `configs/smolvla/so100_pickplace.yaml`, `configs/experiments/baseline.yaml`, `configs/experiments/baseline_extended.yaml`, `configs/experiments/reduced.yaml`, `configs/experiments/imbalanced.yaml`, `configs/experiments/rebalanced.yaml`, `configs/experiments/aloha_coffee_baseline.yaml`

**Where we are:** Module 1 established that this repository owns *procedure* and the engine owns *mechanism*. Procedure starts here. An `ExperimentConfig` is the object every later stage reads, and its fully resolved form is what gets written next to every run. If the config can't be trusted to mean what it says, nothing downstream can be either. That makes this module the right place to test the project's central claim: "every experimental parameter that matters lives here."

**The one-line thesis:** Configuration is a two-layer YAML deep-merge validated by Pydantic. The validators do enforce *semantic* rules: an imbalanced or rebalanced experiment can't run without a declared target. But the schema accepts unknown keys silently, validates several fields that nothing consumes, and has one imbalanced path that silently becomes a full-data run. So the "single object a run is reproducible from" is reproducible, but it isn't always truthful.

---

## 2.1 Prerequisites — the absolute basics

- **YAML.** A human-readable data format. Mappings (`key: value`), lists (`[a, b]` or `- a`), scalars. Two features appear here:
  - `>` (a *folded block scalar*): the following indented lines are joined with spaces into one string. Used for `hypothesis:` and `objective:`.
  - `null`: Python `None`.
- **`yaml.safe_load` vs `yaml.load`.** `safe_load` builds only plain types (dicts, lists, strings, numbers). `load` with the full loader can construct arbitrary Python objects from tags like `!!python/object`. That's a code-execution vector if a config file is untrusted. This code uses `safe_load` and `safe_dump` everywhere.
- **Deep merge.** Combining two nested dicts so that nested dicts are merged key by key, and anything else (scalars, lists) in the override *replaces* the base value.
- **Pydantic (v2).** A library that turns a class with type annotations into a validator. `Model.model_validate(dict)` checks types, applies defaults, coerces where allowed, and raises `ValidationError` on failure. Key terms:
  - `Field(default=..., ge=0.0, le=1.0)`: a default plus numeric bounds (greater-or-equal, less-or-equal).
  - `Field(default_factory=lambda: [...])`: builds a fresh mutable default for each instance, avoiding Python's shared-mutable-default bug.
  - `Literal["a", "b"]`: the value must be exactly one of those.
  - `@model_validator(mode="after")`: a method that runs *after* field validation, on the constructed object. It can check cross-field rules and, as here, mutate fields.
  - **Lax mode** (the default): sensible coercions are allowed, such as `"3"` → `3` for an `int` field or dict key. An `int` is **not** coerced to `str` in v2.
  - **`extra` behaviour:** by default (`extra="ignore"`) unknown keys are **silently dropped**. `extra="forbid"` makes them errors. Nothing in this schema sets it.
  - `model_dump(mode="json")`: export to plain JSON-compatible types (dict keys become strings, and so on).
- **Configuration hierarchy / layered config.** Shared defaults in one file, per-experiment overrides in another. This is the pattern Hydra or OmegaConf implement in fuller form.
- **Resolved config.** The final config after merging and after defaults are applied. Persisting the resolved form means a run doesn't depend on code defaults that might change later.
- **Content-addressed ID vs time-stamped ID.** A content-addressed ID is a hash of the content, so the same content gets the same ID. A time-stamped ID mixes in a clock value, so it's unique per creation.

---

## 2.2 The loader: merge first, validate once

```python
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
```

- **What:** a recursive merge. Dicts merge key by key; everything else is replaced.
- **Why `dict(base)`:** a shallow copy, so the caller's base dict isn't mutated at the top level. Nested dicts that the override never touches are *shared* between `out` and `base`, which is safe only because nothing mutates them afterwards.
- **Lists replace, not append.** That's the right semantics for this project. `aloha_coffee_baseline.yaml` sets `slices: [per_length_bucket]` to *replace* base's `[per_task, per_length_bucket]`, and the comment explains why (one task, so `per_task` would duplicate `overall`). Appending would make it impossible to remove a slice.
- **A key can't be deleted.** An override can set `null`, but it can't remove a key. That's fine here: every schema field has a default, so `null` and absent differ only for fields whose default isn't `None`.
- **Alternative:** Hydra or OmegaConf, which give interpolation, command-line overrides (`training.lr=3e-5`), and structured-config validation.
- **Cost of the hand-rolled version:** no CLI overrides, so changing a learning rate means editing or copying a YAML file. That's arguably a *feature* for an experiment log, since every run has a file.
- **When the answer flips:** once sweeps over hyperparameters (not just data fractions) are routine.

```python
def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config file at {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data
```

- `or {}` handles an empty file: `safe_load("")` returns `None`. Without it, the `isinstance` check would raise a misleading "must contain a mapping".
- The `isinstance(data, dict)` check catches a file whose top level is a list or a bare string. Without it, `_deep_merge` would fail later with an `AttributeError` that doesn't name the file. It's a good defensive branch with a specific error.

```python
def load_experiment_config(path, base_path=None) -> ExperimentConfig:
    path = Path(path)
    data = load_yaml(path)

    if base_path is None:
        candidate = path.parent.parent / "base.yaml"
        base_path = candidate if candidate.exists() else None

    if base_path is not None:
        base_data = load_yaml(base_path)
        data = _deep_merge(base_data, data)

    data["source_config_path"] = str(path)
    return ExperimentConfig.model_validate(data)
```

**Decision: find the base by directory convention (`<grandparent>/base.yaml`).**
- *What:* `configs/experiments/x.yaml` picks up `configs/base.yaml` automatically.
- *Why:* zero ceremony for the layout the project ships.
- *Alternative:* an explicit `base:` key inside each experiment file, or an explicit `--base` flag.
- *Cost:* **what a config file means depends on where it sits on disk.** Copy `baseline.yaml` to `/tmp/` and load it: the grandparent of `/tmp/baseline.yaml` is `/`, there's no `/base.yaml`, and the base is skipped. `baseline.yaml` alone has no `dataset:` block, so this fails loudly (`dataset` is required), which is the lucky case. A copied file that *does* declare `dataset:` would load and run silently with **code defaults** (`steps: 200`, `checkpoint_every: 50`, `batch_size: 32` for eval) instead of base.yaml's values (`1000`, `100`, `16`). The resolved config records which values were used, so it's recoverable, but nothing warns you.

**Merge-then-validate** is the right order, and the module docstring states it. The *merged* dict is what Pydantic sees, so a base value plus an override value are validated together. A validator like "fractions sum to one" sees the final numbers, not each file alone.

**Doc vs code: `--base` doesn't exist.** `configs/smolvla/so100_pickplace.yaml`'s header says:

> include its contents in an experiment file's `training:` block, or point --base at it explicitly

The only CLI caller is `cli.py:67`, `cfg = load_experiment_config(config)`, and `mtrain train` has options `--config` and `--seed` only. **There's no `--base` flag.** `base_path` is reachable only from Python. And `load_experiment_config` takes *one* base, so the "second model-specific layer" that file talks about can't be stacked on top of `base.yaml` at all. The file is effectively documentation. Grep shows nothing loads it. Its values also duplicate `base.yaml`'s exactly, so even if it were wired up it would change nothing.

**`source_config_path` overwrites the user.** It's assigned after merging, so a YAML `source_config_path:` is overwritten. Good: provenance can't be spoofed through this one field. But see 2.6 for the two identity fields that *can* be.

---

## 2.3 `DatasetRef` and the "identity, not a path" decision

```python
class DatasetRef(BaseModel):
    """The dataset version an experiment starts from -- an identity, not a path."""
    dataset_id: str
    version: str = Field(..., description="e.g. 'clean/v1' -- the input to the intervention/build step")
```

**Decision: name data by `(dataset_id, version)`, never by filesystem path.**
- *Why:* a path can be overwritten in place; a catalog version is immutable (the engine's claim, inherited). This is what makes "this model was trained on `so100_pickplace@training/v6`" a stable statement.
- *Alternative:* `data_dir: /data/so100/` (the common LeRobot style).
- *Cost:* you can't train on anything that hasn't been ingested into the engine first. There's no "just point at a folder" path. That's deliberate.

**The input version is `clean/v1`, not `training/v1`.** `base.yaml`'s comment explains it:

```yaml
dataset:
  dataset_id: so100_pickplace
  version: clean/v1   # the intervention/build step consumes this; NOT training/v1,
                       # which is itself an *output* of that step
```

The config names the version the intervention *starts from*. Every run then produces its own `training/vN` (Module 4). A consequence to have ready: **two runs with identical configs produce two different `training/vN` versions.** The README's lineage example is `training/v6`, which tells you at least six had been built.

---

## 2.4 `DatasetIntervention`: semantics enforced, mostly

```python
kind: Literal["full", "fraction", "reduced", "imbalanced", "rebalanced", "custom"] = "full"
fraction: float | None = Field(default=None, ge=0.0, le=1.0)
variable: str | None = None
variable_value: Any = None
episode_ids: list[int] | None = None
episode_weights: dict[int, float] | None = None
oversample_factor: float = 3.0
seed: int = 0
reason: str = ""
```

This is a **tagged union flattened into one class**: `kind` is the discriminator, and which other fields matter depends on it.
- *Alternative:* Pydantic's discriminated unions, i.e. one class per kind (`FullIntervention`, `FractionIntervention(fraction: float)`, and so on), selected by `Field(discriminator="kind")`. Each class would declare *only* its fields, so `kind: full, fraction: 0.3` would be rejected as an unknown field (with `extra="forbid"`).
- *Cost of the flat form:* irrelevant fields are accepted and ignored. `kind: full` with `episode_ids: [1, 2]` is valid, and `resolve_episode_subset` returns *all* episodes for `full` before it ever looks at `episode_ids` (Module 4). It's a silent no-op on a field the user explicitly set.

The validator:

```python
@model_validator(mode="after")
def _check_shape(self) -> "DatasetIntervention":
    if self.kind in ("fraction", "reduced") and self.episode_ids is None and self.fraction is None:
        if self.kind == "reduced":
            self.fraction = 0.5
        else:
            raise ValueError("intervention.kind='fraction' requires 'fraction' or an explicit 'episode_ids'")
    if self.kind == "imbalanced" and self.episode_ids is None and self.variable is None:
        raise ValueError("... requires 'variable' (from a dataset profile) or an explicit 'episode_ids' "
                         "-- the weak dimension must not be presumed")
    if self.kind == "rebalanced" and self.variable is None and not self.episode_weights:
        raise ValueError("... requires 'variable' ... or an explicit non-empty 'episode_weights'")
    if self.kind == "custom" and self.episode_ids is None:
        raise ValueError("intervention.kind='custom' requires an explicit 'episode_ids'")
    return self
```

Line by line:
- **`reduced` defaults `fraction` to 0.5.** "Reduced" means "the project-standard ~50%", so `reduced` is just `fraction` with a default. A *mutating* after-validator is unusual but legal, and Pydantic v2 allows assignment inside `mode="after"`. The cost: `0.5` appears as a magic number in **two** places. The other is `resolve_episode_subset`'s `intervention.fraction if intervention.fraction is not None else 0.5`, which is unreachable for configs that went through this validator. Two sources of one default can drift.
- **`imbalanced` requires `variable` or `episode_ids`.** This encodes the design rule "the weak dimension must not be presumed": you can't ask for an imbalance without naming *along what*. It's a methodological rule enforced as a type error. That's the strongest thing in this file, and it's tested (`test_imbalanced_requires_variable_or_episode_ids`).
- **`rebalanced` requires `variable` or non-empty `episode_weights`.** `not self.episode_weights` treats both `None` and `{}` as missing. That's exactly why the shipped `rebalanced.yaml` template, with `episode_weights: {}`, **fails to load**, as its header promises and `test_rebalanced_template_fails_loudly_until_filled_in` asserts. Doc and code agree here.
- **`custom` requires `episode_ids`, but `[]` passes** (it checks `is None`, not falsiness). That asymmetry is *deliberate*: `imbalanced.yaml` is a `custom` template with `episode_ids: []`, and its header says it is "Loadable as-is ... but resolving it ... will fail loudly with 'no episodes'". It does: `build_intervention_dataset` raises `"resolved to zero episodes"` (Module 4). So the two templates fail at *different* stages, one at load and one at resolution. The headers document this correctly, but it's an inconsistency you could be asked about. The honest answer is that they're placeholders with different shapes, and both fail before any training step.

### What the validator does *not* check

1. **`imbalanced` with `variable` set but `variable_value` missing passes.** Follow it through `resolve_episode_subset` (Module 4): with `variable="task_id"`, `val=None`, it keeps `e` where `task_of.get(e) != None`, which is *every episode that has a task*. With `variable="episode_length_bucket"`, `val=None`, no episode has bucket `None`, so all are kept. The subset equals the full set, so `build_intervention_dataset` sets `clean_version = input_version` and **trains a full-data run labelled `imbalanced`.** That's exactly the "silently run as a no-op" outcome the `rebalanced.yaml` header says the design forbids. Rebalanced has a guard for this (`variable_value is not None` before resolving targets, then "resolved to empty episode_weights" raises). **Imbalanced doesn't.**
2. **`variable_value: Any` isn't coerced.** Hand-write `variable: task_id, variable_value: "0"` (a string) and the comparison `task_of.get(e) != "0"` is true for every integer task id, so again a silent full run. The CLI's `build-intervention` converts `task_id` values to `int` (`value = int(variable_value)`), so the *generated* path is safe. The *hand-written* path isn't.
3. **`oversample_factor` and `episode_weights` values are unbounded.** `oversample_factor: 0` or `-1` passes validation. What a zero or negative sampling weight does inside the engine's sampler is unverified. A `torch.multinomial`-style sampler would reject negatives. Either way, it isn't caught here.
4. **`fraction` with `kind: full` (or `rebalanced`) is accepted and ignored.**

**`episode_weights: dict[int, float]` and YAML string keys.** `mtrain build-intervention` writes weights as `{str(e): oversample_factor}`, so YAML holds `'3': 3.0`. Pydantic lax mode coerces a numeric string dict key to `int`, so this round-trips correctly. It's correct, but it works by coercion and not by design. A non-numeric key would be a validation error, which is the right outcome.

---

## 2.5 `SplitConfig`

```python
class SplitConfig(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    mode: Literal["quantile", "independent"] = "quantile"

    @model_validator(mode="after")
    def _sums_to_one(self) -> "SplitConfig":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {total}")
        return self
```

- **The `1e-6` tolerance** is needed because `0.8 + 0.1 + 0.1` in binary floating point isn't exactly `1.0` (it's `0.9999999999999999` or `1.0000000000000002` depending on order). An exact `== 1.0` would reject the default config. Anything looser would accept typos like `0.8/0.1/0.11`.
- **Missing: non-negativity.** `train: 1.2, val: -0.1, test: -0.1` sums to 1.0 and passes. So does `val: 0.0`. What the engine's `assign_splits` does with a negative fraction is unverified. A zero val split means the trainer's validation pass scores an empty split, `val_metric` comes back missing or `None`, no checkpoint is ever "best", and selection silently falls back to the final checkpoint (Module 6). The fix is one line: `Field(ge=0.0, le=1.0)` on each, plus `val > 0`.
- **`mode: quantile` vs `independent`.** These are the engine's split-assignment modes. `base.yaml`'s comment, "exact split sizes at this episode count (48)", plus the README's 38/5/5, suggests `quantile` assigns *exact* counts (48 × 0.8 = 38.4 → 38, and 4.8 → 5 for each of val and test). `independent` would presumably draw each episode's split independently, giving sizes that vary run to run. Both are inferred from the name and the comment; the engine's code isn't here. **Why it matters at n = 48:** with independent draws, a 10% split of 48 episodes has a standard deviation of about √(48·0.1·0.9) ≈ 2 episodes, so a 5-episode test set could easily come out as 3 or 7. Exact sizes are the right choice at this scale.
- **`seed: 42` (split seed) is separate from `training.seed: 0`.** That separation is a real decision. You can vary the training seed while holding the test split fixed, which is what makes "both SO-100 runs are scored on the identical test split" possible. The cost: there are now three seeds (split, intervention, training), and the CLI's `--seed` touches **none** of them (Module 11).

The docstring says "identical semantics to `rde.transformations.ops.assign_splits`, because it *is* that call." That's honest only partly. The fractions, seed and mode are passed straight through to `DatasetBuilder.build` (Module 4), but `SplitConfig`'s validation is *this* repo's, so the two layers could disagree about what's legal (for example, if the engine accepts fractions that don't sum to 1).

---

## 2.6 `TrainingConfig`, `EvaluationConfig`, `ExperimentConfig`

### Fields validated but never consumed

I grepped every `TrainingConfig` and `EvaluationConfig` field against `src/`:

| Field | Consumed? | Evidence |
|---|---|---|
| `training.num_workers` | **no** | only in `schema.py` |
| `training.mixed_precision` | **no** | only in `schema.py` |
| `evaluation.success_tolerance` | **no** | only in `schema.py`; never passed to `Evaluator` |
| `training.device` | yes | `SmolVLAPolicy(..., device=tc.device)` |
| everything else in `TrainingConfig` | yes | passed to `policy.fit` or used in `Trainer` |

**Doc vs code: `docs/API_REFERENCE.md`** describes `success_tolerance` as "Tolerance band defining `Evaluator`'s `success_rate` proxy." Setting it to 0.05 changes nothing. The evaluator uses its own default (the README's "below 0.1" matches the schema default by coincidence, or because both copy the engine's value; I can't verify which). **These three fields make the config lie:** a resolved config that says `mixed_precision: true` describes a run that used full precision.

### The comment pointing at a module that doesn't exist

```python
#: Metric used for checkpoint selection. Must be computed on validation,
#: never test -- enforced by model_training.experiments.selection, not here.
selection_metric: str = "mse"
```

`model_training/experiments/` contains `__init__.py`, `runner.py` and `data_fraction.py`. **There's no `selection` module.** The val-only rule *is* enforced, but structurally, in `Trainer.on_step`, which hard-codes `split="val"` (Module 6). So the guarantee holds and the citation is stale. Say it exactly that way.

`selection_metric: str` is a free string. If you name a metric the evaluator doesn't produce (`"msee"`), `result.metrics.get(ec.selection_metric)` returns `None`, no checkpoint ever becomes best, and the trainer silently falls back to `final` (Module 6). A `Literal[...]` of known metric names would catch it at load.

### `training.model: str`, the most dangerous free string

`model: str = "smolvla"`. The runner branches on `config.training.model == "smolvla"`, and *anything else* skips training, appends a limitation note, and still completes a "successful" run that scores baselines only. That path is intended for `mean_action` (the runner unit test uses it). But a typo, `model: SmolVLA` or `smol_vla`, produces a run directory, a report, a summary and an index row, with **no fine-tuning and exit code 0.** The only trace is one line under "Limitations" in the report. `Literal["smolvla", "mean_action", "nearest_neighbor"]` would turn this into a load-time error.

### Two sources of defaults

| Field | `schema.py` default | `base.yaml` value |
|---|---|---|
| `training.steps` | 200 | 1000 |
| `training.checkpoint_every` / `validate_every` | 50 / 50 | 100 / 100 |
| `training.warmup_steps` | 0 | 50 |
| `evaluation.batch_size` | 32 | 16 |
| `evaluation.slices` | `["per_task"]` | `["per_task", "per_length_bucket"]` |

Which one applies depends on whether `base.yaml` was found (2.2). Persisting the resolved config makes it *auditable*, but a reader of `schema.py` gets the wrong mental model of a "default" run. Pick one home, ideally the YAML, with schema defaults kept only for fields that are genuinely optional.

### Identity stamping

```python
experiment_id: str | None = None
created_at: str | None = None
source_config_path: str | None = None

@model_validator(mode="after")
def _stamp_identity(self) -> "ExperimentConfig":
    if self.created_at is None:
        self.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    if self.experiment_id is None:
        basis = f"{self.experiment_name}:{self.dataset.dataset_id}:{self.dataset.version}:{self.created_at}"
        digest = hashlib.sha256(basis.encode()).hexdigest()[:10]
        self.experiment_id = f"{self.experiment_name}-{digest}"
    return self
```

- **What:** an ID like `baseline_full-3f9a1c0b2e`. The readable prefix is for humans; the hash makes it unique.
- **Why `created_at` is in the hash:** so two runs of the same experiment get different IDs. Without it, every run of `baseline_full` on `clean/v1` would collide in the index.
- **The hash covers name, dataset and version only, not the hyperparameters.** So the ID isn't content-addressed. Two configs differing only in learning rate get IDs that differ *only because of the timestamp*. It's a unique run identifier wearing a content hash's clothes. Calling it "deterministic" (as the test name `test_experiment_id_is_deterministic_given_same_inputs` does) is true only when `created_at` is pinned, which never happens from YAML.
- **Alternative:** `uuid4()` for uniqueness, plus a *separate* `config_hash = sha256(resolved_dict minus identity fields)`. The second would let you ask "have I already run exactly this config?", which nothing in the system can answer today.
- **Ten hex characters** is 40 bits. Collisions are irrelevant at this scale, so that's fine.
- **Doc vs code: "not user-editable."** The field comment says "Set once at load time; not user-editable via YAML," and API_REFERENCE repeats it. **Nothing prevents it.** `experiment_id: anything` in a YAML is accepted, the `is None` guard skips stamping, and the run is indexed under the user's string. Two runs could then share an ID. It's fixable with a `mode="before"` validator that pops these keys, or with `Field(exclude=True)` plus `init=False` semantics. Meanwhile, `data_fraction.py` *relies* on re-stamping by explicitly setting both to `None` before re-validating (Module 11). That pattern shows the author knew the guard exists.

```python
def resolved_dict(self) -> dict[str, Any]:
    return self.model_dump(mode="json")
```

`mode="json"` ensures the dump contains only JSON-safe types, so `yaml.safe_dump` won't fail on, say, an `int`-keyed dict (its keys become strings). The persisted `config.resolved.yaml` therefore has **string** keys for `episode_weights`, and re-loading it coerces them back to `int`. It round-trips through coercion again.

### `extra` is never forbidden, the root gap

None of the six models sets `model_config = ConfigDict(extra="forbid")`. So:

```yaml
training:
  learning_rat: 3.0e-5     # typo
```

loads cleanly. `learning_rate` keeps its inherited `1.0e-4`, and the run trains at the *wrong* learning rate. Worse, **the resolved config drops the typo**, so the run's own record shows `learning_rate: 1e-4` with no trace that you ever asked for something else. For a project whose schema docstring opens with "Every experimental parameter that matters lives here", this is the single most important one-line fix in the codebase.

---

## 2.7 The shipped configs, read as experiments

| File | Kind | Runnable as-is? | Notes from the code |
|---|---|---|---|
| `baseline.yaml` | `full` | yes | 18 lines; inherits everything else from base |
| `baseline_extended.yaml` | `full` + `resume_from` | **only on the author's machine** | `resume_from: runs/2026-09-03/baseline_full/seed_0/checkpoints/step_1000`, relative to the *current working directory*, a dated path, and git-ignored (Module 1) |
| `reduced.yaml` | `reduced`, fraction 0.5, seed 0 | yes | `test_reduced_config_loads_with_fraction` asserts the 0.5 |
| `imbalanced.yaml` | `custom`, `episode_ids: []` | loads; fails at resolution | intentional template |
| `rebalanced.yaml` | `rebalanced`, empty weights | **fails at load** | intentional template; test asserts it |
| `aloha_coffee_baseline.yaml` | `full` on `aloha_coffee` | yes, if ingested | overrides `slices` to drop `per_task` |

Points to slow down on:

- **`baseline_extended.yaml` resumes weights, not optimizer state.** Its header says the reduced LR is "freshly-warmed-up" (`warmup_steps: 20`, `learning_rate: 3.0e-5`). That's consistent with `Trainer` writing `"optimizer_state_saved": False` into each checkpoint (Module 6). The optimizer's moment estimates are lost on resume, which is *why* a fresh warmup is needed: Adam-style optimizers with zeroed second moments take large steps at first. The comment and the code agree, and it's a genuinely good decision. Learning-rate *schedule* semantics after warmup (API_REFERENCE says "cosine decay") live in the engine and are unverified.
- **The extended run builds a *new* `training/vN`** (`intervention: kind: full`), not the same one baseline_full used. "The test set is identical" therefore depends on `DatasetBuilder` assigning the same splits given the same episodes, seed (42) and mode. That's plausible for a deterministic seeded builder, and the README asserts it, but it isn't checked by any code here. A test comparing `split_integrity.json` split membership across the two runs would prove it. The README shows split *sizes*, not membership.
- **`aloha_coffee_baseline.yaml`'s hypothesis says "generalizes without code changes".** That's a claim about the engine's adapter and `fit()`, which "read camera/action shape from the dataset contract". The smoke test `test_policy_features_concatenate_every_inline_modality_like_state_adapter_does` exists because ALOHA's two inline modalities (28 state dims) exposed a mismatch. So it generalized *after* a fix, which is a better story to tell than "without changes".
- **`baseline_extended.yaml` duplicates many base values** (`model`, `pretrained_checkpoint`, `batch_size`, `checkpoint_every`, ...) that the deep merge would supply anyway. It's harmless, but it contradicts base.yaml's own guidance that "only the fields that differ need to appear there."

---

## 2.8 Honest gaps to own before an interviewer finds them

1. **Unknown keys are silently ignored, and the resolved config erases the evidence** (no `extra="forbid"` on any model). *Consequence:* a typo trains with the wrong hyperparameter, and the run record looks correct.
2. **`training.model` is a free string, and any value except `"smolvla"` silently skips training.** *Consequence:* a case typo produces a complete, exit-0 run with no fine-tuning, flagged only in the report's Limitations.
3. **Three dead fields: `num_workers`, `mixed_precision`, `success_tolerance`.** *Consequence:* the persisted config misdescribes the run, and API_REFERENCE documents an effect `success_tolerance` doesn't have.
4. **`imbalanced` with `variable` but no (or a wrongly typed) `variable_value` silently becomes a full-data run labelled imbalanced.** *Consequence:* the "never silently no-op" guarantee that rebalanced enforces doesn't hold for imbalanced.
5. **`experiment_id` and `created_at` are user-settable despite the "not user-editable" comment.** *Consequence:* ID collisions in the index are possible from a copied resolved config.
6. **`experiment_id` isn't content-addressed** (hyperparameters aren't in the hash). *Consequence:* there's no way to detect "already ran this exact config".
7. **The schema comment cites `model_training.experiments.selection`, which doesn't exist.** *Consequence:* a stale reference. The guarantee itself is real but lives in `Trainer`.
8. **Split fractions aren't bounded to [0, 1] and val can be 0.** *Consequence:* a zero val split silently disables checkpoint selection.
9. **The base-config location depends on directory convention, and there's no `--base` flag despite `configs/smolvla/so100_pickplace.yaml` saying there is.** *Consequence:* a config's meaning changes when the file moves, and the model-specific layer file is dead.
10. **Two sources of defaults (schema vs `base.yaml`) disagree** on steps, cadence, warmup, eval batch size and slices. *Consequence:* there's no single answer to "what's the default run?".
11. **`baseline_extended.yaml` hard-codes a dated, CWD-relative, git-ignored checkpoint path.** *Consequence:* it isn't runnable by anyone else, or by you after cleaning `runs/`.
12. **The flat intervention class accepts fields irrelevant to its `kind`** (`kind: full` plus `episode_ids`). *Consequence:* user intent is silently ignored.
13. **The 0.5 "reduced" default lives in two places.** *Consequence:* drift risk. The second copy is unreachable for validated configs.

---

## The shape to carry into Module 3

```
 configs/base.yaml ─┐  (found by path.parent.parent convention; skipped if absent)
                    ├─ _deep_merge (dicts merge, lists/scalars replace)
 configs/experiments/<name>.yaml ─┘
                    │  + source_config_path (forced)
                    ▼
        ExperimentConfig.model_validate        extra keys: IGNORED (!)
        ├── DatasetRef        (dataset_id, version)   identity, never a path
        ├── DatasetIntervention  kind ∈ {full,fraction,reduced,imbalanced,rebalanced,custom}
        │      after-validator: reduced→0.5; imbalanced/rebalanced need a declared target
        ├── SplitConfig       sums to 1 ± 1e-6 (no bounds), seed 42, quantile
        ├── TrainingConfig    model:str (!), 3 dead fields
        ├── EvaluationConfig  selection on val (enforced in Trainer, not "selection")
        └── identity: created_at=now(), experiment_id=name-sha256(name:ds:ver:ts)[:10]
                    │
                    ▼  resolved_dict() = model_dump(mode="json")
             config.resolved.yaml  (written into the run dir — Module 3)
```

---

## Drill — answer from memory, files closed

1. Walk through what happens, field by field, when `configs/experiments/reduced.yaml` is loaded. Where does `fraction: 0.5` come from, and where else does 0.5 appear?
2. *Why didn't you just* use Hydra or OmegaConf?
3. *What breaks in production:* a teammate writes `learning_rat: 3e-5` in a new experiment file. What happens, and what does the run directory say?
4. Why are there three separate seeds, and which one does `mtrain train --seed 1` change?
5. *Interviewer framing:* "How do you make sure an experiment can't silently run as something other than what was intended?"

---

# Appendix A — Model answers to the drill

**A1. Loading `reduced.yaml`.**
"`load_experiment_config` reads the file, finds `configs/base.yaml` by going two directories up, and deep-merges the experiment file over the base. Nested dicts merge key by key; lists and scalars replace. So `dataset`, `splits`, `training` and `evaluation` all come from base, and `intervention`, `experiment_name` and `tags` come from the experiment. It forces `source_config_path`, then calls `ExperimentConfig.model_validate` once on the merged dict. In this file `fraction: 0.5` is set explicitly. If it weren't, the `DatasetIntervention` after-validator would set it, because `kind: reduced` with no fraction and no episode IDs defaults to 0.5. There's a second 0.5 fallback in `resolve_episode_subset` that validated configs can never reach, which is a small duplication I'd remove. Finally the identity validator stamps `created_at` and an ID of the form `reduced_data-` plus ten hex characters of a SHA-256 over name, dataset, version and timestamp."

*Why that answer lands:* it follows the real order of operations (merge, then validate, then stamp) and volunteers the duplicated default. That shows you've read the code path, not the docstring.

**A2. Why not Hydra?**
"The surface I needed was small: one base layer, one override layer, and strong semantic validation, like 'an imbalanced run must name the dimension it skews'. Pydantic gives me the semantic validators, and the merge is ten lines. Hydra would add command-line overrides and multirun sweeps, but here every run is meant to correspond to a file, because the config file *is* the experiment record and carries a hypothesis and an objective. The cost is no `training.lr=...` overrides and a directory-convention base lookup that changes meaning if you move a file. I'd move to Hydra, or at least OmegaConf with structured configs, once hyperparameter sweeps became routine rather than data-fraction sweeps, which the code already handles by generating configs programmatically."

*Why that answer lands:* it justifies the choice by what the configs are *for* (an experiment log), admits the concrete cost, and names the trigger for switching.

**A3. The `learning_rat` typo.**
"It loads without error, because none of the Pydantic models set `extra='forbid'`, so unknown keys are dropped. `learning_rate` keeps the base value of 1e-4, and the run trains at the wrong rate. The worse part is the record: `config.resolved.yaml` is a dump of the validated model, so the typo is gone. The run directory says `learning_rate: 1e-4`, which is true, and there's no trace that you asked for 3e-5. That's the most important one-line fix in the project: `model_config = ConfigDict(extra='forbid')` on every config model. Together with turning `training.model` into a `Literal`, it closes the two ways a config can silently mean something other than what was written."

*Why that answer lands:* it identifies both the failure and the *evidence-erasure*, which is the subtler point, and gives the exact fix. It sounds like someone who has debugged this kind of bug.

**A4. Three seeds.**
"The split seed (42) decides which episodes are in train, val and test. Keeping it fixed is what lets two runs share an identical test set. The intervention seed decides which episodes a fraction draw keeps, so a reduced run is a reproducible random subset. The training seed decides sampling order and initialization noise during fine-tuning. Separating them lets me vary one source of randomness while holding the others fixed. Honestly, `mtrain train --seed 1` changes none of them. It only names the run directory `seed_1` and records 1 in the summary, while `training.seed` stays at 0. That's a real bug. The flag should set `training.seed`, and probably loop over the config's `seeds` list, which today ignores everything after the first entry."

*Why that answer lands:* it explains the design reason for separating seeds, then volunteers a real bug in your own code before being caught. That's the single most credibility-building move available.

**A5. "How do you make sure an experiment can't silently run as something other than intended?"**
"At the config layer there are semantic validators. An imbalanced intervention can't load without naming the variable it skews, a rebalanced one can't load with empty weights, and split fractions must sum to one. The shipped templates are built to fail before any training step. Downstream, the resolved config, including every default, is persisted with the run. But I have to be straight about the holes I found reviewing it. Unknown keys are ignored, so typos vanish. The model name is a free string, and anything but `smolvla` skips training while still producing a complete run. Three fields, including `mixed_precision`, are validated but never used. And an imbalanced intervention with a variable but no value resolves to the full dataset. The fixes are `extra='forbid'`, `Literal` types for model and metric names, deleting or wiring the dead fields, and the same 'resolved to the full set' guard that rebalanced already has."

*Why that answer lands:* it leads with what works and why, then lists specific, ranked holes with specific fixes. The interviewer learns you can audit your own system, which is what the question is really probing.

---

# Module 2 Summary — the one page to revise

**What this module is.** A two-layer YAML config (base plus experiment) deep-merged and validated once by Pydantic into an `ExperimentConfig` whose resolved dump is persisted with every run. Semantic validators enforce the project's methodology (declare the weak dimension); the type surface is too permissive to guarantee the config is truthful.

**Five claims to make cold**
1. Merge first, validate once. Lists replace, dicts merge, and the resolved config, defaults included, is persisted.
2. Data is named by `(dataset_id, version)` and never by path. The input is `clean/v1`, and each run builds its own `training/vN`.
3. Imbalanced and rebalanced interventions can't load without a declared target. "The weak dimension must not be presumed" is enforced as a validation error.
4. There are three seeds: split 42, intervention 0, training 0. The split seed is what keeps test sets identical across runs.
5. The biggest gap is no `extra="forbid"`. Typos vanish, and the resolved record hides them.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Hand-rolled deep merge plus Pydantic | Hydra/OmegaConf | small surface; file = experiment record | no CLI overrides; location-dependent base |
| Version identity | filesystem path | immutability, traceable lineage | can't train on un-ingested data |
| Flat intervention class with a `kind` tag | discriminated union | one simple class | irrelevant fields silently accepted |
| Mutating after-validator (`reduced`→0.5) | explicit fraction required | ergonomic shorthand | default duplicated in resolver |
| `name-sha256(name:ds:ver:ts)[:10]` | `uuid4` plus config hash | readable and unique | not content-addressed; can't detect re-runs |
| Exact (`quantile`) splits | independent draws | n = 48 makes independent sizes vary ±2 | none at this scale |
| Fresh warmup on resume | restoring optimizer state | optimizer state isn't saved | lost Adam moments; LR schedule restarts |

**Volunteer unprompted:** no `extra="forbid"`; `model: str` silently skips training; `num_workers`, `mixed_precision` and `success_tolerance` are dead; imbalanced can silently no-op; `experiment_id` is user-settable; there's no `--base` flag despite a config comment; the `selection` module citation is stale; split fractions are unbounded.

**Numbers and names:** split 0.8/0.1/0.1, seed 42, `quantile` → 38/5/5 on 48 episodes and 32/4/4 on 40; base training 1000 steps, batch 4, lr 1e-4, wd 1e-5, warmup 50, ckpt/val every 100, `val_batches` 8, keep 3, clip 1.0; extended lr 3e-5, warmup 20; eval batch 16, tolerance 0.1; sum tolerance 1e-6; ID hash 10 hex characters (40 bits).

**Sentences worth memorizing**
- "The config file is the experiment record. It carries the hypothesis as well as the hyperparameters."
- "An imbalanced run that doesn't name its dimension is a validation error, not a judgement call."
- "The resolved config is persisted so a run never depends on an unwritten default. The gap is that it also can't show a key you mistyped."
- "The split seed is separate from the training seed so the test set stays fixed while training varies."
