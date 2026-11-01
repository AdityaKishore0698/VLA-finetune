# Module 1 — Foundations: the domain, the boundary, and the package

**Files covered:** `pyproject.toml`, `.gitignore`, `src/model_training/config/__init__.py`, `src/model_training/data/__init__.py`, `src/model_training/deployment/__init__.py`, `src/model_training/evaluation/__init__.py`, `src/model_training/experiments/__init__.py`, `src/model_training/feedback/__init__.py`, `src/model_training/reports/__init__.py`, `src/model_training/tracking/__init__.py`, `src/model_training/training/__init__.py`

**Where we are:** This is the first module, so there is nothing before it. Its job is to give you the vocabulary for the rest of the manual and to show exactly where this repository stops and its sibling, the Episodic Data Engine (`rde`), starts. That boundary is the most important structural fact about the project, and getting it wrong in an interview means claiming work this repository doesn't contain.

**The one-line thesis:** `vla-finetune` is a thin, import-light orchestration layer of about 2,700 lines. It owns *experimental procedure* (config, interventions, validation cadence, selection, reporting, provenance) and delegates every *mechanism* (data storage, transforms, the training step, scoring, failure detection) to a separately installed engine.

---

## 1.1 Prerequisites — the absolute basics

### Robot learning vocabulary

- **Demonstration / episode.** One recorded attempt at a task, usually a human teleoperating a robot. It's a time-ordered sequence of **frames**. "Episode" is the unit of data in this codebase. Splits, interventions, failure reviews and weights are all per-episode.
- **Frame.** One timestep in an episode: the camera image(s), the robot's **state** (joint positions, sometimes efforts) and the **action** commanded at that moment. Code keys look like `state.joint`, `action.joint` and `rgb.top` (these appear in `tests/smoke/test_smolvla_smoke.py`).
- **Action.** A vector of numbers sent to the robot. For SO-100 it has 6 dimensions (5 joints plus the gripper). For bimanual ALOHA it has 14 (two 7-dimensional arms). The configs record this: the `aloha_coffee_baseline.yaml` header says "14-DoF action across two arms, four cameras".
- **DoF (degrees of freedom).** The number of independently controllable joints.
- **Embodiment.** The physical robot body: its kinematics, cameras and action space. "Cross-embodiment" means the same pipeline or model works across different bodies. The config tag `embodiment: so100` or `aloha` records it.
- **Behaviour cloning (imitation learning).** Training a policy to reproduce the demonstrated action given the observation. It's supervised regression from observation to action. Everything here is behaviour cloning.
- **Policy.** A function from observation to action. The trained model is a policy.
- **VLA (Vision-Language-Action model).** A policy whose inputs are camera images plus a natural-language instruction ("Pick up the cube and place it in the box." is literally a task string in `tests/conftest.py`) plus robot state, and whose output is actions. It's usually built by attaching an action-generating head to a pretrained vision-language model.
- **SmolVLA.** The specific VLA used here, loaded from the Hugging Face checkpoint `lerobot/smolvla_base` (`configs/base.yaml`). Per its public release, which this repository doesn't contain and I can't verify from here, it's a small VLA from the LeRobot project. Its action expert generates *chunks* of future actions with **flow matching**. The repository's own evidence that actions are stochastic is the README's §2.3 note that re-scoring the same frames moves MSE by up to 5%. The evidence for chunking is the smoke-test comment about an "action-queue bug" where a cached action chunk was reused across batches.
- **Flow matching.** A generative-modelling technique: the network learns a vector field that transports random noise into a sample of the target distribution (here, an action chunk). What matters for this project: **prediction is stochastic**. Score the same input twice and you get different actions. That fact matters later for checkpoint selection (Module 6) and for how to read small metric differences (Module 7).
- **LeRobot.** Hugging Face's robot-learning library and dataset format. The source datasets (`lerobot/svla_so100_pickplace`, `lerobot/aloha_static_coffee`) are LeRobot datasets, and the test fixture fakes one with `rde.testing.make_lerobot_dataset`.
- **Fine-tuning.** Continuing to train a pretrained model on new task data. "Fine-tuning SmolVLA" means starting from `smolvla_base` weights and taking gradient steps on so100 or aloha episodes.

### Evaluation vocabulary

- **Offline (open-loop) evaluation.** Feed recorded observations, compare predicted actions with the recorded human actions, and compute error. The robot never acts, so errors can't compound. **This is the only kind of evaluation the project runs.**
- **Closed-loop rollout.** Let the policy actually control a robot or simulator and measure task success. Errors compound because the policy's own actions change what it sees next. The code *refuses* to fake this: `mtrain rollout` writes `"status": "unavailable"`.
- **Why the distinction matters:** a low offline MSE doesn't imply a high success rate. A policy can match demonstrations on average and still fail at the one moment that matters, such as the grasp. Every claim the project makes is offline, and the code is built to keep the two from being confused.
- **MSE in normalized units.** Actions are normalized per dimension (subtract the mean, divide by the standard deviation; the test contract fixtures carry `normalization: {mean, std}`). An MSE of 1.0 means about one standard deviation of error. That's why the mean-action baseline sits near 1.0 (README Table 3: 0.908 and 1.010). Predicting the mean leaves roughly the variance.
- **Baseline.** A trivial model the real model must beat. Here there are two: `mean_action` (always predict the training mean) and `nearest_neighbor` (return the action of the most similar training frame, i.e. retrieval). Both are implemented in `rde`, not here.
- **Train/val/test split.** The **validation** split is used during training to pick a checkpoint. The **test** split is touched once, at the end, to report. Splits are **episode-level**: every frame of an episode goes to the same split. If frames from one episode landed in both train and test, near-identical neighbouring frames would leak and inflate the scores.
- **Leakage.** Any path by which test information influences training or selection.
- **Checkpoint.** A saved snapshot of model weights at a given step.
- **Checkpoint selection.** Choosing which snapshot to report, by a validation metric and never by test.

### Data-engineering vocabulary

These terms are defined by the sibling engine. They're inferred from call sites here, not read from the engine's source.

- **Dataset version.** An immutable, named snapshot such as `raw/v1`, `clean/v1` or `training/v6`. The prefix is a **stage** (`rde.schemas.lineage.VersionStage`, with at least `CLEAN` and `TRAINING`, used in `interventions.py`).
  - `raw/vN` is ingested source data.
  - `clean/vN` is the output of transforms (normalization, episode filtering).
  - `training/vN` is a built training set *with split assignments*.
- **Catalog.** The engine's metadata store: which versions exist and which episodes they contain (`catalog.get_manifest`, `catalog.get_episodes`). It's located by `settings.paths.catalog_db`, and the name suggests a database file. I haven't verified the engine.
- **Lineage.** The recorded chain of operations that produced a version (`ingest → transform → build_training_dataset`). The README shows real output of `rde version lineage`.
- **Training contract.** A dict describing a version's modalities: `observation` (each key with a `plane`, `shape` and optional `normalization`) and `action` (`key`, `shape`, `normalization`). The **plane** says where a modality physically lives. `inline` means numeric columns stored with the frame (state). `external` means stored outside the table (images or video). This repo branches on `plane == "inline"` and `plane == "external"` in `profile.py`, `deployment/loader.py` and `reports/model_card.py`.
- **DatasetView.** The read handle on a version. It has `episode_ids(split=...)`, `episodes`, `contract`, `iter_samples`, `get_episode_frames`, `splits()` and `close()`, and it's a context manager (`with engine.get_dataset(...) as view` in `profile.py`).
- **Adapter.** Converts a list of samples into model input tensors (`VisionLanguageAdapter`, `StateAdapter`). An adapter is the "preprocessing" stage.

### Python packaging vocabulary

- **`pyproject.toml`.** The standard (PEP 621) file declaring a package's name, dependencies and build system.
- **Build backend.** The tool that turns source into an installable wheel. Here that's `hatchling`.
- **Distribution name vs import name.** You `pip install vla-finetune` but `import model_training`. The two names are independent.
- **Console script.** `[project.scripts] mtrain = "model_training.cli:main"` makes installation create an `mtrain` executable that calls `main()`.
- **Extras / optional dependencies.** `pip install pkg[dev]` adds a named group.
- **Editable install.** `-e`: the installed package points at your source tree, so edits take effect without reinstalling.
- **`src/` layout.** Code lives under `src/` so tests can't accidentally import the uninstalled working copy.
- **Namespace package.** A directory with **no `__init__.py`** that Python still imports (PEP 420), merging every same-named directory on `sys.path`. This matters below.
- **`__all__`.** The list of names a package declares public, and what `from pkg import *` exports.
- **Import-light discipline.** Not importing heavy libraries (torch) at module import time, so cheap commands and tests run without them. It's done with **lazy imports**, meaning imports inside functions.

---

## 1.2 What this repository is, measured

| Thing | Measured value |
|---|---|
| Source lines (`src/`) | about 2,700 across 28 files |
| Largest modules | `runner.py` 314, `cli.py` 306, `profile.py` 303, `trainer.py` 256, `interventions.py` 212 |
| Test functions | 49 in `tests/unit`, 3 in `tests/integration`, 4 in `tests/smoke`. The README's "52 unit + integration tests" (line 282) **matches** 49 + 3 |
| CI | **none**: no `.github/`, no other CI config |
| Lockfile | **none**: no `uv.lock` and no `requirements*.txt` |
| Engine present in this checkout | **no**: `import rde` fails, and `../episodic-data-engine` doesn't exist here |

That last row shapes the whole manual. **I couldn't run any test or command while writing it.** Every statement about what `rde` does is either inferred from a call site here (and marked as such) or quoted from this repo's docs (and treated as a claim).

---

## 1.3 The boundary: what `rde` owns and what this repo owns

Here is every `rde` symbol imported by `src/`, grouped by who imports it (from grep):

| `rde` symbol | Imported by (this repo) | What it does for us |
|---|---|---|
| `catalog.Catalog`, `config.Settings/get_settings` | runner, interventions, profile, feedback, cli | locate and read versions |
| `engine.DataEngine`, `engine.DatasetView`, `engine.Sample` | runner, profile, split_integrity, trainer, offline, loader, cli | open a version for reading |
| `transformations.TransformPipeline`, `EpisodeSubsetFilter` | interventions | **create** a `clean/vN` by dropping episodes |
| `dataset_builder.DatasetBuilder`, `build_training_contract` | interventions, profile | **create** a `training/vN` with splits and weights |
| `versioning.next_version`, `schemas.lineage.VersionStage` | interventions | allocate the next version name |
| `exploration.queries.DatasetExplorer` | profile, cli | stats, histograms, raw SQL over frames |
| `model_adapters.models.SmolVLAPolicy` | trainer, runner, deployment artifact and loader, cli | **the model, including `fit()`: the training step itself** |
| `model_adapters.VisionLanguageAdapter`, `MODEL_REGISTRY`, `build_model` | trainer, runner, offline, loader, cli | preprocessing and baselines |
| `evaluation.Evaluator`, `EvaluationResult` | trainer, offline, feedback | scoring predictions against demonstrations |
| `feedback.FeedbackLoop` | feedback | failure detection and episode nomination |

So the split is:

```
              this repo (model_training)                          rde (episodic-data-engine)
  ┌──────────────────────────────────────────────┐      ┌─────────────────────────────────────┐
  │ WHAT to run: config schema, merge, identity  │      │ storage, catalog, versions, lineage │
  │ WHICH episodes: intervention resolution      │─────▶│ transforms, DatasetBuilder, splits  │
  │ WHEN to validate/checkpoint/prune/select     │─────▶│ SmolVLAPolicy.fit (one grad step)   │
  │ WHAT to score: baselines, slices, tails      │─────▶│ Evaluator (pred vs demo)            │
  │ HOW to package: run dir, index, reports      │─────▶│ FeedbackLoop (failure detection)    │
  └──────────────────────────────────────────────┘      └─────────────────────────────────────┘
```

**Decision: put the model adapter, including `SmolVLAPolicy.fit`, inside the *data* engine.**
- *What:* the class that wraps SmolVLA, builds batches and runs forward/backward lives in `rde.model_adapters.models`, not here. `docs/ARCHITECTURE.md` line 35 says so outright: `SmolVLAPolicy` was "Extended in place: `fit()` now performs real gradient steps; `save_checkpoint()` and `from_checkpoint()` were added."
- *Why (inferred):* the adapter has to read the engine's training contract to build feature shapes and normalization statistics. The smoke test calls rde's private `_policy_features_from_contract` and `_dataset_stats_from_contract`. Keeping the model next to the contract keeps those two in lockstep.
- *Alternative:* the model lives in this repo, and the engine exposes only data, a contract and an iterator.
- *Cost:* **the dependency direction is inverted relative to the README's slogan.** The README says "the model layer never owns the data". That's true, but the *data* layer owns the model. The most technically demanding code (batch construction, optimizer, loss, the action-queue fix) isn't in this repository. In an interview, say plainly: "the training step lives in the engine; this repo owns the run around it." Don't claim the optimizer loop as code in this repo.
- *When the answer flips:* once a second model family arrives (a diffusion policy, ACT), its adapter would also have to go into the data engine. At that point a separate `model_adapters` package depended on by both repos is the right shape.

---

## 1.4 `pyproject.toml`, line by line

```toml
[project]
name = "vla-finetune"
version = "0.1.0"
description = "Experiment, training, evaluation, deployment and feedback layer on top of episodic-data-engine"
license = "MIT"
authors = [{ name = "Kushagra Srivastava" }]
requires-python = ">=3.10"
```

- `requires-python = ">=3.10"`: the code uses `X | None` union syntax in annotations at runtime-evaluated positions (for example, Pydantic field types like `float | None` in `schema.py`). Pydantic evaluates those, so Python 3.10 is a real floor and not an arbitrary one. The README installs with 3.11 (`uv venv --python 3.11`), so 3.10 itself is **declared but never shown to have been tested**.
- `authors` and `[project.urls] Repository = "https://github.com/kushagrathisside/VLAFinetuningExperiments"` name a person and a repository other than the checkout this manual was written from. If you present this project as yours, have a one-sentence account of the relationship (fork, collaboration, renamed), because an interviewer who opens the repo will see it.

```toml
dependencies = [
    "episodic-data-engine",
    "pydantic>=2",
    "pyyaml>=6",
    "typer>=0.9",
    "rich>=13",
    "psutil>=5",
    "matplotlib>=3.8",
    "numpy>=1.24",
]
```

- **Floors only, no ceilings, no lockfile.** `pydantic>=2` admits any future major version. The code uses v2-only API (`model_validator`, `model_validate`, `model_dump(mode="json")`), so the floor is load-bearing, but nothing stops a breaking v3.
  - *Alternative:* a committed `uv.lock`.
  - *Cost:* two installs months apart can resolve different versions. The project compensates *after the fact* by recording versions into `environment.json` (Module 3). That records drift but doesn't prevent it.
- `episodic-data-engine` with **no version constraint at all.** The repository is coupled to the engine's *private* functions (the smoke test imports `_policy_features_from_contract`), yet it declares no compatible engine version. The only real pin is the git commit recorded per run in `git.json` (Module 3).
- `psutil` is a **hard** dependency, yet `trainer.py` and `run_dir.py` both wrap `import psutil` in `try/except ImportError` marked `# pragma: no cover`. A declared dependency can't be missing in a correct install, so those branches are dead defensive code. They're harmless, but don't call them "graceful degradation".
- `matplotlib` is a hard runtime dependency of a library, used only by `reports/figures.py`. That's acceptable because every run renders figures. It does make the "light" environment heavier than it needs to be.

```toml
[project.optional-dependencies]
# The heavy VLA stack (torch, lerobot) is pulled in transitively through
# episodic-data-engine's own `vla` extra -- installed once, in .venv-vla, per
# the engine's own import-light discipline. Nothing here re-declares torch.
dev = ["pytest>=8", "pytest-cov>=5"]
```

**Decision: torch and lerobot are not declared here at all.**
- *What:* there's no `vla` extra on this package. You get torch only by installing `episodic-data-engine[vla]`.
- *Why:* single ownership. The engine's `SmolVLAPolicy` is the only code that needs torch, so the engine declares it, and two packages can't pin conflicting torch versions.
- *Alternative:* `vla-finetune[vla] = ["episodic-data-engine[vla]"]`, a pass-through extra that would make `pip install vla-finetune[vla]` work in one step.
- *Cost:* **`pyproject.toml` alone gives no way to get a working training environment.** You have to know to install the engine's extra, which only the README tells you (line 177). And `trainer.py` and `run_dir.py` do `import torch` directly, so this package *does* use torch without declaring it. For `trainer.py` the comment covers it ("safe: policy.load() above already required it to succeed"). It's still an undeclared direct dependency.

`pytest-cov` is declared, but `addopts = "-q"` doesn't enable coverage, and there's no coverage config. Nothing in the repo runs it, so it's an unused dependency.

```toml
[project.scripts]
mtrain = "model_training.cli:main"
```

The docstring in `cli.py` explains the name: "`mtrain`, not a bare `model`, to avoid colliding with the engine's own `rde model ...` sub-app". That's a real rationale, and it's verifiable because `cli.py` exists and is the target.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/model_training"]

[tool.hatch.metadata]
allow-direct-references = true
```

- `packages = ["src/model_training"]` tells hatch where the import package is. That's needed because the import name differs from the distribution name.
- `allow-direct-references = true` permits dependencies written as direct URLs (`pkg @ file:///...` or `git+https://...`). **No dependency in this file is a direct reference.** `episodic-data-engine` is a bare name. The setting is vestigial, probably left over from an earlier `episodic-data-engine @ file:../...` form. Harmless, but it's a loose thread.

```toml
[tool.uv.sources]
episodic-data-engine = { path = "../episodic-data-engine", editable = true }
```

**Decision: resolve the engine from a sibling directory via `uv`.**
- *What:* when installed with `uv`, the bare dependency name resolves to the local checkout `../episodic-data-engine`, editable.
- *Why:* the two repos are developed together, so engine edits (like the in-place `SmolVLAPolicy` extensions) take effect immediately.
- *Alternative:* publish the engine to PyPI or a private index and pin a version, or vendor it as a git submodule.
- *Cost:* `[tool.uv.sources]` is **a uv-only table**. Plain `pip install -e .` ignores it and tries to fetch `episodic-data-engine` from PyPI. I couldn't verify whether a package of that name exists there. If it doesn't, the install fails; if a different package squats the name, it installs the wrong code. The install also needs a specific *directory layout* (`../episodic-data-engine` next to this repo), which is why `runner.py` hard-codes `parents[4] / "episodic-data-engine"` (Module 11). The same layout assumption appears in three places, with nothing enforcing it.
- *When the answer flips:* once anyone other than the author installs it, or once CI exists. CI would need to check out both repos side by side.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`testpaths = ["tests"]` matters because there's a file called `scripts/test.py`. Without `testpaths`, pytest's default discovery pattern (`test_*.py` or `*_test.py`) wouldn't match `test.py` anyway, so this is belt-and-braces, not a fix for a real problem.

---

## 1.5 `.gitignore`: what the repo refuses to version

```
runs
experiments/*.jsonl
paper_ready/
...
.venv/

# Internal code-review audit -- kept locally as a private record, not published.
docs/CODE_REVIEW.md
```

- `runs` (no trailing slash, no leading slash) ignores **any file or directory named `runs` at any depth**. That's intended for `<repo>/runs/`, where `ExperimentRunner` writes by default.
- **Consequence: the README's results can't be reproduced from the repository alone.** Every run directory, checkpoint and metric file is git-ignored. `docs/figures/results.svg` is committed, but the `runs/...` inputs to `scripts/plot_results.py` aren't. The numbers in README Table 2 are claims backed by files that aren't in the repo. That's normal for ML (checkpoints are gigabytes), but say it before you're asked: "the run artifacts are local; the figure script regenerates from them."
- `experiments/*.jsonl` ignores the cross-run index (Module 3). The pattern is unanchored, so it would also match `src/model_training/experiments/*.jsonl`. No such files exist, so it's harmless.
- `.venv/` is listed, but the README creates both venvs at `../.venv` and `../.venv-vla`, *outside* the repo. The entry is defensive, so the missing `.venv-vla/` entry isn't a real gap. (I raised it as a possible gap in my first survey; on reading the README's install table it isn't one.)
- `docs/CODE_REVIEW.md` is deliberately excluded: there's an internal audit the public repository doesn't contain. Several code comments read like responses to it. For example, `trainer.py`'s crash handler says "A crash here ... previously left the run directory in an undetectable half-finished state", and `registry.py`'s `get` docstring says "returning the first match returned the oldest". Those are fixed bugs narrated in comments. In an interview this is useful: "I ran an adversarial review and fixed what it found; the comments record the before-state."

---

## 1.6 The package surface: the `__init__.py` files

### There is no `src/model_training/__init__.py`

```
$ ls src/model_training/__init__.py
ls: cannot access 'src/model_training/__init__.py': No such file or directory
```

Every *subpackage* has an `__init__.py`, but the top-level `model_training` doesn't. So `model_training` is an **implicit namespace package** (PEP 420).
- *Probably why:* it was never needed, and nothing reads `model_training.__version__`. `collect_environment` gets the version from installed metadata via `importlib.metadata.version("vla-finetune")` instead.
- *Alternative:* an `__init__.py`, which costs one empty file.
- *Cost:* a namespace package **merges** every directory named `model_training` on `sys.path`. Any other installed project with a top-level `model_training` directory, which is a generic name, would silently interleave with this one, and a `model_training/cli.py` from the wrong tree could be imported. It's also a less common setup, which some tooling (mypy without `namespace_packages`, older coverage configs) handles worse. This looks accidental rather than designed; nothing in the code or docs mentions it.

### The nine subpackage `__init__.py` files are re-export façades

Each one re-exports a curated public API with an explicit `__all__`, for example:

```python
# src/model_training/deployment/__init__.py
from .artifact import ModelArtifact, build_model_artifact
from .loader import DeployedModel, DeploymentHealthError, LatencyBreakdown
from .registry import ModelRegistry

__all__ = ["DeployedModel", "DeploymentHealthError", "LatencyBreakdown",
           "ModelArtifact", "ModelRegistry", "build_model_artifact"]
```

- **Pattern:** *Façade* at the package level. Callers write `from model_training.deployment import DeployedModel` and don't depend on the file layout.
- **Why:** module files can be reorganized without breaking importers.
- **Cost 1: being exported is not being used.** The façades make names look first-class. Grep shows several exported names that **nothing in `src/` or `scripts/` calls**:

| Exported name | Used by `src/`/`scripts/`? | Used by tests? |
|---|---|---|
| `deployment.ModelRegistry` | **no** (the runner writes models via `ExperimentIndex.record_model`, not the registry) | yes, `test_deployment.py` |
| `deployment.DeployedModel` | **no** (only named in a model-card string) | yes |
| `experiments.run_data_fraction_sweep` | **no** (no CLI command calls it) | needs checking in Module 13 |
| `reports.build_model_dataset_matrix` | **no** | no |

  These are covered properly in Modules 9, 10 and 11. For now the point is that **the public API is wider than the wired system.**
- **Cost 2: eager imports defeat laziness.** Importing a subpackage executes every module it re-exports. `from model_training.experiments import ExperimentRunner` imports `runner.py`, which at module top imports `..deployment.artifact`, which at module top does `from rde.model_adapters.models import SmolVLAPolicy`. So *any* import of the runner imports the engine's SmolVLA module. The runner's careful lazy `from ..training.trainer import Trainer` inside `if config.training.model == "smolvla":` **saves nothing on the rde side**, because `deployment/artifact.py` has already pulled in the same `models` module. It works in the light venv only if the engine's `models.py` itself imports torch lazily. The trainer comment ("`import torch # safe: policy.load() above already required it`") implies it does, and the README says `test_experiment_runner.py` passes without torch. **I couldn't verify this without the engine.** If the engine ever adds a top-level `import torch` to `models.py`, the whole light test suite breaks, and nothing in this repo would show why.

### The CLI is the one place laziness is done right

`cli.py` imports only `typer`, `yaml` and `rich` at the top. Every `rde` and `model_training` import sits inside a command function, so `mtrain --help` and `mtrain rollout` don't load the engine at all. That's the import-light discipline applied where it pays off most: a CLI should start fast and fail only for the command you actually ran. Module 12 covers it.

---

## 1.7 The two-environment design (from the README, partly verified)

The README (lines 176–177) and `docs/ARCHITECTURE.md` (lines 143–144) describe:

| Venv | Contents | What runs there |
|---|---|---|
| `../.venv` (light) | engine core plus this package, no torch | config, profiling, interventions, tracking, reports, CLI, unit and integration tests, baselines |
| `../.venv-vla` (full) | `episodic-data-engine[vla]` (torch, lerobot) | real fine-tuning, checkpoint evaluation, the smoke test |

**What the code confirms:** the smoke test begins `torch = pytest.importorskip("torch")` and `pytest.importorskip("lerobot")`, so it skips cleanly in the light venv. The integration test docstring says it "Uses the baseline models (no torch needed)". The runner test uses `"training": {"model": "mean_action"}` precisely so that "the whole runner ... without requiring torch/lerobot" runs.

**Decision: two environments rather than one.**
- *Why:* torch plus lerobot is gigabytes, and CUDA builds are platform-specific (`docs/IMPLEMENTATION_PLAN.md` line 254 records reinstalling torch with CUDA). Fifty-two tests that need no GPU shouldn't pay for that.
- *Alternative:* one environment with torch always installed, plus test markers.
- *Cost:* two environments drift independently, and **nothing tests that the light environment stays light**. No test asserts `"torch" not in sys.modules` after importing the runner. The design is a convention, not an enforced invariant.

---

## 1.8 Honest gaps to own before an interviewer finds them

1. **The training step isn't in this repository.** `SmolVLAPolicy.fit`, batch construction, the optimizer and the loss all live in `rde.model_adapters.models`. *Consequence:* claiming "I wrote the fine-tuning loop" is false for this repo. What this repo owns is the run around it (Module 6).
2. **Nothing can be run from this checkout.** The engine isn't present, `[tool.uv.sources]` needs a sibling directory, and plain `pip` ignores that table. *Consequence:* a reviewer can't reproduce anything without reconstructing your directory layout.
3. **No lockfile, floors-only versions, and an unpinned engine.** *Consequence:* reproducibility depends on after-the-fact `environment.json` and `git.json` capture (Module 3), which records drift but doesn't prevent it.
4. **Coupling to the engine's private API.** `tests/smoke/test_smolvla_smoke.py` imports `_policy_features_from_contract` and `_dataset_stats_from_contract`. *Consequence:* an engine refactor that's legitimate from the engine's side breaks this repo's tests, and the missing version constraint means nothing warns you.
5. **No CI.** *Consequence:* the "52 tests pass" claim is a statement about the author's machine on some date. It isn't continuously enforced.
6. **The top-level package has no `__init__.py`** (implicit namespace package under a generic name). *Consequence:* there's a silent import-merging risk with any other `model_training` on the path.
7. **Light-environment laziness is unenforced** and partly illusory. The runner's lazy trainer import is undone by `deployment/artifact.py`'s eager import. *Consequence:* a top-level torch import added in the engine would break the light suite with no local cause.
8. **Unused declarations.** `allow-direct-references = true` (no direct references), `pytest-cov` (never invoked), `psutil` `ImportError` fallbacks (it's a hard dependency). *Consequence:* small, but each one is a "why is this here?" you should answer as "vestigial", not invent a reason for.
9. **The public API is wider than the wired system** (`ModelRegistry`, `DeployedModel`, `run_data_fraction_sweep`, `build_model_dataset_matrix`). *Consequence:* the package surface overstates what the pipeline actually exercises.
10. **Undeclared direct torch dependency.** `trainer.py` and `run_dir.py` import torch, but this package never declares it. It's deliberate, delegated to the engine's extra, and the README explains it, but the metadata alone is incomplete.
11. **The `pyproject` author and repository URL point elsewhere.** *Consequence:* you need a one-line provenance story ready.

---

## The shape to carry into Module 2

```
                   ┌──────────────────────── vla-finetune (model_training) ────────────────────────┐
  YAML configs ──▶ │ config ──▶ experiments.runner ──▶ data.{interventions,split_integrity,profile} │
                   │               │                 ──▶ training.trainer                           │
  mtrain (cli) ──▶ │               │                 ──▶ evaluation.offline ──▶ feedback.artifact   │
                   │               │                 ──▶ deployment.{artifact,registry*,loader*}    │
                   │               └───────────────▶ tracking.{run_dir,index} ──▶ reports           │
                   └────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │ imports (no version pin)
                   ┌────────────────────────────────▼──────────────────────────────────────────────┐
                   │ rde: Catalog · DataEngine/DatasetView · TransformPipeline · DatasetBuilder     │
                   │      SmolVLAPolicy.fit/predict/save · Evaluator · FeedbackLoop · baselines     │
                   └────────────────────────────────┬──────────────────────────────────────────────┘
                                                    │ torch, lerobot (only in .venv-vla)
                                                    ▼
                                        lerobot/smolvla_base weights
   * = exported but not called by src/
```

---

## Drill — answer from memory, files closed

1. What's the difference between offline evaluation and closed-loop rollout, and which one does this project do? Why can a model with low offline MSE still fail on the robot?
2. *Why didn't you just* put torch in this package's dependencies, or make one environment for everything?
3. Where does the gradient step actually happen, and what exactly does this repository own about training?
4. *What breaks in production* if a teammate runs `pip install -e .` in a fresh clone?
5. *Interviewer framing:* "Walk me through the dependency structure of this project. Is there anything about it you'd change?"

---

# Appendix A — Model answers to the drill

**A1. Offline vs closed-loop.**
"Offline evaluation feeds the model recorded observations from held-out demonstrations and compares its predicted action with what the human actually did, so it's regression error, in normalized action units here. Closed-loop rollout lets the policy drive a robot or simulator and measures whether the task succeeds. This project only does offline evaluation. There's no simulator or robot attached, and the rollout command writes 'unavailable' instead of inventing a number. Low offline MSE can still fail in closed loop for two reasons. First, compounding error: a small deviation puts the robot into states the demonstrations never visited, and the policy has never seen how to recover. Second, averaging: MSE is dominated by the many easy frames, and a policy can be accurate on average and wrong on the few frames that decide success, like the moment of grasp. That's why I report tails (p99 and max) and per-phase error rather than just the mean, and why I never call the tolerance metric a success rate."

*Why that answer lands:* it names both mechanisms (distribution shift and averaging) instead of saying "sim-to-real is hard", and it ties them to concrete design choices in the codebase. It also shows you know the limits of your own number before the interviewer has to point them out.

**A2. Why not declare torch here or use one environment?**
"Torch and lerobot are only needed by the SmolVLA adapter, which lives in the engine, so the engine owns that dependency through its `vla` extra. One owner, and no chance of two packages pinning conflicting torch builds. The two environments exist because about 52 of the tests and most of the commands (config, profiling, interventions, reports, baselines) don't need a GPU stack, and a CUDA torch install is several gigabytes and platform-specific. The cost is that it's a convention, not an invariant. Nothing asserts the light environment stays light, and in fact the runner transitively imports the engine's model module, so this relies on the engine importing torch lazily. I'd add a test that imports the runner and asserts torch isn't in `sys.modules`. I'd flip to one environment the moment CI runs on a GPU image anyway."

*Why that answer lands:* it gives the reason, then names the cost without being asked, then gives the concrete enforcement fix and the condition under which you'd abandon the design. That's the "no, until X" shape.

**A3. Where the gradient step happens.**
"The step itself (batch construction, forward, loss, backward, optimizer, warmup) is `SmolVLAPolicy.fit` in the data engine's `model_adapters` package. I extended that class there because it has to build feature shapes and normalization stats from the engine's training contract. This repository owns the run around the step. It passes an `on_step` callback that does periodic validation on the val split only, periodic checkpointing with verification, pruning that never deletes the best checkpoint, a forced final checkpoint, a crash marker, peak memory tracking, and freeing the GPU before the evaluation model loads. So the procedure is here and the mechanics are there."

*Why that answer lands:* it's precise about ownership and doesn't overclaim, which interviewers test for directly. And it turns "I didn't write the loop here" into a clean separation of mechanism and procedure, which is a real design idea.

**A4. What breaks with `pip install -e .`.**
"The engine dependency is declared by bare name, and the only thing pointing it at the local checkout is `[tool.uv.sources]`, which only uv reads. Plain pip would go to PyPI for `episodic-data-engine`. That either fails, or worse, installs whatever has that name. Even with uv it needs the engine checked out as a sibling directory, and the runner separately assumes that layout to find the engine's git commit. So a fresh clone isn't installable on its own. The fix is either publishing the engine with a version and pinning a compatible range, or a submodule plus a lockfile, and CI that does exactly the fresh-clone install so this is tested and not just documented."

*Why that answer lands:* it shows you know the difference between pip and uv semantics, spots the dependency-confusion risk, and ends on a fix that's verified by CI rather than by documentation.

**A5. "Walk me through the dependency structure. Anything you'd change?"**
"Two layers. The engine owns data: versions, lineage, transforms, the builder that assigns splits, the evaluator, failure detection, and also the SmolVLA adapter including `fit`. This package owns experimental procedure: a validated config, resolving an intervention into a lineage-tracked dataset version, the training cadence and checkpoint selection, stratified evaluation against baselines, and packaging everything into a run directory and an index. All rde imports in the CLI are lazy, so commands start fast. Three things I'd change. First, the model adapter living in the data engine inverts the dependency. It's fine with one model, but with a second model family I'd pull adapters into their own package. Second, the engine is unpinned and the smoke test imports two of its private functions, so I'd pin a version range and promote those functions to public. Third, I'd add a lockfile and CI that checks out both repos, because right now 'the tests pass' isn't continuously true."

*Why that answer lands:* it describes the structure in terms of responsibility, not files, then volunteers three ranked, concrete changes, each tied to a trigger. Being critical of your own architecture is what separates a designer from someone who only implemented it.

---

# Module 1 Summary — the one page to revise

**What this module is.** The vocabulary of VLA fine-tuning and offline evaluation, and the shape of the codebase: a thin procedural layer (`model_training`, about 2,700 lines) over a separately installed data engine (`rde`) that also owns the model and its training step.

**Five claims to make cold**
1. All evaluation is offline (predicted vs demonstrated action, normalized units). Rollout is explicitly reported as unavailable.
2. Splits are episode-level, to prevent near-duplicate-frame leakage.
3. `SmolVLAPolicy.fit` lives in the engine. This repo owns cadence, selection, checkpointing and packaging.
4. Torch is owned by the engine's `vla` extra. A light venv runs 52 tests without it, by convention and not by enforcement.
5. The repo isn't installable alone: there's a sibling-directory `uv` source, no lockfile, no CI, and an unpinned engine.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Model adapter and `fit` inside the engine | adapter in this repo | must share the training contract's shapes and normalization | data layer owns the model; hardest code isn't here |
| torch via `episodic-data-engine[vla]` | a `vla` extra here | single owner of the heavy stack | `pyproject` alone can't build a training env; undeclared direct torch import |
| Two venvs | one env | 52 tests don't need CUDA | unenforced "lightness" |
| `[tool.uv.sources]` sibling path | published, pinned engine | co-development with editable engine | pip-incompatible; layout assumption repeated in code |
| Re-export façades with `__all__` | deep imports | stable public API | eager imports defeat laziness; API wider than what's wired |
| `mtrain` script name | `model` | avoids `rde model` collision | none |

**Volunteer unprompted:** no CI; no lockfile; private-API coupling; run artifacts are git-ignored so README numbers aren't reproducible from the repo; no top-level `__init__.py`; authorship and URL point elsewhere.

**Numbers and names:** about 2,700 lines in `src`; 49 unit, 3 integration and 4 smoke tests (README's 52 matches); Python ≥3.10 declared, 3.11 used; `lerobot/smolvla_base`; datasets `so100_pickplace` (6-DoF, 2 cameras, 30 fps) and `aloha_coffee` (14-DoF, 4 cameras, 50 fps); console script `mtrain`; build backend `hatchling`.

**Sentences worth memorizing**
- "The engine owns the mechanics of one gradient step; this repo owns the procedure around it."
- "Every number I report is offline. The code refuses to produce a rollout number it can't measure."
- "Light-environment import discipline is a convention I'd turn into a test."
- "The run artifacts are deliberately not versioned; the figure script regenerates the README figure from them."
