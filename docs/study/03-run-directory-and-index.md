# Module 3 — The run directory and the cross-run index

**Files covered:** `src/model_training/tracking/run_dir.py`, `src/model_training/tracking/index.py`

**Where we are:** Module 2 produced a validated `ExperimentConfig`. Before any data is touched, the runner gives that config a place to live: a run directory, plus snapshots of the environment and git state, plus rows in a project-wide index. Every later module writes *into* the structure defined here, so the guarantees and non-guarantees of this module are inherited by everything after it.

**The one-line thesis:** Run tracking is plain files: one directory per run that refuses to be overwritten, plus four append-only JSONL files queryable without this code. The design is honest and tool-agnostic, but none of the writes are atomic, the index isn't transactional with the run, and the provenance it captures (git commit of both repos) can silently come out as `null` for the repo that matters most.

---

## 3.1 Prerequisites — the absolute basics

- **Experiment tracking.** Recording, for every training run, what went in (config, data, code version, environment) and what came out (metrics, checkpoints), so runs can be compared and reproduced. Off-the-shelf tools: MLflow, Weights & Biases, TensorBoard, DVC, Aim.
- **Run directory.** A folder holding one run's artifacts. Here it's `runs/<date>/<experiment_name>/seed_<seed>/`.
- **JSON / JSONL.** JSON is one structured document. **JSONL** (JSON Lines) is one JSON object per line. Its virtue is that you can *append* a record without rewriting the file, and tools like `jq`, `pandas.read_json(lines=True)` and DuckDB `read_json_auto` read it directly.
- **Append-only log.** A file that's only ever added to, never edited. History is preserved, and "the current value" for a key is the *latest* row for that key (the event-sourcing idea in miniature).
- **Atomic write.** A write that either fully happens or doesn't happen at all, so no reader ever sees half a file. The standard recipe is to write to a temporary file in the same directory, `fsync`, then `os.replace(tmp, target)`. `os.replace` is atomic on POSIX filesystems.
- **Torn write.** What you get *without* atomicity when a process dies mid-write: a truncated file or a half line.
- **TOCTOU (time-of-check to time-of-use).** A race where you check a condition ("is this directory empty?") and then act on it, and another process changes the condition in between.
- **`O_APPEND`.** The OS flag behind Python's `open(..., "a")`. Each `write()` syscall goes to the current end of file. On a local POSIX filesystem a single `write()` of a modest size won't interleave with another process's append. Python's buffered text I/O may split one logical line into several syscalls if it exceeds the buffer (8 KiB by default), and network filesystems (NFS) don't guarantee append atomicity at all.
- **Dirty working tree.** A git checkout with uncommitted changes (`git status --porcelain` non-empty). A commit hash alone doesn't identify the code that ran from a dirty tree.
- **CUDA compute capability vs driver version.** *Compute capability* (for example 8.9) is the GPU architecture's feature level. The *driver version* is the installed NVIDIA kernel driver. `torch.version.cuda` is a third thing: the CUDA toolkit version PyTorch was *compiled* against.

---

## 3.2 `RunDir` and `make_run_dir`: the no-overwrite rule

```python
@dataclass
class RunDir:
    root: Path
    experiment_name: str
    seed: int
    date: str

    _SUBDIRS = ("checkpoints", "predictions", "evaluation", "failures", "reports", "logs", "metrics")

    @property
    def path(self) -> Path:
        return self.root / self.date / self.experiment_name / f"seed_{self.seed}"
```

- `_SUBDIRS` has no type annotation, so `@dataclass` treats it as a **class attribute, not a field**. That's intentional: it's a constant of the layout, not per-instance state.
- **Layout `date / name / seed`.** It groups by day first, which suits a researcher scanning what they ran today. *Alternative:* `name / date / seed` groups all runs of one experiment together, which suits comparison. *Cost of the choice:* every reference to a run embeds its date. `baseline_extended.yaml` hard-codes `runs/2026-09-03/...` (Module 2), and `scripts/plot_results.py` needs three separate dated paths.
- **`seed` in the path.** Multi-seed runs of the same experiment sit side by side. As Module 2 showed, the `seed` here is *the CLI/summary seed*, which doesn't control `training.seed`. So `seed_1/` and `seed_0/` from `mtrain train --seed 1` vs the default **differ only in name** (Module 11 covers this in detail).

```python
def make_run_dir(base_runs_dir, experiment_name, seed, date=None) -> RunDir:
    date = date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    rd = RunDir(root=Path(base_runs_dir), experiment_name=experiment_name, seed=seed, date=date)
    if rd.path.exists() and any(rd.path.iterdir()):
        raise FileExistsError(
            f"run directory already exists and is non-empty: {rd.path} -- "
            "never silently overwrite an experiment run"
        )
    rd.ensure()
    return rd
```

**Decision: refuse a non-empty run directory and never overwrite.**
- *What:* if `runs/<today>/<name>/seed_<n>/` has anything in it, raise.
- *Why:* overwriting would destroy checkpoints and metrics. The guarantee is tested (`test_runner_refuses_to_overwrite_an_existing_run` expects `FileExistsError` on the second run).
- *Alternative 1:* auto-suffix (`seed_0_2`, or a timestamp or UUID directory). *Alternative 2:* MLflow-style opaque run IDs, with names as tags.
- *Cost:*
  1. **A failed run blocks its own retry for the rest of the (UTC) day.** The runner creates the directory and writes `environment.json`, `git.json` and `config.resolved.yaml` *before* the intervention step (Module 11). So a typo in a dataset version, a missing GPU, anything at all that fails, leaves a non-empty directory. The next attempt raises `FileExistsError`. The message explains what to do, which is the right behaviour for a guard, but there's no `--force` and no crash marker outside the trainer (Module 6 has `crash.json`; earlier failures have nothing).
  2. **The date is UTC.** A run started at 23:50 and retried at 00:10 UTC gets a *different* directory and doesn't collide. Collision behaviour therefore depends on the wall clock, which is surprising but harmless.
  3. **Check-then-create is a TOCTOU race.** Two processes launching the same experiment and seed at once can both see "empty", both call `ensure()`, and both write into the same directory. It's unlikely for a single researcher on one GPU. It's guaranteed to bite with a job scheduler. *Fix:* `path.mkdir(parents=True, exist_ok=False)` on the leaf, which is atomic at the filesystem level, instead of check-then-`exist_ok=True`.
- **An *empty* existing directory is reused on purpose** (`any(rd.path.iterdir())`), which is reasonable: an empty directory holds nothing to destroy.

```python
def ensure(self) -> Path:
    p = self.path
    p.mkdir(parents=True, exist_ok=True)
    for sub in self._SUBDIRS:
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p
```

Seven subdirectories are pre-created. **`logs/` is created and never written.** Grep for `"logs"` finds only the `_SUBDIRS` tuple, and **the package has no logging at all** (there's no `import logging`; output goes to the terminal through `rich.Console`). Training progress printed by the engine isn't captured into the run directory, so an empty `logs/` folder in every run *implies* a log that doesn't exist. Either tee stdout/stderr into `logs/run.log`, or delete the directory.

```python
def write_json(self, name: str, data: Any) -> Path:
    target = self.path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, default=str))
    return target

def append_jsonl(self, relpath: str, record: dict[str, Any]) -> None:
    target = self.path / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")

def sub(self, name: str) -> Path:
    p = self.path / name
    p.mkdir(parents=True, exist_ok=True)
    return p
```

- **`write_json` isn't atomic.** `write_text` truncates and then writes. A kill between the two leaves an empty or truncated `summary.json` or `training_outcome.json`. `mtrain validate` and `mtrain compare` then fail with `JSONDecodeError` and no hint of what happened. The code goes to some length to defend checkpoints against partial writes (the trainer verifies each one, Module 6) but not its own metadata.
- **`default=str` is a silent type-coercion escape hatch.** Anything not JSON-serializable (a `Path`, a `datetime`, a NumPy scalar that isn't a `float` subclass, such as `np.float32`) becomes a string instead of raising. For `Path`, that's what you want. For a metric, it's a hazard: an `np.float32` MSE would be written as the *string* `"0.0239"`, and every downstream comparison or plot would then deal in strings. Whether the engine's `Evaluator` returns Python floats or NumPy scalars is **unverified**. `np.float64` is a `float` subclass and serializes correctly, so the risk is limited to other dtypes. It's still a correctness question settled by a fallback instead of a type.
- **Not every writer uses `RunDir`.** The runner writes `config.resolved.yaml`, `evaluation/*.json`, `predictions/test_outcomes.jsonl`, `failures/feedback_artifact.json` and `reports/*.md` by building paths and calling `write_text` or `open` directly (Module 11). So the run directory has no single writer through which atomicity or a manifest could be added later. The abstraction exists but is only partly used.
- `sub()` creates on every call, so it's idempotent and cheap. Callers use it as "path to subdirectory, guaranteed to exist."

### The artifacts that end up in a run directory

Collected by grepping every `write_json`/`append_jsonl` call plus the runner's direct writes:

| File | Written by | When |
|---|---|---|
| `environment.json`, `git.json`, `config.resolved.yaml` | runner | first, before data |
| `dataset.json` | runner | after intervention |
| `split_integrity.json`, `dataset_profile.json` | runner | after the leakage check and profile |
| `metrics/train.jsonl`, `metrics/validation.jsonl`, `metrics/checkpoint_errors.jsonl` | trainer | during training |
| `resume_source.json`, `crash.json` | trainer | on resume, or on a training exception |
| `training_outcome.json`, `model.json` | runner | after training |
| `evaluation/{baselines,test_overall,stratified}.json` | runner | after evaluation |
| `predictions/test_outcomes.jsonl` | runner | per-frame test outcomes |
| `model_artifact.json`, `failures/feedback_artifact.json` | runner | after a SmolVLA evaluation |
| `metrics/test.jsonl`, `metrics/rollout.jsonl` | runner | end |
| `reports/experiment_report.md`, `reports/model_card.md`, `reports/figures/*.png` | runner | end |
| `summary.json` | runner | **last**, which makes it the de facto "completed" marker |
| `manifest.json` | **nobody** | read by `reports/comparison.py` and `cli.py export`, but never written (gap S12) |

`summary.json` being written last is the closest thing to a completion marker, though no code *uses* it as one. `compare` and `export` read whatever exists. Making it explicit (for example, a `status` field written first as `running` and atomically replaced with `completed` or `failed`) would turn the directory into a reliable state machine.

---

## 3.3 Provenance capture: git and environment

```python
def _git_info(repo_path: Path) -> dict[str, Any]:
    if not (repo_path / ".git").exists():
        return {"repo": str(repo_path), "commit": None, "dirty": None, "branch": None,
                 "note": "not a git repository"}
    try:
        def _run(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(repo_path), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        commit = _run("rev-parse", "HEAD")
        dirty = bool(_run("status", "--porcelain"))
        branch = _run("rev-parse", "--abbrev-ref", "HEAD")
        return {"repo": str(repo_path), "commit": commit, "dirty": dirty, "branch": branch}
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {..., "error": str(exc)}
```

**Decision: record git identity of *both* repositories.**
- *Why:* Module 1 established that the training step lives in the engine, so the engine's commit is *more* important to reproducibility than this repo's. Recording both is correct and unusual. Most projects record only their own repo.
- `.exists()` is true for a `.git` *file* too (worktrees and submodules use one), so those cases work.
- **`dirty` is recorded, but the diff isn't.** A dirty run is flagged as unreproducible-from-commit, but the uncommitted change isn't saved. *Fix:* write `git diff HEAD` to `git_<repo>.patch` when dirty.
- **`status --porcelain` counts untracked files.** That's why the `.gitignore` entries `runs` and `experiments/*.jsonl` (Module 1) matter: without them, every run after the first would mark the repo dirty because of the previous run's outputs. That's a real, non-obvious reason those ignore rules exist.
- **`stderr=DEVNULL` plus a broad `except`:** git failures (git not installed, "dubious ownership" safe-directory errors in containers) become an `"error"` string in `git.json`, and the run continues. That's the right failure mode, since provenance shouldn't block training, but it's silent at the terminal.

**Where the paths come from (in `runner.py`, shown here because it decides what this captures):**

```python
RDE_REPO = Path(os.environ.get("RDE_REPO_PATH", str(_default_repo_root() / "episodic-data-engine")))
MT_REPO  = Path(os.environ.get("MODEL_TRAINING_REPO_PATH", str(Path(__file__).resolve().parents[3])))
```

- The engine path is a **guess from directory layout** (`parents[4]`, the parent of this repo, plus `/episodic-data-engine`), overridable by an environment variable.
- **The commit recorded isn't necessarily the code that ran.** The git info is read from wherever `RDE_REPO` points, not from where the imported `rde` package actually lives (`rde.__file__`). If you installed the engine from a different checkout, or non-editable from a wheel, `git.json` reports the commit of an unrelated directory, or `"not a git repository"`, and the run proceeds. **For the repository that owns `fit()`, provenance can be null or wrong without any warning.** *Fix:* resolve the repo from `Path(rde.__file__).resolve()` and walk up to the `.git`.
- `MT_REPO` = `parents[3]` is correct only for an editable `src/` install. For a regular wheel install it resolves to somewhere like `.../lib/python3.11/`, which isn't a git repo. The runner's default `runs/` and `experiments/` directories are derived from the same `MT_REPO` (Module 11), so they'd be created *inside the Python installation*. Because of the sibling-path dependency, the project is effectively editable-only (Module 1), so in practice this is latent.

**Doc vs code: the report's git line is always `n/a`.** `collect_git_info` returns keys `"episodic_data_engine"` and **`"vla_finetune"`**, while `reports/experiment_report.md`'s renderer reads `env.get('git', {}).get('model_training', ...)` (Module 10). The model_training commit therefore always prints `n/a` in every report, even though `git.json` has it. A one-word bug.

```python
def collect_environment() -> dict[str, Any]:
    ...
    info = {
        "python_version": sys.version,
        "os": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": ram_total_gb,
        "packages": {
            "episodic-data-engine": _package_version("episodic-data-engine"),
            "vla-finetune": _package_version("vla-finetune"),
            "numpy": _package_version("numpy"),
            "pydantic": _package_version("pydantic"),
        },
        "torch": None, "cuda": None, "gpu": None, "lerobot": None, "notes": [],
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "driver_capability": f"{props.major}.{props.minor}",
            }
    except ImportError:
        info["notes"].append("torch not installed in this environment (light .venv, no `vla` extra)")
```

- **Pattern: "observe, don't fabricate."** Missing dependencies produce `None` plus a note, never a guess. That matches the project's stated principle, and it's what the runner keys its `limitations` list on (`environment.get("gpu") is None`, Module 11).
- **`driver_capability` is mislabelled.** `props.major` and `props.minor` are the GPU's **compute capability** (8.9 for an RTX 4060), not a driver version. The driver version isn't recorded anywhere. Neither is cuDNN (`torch.backends.cudnn.version()`). And `cuda.version` is `torch.version.cuda`, the *compile-time* toolkit, not what the driver supports.
- **The package list misses what actually determines model behaviour.** The versions that matter for SmolVLA numerics are `torch` (recorded), `lerobot` (recorded via `__version__`), and `transformers` plus `safetensors` (SmolVLA's backbone loader; *not recorded*). `numpy` and `pydantic` are recorded instead. `vla-finetune` is always `0.1.0` and carries no information; the git commit is the real identity. A full `pip freeze` (or `importlib.metadata.distributions()`) dump would be cheap and complete.
- **`import torch` at run start** costs a second or two and initializes CUDA. That's harmless here, since training follows immediately.
- **Only GPU 0 is described.** That's fine for a single-GPU laptop (README: RTX 4060 Laptop, 8 GB). The record would be wrong under `CUDA_VISIBLE_DEVICES` remapping on a multi-GPU box, where device 0 might not be the one used.
- **`hostname` is recorded.** That's fine for local use. Note that `mtrain export` doesn't copy `environment.json` into the paper-ready bundle (it copies configs and manifests only; Module 12), so hostnames don't leak into exported artifacts. It's unclear whether that was deliberate, but the effect is right.

---

## 3.4 `JsonlIndex` and `ExperimentIndex`: the cross-run index

```python
class JsonlIndex:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def query(self, **filters: Any) -> Iterator[dict[str, Any]]:
        for row in self.read_all():
            if all(row.get(k) == v for k, v in filters.items()):
                yield row
```

**Decision: JSONL files, not a database or a tracking server.**
- *What:* four files under `experiments/`, namely `experiments.jsonl`, `metrics.jsonl`, `datasets.jsonl` and `models.jsonl`, one row per event.
- *Why:* the module docstring says "queryable with `jq`, `pandas.read_json(lines=True)`, or DuckDB's `read_json_auto` without this project running". That's the "scrapable output" principle. No server, no schema migration, readable decades later, diffable.
- *Alternative 1:* MLflow (a tracking server plus artifact store, with a UI). *Alternative 2:* SQLite, which is transactional and indexed, and is plausibly what the engine's own catalog already uses (unverified). *Alternative 3:* W&B (hosted, rich UI, network dependency).
- *Cost:*
  1. **No atomicity or crash tolerance on read.** `read_all` does `json.loads` on every line with no `try`. If a process dies mid-append, leaving a half line, **every subsequent `read_all`, `query`, `ModelRegistry.get` and `ModelRegistry.list` raises `JSONDecodeError` forever**, until someone hand-edits the file. One torn write poisons the whole index. *Fix:* skip and count undecodable lines (and warn), or write each record with a single `os.write` on an `O_APPEND` file descriptor.
  2. **Concurrent appends from several processes** are safe on local disk *only* while each line goes out in one `write()` syscall. `models.jsonl` rows carry the full observation and action schema plus both repos' git info (Module 9), so they can approach or exceed the 8 KiB buffer, at which point lines from two processes can interleave. It's unlikely on one laptop and plausible under a scheduler or NFS.
  3. **Every query is a full scan.** That's `O(rows)` per lookup, which is irrelevant at tens or hundreds of runs. It becomes painful only at tens of thousands, and by then DuckDB over the same files is the natural upgrade with no format change. That's a real strength of the format choice.
  4. **No schema.** `record_experiment(**fields)` accepts anything. Rows written by different code versions differ silently, and there's no `schema_version` field.

```python
class ExperimentIndex:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.experiments = JsonlIndex(self.root / "experiments.jsonl")
        self.metrics = JsonlIndex(self.root / "metrics.jsonl")
        self.datasets = JsonlIndex(self.root / "datasets.jsonl")
        self.models = JsonlIndex(self.root / "models.jsonl")

    def record_experiment(self, **fields): self.experiments.append(fields)
    def record_metric(self, **fields):     self.metrics.append(fields)
    def record_dataset(self, **fields):    self.datasets.append(fields)
    def record_model(self, **fields):      self.models.append(fields)
```

- **Pattern:** a *Façade* over four *append-only logs*, close to the *Repository* pattern without the query side. Only `ModelRegistry` (Module 9) ever reads one of these files back. **Nothing in `src/` reads `experiments.jsonl`, `metrics.jsonl` or `datasets.jsonl`.** `mtrain compare` reads run *directories*, not the index (Module 10). So the index is currently **write-only from the system's point of view**. It's a sound design, but it's a log for external tools rather than a component the system relies on.
- **The index isn't transactional with the run directory.** The runner writes `record_dataset` right after the intervention, *before* training (Module 11). `record_model`, `record_experiment` and `record_metric` come at the end. So a run that crashes in training leaves a `datasets.jsonl` row pointing at a `training/vN` with no matching `experiments.jsonl` row, and there's no row anywhere saying the run failed. **You can't list failed runs from the index.** Even orphaned dataset versions are invisible unless you join `datasets.jsonl` against `experiments.jsonl` by `experiment_id`.
- **Doc check:** the index docstring lists the four files and their row semantics ("one row per run", "one row per (run, split, metric snapshot)", ...). That matches the call sites, with one nuance: `models.jsonl` gets one row per *SmolVLA* run only, because baseline-only runs never build a model artifact.

---

## 3.5 Honest gaps to own before an interviewer finds them

1. **No write is atomic** (`write_json` truncates then writes; `append_jsonl` and `JsonlIndex.append` are plain buffered appends). *Consequence:* a kill mid-write corrupts `summary.json` or a JSONL line, and readers fail with an unexplained `JSONDecodeError`.
2. **One torn line poisons the whole index.** `read_all` has no per-line error handling. *Consequence:* `ModelRegistry.get`/`list` and every `query` fail until the file is repaired by hand.
3. **A failed run blocks its own same-day retry**, because the directory is created and populated before any validation of data or environment, and there's no `--force` or crash marker. *Consequence:* the user has to rename the experiment or delete the directory, which defeats the no-overwrite guarantee's purpose.
4. **`make_run_dir` has a TOCTOU race** (check non-empty, then `mkdir(exist_ok=True)`). *Consequence:* two concurrent identical launches share one directory.
5. **Engine provenance can be silently wrong or null.** The git info is read from a layout-guessed path, not from the imported `rde` package's location. *Consequence:* for the code that owns `fit()`, `git.json` may record nothing, or an unrelated checkout.
6. **Dirty trees are flagged but the diff isn't saved.** *Consequence:* a dirty run is unreproducible by construction.
7. **`environment.json` mislabels compute capability as `driver_capability`, omits the driver, cuDNN and `transformers` versions, and records a meaningless `vla-finetune` 0.1.0.** *Consequence:* the record misses exactly the versions most likely to explain a numerics difference.
8. **The report's git line always prints `n/a`** (key `vla_finetune` written, `model_training` read). *Consequence:* the human-readable report hides provenance that `git.json` holds.
9. **`logs/` is created in every run and never written, and there's no logging at all.** *Consequence:* engine training output isn't captured, and the empty folder implies otherwise.
10. **The index is write-only inside the system, not transactional with the run, and has no status field.** *Consequence:* failed runs and orphaned dataset versions are undiscoverable from the index.
11. **`default=str` silently stringifies non-JSON types.** *Consequence:* a non-`float64` NumPy metric would be stored as a string. Unverified whether the engine produces one.
12. **Writers bypass `RunDir`** for about half the artifacts. *Consequence:* there's no single choke point to add atomicity, a manifest or checksums later.
13. **Default `runs/` and `experiments/` locations derive from `parents[3]` of the installed file.** *Consequence:* with a non-editable install they'd land inside the Python installation (latent, because the project is editable-only).

---

## The shape to carry into Module 4

```
 runs/                                         experiments/            (both git-ignored)
 └── <UTC date>/                               ├── experiments.jsonl   one row per COMPLETED run
     └── <experiment_name>/                    ├── metrics.jsonl       one row per test eval
         └── seed_<n>/   ← make_run_dir:       ├── datasets.jsonl      one row per intervention
             │            raise if non-empty   └── models.jsonl        one row per SmolVLA artifact
             ├── environment.json  (torch/cuda/gpu or None+note)       append-only, full-scan,
             ├── git.json          (engine + this repo; dirty flag)    no locking, no schema,
             ├── config.resolved.yaml                                  write-only inside system
             ├── checkpoints/ predictions/ evaluation/
             ├── failures/ reports/ metrics/
             ├── logs/            ← always empty
             └── summary.json     ← written LAST (implicit "completed")
   writes: write_text / open("a")  — none atomic
```

---

## Drill — answer from memory, files closed

1. What happens if `mtrain train` fails during the intervention step and you immediately rerun the same command?
2. *Why didn't you just* use MLflow or Weights & Biases?
3. *What breaks in production* if the machine loses power while the runner is appending to `experiments/models.jsonl`?
4. Why does `.gitignore` excluding `runs` matter for the correctness of `git.json`, beyond keeping the repo small?
5. *Interviewer framing:* "How would I reproduce one of your runs six months from now? What would stop me?"

---

# Appendix A — Model answers to the drill

**A1. Fail-then-rerun.**
"The runner creates the run directory and writes `environment.json`, `git.json` and the resolved config before it touches the data. So if the intervention fails, say a wrong dataset version, the directory is already non-empty, and the rerun raises `FileExistsError` from `make_run_dir`, which never reuses a non-empty run directory. The guard is deliberate, because overwriting a real run would destroy checkpoints. But here it protects nothing, since the failed run has no results. I'd fix that by validating the dataset and environment before creating the directory, writing a status marker that a failed run can be identified by, and allowing a retry over a directory whose status is `failed`. Today you rename the experiment or delete the directory, and the error message tells you that."

*Why that answer lands:* it traces the exact order of writes, defends the guard's intent, and then shows that the guard misfires in one specific case, with a fix that keeps the guarantee.

**A2. Why not MLflow or W&B?**
"The requirements were reproducibility and inspectability, not a dashboard. Run directories plus append-only JSONL mean every artifact is a plain file you can query with `jq`, pandas or DuckDB without this code, a server, a login or network access, and the format will still be readable in ten years. There's also no second source of truth: the run directory is the record. The costs are that I own the durability problems myself (none of my writes are atomic yet), there's no UI, and every query is a full scan. I'd switch, or add MLflow as a sink alongside the files, once more than one person was running experiments or once runs numbered in the thousands. At that point locking and a UI earn their keep, and the JSONL files map directly onto MLflow params and metrics."

*Why that answer lands:* it argues from requirements, admits the costs you inherited by rolling your own, and names a migration path that keeps the existing format, not a rewrite.

**A3. Power loss during an index append.**
"The append is a buffered text write with no fsync, so after a power loss the last line can be truncated. The reader parses every line with `json.loads` and no error handling, so from then on every read of that file raises, including the model registry's `get` and `list`. One torn write takes out the whole registry until someone hand-edits the file. The fix has two halves: make the reader skip and report undecodable lines, and make the writer emit each record as a single `os.write` on an `O_APPEND` descriptor, with fsync if durability matters. The same class of bug exists for the run directory's JSON files, which are truncated and rewritten in place, so I'd move those to write-to-temp plus `os.replace`."

*Why that answer lands:* it gets the failure exactly right (a single bad line breaks all reads, not just the lost record), gives both halves of the fix, and generalizes to the sibling bug unprompted.

**A4. Why ignoring `runs` matters for `git.json`.**
"`git.json` records a dirty flag from `git status --porcelain`, which counts untracked files. If run outputs lived unignored inside the repo, every run after the first would see the previous run's files as untracked and mark the tree dirty, so the flag would be permanently true and carry no information. Ignoring `runs` and the index files keeps 'dirty' meaning 'the code changed', which is the only meaning that's useful for reproducibility. The remaining gap is that when the tree really is dirty, I record the flag but not the diff, so I'd save `git diff HEAD` as a patch next to it."

*Why that answer lands:* it connects two files that look unrelated, showing systems-level reading, and ends with the next gap.

**A5. "How would I reproduce one of your runs in six months?"**
"Start from the run directory. `config.resolved.yaml` has every parameter including defaults, `dataset.json` names the exact training version in the engine's catalog, and the engine's lineage resolves that back to raw episodes. `git.json` has the commit of both repositories, and `environment.json` has the Python, torch, lerobot and GPU details. Then check out both commits, install, and rerun the config against the same input version. What would stop you, honestly: there's no lockfile, so transitive versions like `transformers` may differ and aren't recorded. If either tree was dirty, the diff wasn't saved. The engine's commit is read from a path guessed from directory layout, not from the package that was actually imported, so it could be wrong. And the checkpoints and run directories aren't versioned, so you need my disk, not just the repo. Even with all of that fixed, SmolVLA's flow-matching sampling is stochastic, so you'd reproduce the metrics within noise, a few percent, not bit-for-bit."

*Why that answer lands:* it gives the full chain first, then lists the breaks precisely, and closes with the irreducible stochasticity. That's the sign of someone who has actually tried to reproduce ML results.

---

# Module 3 Summary — the one page to revise

**What this module is.** One directory per run (`runs/<UTC date>/<name>/seed_<n>/`) that's never overwritten, populated with environment and git snapshots and every artifact, plus four append-only JSONL index files that external tools can query without this code. The format is good; durability, transactionality and provenance accuracy are the weak points.

**Five claims to make cold**
1. A non-empty run directory is never reused: `FileExistsError`, tested.
2. Both repos' git commits and dirty flags are recorded, because the engine owns `fit()`.
3. Missing dependencies become `None` plus a note, never a guess, and the runner turns these into report limitations.
4. The index is four JSONL files, append-only, readable by `jq`, pandas or DuckDB with no server.
5. No write is atomic, and one torn JSONL line breaks every subsequent read of that file.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Plain run directories plus JSONL | MLflow/W&B | inspectable, serverless, durable format | durability, locking, UI are on me |
| Refuse non-empty directory | auto-suffix/UUID | never destroy a run | failed runs block same-day retry; TOCTOU race |
| `date/name/seed` layout | `name/date/seed` | scan by day | dated paths leak into configs and scripts |
| Record git of both repos | own repo only | engine owns training | engine path guessed, can be null or wrong |
| `default=str` in JSON dumps | strict serialization | never crash on `Path`/`datetime` | silently stringifies odd types |
| Pre-create 7 subdirectories | create on demand | predictable layout | `logs/` always empty |

**Volunteer unprompted:** the torn-write poisoning; the retry-blocking guard; the `driver_capability` mislabel; the report's `n/a` git line; no diff saved for dirty trees; the index is write-only and has no failed-run status; `manifest.json` is read but never written.

**Numbers and names:** 7 subdirectories (`checkpoints predictions evaluation failures reports logs metrics`); 4 index files; env vars `RDE_REPO_PATH` and `MODEL_TRAINING_REPO_PATH`; engine path `parents[4]/episodic-data-engine`; repo root `parents[3]`; `summary.json` written last; Python's default text buffer is 8 KiB.

**Sentences worth memorizing**
- "Every artifact is a plain file you can query without my code; that's the design, and durability is the price I still owe."
- "I record the engine's commit as well as mine, because the engine owns the training step."
- "One torn line in a JSONL index breaks every read after it, so the reader has to be tolerant and the writer atomic."
- "Ignoring `runs/` isn't just tidiness; it keeps the dirty flag meaningful."
