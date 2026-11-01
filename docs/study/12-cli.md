# Module 12 — The CLI surface: `mtrain` and the script wrappers

**Files covered:** `src/model_training/cli.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/test.py`, `scripts/rollout.py`, `scripts/compare.py`, `scripts/analyze_failures.py`, `scripts/export.py` (`scripts/plot_results.py` was covered in Module 10)

**Where we are:** Module 11 was the programmatic entry point (`ExperimentRunner.run`). This module is the human one, the eleven `mtrain` commands a researcher actually types. Most commands are thin views over run-directory artifacts. Three of them *change* state: `train` runs the pipeline, `evaluate`/`test` re-score and overwrite, and `build-intervention` rewrites a config file. Those three are where the CLI can undo the guarantees the rest of the system builds.

**The one-line thesis:** The CLI's structure is right: lazy imports, a dry run by default for config rewriting, and honest "unavailable" commands. But `mtrain evaluate` is a divergent reimplementation of the runner's evaluation stage. It **overwrites the published test numbers in place**, with different slices and batch settings, and leaves the rest of the run directory describing the old numbers. Its own reference doc recommends using it with `--split val`, which writes validation results into `test_overall.json`.

---

## 12.1 Prerequisites — the absolute basics

- **Typer.** A library that builds a CLI from Python function signatures. `@app.command()` registers a function as a sub-command, and each parameter becomes an option via `typer.Option(default, "--flag")`. `...` (Ellipsis) as the default means *required*. `list[Path]` options are given by **repeating the flag** (`--runs a --runs b`), not as space-separated values. `"--dry-run/--apply"` declares a boolean with paired on and off flags.
- **`typer.Exit(code)`.** Ends the command with an exit code, and no traceback.
- **`rich`.** A terminal-formatting library: `Console.print` (with `[yellow]…[/yellow]` markup), `print_json` and `Table`.
- **Command–query separation (CQS).** Distinguishing commands that *change state* from queries that only *read* it. CLIs that mix the two in one verb (e.g. "evaluate" that also overwrites) surprise users.
- **Single source of truth.** Each behaviour implemented in exactly one place. Two implementations of "evaluate a run" will drift.
- **Idempotence.** Running a command twice has the same effect as running it once. For a stochastic model, re-scoring is *not* idempotent: each run produces new numbers.
- **YAML round-trip loss.** `yaml.safe_load` followed by `yaml.safe_dump` discards comments, and reflows quoting and folded strings. Preserving them needs a round-trip parser such as `ruamel.yaml`.

---

## 12.2 Structure: lazy imports and thin views

```python
app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

def _print_json(data) -> None:
    console.print_json(json.dumps(data, indent=2, default=str))
```

- **Only `typer`, `yaml` and `rich` load at import.** Every `rde` and `model_training` import is inside a command function. So `mtrain --help`, `mtrain rollout` and `mtrain analyze-failures` start without loading the engine or torch. This is the one place where the import-light discipline (Module 1) is applied *fully*, and it's where it matters most, because CLI startup latency is user-facing.
- **`no_args_is_help=True`** prints help instead of an error for a bare `mtrain`. `add_completion=False` hides Typer's shell-completion install commands, so there's less surface.
- **The name `mtrain`** avoids colliding with the engine's `rde model` (module docstring, Module 1).

### Read-only views

| Command | Reads | Writes | Notes |
|---|---|---|---|
| `inspect --dataset --version` | engine explorer `overview` | — | `--version` defaults to `"training/v1"`, an arbitrary version that may not exist for a given dataset. That's the only CLI default that names data. |
| `profile --dataset --version [--out]` | Module 5's `build_dataset_profile` | optional JSON file | the read-only profile the imbalance choice must come from |
| `validate --run` | `training_outcome.json`, `metrics/validation.jsonl` | — | see below |
| `analyze-failures --run` | `failures/feedback_artifact.json` | — | prints the Module 8 artifact verbatim |
| `compare --runs … [--metric] [--out]` | Module 10 `build_comparison_table` | optional CSV | terminal-table misalignment (Module 10 gap 7) |

**`validate` misdiagnoses.**

```python
outcome_path = run / "training_outcome.json"
if not outcome_path.exists():
    console.print(f"[yellow]no training_outcome.json in {run} -- this run had no gradient-based training[/yellow]")
    raise typer.Exit(1)
```

A missing `training_outcome.json` means *either* a baseline-only run *or* a run that crashed in training (Module 6: `crash.json` is written, `training_outcome.json` isn't). The message asserts the first. A one-line check for `crash.json` would give the right diagnosis. `analyze-failures` has the same pattern: "this run had no test-set model evaluation" could also mean "crashed before evaluation".

**`compare` usage is documented two ways.** `scripts/compare.py`'s docstring says `python scripts/compare.py --runs a b c`. With a Typer `list[Path]` option, that form fails: after the first value, `b` and `c` are unexpected extra arguments. The README (line 186, "`--runs` repeated once per run") and `docs/API_REFERENCE.md` §4.7 show the correct repeated-flag form. The script wrapper's docstring is wrong.

---

## 12.3 `train`

```python
@app.command()
def train(config: Path = typer.Option(..., "--config"), seed: Optional[int] = typer.Option(None, "--seed")):
    """Run one experiment end to end: intervention -> train -> evaluate -> report."""
    cfg = load_experiment_config(config)
    result = ExperimentRunner().run(cfg, seed=seed)
    console.print(f"run complete: {result.run_dir}")
    _print_json(result.summary)
```

- It's a thin wrapper over Modules 2 and 11, which is correct: one implementation.
- **`--seed` is the Module 11 bug at its entry point.** The help text gives no hint that it only names the directory.
- **No `--base`** (Module 2 gap 9), **no seed loop** over `config.seeds`, and **no `--force`** for a failed same-day run (Module 3 gap 3).
- **Exceptions surface as raw tracebacks.** For a research CLI, that's defensible: the traceback is the diagnosis. But a `LookupError("unknown version: so100_pickplace@clean/v9")` would be more readable as a one-line error with exit code 2.

---

## 12.4 `evaluate` and `test`: the divergent re-implementation

```python
@app.command()
def evaluate(run: Path = typer.Option(..., "--run"), split: str = typer.Option("test", "--split")):
    """Re-run stratified + baseline offline evaluation for an existing run's
    selected checkpoint against SPLIT, overwriting its evaluation/ artifacts."""
    from rde.catalog import Catalog
    ...
    from model_training.evaluation import episode_metrics, error_distribution, evaluate_baselines, stratified_evaluate, temporal_error

    dataset_json = json.loads((run / "dataset.json").read_text())
    model_json = json.loads((run / "model.json").read_text())
    checkpoint = model_json.get("selected_checkpoint")
    if not checkpoint:
        console.print(f"[yellow]no selected_checkpoint recorded for run {run}[/yellow]")
        raise typer.Exit(1)

    settings = get_settings()
    view = DataEngine(settings).get_dataset(dataset_json["dataset_id"], dataset_json["training_version"])
    try:
        policy = SmolVLAPolicy.from_checkpoint(checkpoint)
        adapter = VisionLanguageAdapter(view.contract)
        strat = stratified_evaluate(view, policy, adapter, base_split=split, slices=["per_task"])
        overall = strat.pop("overall")
        payload = {"summary": overall.summary(), "metrics": overall.metrics,
                   "error_distribution": error_distribution(overall),
                   "episode_metrics": episode_metrics(overall),
                   "temporal_error": temporal_error(overall, view)}
        eval_dir = run / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        (eval_dir / "test_overall.json").write_text(json.dumps(payload, indent=2, default=str))
        (eval_dir / "stratified.json").write_text(json.dumps({...}, indent=2, default=str))
        _print_json(payload["summary"])
    finally:
        view.close()
```

**What's right:** it reopens the *exact* `training_version` recorded in `dataset.json` (data by identity, never by path), reloads the *selected* checkpoint, and closes the view in `finally`. Re-scoring the same version on the same split scores the same episodes.

**What diverges from the runner's stage 4 (Module 11):**

| Aspect | Runner | `mtrain evaluate` |
|---|---|---|
| Baselines | re-fit and scored | **not run.** `evaluate_baselines` is imported and never called, despite the docstring's "Re-run stratified + baseline" |
| Slices | `config.evaluation.slices` (`[per_task, per_length_bucket]` from `base.yaml`) | **hard-coded `["per_task"]`** |
| `batch_size` / `max_frames` | from config (16 / `null`) | `stratified_evaluate` defaults (32 / `None`) |
| Predictions file | `predictions/test_outcomes.jsonl` written | **not rewritten**, so it now describes the *old* predictions |
| `metrics/test.jsonl`, `summary.json`, `model_artifact.json`, report, model card, feedback, index | written | **none updated** |

**Consequences:**
1. **It overwrites the reported numbers in place.** SmolVLA is stochastic (Module 7: up to 5% between re-scorings), so running `mtrain test` on a published run *changes* `test_overall.json`. The README's numbers can no longer be traced to the file they came from, and there's no backup or history. The rest of the system treats a run directory as write-once (`make_run_dir` refuses to overwrite). This command quietly breaks that rule for the most important file in it.
2. **It destroys the slices.** The original `stratified.json` (task plus three length slices) is replaced by per-task only. That's information loss, not just a re-score.
3. **It leaves the directory internally inconsistent.** `test_overall.json` disagrees with `summary.json`, `metrics/test.jsonl`, `model_artifact.json["metrics"]`, the report, the model card, `predictions/test_outcomes.jsonl` and the feedback artifact computed from the old outcomes.
4. **`--split val` writes validation results into `test_overall.json`.** `docs/API_REFERENCE.md` §4.5 and §9.3 *recommend* `mtrain evaluate --run <dir> --split val`. The doc honestly says it overwrites `test_overall.json`, but the *filename* then lies, and `mtrain compare`, `plot_results.py` and `load_run_summary` will all read validation metrics as test metrics. On a project whose central discipline is val/test separation, this is the most dangerous documented workflow in the repo.
5. **Unused imports** (`Catalog`, `evaluate_baselines`) are leftovers from the fuller version the docstring still describes.

**What it should be:** a call into the *same* evaluation function the runner uses, factored out of `ExperimentRunner.run` (a `run_evaluation_stage(run_dir, config, view, checkpoint)` that takes its settings from the run's own `config.resolved.yaml`), writing to a **new, versioned** directory (`evaluation/rescore_<timestamp>_<split>/`) and never over the original. That gives one implementation, no overwrite, and filenames named for the split.

```python
@app.command()
def test(run: Path = typer.Option(..., "--run")):
    """Alias for `evaluate --split test` -- the offline test-set score."""
    evaluate(run=run, split="test")
```
Calling a Typer command function directly works, because it's still a plain Python function. Note that the alias inherits everything above. "`mtrain test`" sounds like a harmless check, but it overwrites the published test numbers.

---

## 12.5 `rollout`: honesty as a command

```python
@app.command()
def rollout(run: Path = typer.Option(..., "--run")):
    """Closed-loop rollout evaluation. No simulator or robot is connected in
    this environment, so this reports 'unavailable' rather than fabricating a
    success rate -- never conflated with offline evaluation."""
    payload = {"status": "unavailable", "reason": "no simulator or physical robot connected in this environment"}
    (run / "metrics").mkdir(exist_ok=True)
    with (run / "metrics" / "rollout.jsonl").open("a") as f:
        f.write(json.dumps(payload) + "\n")
    _print_json(payload)
```

**Decision: a command whose only job is to record that something can't be measured.** *Why:* `docs/API_REFERENCE.md` §4.6 says "absence is a recorded, honest fact rather than a silently missing file". It also gives the interface a slot where a real rollout harness would plug in. *Cost:* it appends a duplicate record each time it's run (the runner already wrote one), with no timestamp to distinguish them. `mkdir(exist_ok=True)` without `parents=True` raises on a mistyped run path whose parent doesn't exist. That's arguably the right failure, though it's a raw traceback.

---

## 12.6 `export`: "paper-ready" with gaps

```python
output.mkdir(parents=True, exist_ok=True)
for sub in ("figures", "tables", "metrics", "configs", "manifests"):
    (output / sub).mkdir(parents=True, exist_ok=True)
# figures ← reports/figures/*; metrics ← evaluation/*.json; configs ← config.resolved.yaml;
# manifests ← manifest.json, dataset.json, model.json, dataset_profile.json (if present)
(output / "experiment_summary.md").write_text(
    "... Every file under figures/, tables/, metrics/, configs/, manifests/ traces back to the "
    f"run directory above. Regenerate with `mtrain export --run {run} --output {output}`.\n")
```

- **`tables/` is created and never filled.** No code copies anything into it.
- **`manifest.json` never exists** (Module 3, S12).
- **`git.json` and `environment.json` aren't exported.** The bundle that claims everything "traces back" omits the commits and environment, which are exactly what a reader of a paper artifact needs to reproduce it. (It does keep the hostname out, as Module 3 noted, but it drops the commits along with it.)
- **`reports/experiment_report.md`, `reports/model_card.md` and `predictions/test_outcomes.jsonl` aren't exported**, so neither the narrative nor the per-frame evidence is included.
- **It overwrites an existing export silently** (`exist_ok=True` plus `copy2`). That's the opposite policy to `make_run_dir`. For a derived bundle that can be regenerated, it's defensible, but stale files from a previous export into the same directory remain.
- **"Nothing is recomputed or summarized lossily"** (API_REFERENCE §4.9) is true for what *is* copied, but the selection of what to copy is lossy in the most important places.

---

## 12.7 `build-intervention`: resolving the design rule

```python
@app.command("build-intervention")
def build_intervention_cmd(dataset, version, kind, variable=None, variable_value=None,
                           oversample_factor: float = 3.0, target_config: Path = ...,
                           reason: str = "", dry_run: bool = typer.Option(True, "--dry-run/--apply")):
    """... Deliberately does not build the dataset version itself -- `mtrain train`
    does that as part of a normal run, with real lineage. This command only
    resolves *which* episodes/weights the intervention names, from the
    dataset's actual per-episode metadata, and never presumes them."""
```

**What's right:**
- **Dry run by default.** It prints the resolved block, and you must pass `--apply` to write. For a command that rewrites a config file, that's exactly the right default.
- **Resolution and materialization are separate.** It *names* episodes; `train` *builds* the version, so lineage is created once, by the normal path. This is the command that operationalizes "the weak dimension must not be presumed" (Module 2): it reads the real catalog and resolves from real metadata.
- **`task_id` values are converted to `int`** (`value = int(variable_value)`). That avoids Module 2's string-vs-int silent no-op on the generated path.
- **An empty result exits with an error** for rebalanced ("resolved zero episodes").

**What's wrong:**
- **`--apply` destroys the config's comments.** `yaml.safe_load` followed by `yaml.safe_dump(data, sort_keys=False)` rewrites the whole file. The shipped `imbalanced.yaml` and `rebalanced.yaml` carry 15–20-line header comments explaining *why* they're templates and what the result must and mustn't claim ("do not claim a fixed weak regime unless the intervention and the evidence both support it"). **Applying the resolved intervention deletes that reasoning** and reflows the `>` folded hypothesis strings. The safeguards written for the human are erased at the moment they matter. `ruamel.yaml` round-trip mode, or writing the block to a separate `*.intervention.yaml` that the loader merges, fixes this.
- **Imbalanced is written as `kind: custom`.** `block = {"kind": "custom", "episode_ids": ..., "variable": ..., "variable_value": ..., "reason": ...}`. The run's `dataset.json` will then say `kind: custom`, and `summary.intervention`, the index, and `mtrain compare`'s "intervention" column all lose the fact that this was the *imbalanced* experiment. Only `tags.intervention: imbalanced_data` preserves it, and only because the template sets that tag.
- **Imbalanced inherits the silent no-op** (Module 2 gap 4). `--kind imbalanced --variable episode_length_bucket` with **no `--variable-value`** passes `None`, `resolve_episode_subset` keeps every episode, and the command prints and (with `--apply`) writes a `custom` block listing **all** episode IDs. The rebalanced path checks for an empty result; the imbalanced path doesn't check for a *full* result.
- **`_target_episodes` is copied from `interventions.py`.** The CLI defines its own `_target_episodes(var, val)` with the same body as the closure inside `build_intervention_dataset`. That's two implementations of "which episodes have attribute = value", which will drift. It should be one public function in `data/interventions.py`.
- **A missing target config produces a file with only `intervention:`.** `data = yaml.safe_load(...) if target_config.exists() else {}`, then writing it gives a config with no `experiment_name` or `dataset`. That fails later at load, which is acceptable, but it's an odd thing to allow.
- **It resolves against `--version`, which may differ from the config's `dataset.version`.** Nothing checks that the version the episodes were resolved against matches the version the config will train from. If they differ, the episode IDs may not exist in the training input.

---

## 12.8 The script wrappers

```python
#!/usr/bin/env python
"""Thin wrapper: `python scripts/train.py --config configs/experiments/baseline.yaml`.
Identical to `mtrain train`; provided for environments where the console script is not on the path."""
import sys
from model_training.cli import app

if __name__ == "__main__":
    app(["train", *sys.argv[1:]])
```

- **Seven 8–9-line files, one per command**, each prepending the sub-command name. They're all identical in shape. *Why:* environments where `mtrain` isn't on `PATH`. *Cost:* they still `import model_training`, so the package must be installed. They help only when the virtualenv's `bin/` isn't on `PATH`, and `python -m model_training.cli train ...` already covers that, because `cli.py` has `if __name__ == "__main__": main()`. So they're redundant convenience. That's harmless, but it's seven more files whose docstrings can drift, and `compare.py`'s already has.
- **There are no wrappers for `inspect`, `profile`, `validate` or `build-intervention`**, so the set is partial.
- **`scripts/test.py`'s name.** Running `python scripts/test.py` puts `scripts/` first on `sys.path`, which shadows the standard library's `test` package for that process. It's harmless here (nothing imports `test`), but it's the sort of name that bites later.

---

## 12.9 Honest gaps to own before an interviewer finds them

1. **`mtrain evaluate`/`test` overwrites `test_overall.json` and `stratified.json` in place**, for a stochastic model, with no history. *Consequence:* published numbers can't be traced to their file after a re-score, and a write-once run directory becomes mutable.
2. **`--split val` writes validation metrics into `test_overall.json`, and the docs recommend it.** *Consequence:* `compare`, `plot_results.py` and `load_run_summary` will read val as test.
3. **`evaluate` diverges from the runner**: no baselines (despite its docstring), hard-coded `per_task` slices, default batch settings, no update to predictions, summary, metrics, report, card, artifact or feedback. *Consequence:* a re-scored run directory is internally inconsistent.
4. **`build-intervention --apply` erases the template's explanatory comments** and reflows text. *Consequence:* the written-down safeguards disappear at the moment of use.
5. **Imbalanced is written as `kind: custom`.** *Consequence:* the intervention label is lost from `dataset.json`, the summary, the index and the comparison.
6. **`build-intervention --kind imbalanced` without `--variable-value` silently selects every episode.** *Consequence:* a "resolved" imbalance that's a full run.
7. **`_target_episodes` is duplicated between the CLI and `interventions.py`.** *Consequence:* drift risk in the core selection rule.
8. **The resolution version isn't checked against the config's training input version.** *Consequence:* episode IDs may not exist where training happens.
9. **`export` omits `git.json`, `environment.json`, reports and predictions; `tables/` is always empty; `manifest.json` never exists.** *Consequence:* the "paper-ready" bundle lacks the provenance it claims.
10. **`validate` and `analyze-failures` misdiagnose crashed runs** as "no training" or "no evaluation". *Consequence:* misleading triage.
11. **`scripts/compare.py`'s docstring shows `--runs a b c`, which Typer rejects.** *Consequence:* a wrong usage example.
12. **`--seed` help doesn't say it only names the directory; there's no `--base`, `--force`, seed loop, `report` or `sweep` command.** *Consequence:* the CLI can't express multi-seed studies, report regeneration or sweeps.
13. **`inspect --version` defaults to `training/v1`.** *Consequence:* an arbitrary data default that may not exist.
14. **Script wrappers are redundant with `python -m model_training.cli`, incomplete, and one shadows stdlib `test`.** *Consequence:* maintenance surface for no capability.
15. **No CLI tests** (grep finds none). *Consequence:* every CLI-only behaviour above is unexercised.

---

## The shape to carry into Module 13

```
 mtrain (Typer; only typer/yaml/rich at import — rde loaded per command)
 ├─ STATE-CHANGING
 │   train --config [--seed≠seed]        → load_experiment_config → ExperimentRunner.run   (one impl ✓)
 │   evaluate --run [--split] / test     → OWN re-implementation: no baselines, slices=[per_task],
 │                                          default batch → OVERWRITES evaluation/test_overall.json
 │                                          (val → "test_overall.json") · leaves rest of run stale
 │   build-intervention ... [--apply]    → catalog episodes → block → yaml.safe_dump (comments lost)
 │                                          imbalanced → kind: custom · dup _target_episodes
 │   rollout --run                       → append "unavailable"  (honest placeholder)
 │   export --run --output               → copy figs/eval/config/manifests (no git/env/reports; tables/ empty)
 └─ READ-ONLY
     inspect · profile · validate · analyze-failures · compare (rich table positional; CSV by key)
 scripts/*.py = app(["<cmd>", *argv])  (7 of 11 commands; == python -m model_training.cli)
```

---

## Drill — answer from memory, files closed

1. Which `mtrain` commands change state, and what does each one change?
2. *Why didn't you just* have `mtrain evaluate` call the runner's evaluation code?
3. *What breaks in production:* a colleague runs `mtrain test --run <published run>` to "check it still works" the day before a paper deadline. What has happened?
4. What does `build-intervention --apply` do to `configs/experiments/imbalanced.yaml`, exactly?
5. *Interviewer framing:* "Walk me through how a researcher uses your tool from a fresh dataset to a comparison table."

---

# Appendix A — Model answers to the drill

**A1. State-changing commands.**
"Four. `train` runs the whole pipeline, so it creates a run directory, engine dataset versions, index rows and engine failure records. `evaluate`, and its alias `test`, reload the selected checkpoint and overwrite `test_overall.json` and `stratified.json` in that run's evaluation directory. `build-intervention --apply` rewrites the target config file's intervention block. And `rollout` appends an 'unavailable' record to the run's rollout log. Everything else (inspect, profile, validate, analyze-failures, compare, and export apart from writing its own output directory) only reads. The one I'd change is `evaluate`, because it's the only command that mutates an existing run's results."

*Why that answer lands:* it's a complete inventory organized by command–query separation, and it ends on the problem command.

**A2. Why not reuse the runner's evaluation?**
"I should have. The evaluation stage is inline in `ExperimentRunner.run`, so there was no function to call, and `evaluate` re-implemented a subset of it. The result drifted. It doesn't re-run baselines even though its docstring says it does, it hard-codes the per-task slice instead of reading the run's configured slices, it uses default batch settings, and it only rewrites two of the files the runner writes, so the predictions, summary, report and feedback all describe the old numbers. The fix is to factor stage 4 out into one function that takes the run's own resolved config, call it from both places, and have re-scores write to a new versioned directory instead of overwriting."

*Why that answer lands:* it gives the root cause (no factored function), the concrete drift, and the fix, including the overwrite policy. That shows you know why duplication happened, not just that it did.

**A3. `mtrain test` the day before a deadline.**
"They've changed the published numbers. `test` is an alias for `evaluate --split test`, which reloads the selected checkpoint and overwrites `test_overall.json` and `stratified.json`. SmolVLA samples actions stochastically, so the new MSE differs by up to a few percent from what's in the paper's tables, and there's no backup of the old file. The stratified file has also lost its length slices, because the command only computes per-task. Meanwhile `summary.json`, the report, the model card and the predictions file still have the original numbers, so the run directory now contradicts itself. Recovery is from git if the paper-ready export happened earlier, but the export doesn't include the predictions or the report. The fix is that re-scoring never overwrites, writing a timestamped sibling directory instead."

*Why that answer lands:* it gives the precise mechanism, the stochastic amplification, the internal inconsistency and the incomplete recovery path. It shows real operational thinking.

**A4. `--apply` on `imbalanced.yaml`.**
"It loads the YAML, replaces the whole `intervention` block with the resolved one, a `custom` block listing the kept episode IDs plus the variable, value and reason, and dumps the file back with PyYAML. Two side effects. First, PyYAML doesn't preserve comments, so the twenty-line header explaining that it's a template, why the variable must come from a profile and what the result can't claim is deleted, and the folded hypothesis strings are reflowed. Second, the kind becomes `custom`, so the run records the intervention as custom, not imbalanced, everywhere except the tags. I'd use a round-trip YAML library, or write the resolved block to a separate file the loader merges, and record `kind: imbalanced` with explicit episode IDs, which the schema already allows."

*Why that answer lands:* it's exact about what happens and why (PyYAML semantics), identifies both the human-facing and the data-facing consequences, and offers fixes that fit the existing schema.

**A5. "From a fresh dataset to a comparison table."**
"The dataset has to be ingested and cleaned in the data engine first, giving something like `clean/v1`. Then `mtrain profile` on that version to see what actually varies. For SO-100 that showed a single task, which is why episode length became the imbalance variable. `mtrain train` with the baseline config builds a training version with lineage, checks the splits, fine-tunes, evaluates against both baselines, and writes the report, model card and feedback artifact. For a targeted experiment, `mtrain build-intervention` in dry-run mode resolves the episodes from the real profile; you inspect them and then `--apply`. Train that config, then `mtrain compare` with `--runs` repeated for each run. Three cautions: excluding interventions are scored on different test splits, `--seed` doesn't change the training seed yet, and `mtrain test` overwrites results, so I'd tell a new user to treat run directories as read-only."

*Why that answer lands:* it's a correct end-to-end workflow, grounded in the real data story, and it volunteers the three operational traps a new user would hit.

---

# Module 12 Summary — the one page to revise

**What this module is.** An eleven-command Typer CLI with lazy engine imports. `train` delegates to the runner, `build-intervention` resolves template configs from real metadata with a dry run by default, `rollout` records "unavailable", and the rest are read-only views. `evaluate`/`test` is a divergent re-implementation that overwrites published results.

**Five claims to make cold**
1. Only `typer`, `yaml` and `rich` load at import, so `--help` and the read-only commands never load the engine or torch.
2. `build-intervention` is dry-run by default and resolves without building. `train` creates the version and its lineage.
3. `evaluate`/`test` overwrite `test_overall.json` for a stochastic model, with different slices and no baselines, and leave the run inconsistent.
4. `--split val` writes validation into `test_overall.json`, and the API doc recommends it.
5. `--apply` deletes the template's explanatory comments and relabels imbalanced as `custom`.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Lazy imports per command | top-level imports | fast startup; no torch for views | none |
| `train` delegates to the runner | CLI-side logic | single implementation | `--seed` inherits the runner bug |
| `evaluate` re-implemented inline | shared stage function | runner stage wasn't factored | drift; overwrite; inconsistency |
| Dry run by default | apply by default | config rewrites need review | none |
| PyYAML round trip | ruamel round-trip | already a dependency | comments and formatting lost |
| `rollout` placeholder command | omit until real | absence is recorded | duplicate untimestamped records |
| Script wrappers | rely on `python -m` | `PATH` convenience | redundant, drifting docs |

**Volunteer unprompted:** the overwrite plus stochasticity; val into `test_overall.json`; comment erasure; the `custom` relabel; `export` missing git and env; the `compare.py` usage error; no CLI tests.

**Numbers and names:** 11 commands (`inspect, profile, train, validate, evaluate, test, rollout, compare, analyze-failures, export, build-intervention`); 7 script wrappers; `--dry-run/--apply` default dry; `--oversample-factor 3.0`; `inspect --version` default `training/v1`; `evaluate` slices `["per_task"]`, batch 32; export subdirectories `figures, tables (empty), metrics, configs, manifests`.

**Sentences worth memorizing**
- "Every state-changing command should be obvious from its name; `mtrain test` overwrites published numbers, and that's the first thing I'd fix."
- "Two implementations of evaluation drifted within one codebase. Factor the stage once, call it twice."
- "A re-score of a stochastic model should be a new artifact, never an overwrite."
- "My config rewriter deleted the comments that told the user what not to claim."
