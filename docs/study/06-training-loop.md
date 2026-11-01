# Module 6 — The training loop around `fit`

**Files covered:** `src/model_training/training/trainer.py` (`training/__init__.py` was covered in Module 1)

**Where we are:** Modules 4 and 5 produced a verified, profiled `training/vN`. This module spends the GPU budget. Module 1 established that the gradient step itself, `SmolVLAPolicy.fit`, lives in the engine. What's here is everything *around* the step: the callback that validates, checkpoints, selects and prunes, plus crash recording and memory release. This is the code you can honestly say you own in "the training loop."

**The one-line thesis:** `Trainer` is a careful *Template-Method-by-callback* wrapper. It runs validation only on the val split, verifies every checkpoint before it can become "best," never prunes the best one, and always saves a final checkpoint. But the checkpoint it selects is chosen by **32 validation frames of a stochastic policy**. That's small enough that the README's validation MSE is about 13× lower than its test MSE. The crash handler also catches fewer failures than its comment claims.

---

## 6.1 Prerequisites — the absolute basics

- **Training step.** One forward pass, loss, backward pass and optimizer update on one batch. "1,000 steps at batch size 4" means 4,000 sampled frames (or sample windows), not 1,000 epochs.
- **Gradient accumulation (`grad_accum_steps`).** Sum the gradients of `k` micro-batches before one optimizer update. The effective batch is `k × batch_size`. Here `k = 1`.
- **Gradient clipping (`max_grad_norm = 1.0`).** If the gradient vector's L2 norm exceeds the threshold, rescale it down. This stops one bad batch from blowing up the weights.
- **Weight decay (`1e-5`).** A small pull toward zero on the weights each step (L2-style regularization).
- **Learning-rate warmup and schedule.** Start the LR near zero and ramp it up linearly over `warmup_steps`. A sudden full LR on pretrained weights, with an optimizer whose statistics are still empty, can wreck them. `docs/API_REFERENCE.md` says the schedule is "warmup then cosine decay". That lives in the engine and is **unverified** here.
- **Optimizer state.** Adam-family optimizers keep two running averages per parameter (first and second moments). Restarting without them behaves differently from continuing, which is why resuming needs a fresh warmup (Module 2).
- **Checkpoint cadence.** Save every `checkpoint_every` steps. **Validation cadence** means scoring on held-out data every `validate_every` steps.
- **Model selection / early stopping.** Pick the checkpoint with the best validation score instead of the last one. That's a guard against overfitting.
- **Winner's curse (selection bias).** If you take the minimum of `m` noisy measurements, the selected value is biased *low*. The more candidates and the noisier the estimate, the bigger the bias. The selected checkpoint's validation score then overstates how good it is.
- **Effective sample size.** Consecutive frames of one episode are highly correlated, so 32 adjacent frames carry roughly the information of a handful of independent samples, not 32.
- **Template Method pattern.** A fixed algorithm skeleton (here, `fit`'s loop) with hook points that subclasses or callers fill in (here, `on_step`). When the hook is a function argument rather than a subclass override, it's often called *inversion of control* or a *callback hook*.
- **Closure / `nonlocal`.** A nested function that captures variables from its enclosing function. `nonlocal x` lets the nested function rebind `x`.
- **`BaseException` vs `Exception`.** `KeyboardInterrupt` (Ctrl-C) and `SystemExit` inherit from `BaseException`, *not* `Exception`, so `except Exception` doesn't catch them. SIGTERM's default action terminates the process without raising anything. SIGKILL (the Linux OOM-killer) can't be caught at all.
- **PyTorch CUDA memory.** `torch.cuda.max_memory_allocated()` reports the peak bytes *handed out by the caching allocator to tensors*. It excludes the allocator's cached-but-free blocks and the CUDA context (often several hundred MB), so it's lower than what `nvidia-smi` shows. `empty_cache()` returns *unreferenced* cached blocks to the driver. It can't free memory that something still references.

---

## 6.2 Shape of `Trainer.train`

```
train(train_view, val_view, episode_ids, episode_weights)
  ├─ guard: model must be "smolvla"
  ├─ build SmolVLAPolicy(resume_from or pretrained); load(contract or None)
  ├─ reset CUDA peak stats; resolve train ids (raise if empty)
  ├─ define on_step(record)            ← the hook: log · validate · checkpoint · verify · select · prune
  ├─ try:   fit_result = policy.fit(..., on_step=on_step)      ← engine owns the loop
  │  except Exception: write crash.json; del policy; empty_cache; raise
  ├─ save checkpoints/final (always)
  ├─ peak GPU; del policy; empty_cache
  └─ return TrainingOutcome(best_checkpoint = best or final, ...)
```

**Decision: drive the engine's `fit` loop through a callback rather than writing the loop here.**
- *What:* `fit` owns the iteration. This module injects behaviour at each step through `on_step(record)`.
- *Why:* the step mechanics (batching against the dataset contract, weighted sampling, the optimizer) are engine-coupled (Module 1). The run *policy* (when to validate, what to keep) is experiment-coupled. The callback is the seam between them.
- *Alternative:* `fit(steps=100)` called in a loop of chunks, returning control between chunks. That's simpler to reason about, but it needs optimizer and schedule state carried across calls (`fit` does accept `optimizer_state=` per `docs/API_REFERENCE.md` line 792, which is unverified). The other alternative is a framework trainer (PyTorch Lightning, HF `Trainer`) with its own callback system.
- *Cost:* the callback runs **inside** the engine's loop, so validation, checkpoint I/O and pruning all happen on the training thread, between steps, and their time is counted in `wall_clock_s` (the README's "Training time 324 s" includes them). An exception inside `on_step` (a full disk during `save_checkpoint`) propagates out of `fit` and is recorded as a training crash.

---

## 6.3 Setup: model construction and resume

```python
if tc.model != "smolvla":
    raise ValueError(f"Trainer implements real gradient-based fine-tuning for model='smolvla' only ...")
```
This duplicates the runner's own branch, which already only constructs a `Trainer` when `model == "smolvla"` (Module 11). It's defensive and correct, and it makes `Trainer` safe to call directly.

```python
checkpoint_dir = self.run_dir.sub("checkpoints")
policy = SmolVLAPolicy(checkpoint=tc.resume_from or tc.pretrained_checkpoint, device=tc.device)
# Resuming one of our own checkpoints: it is already configured (shapes
# and normalization stats) for this exact dataset -- reload as-is. Only
# a *fresh* pretrained checkpoint (which may target a different
# embodiment than train_view's) needs its shapes/stats overridden from
# the dataset contract ...
policy.load(contract=None if tc.resume_from else train_view.contract)
if tc.resume_from:
    meta = SmolVLAPolicy.verify_checkpoint(tc.resume_from)
    self.run_dir.write_json("resume_source.json", meta)
```

**Decision: pass the dataset contract only for a fresh start.**
- *Why:* `smolvla_base` was pretrained with *its own* input and output shapes and normalization. The SO-100 run needs 6-dimensional action, and ALOHA needs 14-dimensional action plus 28-dimensional state (the smoke test's cross-embodiment cases). Loading with the contract rewrites the features and normalization stats from *this* dataset. A resumed checkpoint already has them. Passing the contract again could overwrite the checkpoint's stats with the new version's (identical here, but not in general).
- *Cost:* **a resume isn't checked against the dataset.** `resume_source.json` records the source checkpoint's `dataset_id` and `dataset_version` (written into every checkpoint's `extra`, see below) but never compares them with `train_view`. Resume an SO-100 checkpoint against ALOHA by mistake and the shapes disagree. The failure would surface somewhere inside the engine's batching, as a tensor-shape error far from the cause. `if meta["dataset_id"] != train_view.dataset_id: raise` is one line.
- **Order:** `load` runs *before* `verify_checkpoint`, so a corrupt resume directory fails inside `load` with whatever error the engine raises, and the clear "incomplete checkpoint" message from `verify_checkpoint` (asserted by `test_corrupted_checkpoint_directory_rejected`) is never reached. Swap the two lines.

**Optimizer state isn't saved, and it could be.** Every periodic checkpoint is written with `"optimizer_state_saved": False`, and resume passes no optimizer state. Module 2 framed the fresh warmup on resume as the consequence. Per `docs/API_REFERENCE.md`, the engine's `fit` *returns* `"optimizer_state"` and *accepts* `optimizer_state=`. So this is a choice, not an engine limitation (unverified, since I can't read the engine). The likely reason is disk: Adam's two moments are twice the parameter count, roughly tripling checkpoint size. That's a reasonable choice on a laptop. It isn't recorded as a reason anywhere, and the API doc's §9.2 presents the LR restart as inherent.

```python
import torch  # safe: policy.load() above already required it to succeed
```
This is the lazy import that keeps the module importable without torch (Module 1), but `trainer.py`'s *top-level* `from rde.model_adapters.models import SmolVLAPolicy` means importing this file still loads the engine's model module. The comment is accurate about torch specifically.

```python
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
peak_rss = proc.memory_info().rss if proc else None
```
**Peak stats are reset *after* the model is loaded**, so `peak_gpu_mb` measures training, not the load. The README's "Peak GPU memory 2.2 GB / 3.0 GB" is therefore allocator-reported peak during training. It's lower than `nvidia-smi` would show (context plus cache excluded), so it understates what you'd need to provision on a GPU.

```python
ids = episode_ids if episode_ids is not None else train_view.episode_ids(split="train")
if not ids:
    raise ValueError("no training episodes resolved -- nothing to fine-tune on")
```
The fallback to `split="train"` is for direct callers. The runner always passes train IDs. The empty guard runs **after** the model has been loaded onto the GPU, so an empty-train failure costs a full model load first. Moving it to the top is free.

---

## 6.4 The hook: `on_step`

```python
val_adapter = VisionLanguageAdapter(val_view.contract)
evaluator = Evaluator()
max_val_frames = tc.val_batches * tc.batch_size     # 8 × 4 = 32 with base.yaml

checkpoints: list[dict[str, Any]] = []
best: dict[str, Any] = {"metric": None, "checkpoint": None}
last_step = 0

def on_step(record: dict[str, Any]) -> None:
    nonlocal peak_rss, last_step
    last_step = record["step"] + 1
    self.run_dir.append_jsonl("metrics/train.jsonl", {"experiment_id": self.config.experiment_id, **record})
    if proc is not None:
        peak_rss = max(peak_rss or 0, proc.memory_info().rss)

    step = last_step  # steps completed so far, 1-indexed
    do_ckpt = bool(tc.checkpoint_every) and step % tc.checkpoint_every == 0
    do_val = bool(tc.validate_every) and step % tc.validate_every == 0
    if not (do_ckpt or do_val):
        return
```

- **`best` is a dict, not two variables,** so the closure can *mutate* it without `nonlocal`. `peak_rss` and `last_step` are rebound, so they need `nonlocal`. Mixing both styles in one closure is legal but inconsistent.
- **`record["step"] + 1`** assumes the engine reports 0-indexed step numbers. With `steps = 1000` and `checkpoint_every = 100`, saves land at 100, 200, …, 1000, which matches the README's "step_1000". With `grad_accum_steps > 1` it's unknown whether `record` fires per micro-batch or per optimizer step. If per micro-batch, every cadence would be off by a factor of `k`. That's unverifiable here and unexercised (`grad_accum_steps: 1` everywhere).
- **`bool(tc.checkpoint_every) and …`** makes `0` mean "never" and avoids `step % 0` (a `ZeroDivisionError`). That's the defensive branch, and it has a purpose.
- **One JSONL append per step** means 1,000 file open/write/close cycles. That's negligible next to a SmolVLA step (around 300 ms per step at the README's 324 s per 1,000 steps), but it's an fsync-free append, so the Module 3 torn-write caveat applies.
- **Peak RSS is sampled once per step.** It misses transient spikes between samples, which is fine as a rough number.

### Validation: what "best" is chosen by

```python
    val_metric = None
    if do_val:
        policy.eval_mode()
        result = evaluator.evaluate(
            val_view, policy, val_adapter, split="val", batch_size=tc.batch_size,
            max_frames=max_val_frames, persist=False,
        )
        policy.train_mode()
        val_metric = result.metrics.get(ec.selection_metric)
        self.run_dir.append_jsonl("metrics/validation.jsonl",
                                  {"experiment_id": self.config.experiment_id, "step": step, **result.metrics})
```

- **`split="val"` is hard-coded here,** and *this* is where "selection never touches test" is actually enforced. It isn't in the nonexistent `model_training.experiments.selection` that the schema comment cites (Module 2). Say it that way: it's enforced structurally, by the only call site.
- **`persist=False`** stops the evaluator from writing these interim results into the engine's catalog (inferred from the flag name). Contrast this with the feedback step, which uses `persist=True` (Module 8).
- **`eval_mode()` / `train_mode()` toggling** switches dropout and normalization behaviour. If `evaluate` raises, `train_mode()` is never called, but the exception aborts training anyway, so it doesn't matter.

**The validation sample is tiny.** `max_val_frames = val_batches × batch_size = 8 × 4 = 32` frames. The SO-100 val split is 5 episodes. The README's test split of 5 episodes has 1,908 frames, so val is about 1,900 frames. **Validation scores roughly 1.7% of the validation set.** Which 32 frames is decided by the engine's `Evaluator` with `max_frames`, and I can't verify that. If it takes the first `N` frames in iteration order (the most common implementation), the 32 frames are **the first ~1 second (at 30 fps) of the first validation episode**.

**Circumstantial evidence that this matters, from the README itself:**
- README §2.2 says SO-100 error "is lowest at the start (0.051) ... of an episode and peaks in the middle ... at 0.144". That's mean absolute error by progress decile on test.
- README result 2 says validation MSE "reached its minimum (**0.0018**) at step 1,800", while the selected checkpoint's **test** MSE is **0.0239**. **Test is about 13× worse than validation.**
- An MAE of about 0.05 corresponds to an MSE on the order of 0.003, the same order as the validation numbers. Validation on start-of-episode frames would produce exactly this pattern.

I can't *prove* the first-N behaviour without the engine. But the 13× val–test gap is a real number in the README, and nothing else in the code explains it as naturally. (Distribution shift between val and test episodes is possible with 5 episodes each, but 13× is large.) Even if the frames were drawn at random, the setup has three compounding problems:
1. **Correlated frames.** 32 frames from one or two episodes have a tiny effective sample size.
2. **A stochastic policy.** README §2.3 says flow-matching sampling moves MSE by up to 5% when re-scoring the *same* frames.
3. **Selection over 10–20 noisy estimates** (every 100 steps across 1,000–2,000 steps) means a winner's-curse bias in `best_val_metric`. The reported "selection value" is optimistic by construction.

**Decision: a fixed small validation budget instead of the full val split.**
- *Why (inferred):* a full validation pass over about 1,900 frames at the README's roughly 47–50 ms per frame is about 90 seconds per pass, about 15 minutes over ten passes. That would triple a 5-minute training run on a laptop GPU.
- *Cost:* checkpoint selection is close to noise on SO-100, where validation MSE was still falling. It probably picked the late checkpoint correctly *because* the trend dominated the noise.
- *The fix that keeps the budget:* choose the 32 (or 128) validation frames **once**, stratified across every val episode and across episode progress, with a fixed seed. Score every checkpoint on that **same frozen subset**, and average two stochastic samples per frame. The cost stays the same, the noise drops sharply, and every checkpoint is compared on identical inputs.

### Checkpoint, verify, select, prune

```python
    if do_ckpt:
        ckpt_path = checkpoint_dir / f"step_{step}"
        policy.save_checkpoint(ckpt_path, extra={
            "experiment_id": ..., "step": step,
            "dataset_id": train_view.dataset_id, "dataset_version": train_view.version,
            "optimizer_state_saved": False,
        })
        # A process killed mid-write (disk full, OOM-killer, Ctrl-C)
        # can leave a partial checkpoint directory; verifying before
        # trusting it as `best` (and therefore protected from
        # pruning) keeps a corrupt save from silently displacing a good one.
        try:
            SmolVLAPolicy.verify_checkpoint(ckpt_path)
            verified = True
        except (FileNotFoundError, ValueError) as exc:
            verified = False
            self.run_dir.append_jsonl("metrics/checkpoint_errors.jsonl", {...})
        checkpoints.append({"step": step, "path": str(ckpt_path), "val_metric": val_metric, "verified": verified})
        if verified and val_metric is not None:
            better = (best["metric"] is None
                      or (ec.selection_mode == "min" and val_metric < best["metric"])
                      or (ec.selection_mode == "max" and val_metric > best["metric"]))
            if better:
                best.update(metric=val_metric, checkpoint=str(ckpt_path))
        self._prune_checkpoints(checkpoint_dir, checkpoints, tc.keep_last_n_checkpoints, best)
```

- **`extra` metadata** stamps every checkpoint with its experiment, step and dataset version. That's what lets `build_model_artifact` later trust the checkpoint's own claim about its training data (Module 9). It's a good provenance detail.
- **The verify-before-trust comment describes a scenario its mechanism can't handle.** "A process killed mid-write (disk full, OOM-killer, Ctrl-C)": if the process is *killed*, the verification line never runs in that process, and there's no subsequent `best` to protect. A disk-full error during `save_checkpoint` would *raise*, not return, so it would crash training instead of producing an unverified checkpoint. **Verification only catches a `save_checkpoint` that returns normally but wrote an incomplete directory.** What `verify_checkpoint` checks is, per `test_corrupted_checkpoint_directory_rejected`, that all required files exist ("only one of five required files" gives "incomplete"). That's a presence check, not an integrity check: a truncated `model.safetensors` would pass. The mechanism is sound as cheap insurance, but the comment oversells the threat model. The case where partial checkpoints really bite, resuming from one after a kill, goes through the resume path (6.3), where `load` runs first.
- **Validation and checkpoint cadences must coincide for selection to work.** A checkpoint saved at a step with no validation has `val_metric = None` and can never become `best`. A validation at a step with no checkpoint has no weights to select. With `checkpoint_every: 50, validate_every: 30`, **no checkpoint is ever eligible**, and selection silently falls back to `final` (below). Nothing validates that `checkpoint_every % validate_every == 0`. Both are 100 in `base.yaml`, so the shipped configs are fine.
- **A NaN validation metric can lock selection.** `val_metric is not None` admits `float("nan")`. If the *first* validated checkpoint has a NaN metric, `best["metric"] is None` makes it `better`, so `best = NaN`. After that, `x < NaN` is always `False`, and **no later checkpoint can ever displace it.** Whether the engine's `fit` raises on a NaN loss (the crash-handler comment says "a NaN loss" as if it raises) or keeps going is unverified. If it keeps going, this is a real path. Guard with `math.isfinite(val_metric)`.
- **`selection_mode` is checked with string equality.** `Literal["min", "max"]` at the schema level (Module 2) makes the third case impossible. Good.

### `_prune_checkpoints`

```python
@staticmethod
def _prune_checkpoints(checkpoint_dir, checkpoints, keep_last_n, best) -> None:
    """Drop periodic checkpoints beyond ``keep_last_n``, always keeping the
    best-by-validation one regardless of recency."""
    if keep_last_n <= 0:
        return
    import shutil

    periodic = [c for c in checkpoints if not c.get("final")]
    periodic.sort(key=lambda c: c["step"])
    keep_paths = {best.get("checkpoint")} | {c["path"] for c in periodic[-keep_last_n:]}
    for c in periodic[:-keep_last_n] if len(periodic) > keep_last_n else []:
        if c["path"] in keep_paths:
            continue
        p = Path(c["path"])
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        c["pruned"] = True
```

Walk it for `keep_last_n = 3`, with saves at 100…500 and the best at 200: `periodic[-3:]` = {300, 400, 500}, `keep_paths` = those plus 200, and the candidates `periodic[:-3]` = {100, 200}. So 100 is deleted and 200 is kept. That's correct.

- **`keep_last_n <= 0` means "keep everything".** That's a reasonable reading of "keep the last 0", read as "pruning disabled", but it's undocumented and opposite to the literal meaning.
- **Unverified (possibly corrupt) checkpoints count toward the last-N.** A corrupt `step_500` occupies a slot and pushes a good `step_200` out, unless 200 is best.
- **`ignore_errors=True` plus `c["pruned"] = True` unconditionally** means a deletion that fails (permissions, a file handle still open on network storage) is marked pruned anyway. The record and the disk disagree, silently.
- **Pruned entries stay in `checkpoints`** with `pruned: True`, so `training_outcome.json` lists paths that no longer exist. They're marked, so a reader can tell. That's fine.
- **This is a pure function with no I/O except `rmtree`, and it has no unit test.** Grep for `_prune` in `tests/` finds nothing. The only coverage is the smoke test (torch required), which sets `keep_last_n_checkpoints: 1` with checkpoints at steps 2 and 4 and doesn't assert what was pruned. It's the easiest function in the trainer to test with `tmp_path`, and it's untested.

---

## 6.5 Calling `fit`, the crash path, and the final checkpoint

```python
started = time.perf_counter()
try:
    fit_result = policy.fit(
        train_view, episode_ids=ids, episode_weights=episode_weights,
        steps=tc.steps, batch_size=tc.batch_size, learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay, grad_accum_steps=tc.grad_accum_steps,
        max_grad_norm=tc.max_grad_norm, warmup_steps=tc.warmup_steps,
        seed=tc.seed, on_step=on_step,
    )
```

What's passed and what isn't: `num_workers`, `mixed_precision` and `device` (device went to the constructor) **aren't passed to `fit`**. This confirms Module 2's dead-field finding at the only call site. `seed=tc.seed` means the training seed comes from config only. The CLI `--seed` never reaches here (Module 11).

```python
except Exception as exc:
    # A crash here (OOM, a NaN loss, SIGTERM) previously left the run
    # directory in an undetectable half-finished state ...
    self.run_dir.write_json("crash.json", {
        "experiment_id": ..., "error_type": type(exc).__name__, "error": str(exc),
        "steps_completed": last_step, "checkpoints": checkpoints,
        "wall_clock_s": time.perf_counter() - started,
    })
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise
```

**Decision: record a crash marker and re-raise; never swallow.**
- *Why:* the comment narrates a real bug that was fixed. A crashed run used to be indistinguishable from one that never started. `crash.json` with `steps_completed` and the surviving checkpoint list makes a crashed run diagnosable and partly salvageable, since a verified `step_N` can be resumed from.
- **What it actually catches, versus what the comment claims:**

| Failure | Caught by `except Exception`? | `crash.json` written? |
|---|---|---|
| CUDA out-of-memory (`torch.OutOfMemoryError`, a `RuntimeError` subclass) | yes | yes |
| NaN loss | only if the engine *raises* on NaN (unverified) | conditional |
| Exception inside `on_step` (full disk during save, JSON error) | yes | yes |
| **Ctrl-C** (`KeyboardInterrupt`, a `BaseException`) | **no** | **no** |
| **SIGTERM** (scheduler or `kill`) | **no**: default handler terminates without an exception | **no** |
| **Host OOM-killer** (SIGKILL) | **no**: uncatchable | **no** |

  The comment's "SIGTERM" is wrong, and the run-directory status gap from Module 3 remains for the two most common ways a long job dies on a shared machine. *Fix:* `except BaseException`, for Ctrl-C, plus a `signal.signal(SIGTERM, …)` handler that raises, and a `status: running` marker written *before* `fit`, so that a SIGKILL'd run is at least identifiable as "running with no heartbeat".
- **`del policy; empty_cache()` frees less than it appears to.** At this point the exception object `exc` holds a traceback whose frames include `fit`'s frame, and `fit`'s `self` *is* the policy. That's a live reference to every parameter and optimizer tensor until the exception is fully handled upstream. So `empty_cache()` here can't return that memory. The `del` is harmless, but the stated purpose, "release the GPU memory this policy was holding rather than leaving it for the caller's next model", isn't achieved while the exception propagates. The runner doesn't load another model after a training exception anyway (it re-raises out of `run`), so the practical impact is nil. The comment just misdescribes the effect.

```python
wall_clock_s = time.perf_counter() - started

# Always checkpoint the final step, even off-cadence, so resume and
# deployment never require rerunning training to get a usable artifact.
final_path = checkpoint_dir / "final"
policy.save_checkpoint(final_path, extra={..., "step": fit_result["steps"], ..., "final": True})
checkpoints.append({"step": fit_result["steps"], "path": str(final_path), "val_metric": None, "final": True})
```

- **`wall_clock_s` excludes the final save** but includes every periodic validation and save. It's "training plus in-loop overhead", not pure optimization time.
- **The final checkpoint is never verified** (no `verify_checkpoint` call, unlike the periodic ones). It's verified later only if it becomes the selected checkpoint, inside `build_model_artifact` (Module 9).
- **The final checkpoint is never validated** (`val_metric: None`), so it can never compete for `best`. When `steps % checkpoint_every == 0` (the shipped configs: 1000 % 100), the last periodic checkpoint `step_1000` and `final` are **the same weights saved twice**: one is validated and selectable, the other isn't. SmolVLA is reported publicly as about 450M parameters, so each save is plausibly around 1–2 GB (I didn't measure this). Under the shipped configs it's a duplicate on every run.
- **Placing the final save outside the `try`** means a failure while saving it (disk full) raises *without* a `crash.json`, though training itself finished.

```python
peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
del policy
if torch.cuda.is_available():
    torch.cuda.empty_cache()

return TrainingOutcome(
    steps_completed=fit_result["steps"],
    final_train_loss=fit_result["final_loss"],
    best_checkpoint=best["checkpoint"] or str(final_path),
    best_val_metric=best["metric"],
    ...
)
```

- **Releasing memory before return** is correct here: the runner immediately loads a second `SmolVLAPolicy` from the selected checkpoint for evaluation (Module 11), and two copies resident at once is exactly what an 8 GB laptop GPU can't afford. On this path there's no traceback pinning the model. `fit_result` may still hold `optimizer_state` (if the engine returns it, per the API doc) until `train` returns, which is moments later.
- **The `/ 1e6` is megabytes in SI units**, not MiB. The README's "2.2 GB" is consistent with SI.
- **`best["checkpoint"] or str(final_path)` is a silent fallback.** If no checkpoint was ever eligible (cadence mismatch, a wrong `selection_metric` name, a zero-size val split from Module 5, every checkpoint unverified), the run reports `best_checkpoint = final` with `best_val_metric = None`. The runner writes that into `model.json` as `selection_value: null`, and **no limitation is recorded**. The run looks selected, but it was just last-step. Add a limitation string whenever the fallback fires.

---

## 6.6 Honest gaps to own before an interviewer finds them

1. **Checkpoint selection uses 32 validation frames** (`val_batches × batch_size`), probably the opening second of one episode (unverified engine behaviour), on a stochastic policy, with a min over 10–20 candidates. *Consequence:* the README's selected val MSE (0.0018) is about 13× lower than test (0.0239). Selection is near-noise, and `best_val_metric` is optimistically biased.
2. **The crash handler misses Ctrl-C, SIGTERM and the OOM-killer**, contrary to its comment. *Consequence:* the most common ways long jobs die leave no `crash.json`.
3. **A silent fallback to `final`** when no checkpoint is eligible, with no limitation recorded. *Consequence:* "selected checkpoint" can mean "last step" with no indication.
4. **A NaN first validation locks `best` forever** (NaN passes `is not None`, and every later comparison is false). *Consequence:* selection is pinned to a broken checkpoint (if `fit` doesn't raise on NaN).
5. **Validation and checkpoint cadences aren't checked for alignment.** *Consequence:* misaligned settings silently disable selection.
6. **A resume isn't checked against the training dataset, and `load` precedes `verify_checkpoint`.** *Consequence:* a wrong resume fails as an obscure tensor error, and the clear verification message is unreachable.
7. **Optimizer state is never saved or restored, though the engine's `fit` reportedly supports it.** *Consequence:* resume restarts warmup and schedule. It's a defensible disk trade-off, but undocumented as a choice.
8. **`final` duplicates the last periodic checkpoint, is never verified or validated, and is saved outside the `try`.** *Consequence:* about one wasted model-size copy per run, and a save failure without a crash marker.
9. **`verify_checkpoint` is a presence check, and the comment's threat model (process killed mid-write) can't be caught in-process.** *Consequence:* the verification is narrower than described.
10. **`_prune_checkpoints` is untested; failed deletions are marked pruned; corrupt checkpoints occupy keep slots.** *Consequence:* disk state and records can diverge silently.
11. **`del policy; empty_cache()` in the crash path can't free memory pinned by the traceback.** *Consequence:* a comment that misdescribes the effect (no practical harm today).
12. **`peak_gpu_mb` is allocator-peak, measured after load.** *Consequence:* the README's GPU figures understate what you'd need to provision.
13. **The empty-train-IDs guard runs after the model load.** *Consequence:* a wasted GPU load on a trivially detectable error.
14. **Step semantics under `grad_accum_steps > 1` are unknown.** *Consequence:* the cadence could be off by a factor of `k` (unexercised).

---

## The shape to carry into Module 7

```
 Trainer.train(view, view, train_ids, weights)
   SmolVLAPolicy(resume_from | smolvla_base).load(contract | None)
   │
   │  engine fit loop (steps, bs=4, lr, wd, clip, warmup, seed) ──calls──▶ on_step(record)
   │                                                                     ├─ train.jsonl  (every step)
   │                                                                     ├─ every validate_every:
   │                                                                     │    Evaluator(val, max_frames=32) ─▶ validation.jsonl
   │                                                                     ├─ every checkpoint_every:
   │                                                                     │    save step_N (no optimizer) → verify (presence)
   │                                                                     │    best ← min(val_metric)  [needs val at same step]
   │                                                                     └─ prune: keep last N + best
   ├─ except Exception → crash.json, re-raise     (not Ctrl-C / SIGTERM / SIGKILL)
   ├─ save final (unverified, unvalidated)
   └─ TrainingOutcome(best_checkpoint = best or final  ← silent fallback, best_val_metric, peaks, wall clock)
                     │
                     ▼  runner reloads best_checkpoint for test evaluation (Module 7)
```

---

## Drill — answer from memory, files closed

1. Walk through exactly what happens at step 300 of a run with `checkpoint_every: 100`, `validate_every: 100`, `keep_last_n_checkpoints: 3`.
2. *Why didn't you just* validate on the whole validation split every time?
3. *What breaks in production:* the job scheduler sends SIGTERM at a time limit, 70% through training. What's on disk, and what can you recover?
4. The README reports validation MSE 0.0018 and test MSE 0.0239 for the same checkpoint. Explain the gap.
5. *Interviewer framing:* "How did you choose which checkpoint to evaluate, and how confident are you that it was the right one?"

---

# Appendix A — Model answers to the drill

**A1. Step 300.**
"The engine's `fit` finishes optimizer step 300 and calls my `on_step` callback. It appends the step's loss, learning rate and grad norm to `train.jsonl` and samples resident memory. 300 is a multiple of both cadences, so it switches the policy to eval mode, runs the evaluator on the val split capped at 32 frames, switches back to train mode, and logs the full metrics to `validation.jsonl`. Then it saves `checkpoints/step_300` with metadata (experiment, step, dataset version, and a flag that optimizer state wasn't saved), and verifies the directory has all required files. If it's verified and its val MSE beats the running best, it becomes best. Finally pruning keeps the last three periodic checkpoints plus the best. At step 300 there are only three, so nothing is deleted yet. At step 400, step 100 would be deleted unless it was the best."

*Why that answer lands:* it's the exact sequence, including the eval/train toggle and the conditions for becoming best, plus a correct worked pruning example. It shows you could reimplement it.

**A2. Why not validate on the full split?**
"Cost. The val split is about 1,900 frames, and SmolVLA inference ran at roughly 50 ms per frame on the laptop GPU, so a full pass is about a minute and a half. Ten of them would roughly triple a five-minute run. So I capped it at `val_batches × batch_size`, 32 frames. That turned out to be the weak point of the design. Thirty-two correlated frames, probably all from the start of one episode, scored with a stochastic flow-matching sampler, and then a minimum taken over ten or twenty checkpoints. The selected validation number is noisy and biased low. The fix that keeps the cost is to choose a fixed, stratified subset once (frames spread across every val episode and across episode progress), score every checkpoint on that same subset, and average a couple of samples per frame."

*Why that answer lands:* it gives the real budget arithmetic, owns the consequence precisely, and offers a fix that respects the original constraint rather than just saying "use more data."

**A3. SIGTERM at 70%.**
"Python's default SIGTERM handler terminates the process without raising, so my `except Exception` never runs. No `crash.json`, no final checkpoint, no `training_outcome.json` or `summary.json`. On disk you'd have `train.jsonl` and `validation.jsonl` up to about step 700, and the last three periodic checkpoints plus the best, each verified when saved, so you can resume from one of them in a new run with `resume_from`. You'd lose the optimizer state, so the resume restarts warmup. The run directory itself is now non-empty, so retrying the same experiment the same day is blocked. The fix is a SIGTERM handler that raises, catching `BaseException` so Ctrl-C is recorded too, and a 'running' status file written before training, so a hard kill is at least identifiable."

*Why that answer lands:* it knows that SIGTERM doesn't raise by default (a classic gap), inventories exactly what survives, and connects it to the retry-blocking behaviour from another module.

**A4. 0.0018 val vs 0.0239 test.**
"Several things push validation down, and I think one dominates. Validation scores only 32 frames, and if the evaluator takes the first frames in order, that's the first second of one validation episode. My own per-phase analysis shows error is lowest at the start of episodes and peaks mid-episode. Second, the selected value is the minimum over about twenty noisy checkpoint evaluations, so it's biased low by selection alone. Third, the val and test splits are five episodes each, so episode-to-episode variation is large. I can't fully separate these without the engine code in front of me, but I wouldn't report that validation number as an estimate of anything. The test number is the honest one, and the fix is a frozen, stratified validation subset."

*Why that answer lands:* it gives a mechanistic hypothesis tied to your own data, then two other contributing causes, and states the uncertainty explicitly. That's the posture of a careful experimentalist.

**A5. "How did you choose which checkpoint, and how confident are you?"**
"Checkpoints every 100 steps, each validated on the val split only (the trainer hard-codes val, so test never influences selection) and verified for completeness before it's eligible. The lowest validation MSE wins, and pruning never deletes it. If nothing is eligible, it falls back to the final checkpoint, which I'd now flag explicitly in the report, because today it's silent. Confidence: moderate on SO-100, low as a general method. On SO-100, validation was still decreasing at the end of training, so 'pick a late checkpoint' was almost certainly right regardless of noise. But the selection signal itself is 32 frames of a stochastic policy, and the validation minimum is about 13× below test, so it's not a precise estimator. If the curve had been flat, I'd have been picking among checkpoints at random."

*Why that answer lands:* it separates the *mechanism's* correctness (no test leakage, verified, protected) from the *estimate's* quality. It also explains why the result was probably right anyway, which is more convincing than either defending or disowning it.

---

# Module 6 Summary — the one page to revise

**What this module is.** A wrapper around the engine's `SmolVLAPolicy.fit` that injects run policy through an `on_step` callback: val-only validation, checkpointing with presence verification, best-by-validation selection, pruning that protects the best, a forced final checkpoint, a crash marker, and GPU release before evaluation.

**Five claims to make cold**
1. The gradient step is the engine's. The trainer owns cadence, selection, pruning, crash recording and memory release.
2. Selection never sees test, because the only validation call hard-codes `split="val"`.
3. Only verified checkpoints with a validation score can be best, and the best is never pruned.
4. Selection uses 32 frames of a stochastic policy, so the val minimum (0.0018) is about 13× below test (0.0239). The fix is a frozen, stratified subset.
5. The crash marker covers in-process exceptions, not Ctrl-C, SIGTERM or SIGKILL.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Callback into engine `fit` | own loop / Lightning | step mechanics are contract-coupled | callback time counted as training; `on_step` errors look like training crashes |
| Contract only on fresh start | always pass contract | resumed checkpoint already has its stats | resume not checked against dataset |
| No optimizer state in checkpoints | save moments | about 3× smaller checkpoints (inferred) | resume restarts warmup and schedule |
| 32-frame validation | full val split | about 90 s per full pass vs a 5-minute run | near-noise selection, winner's curse |
| Verify before eligible | trust every save | a corrupt save can't become a protected best | presence check only |
| Keep last N plus best | keep all / last N only | bounded disk, best preserved | untested; failed deletes marked pruned |
| Always save `final` | rely on periodic | a usable artifact even off-cadence | duplicate of `step_N` when aligned; unverified |
| Record `crash.json` and re-raise | swallow / no marker | diagnosable crashes | misses BaseException and signals |

**Volunteer unprompted:** the 13× val–test gap and its likely cause; the silent fallback to `final`; NaN locking `best`; the signals not caught; `_prune_checkpoints` has no test; `final` duplicates `step_1000`.

**Numbers and names:** base config 1,000 steps, batch 4, lr 1e-4, warmup 50, wd 1e-5, clip 1.0, ckpt and val every 100, `val_batches` 8 (so 32 frames), keep 3; extended lr 3e-5, warmup 20; README val min 0.0018 at step 1,800 vs test 0.0239; SO-100 324 s and 385 s, ALOHA 576 s; peak GPU 2.2 / 3.0 GB (allocator peak, SI); about 50 ms per frame for SO-100 inference; files `train.jsonl`, `validation.jsonl`, `checkpoint_errors.jsonl`, `crash.json`, `resume_source.json`, `checkpoints/step_N`, `checkpoints/final`.

**Sentences worth memorizing**
- "The engine owns the step; I own what happens between steps."
- "Test can't influence selection because the only validation call in the trainer hard-codes the val split."
- "My validation signal was 32 frames of a stochastic policy, which is why val sits 13× below test. Selection was right on SO-100 because the curve was still falling, not because the estimate was good."
- "`except Exception` doesn't see Ctrl-C, and SIGTERM doesn't raise at all; the crash marker covers less than its comment says."
