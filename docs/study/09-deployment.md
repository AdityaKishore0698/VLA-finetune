# Module 9 — Model artifact, registry and deployment loader

**Files covered:** `src/model_training/deployment/artifact.py`, `src/model_training/deployment/registry.py`, `src/model_training/deployment/loader.py` (`deployment/__init__.py` was covered in Module 1)

**Where we are:** Modules 6–8 produced a selected checkpoint, its test metrics and a feedback artifact. This module answers "what exactly is this checkpoint, and how would I run it?" `build_model_artifact` packages its identity, `ModelRegistry` is meant to let you look it up, and `DeployedModel` is meant to load and serve it with health checks and latency timing. The README and `docs/DEPLOYMENT.md` present this as the project's deployment layer.

**The one-line thesis:** The artifact is a good identity record, and it prefers the checkpoint's own claims about its training data. But the "deployment" layer is **never executed by the pipeline**, its health check **never compares the checkpoint to the contract** despite its docstring, it checks normalization on the object inference doesn't use, and it serves **dataset `Sample` objects, not robot observations**. It's a replay harness, not a deployment.

---

## 9.1 Prerequisites — the absolute basics

- **Model artifact.** A trained model packaged with the metadata needed to use and trust it: weights, configuration, input/output schema, training data identity, code version, dependencies and metrics.
- **Model registry.** A catalogue of artifacts that answers "which model is this?" and, in mature systems (MLflow Model Registry, SageMaker), "which version is *promoted* to staging or production?", through stages, aliases and transitions. Lookup alone is a catalogue. Promotion is what makes it a registry.
- **Pre- and postprocessor.** The input normalization and tokenization before the network, and the action de-normalization after it. In LeRobot-style checkpoints these are saved as separate files (`policy_preprocessor.json` and `policy_postprocessor.json`, per `docs/DEPLOYMENT.md` Table 1), because the normalization statistics live there, not in the weights.
- **Health check / readiness probe.** Verifying, before serving, that a model can actually serve the inputs it will get. The valuable checks are *compatibility* checks (does this model's expected input match what I'll feed it?), not just *presence* checks.
- **Training/serving skew.** Any difference between how inputs were processed at training time and at serving time (different normalization stats, camera order, image size). It's the most common silent production ML failure.
- **Action chunking.** SmolVLA predicts a *sequence* of future actions per call. A robot controller executes some of the chunk, then re-queries.
- **Latency measurement on a GPU.** CUDA calls are asynchronous: Python returns before the GPU finishes. Timing with `time.perf_counter()` measures real GPU work only if something forces a synchronization (`torch.cuda.synchronize()`, or copying results to the CPU). The first call also includes one-off warm-up costs (kernel selection, memory allocation).

---

## 9.2 `build_model_artifact`

```python
@dataclass
class ModelArtifact:
    model_id: str; model_version: str; artifact_path: str; architecture: str
    pretrained_model: str; training_run_id: str; experiment_id: str
    dataset_id: str; dataset_version: str
    observation_schema: dict; action_schema: dict; embodiment: str | None
    dependencies: dict; git_commit: dict; created_at: str; metrics: dict

def build_model_artifact(*, checkpoint_path, contract, run_dir_path, experiment_id, tags,
                         dataset_id, dataset_version, environment, git_info, metrics) -> ModelArtifact:
    """Assemble a ModelArtifact from a checkpoint already saved by
    SmolVLAPolicy.save_checkpoint -- verified, not assumed complete."""
    meta = SmolVLAPolicy.verify_checkpoint(checkpoint_path)
    model_id = f"{tags.get('model', 'smolvla')}-{experiment_id}"
    return ModelArtifact(
        model_id=model_id,
        model_version=str(meta.get("step", "final")),
        artifact_path=str(checkpoint_path),
        architecture="smolvla",
        pretrained_model=meta.get("checkpoint_source", ""),
        training_run_id=str(run_dir_path),
        experiment_id=experiment_id,
        dataset_id=meta.get("dataset_id", dataset_id),
        dataset_version=meta.get("dataset_version", dataset_version),
        observation_schema=contract.get("observation", {}),
        action_schema=contract.get("action", {}),
        embodiment=tags.get("embodiment"),
        dependencies={"torch": environment.get("torch"), "lerobot": environment.get("lerobot")},
        git_commit=git_info,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        metrics=metrics,
    )
```

What's good:
- **It verifies the selected checkpoint.** This is the only place the `final` checkpoint is ever verified, when it's the fallback selection (Module 6). So an incomplete selected checkpoint fails *here*, before the artifact is registered.
- **The dataset identity comes from the checkpoint's own metadata first.** `meta.get("dataset_id", dataset_id)` prefers what `Trainer` stamped into the checkpoint at save time (`extra={"dataset_id": ..., "dataset_version": ...}`, Module 6) over what the caller says. The checkpoint is the authority on what it was trained on. That's the right provenance direction, and it would catch a caller passing the wrong version, though only by silently preferring the checkpoint's answer, not by flagging the disagreement.
- **Keyword-only arguments** (`*`) make an eleven-field constructor call impossible to get wrong by position.

What's weaker:
- **`metrics` are the *test* metrics** (`test_overall.metrics` in the runner, Module 11). `docs/DEPLOYMENT.md` Table 2 and the registry docstring both say `row["metrics"]` answers "**What metrics justified selection?**". **Selection was justified by the *validation* metric** (Module 6), which isn't in the artifact at all. Putting test metrics under that label invites exactly the confusion the project otherwise works hard to avoid.
- **`model_id = f"{tags['model']}-{experiment_id}"`** takes its prefix from a free-form tag. `experiment_id` is unique per load (Module 2), so IDs don't collide, but the prefix is cosmetic and can disagree with `architecture` (hard-coded `"smolvla"`).
- **`model_version = str(meta.get("step", "final"))`.** The `final` checkpoint is saved with `"step": fit_result["steps"]` (Module 6), so this is always a step number and the `"final"` default is unreachable for this codebase's checkpoints. "Version" here means "training step", not a registry version.
- **`pretrained_model = meta["checkpoint_source"]`.** For a resumed run (`baseline_extended`), this is plausibly the *resume path* (the source the policy was constructed from), not `lerobot/smolvla_base`. That's engine-side and unverified. If so, the lineage of weights is one hop deep: you'd follow `resume_source.json` (Module 6) by hand to get to the base model.
- **`observation_schema` and `action_schema` come from the *view's* contract**, not the checkpoint, so the artifact can't show a mismatch between them either.
- **`dependencies` records only torch and lerobot.** It misses `transformers` and `safetensors`, as Module 3 found for `environment.json`.

---

## 9.3 `ModelRegistry`

```python
class ModelRegistry:
    def __init__(self, path: Path):
        self._index = JsonlIndex(Path(path))

    def register(self, artifact: ModelArtifact) -> None:
        self._index.append(artifact.to_dict())

    def get(self, model_id: str) -> dict[str, Any] | None:
        """The most recent record for ``model_id``. ... returning the first match
        returned the oldest, stalest one. Later rows supersede earlier ones for the same id."""
        row = None
        for row in self._index.query(model_id=model_id):
            pass
        return row

    def list(self) -> list[dict[str, Any]]:
        return self._index.read_all()

    def for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        return list(self._index.query(experiment_id=experiment_id))
```

- **"Last write wins" over an append-only log** is the correct read semantics, and the docstring records the bug it fixed (the earlier version returned the first match, which was the stalest). The `for row in ...: pass` idiom is terse: it leaves the loop variable bound to the last yielded row, or `None` if there were none. Some reviewers will find it too clever. `rows = list(...); return rows[-1] if rows else None` reads more plainly.
- **It isn't wired in.** `docs/DEPLOYMENT.md` §7 says so honestly: "`ExperimentRunner.run` writes each model record directly through `ExperimentIndex.record_model` ... and never calls `ModelRegistry.register`." Grep confirms that `ModelRegistry` appears only in `deployment/__init__.py`, `registry.py` and `test_deployment.py`. It's *compatible* with the runner's file, because `record_model(**model_artifact.to_dict())` writes the same row shape, so `ModelRegistry("experiments/models.jsonl").get(...)` would work. It's just never constructed outside the test.
- **It's a catalogue, not a registry.** There's no stage, alias, promotion, or "which model is deployed" field. The question a registry exists to answer, "what's in production?", has no representation. For a project with no production, that's honest scope. Just don't call it a registry in the MLflow sense.
- **It inherits every `JsonlIndex` weakness** (Module 3): a full scan per lookup, no locking, and one torn line breaks `get` and `list` for good.
- **The one test** (`test_model_registry_round_trips`) covers register, get, for_experiment and the missing-ID case. It doesn't cover the "most recent wins" fix that the docstring narrates. A two-row test would pin it.

---

## 9.4 `DeployedModel`: what it checks, and what it doesn't

```python
class DeploymentHealthError(Exception):
    """Raised by DeployedModel.health_check on any checkpoint/contract
    mismatch, detected before inference rather than surfacing as a confusing
    tensor-shape error mid-prediction."""
```

```python
def health_check(self) -> dict[str, Any]:
    """Everything that must be verified before inference. ..."""
    meta = SmolVLAPolicy.verify_checkpoint(self.checkpoint_path)  # raises on incomplete/foreign dir

    action = self.contract.get("action", {})
    if not action.get("shape"):
        raise DeploymentHealthError("dataset contract has no action shape -- cannot verify compatibility")

    obs = self.contract.get("observation", {})
    images = [k for k, v in obs.items() if v.get("plane") == "external"]
    if not images:
        raise DeploymentHealthError("dataset contract has no image modality -- SmolVLA requires at least one camera")
    for k in images:
        shape = obs[k].get("shape")
        if not shape or len(shape) != 3:
            raise DeploymentHealthError(f"camera {k!r} has no valid (H, W, C) shape in the contract: {shape}")

    state_keys = [k for k, v in obs.items() if v.get("plane") == "inline"]
    for k in state_keys:
        if not obs[k].get("normalization"):
            raise DeploymentHealthError(
                f"state modality {k!r} carries no normalization statistics in the dataset contract -- "
                "the checkpoint's preprocessor expects normalized state and per-run normalization is missing")

    return {"status": "ok", "checkpoint_meta": meta, "cameras": images, "state_keys": state_keys}
```

**Read it adversarially.** `meta` is loaded, and then **never used again** except to be returned. Every other check looks at `self.contract` alone. So:

| Promised (docstrings, `DEPLOYMENT.md` §4) | Actually checked |
|---|---|
| "any checkpoint/contract **mismatch**" | **No comparison is made.** It checks the checkpoint for completeness and the contract for well-formedness, independently. |
| "cannot verify compatibility" (the action-shape error text) | Only that a shape *exists*, not that it equals the checkpoint's action dimension, which `rde_checkpoint_meta.json` carries per `DEPLOYMENT.md` Table 1 ("action dimensionality"). |
| SmolVLA needs cameras | That at least one camera exists with a 3-dimensional shape. **Not** that the camera *count* or names match what the checkpoint was trained with. |
| state normalization is present | Present **in the contract**. But see below: inference doesn't use it. |

**A concrete passing mismatch.** Load an ALOHA checkpoint (14-dimensional action, 4 cameras, 28-dimensional state) with the SO-100 contract (6-dimensional action, 2 cameras, 6-dimensional state). Every check passes: the checkpoint is complete, and the contract has an action shape, 3-dimensional cameras and normalized state. `health_check` returns `{"status": "ok"}`. Then `predict` fails mid-inference with *exactly* the "confusing tensor-shape error mid-prediction" the exception class says it exists to prevent. Or, worse, if the shapes happen to line up, it runs and produces nonsense. The test suite proves the gap: `_make_checkpoint` writes metadata with `"dataset_id": "d"` and an **empty** `model.safetensors` (`write_bytes(b"")`), and `test_health_check_passes_on_well_formed_checkpoint_and_contract` passes it against an SO-100-shaped contract.

**The normalization check is on the wrong object.** `load()` constructs the policy with `SmolVLAPolicy.from_checkpoint(...)`, which per `docs/API_REFERENCE.md` "loads with `contract=None`". The checkpoint uses **its own saved preprocessor statistics**. `DEPLOYMENT.md` §5 confirms that `predict` "applies the checkpoint's own preprocessor (normalization and tokenization)". So the contract's normalization statistics, the only thing the health check inspects, **aren't used at inference at all.** The real training/serving-skew risk is the opposite one: the checkpoint's stats differing from the live data's distribution. That would need a comparison of the checkpoint preprocessor's mean and std against the contract's, and that comparison isn't made.

**The fix is cheap,** because the metadata already exists:

```python
if meta.get("action_dim") not in (None, action["shape"][0]):
    raise DeploymentHealthError(f"checkpoint action_dim={meta['action_dim']} != contract {action['shape']}")
if meta.get("dataset_id") not in (None, expected_dataset_id): ...
# and compare camera keys/count and the preprocessor's state mean/std against the contract's
```

(The exact metadata key names are engine-side and unverified. `DEPLOYMENT.md` Table 1 says the metadata includes "action dimensionality, chunk size".)

```python
def load(self) -> "DeployedModel":
    self.health_check()
    t0 = time.perf_counter()
    self._policy = SmolVLAPolicy.from_checkpoint(self.checkpoint_path, device=self.device)
    self.load_time_ms = (time.perf_counter() - t0) * 1000.0
    return self
```

- **Verification runs twice**, once in `health_check` and again inside `from_checkpoint`. It's cheap (file-presence checks) and harmless.
- **`load_time_ms` excludes the health check**, and it includes weight transfer to the device. It's measured once.
- **Returning `self`** enables `DeployedModel(...).load()` chaining, as in the `DEPLOYMENT.md` example.
- **No fallback model** on failure. `DEPLOYMENT.md` §4 makes "no silent substitution" a stated rule. That's good.

### `predict` and the latency breakdown

```python
def predict(self, samples: Sequence[Sample]) -> tuple[Prediction, LatencyBreakdown]:
    if self._policy is None:
        raise RuntimeError("call load() before predict()")
    t0 = time.perf_counter()
    model_input = self._adapter.adapt(samples)
    t1 = time.perf_counter()
    prediction = self._policy.timed_predict(model_input)
    t2 = time.perf_counter()

    preprocessing_ms = (t1 - t0) * 1000.0
    inference_ms = prediction.latency_ms
    postprocessing_ms = max(0.0, (t2 - t1) * 1000.0 - inference_ms)
    latency = LatencyBreakdown(model_load_ms=self.load_time_ms, preprocessing_ms=preprocessing_ms,
                               inference_ms=inference_ms, postprocessing_ms=postprocessing_ms,
                               total_ms=preprocessing_ms + inference_ms + postprocessing_ms)
    return prediction, latency
```

- **The input is `Sequence[Sample]`, an engine dataset object.** Samples come from `DatasetView.iter_samples`, as in the integration test. A robot controller has camera frames, a joint-state vector and an instruction string, not engine `Sample`s. **So `DeployedModel` can only replay recorded data.** It can't be dropped into a control loop without an adapter from raw observations to `Sample`. That's the most important framing point in this module: this is an *offline inference harness with timing*, not deployment.
- **"`postprocessing_ms`" isn't postprocessing.** Per `DEPLOYMENT.md` §5, `SmolVLAPolicy.predict` applies the postprocessor (action de-normalization) *inside* the predict call, so it's already included in `prediction.latency_ms`. What's left in `(t2 − t1) − inference_ms` is the overhead of the `timed_predict` wrapper itself, which is near zero. The label promises a stage that this subtraction doesn't isolate.
- **`max(0.0, …)` hides clock disagreement.** If `inference_ms` (measured inside the engine) exceeds the outer interval, the negative value is clamped to 0, and `total_ms` (a *sum* of parts, not a measured interval) can then **exceed the real wall time**. Measuring `total_ms = (t2 − t0) × 1000` directly would make the parts auditable against the whole.
- **GPU synchronization is unverified.** If `timed_predict` returns tensors still on the GPU without synchronizing, `inference_ms` measures kernel *launch* time. The integration test's `prediction.actions.shape` suggests NumPy arrays, which would force a device-to-host copy and therefore a sync, but I can't confirm that from here.
- **No warm-up, no repetition, no percentiles.** `DEPLOYMENT.md` §6 explicitly leaves p50, p95 and p99 to the caller, as "a benchmark-design decision". That's a defensible scope line, but nothing in the repo *is* that caller.
- **The `LatencyBreakdown.note`** (preprocessing isn't split into decode, resize and tensor construction) is the project's honest-metadata habit applied well. But it names `VisionStateAdapter`, while this class uses `VisionLanguageAdapter`, which is related but not the same class.
- **Action chunks:** whether `Prediction.actions` is the first action of each chunk or the whole chunk isn't visible here. A control loop needs to know.

### Nothing calls it

Grep shows `DeployedModel` used only in `deployment/__init__.py`, `tests/unit/test_failure_modes.py` and `tests/unit/test_deployment.py` (health checks only, never `load` or `predict`), and as a string inside `reports/model_card.py`. **`DeployedModel.predict` has never run in any test, any script or the pipeline.** The README's latency figures ("47–50 ms per frame on SO-100 and 86 ms per frame on ALOHA") come from `Evaluator`'s `latency_ms_per_frame` during test evaluation (Module 7). The README correctly says they "are not a deployment latency measurement". **No deployment latency has ever been measured by this code.**

**Doc vs doc:** `DEPLOYMENT.md` §5 says `SmolVLAPolicy.predict` "invokes `select_action`". `docs/API_REFERENCE.md` §7.4 says predict "Uses `predict_action_chunk` ... **not** `select_action` (which maintains a stateful action queue ... reusing it across independent evaluation batches silently replays stale actions)". That's the bug the smoke test's comment describes as fixed. `DEPLOYMENT.md` was written before the fix and never updated. Since you can't read the engine here, cite API_REFERENCE and the smoke test as the current behaviour.

---

## 9.5 Honest gaps to own before an interviewer finds them

1. **`health_check` never compares the checkpoint to the contract**, despite "raised on any checkpoint/contract mismatch". *Consequence:* an ALOHA checkpoint against an SO-100 contract passes and then fails, or silently misbehaves, mid-inference. That's the exact failure the class exists to prevent.
2. **It checks normalization on the contract, but inference uses the checkpoint's own preprocessor stats.** *Consequence:* the check guards an unused object, and real training/serving skew between checkpoint stats and live data is unchecked.
3. **`DeployedModel` consumes engine `Sample`s, not robot observations.** *Consequence:* it's a replay harness. Deploying to a robot needs an observation adapter that doesn't exist.
4. **`DeployedModel.load` and `predict` have never been executed** by any test, script or pipeline. *Consequence:* the deployment path is unverified code, and no deployment latency exists.
5. **`postprocessing_ms` measures wrapper overhead, not postprocessing; `total_ms` is a sum, not an interval; negatives are clamped.** *Consequence:* the breakdown can misattribute time and overstate the total.
6. **The registry is a catalogue** (no stages, promotion or "deployed" concept) and **isn't wired in** (honestly documented). *Consequence:* "model registry" overclaims relative to the industry meaning.
7. **The artifact's `metrics` are test metrics, labelled as "what metrics justified selection".** *Consequence:* the validation metric that actually justified selection is absent from the artifact.
8. **`model_version` is always a step number, `pretrained_model` may be a resume path (unverified), and `model_id`'s prefix comes from a free tag.** *Consequence:* identity fields are less meaningful than their names suggest.
9. **`DEPLOYMENT.md` §5 says predict uses `select_action`.** API_REFERENCE and the smoke test say `predict_action_chunk`. *Consequence:* a stale doc describing the pre-fix bug as current behaviour.
10. **Health-check tests use an empty `model.safetensors` and mismatched metadata, and pass.** *Consequence:* the suite encodes the presence-only behaviour as correct.
11. **The registry's "most recent wins" fix isn't tested.** *Consequence:* a regression to first-match wouldn't be caught.
12. **Dependencies recorded: torch and lerobot only.** *Consequence:* the artifact can't explain a `transformers`-driven behaviour change.

---

## The shape to carry into Module 10

```
 selected checkpoint ──verify_checkpoint──▶ meta {step, checkpoint_source, dataset_id, dataset_version, ...}
        │
        ▼
 build_model_artifact  (dataset identity: checkpoint's own claim wins)
   ModelArtifact{model_id=tag-expid, version=step, schema=VIEW contract, metrics=TEST (labelled "justified selection")}
        │                               │
        ├─▶ model_artifact.json         └─▶ ExperimentIndex.record_model → experiments/models.jsonl
        │                                         ▲ same row shape
        │                                 ModelRegistry(get = last row wins)   ← never constructed by src/
        ▼
 DeployedModel(checkpoint, contract)                                             ← never constructed by src/
   health_check: ckpt files present? contract has action shape / 3-D cams / state norm?   (no cross-check!)
   load: from_checkpoint(contract=None) → uses CHECKPOINT preprocessor stats
   predict(Sequence[Sample])  ← engine samples only, not robot observations
     → Prediction + LatencyBreakdown(pre | inference | "post" = wrapper overhead | total = sum)
```

---

## Drill — answer from memory, files closed

1. What does `DeployedModel.health_check` actually verify, and name one incompatible checkpoint/contract pair it would pass.
2. *Why didn't you just* build a FastAPI server around the model?
3. *What breaks in production:* you hand `DeployedModel` to a robotics engineer and say "here's the deployment layer." What do they hit first?
4. Your model card says the registry answers "what metrics justified selection." What's in that field, and is the label right?
5. *Interviewer framing:* "How did you deploy this model, and what was its inference latency?"

---

# Appendix A — Model answers to the drill

**A1. What `health_check` verifies.**
"Four things: the checkpoint directory has all its required files; the contract declares an action shape; it has at least one camera with a three-dimensional shape; and every state modality in the contract has normalization stats. What it doesn't do is compare the two. It reads the checkpoint's metadata and then never uses it. So an ALOHA checkpoint with 14-dimensional actions and four cameras, checked against the SO-100 contract with 6-dimensional actions and two cameras, passes, then fails inside inference with a shape error, which is exactly what the exception class says it prevents. And the normalization check looks at the contract's stats, while inference loads the checkpoint's own preprocessor stats, so it guards something that isn't used. The fix is to compare the checkpoint's recorded action dimension, camera set and normalization stats against the contract's. All of that metadata is already saved."

*Why that answer lands:* it states what's checked precisely, constructs a concrete passing counterexample, and spots the wrong-object normalization check. That's an adversarial reading of your own safety code.

**A2. Why not a FastAPI server?**
"The brief was to avoid infrastructure that doesn't earn its keep, and there's no consumer for a network endpoint. No robot or simulator is connected, so a server would only add serialization, a process boundary and failure modes. The deployment layer is deliberately a function call. If a network interface were needed, it would be a thin wrapper around `predict`. The honest limitation isn't the missing server, though. It's that `predict` takes the engine's dataset `Sample` objects, so even in-process it can only replay recorded data. Before any server, I'd add an adapter from raw observations (camera frames, a joint vector, an instruction) to model input, and a closed-loop harness that executes action chunks. I'd add a server only once a separate process, like a robot controller, needed to call it."

*Why that answer lands:* it defends the minimalism with a reason, then volunteers the real gap, which is input type, not transport, and orders the next steps correctly.

**A3. A robotics engineer gets `DeployedModel`.**
"First, it doesn't accept their inputs. `predict` wants a sequence of engine `Sample` objects from a dataset view, and they have camera images and joint states from a live robot. Second, if they wrap that, they'll find it returns predictions per call with no handling of action chunks or control frequency, and SmolVLA plans in chunks. Third, the health check would pass a checkpoint trained on a different embodiment, so a wrong model reaches inference. And fourth, none of it has ever been executed: `load` and `predict` have no test, and no latency was ever measured through this class. I'd tell them it's a replay and timing harness, and that a deployment needs an observation adapter, a chunk-execution loop, a real compatibility check and a warm-up-plus-percentile benchmark."

*Why that answer lands:* it ranks the failures in the order they'd actually be hit, and closes with a precise, honest reclassification of what the code is.

**A4. "What metrics justified selection."**
"That field holds the test-split metrics, the MSE and so on of the selected checkpoint on the test set. But selection was justified by the validation metric, because the trainer only ever compares checkpoints on val, and that number isn't in the artifact at all. So the label is wrong, and it's wrong in the most sensitive direction, because it invites someone to think test was used for selection. I'd store both, under 'selection: val MSE at step N' and 'reported: test metrics', and rename the registry's question."

*Why that answer lands:* it catches a labelling error that touches the project's core validity principle, and fixes it in a way that reinforces the val/test separation.

**A5. "How did you deploy this model, and what was its inference latency?"**
"I didn't deploy it to a robot. There was no robot or simulator connected, and the system says so rather than reporting a rollout number. What exists is a local loader class with a health check and a per-stage latency breakdown, but that path has never been run end to end, and its health check doesn't cross-check checkpoint and contract, which I'd fix first. The latency numbers I can give are evaluation-harness numbers: about 47–50 milliseconds per frame on SO-100 and 86 on ALOHA, which has four cameras. That's prediction time divided by batch size during batched offline evaluation on a laptop RTX 4060, excluding data loading. That isn't control-loop latency. A real number would be single-sample, warmed up, synchronized, reported as p50 and p99, and measured inside a control loop that executes action chunks."

*Why that answer lands:* it refuses the overclaim the question invites, gives the real numbers with their exact provenance, and describes what a correct measurement would look like. Precision here is what interviewers test.

---

# Module 9 Summary — the one page to revise

**What this module is.** An identity record for the selected checkpoint (provenance taken from the checkpoint's own metadata), a JSONL "registry" that's a correct but unwired catalogue, and a `DeployedModel` loader/predictor that's never executed, doesn't cross-check checkpoint and contract, and only accepts dataset samples.

**Five claims to make cold**
1. The artifact verifies the checkpoint and prefers the checkpoint's own dataset identity over the caller's.
2. The registry reads last-write-wins over an append-only log, isn't wired into the runner (and the docs admit it), and has no promotion concept.
3. `health_check` checks presence and well-formedness, not compatibility, and its normalization check targets the contract, which inference doesn't use.
4. `DeployedModel` takes engine `Sample`s, so it's a replay harness, and it has never run.
5. README latency (about 50 ms/frame SO-100, 86 ms ALOHA) is evaluator-batched, not deployment latency.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Function-call deployment | network service | no consumer; avoid infrastructure | none yet, but the input type is the real gap |
| Checkpoint metadata as identity authority | caller-supplied identity | the checkpoint knows its own data | disagreement silently resolved, not flagged |
| JSONL registry | MLflow registry | consistent with the index; no server | no stages or promotion; inherits JSONL fragility |
| Load with checkpoint preprocessor | re-derive stats from contract | avoids training/serving skew from new stats | the contract check is then irrelevant |
| Single preprocessing latency bucket | fabricated sub-stage split | can't measure inside the adapter | "post" bucket mislabelled instead |
| No fallback model | load a default | never silently substitute | none |

**Volunteer unprompted:** no cross-check in `health_check`; the wrong-object normalization check; `Sample`-only input; never executed; the test-metrics-as-selection label; `DEPLOYMENT.md` §5's stale `select_action` claim.

**Numbers and names:** 5 checkpoint files (`config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`, `rde_checkpoint_meta.json`); `model_id = <tag model>-<experiment_id>`; latency fields `model_load_ms`, `preprocessing_ms`, `inference_ms`, `postprocessing_ms`, `total_ms`; README evaluator latency 47–50 ms/frame (SO-100), 86 ms/frame (ALOHA), batch 16.

**Sentences worth memorizing**
- "My health check verified that the checkpoint and the contract each existed, not that they agreed."
- "It takes dataset samples, not robot observations, so it's a replay harness with timing, not a deployment."
- "The latency I can quote is evaluator throughput per frame, not control-loop latency, and I say that every time."
- "The registry labelled test metrics as the selection justification; selection was on validation, and the artifact should say so."
