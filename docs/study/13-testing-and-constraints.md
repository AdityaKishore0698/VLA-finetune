# Module 13 — The test suite, and the constraint → decision map

**Files covered:** `tests/conftest.py`, `tests/unit/test_config.py`, `tests/unit/test_deployment.py`, `tests/unit/test_evaluation_offline.py`, `tests/unit/test_experiment_runner.py`, `tests/unit/test_failure_modes.py`, `tests/unit/test_interventions.py`, `tests/unit/test_profile.py`, `tests/unit/test_split_integrity.py`, `tests/integration/test_full_chain.py`, `tests/smoke/test_smolvla_smoke.py`

**Where we are:** Modules 1–12 read the code adversarially. This module reads the tests the same way: what they prove, what they assume, and which of the gaps found so far a test *should* have caught. It then closes the code modules with the **capstone**, a table tracing every non-obvious design choice back to the constraint that forced it, plus the choices no constraint explains.

**The one-line thesis:** The suite's distinctive strength is that it tests against a **real engine on a real synthetic dataset**, not mocks, so the integration seams are exercised for real. But the fixture is too uniform to exercise the logic that matters (length buckets), the most important guarantees (val-only selection, seed propagation, no-overwrite of results) have **no test at all**, and several tests pin the presence-only behaviour that later modules showed to be wrong.

---

## 13.1 Prerequisites — the absolute basics

- **pytest fixture.** A function decorated `@pytest.fixture` whose return value is injected into any test that names it as a parameter. Fixtures can depend on other fixtures, forming a graph. Default scope is per test: rebuilt for each test function.
- **`tmp_path`.** A built-in fixture giving each test a fresh temporary directory.
- **`pytest.importorskip("torch")`.** Import the module, or skip the whole test file with a reason if it's missing.
- **Mocks vs fakes vs real collaborators.** A *mock* records calls and returns canned answers. A *fake* is a small working substitute (`_FakeView` in Module 5). A *real collaborator* is the production object. Tests with real collaborators catch integration bugs and are slower and coupled to the collaborator's behaviour.
- **Test pyramid.** Many fast unit tests, fewer integration tests, very few end-to-end tests. In practice, a project's "unit" folder may contain integration tests in disguise.
- **Characterization test.** A test that pins *current* behaviour, whether or not that behaviour is correct. It's useful for refactoring, and dangerous when it pins a bug as a contract.
- **Coverage vs assurance.** A line being executed by a test (coverage) is different from its *outcome* being checked (assurance). A test that only asserts a file exists covers a lot and assures little.
- **Global state in tests.** `set_settings(s)` mutates a process-wide singleton. Tests sharing it can't safely run in parallel (`pytest-xdist`).

---

## 13.2 Inventory, verified

| File | Tests | Needs torch? | What it exercises |
|---|---:|---|---|
| `unit/test_config.py` | 8 | no | merge, templates, sum-to-one, identity, resolved dump |
| `unit/test_deployment.py` | 6 | no | `health_check` (5), registry round-trip (1) |
| `unit/test_evaluation_offline.py` | 6 | no | baselines, error distribution, episode metrics, temporal, stratified, unknown slice |
| `unit/test_experiment_runner.py` | 3 | no | baseline-only full run, no-overwrite, reduced recorded |
| `unit/test_failure_modes.py` | 10 | 1 self-skips if torch *is* present | unknown dataset/version/variation, zero fraction, corrupt checkpoint, bad contract |
| `unit/test_interventions.py` | 9 | no | full/reduced/imbalanced/rebalanced against a real catalog, lineage, determinism |
| `unit/test_profile.py` | 3 | no | clean vs training profile, JSON-shaped |
| `unit/test_split_integrity.py` | 4 | no | real view OK, fake leaky view raises or reports |
| `integration/test_full_chain.py` | 3 | no | view → adapter → model → Prediction → Evaluator → feedback |
| `smoke/test_smolvla_smoke.py` | 4 | **yes** (`importorskip` at module top) | three contract-to-feature builders, one 4-step SmolVLA end-to-end run |
| **Total** | **56** | | 52 run in the light venv (README line 282 says "52 unit + integration": **verified**) |

**I couldn't run any of them** (Module 1: the engine isn't in this checkout). Everything below is from reading.

---

## 13.3 `conftest.py`: the real-engine fixture graph

```python
"""Shared fixtures: a real synthetic dataset taken through raw -> clean ->
training, exactly the way `rde run` would, so every layer of model_training is
tested against real rde objects rather than mocks."""

@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(paths=Paths(tmp_path / "rde_data"), row_group_size=64, batch_size=32)
    s.paths.ensure()
    set_settings(s)
    return s

@pytest.fixture
def source_dataset(tmp_path) -> Path:
    # Two tasks (not the single-task default) so episode_id-by-task_id
    # selection -- what the imbalanced/rebalanced interventions use -- has
    # something real to select between.
    return make_lerobot_dataset(tmp_path / "source", num_episodes=8, frames_per_episode=20, fps=10,
        tasks=("Pick up the cube and place it in the box.", "Push the block to the target."))

@pytest.fixture
def raw(settings, source_dataset, catalog):
    return ingest(source_dataset, "so100_test", "lerobot", "raw/v1", settings=settings, catalog=catalog)

@pytest.fixture
def clean_version(settings, catalog, raw) -> str:
    TransformPipeline([NormalizeVector("state.joint"), NormalizeVector("action.joint")], ...).run(
        "so100_test", "raw/v1", "clean/v1")
    # `rde run` always validates before a version is considered usable
    # downstream; without this, every episode sits at quality_status=UNKNOWN
    # and DatasetBuilder.build() (which admits only PASSED/WARNING by
    # default) would refuse to build a training version from it.
    QCEngine(settings=settings, catalog=catalog).run("so100_test", "clean/v1")
    return "clean/v1"

@pytest.fixture
def training_version(settings, catalog, clean_version) -> str:
    DatasetBuilder(...).build("so100_test", clean_version, "training/v1",
                              splits={"train": 0.6, "val": 0.2, "test": 0.2}, seed=7)
    return "training/v1"
```

**Decision: real engine objects, not mocks.**
- *Why:* this repository's risk is almost entirely at the boundary with the engine (Module 1's import table). A mocked `DatasetBuilder` would pass while the real one changed. `test_reduced_intervention_selects_half_and_records_lineage` can check *real* lineage records only because the catalog is real. That's the strongest test in the suite.
- *Alternative:* mock `rde`, so tests run without the engine and fast.
- *Cost:* (1) **the suite can't run without the sibling engine checkout**, so nothing runs in this checkout; (2) every test pays for ingest, normalize, QC and build per test (function scope, rebuilt each time); (3) tests are coupled to engine behaviour, and an engine change can break a test here without any change here, with no version pin to warn you (Module 1).
- **The comments are excellent.** They explain *why* each fixture step exists: two tasks so task-based selection has something to select; QC so the builder admits episodes. Those are the non-obvious facts a new contributor needs.
- **`set_settings(s)` mutates global state.** It's correct for sequential runs. With `pytest -n auto` (xdist), workers are separate processes, so it's actually safe there too, but any thread-based parallelism or test that forgets the `settings` fixture would pick up the previous test's settings.

**The fixture's shape limits what can be tested:**
- **8 episodes × 20 frames, all the same length.** Every length-bucket computation over the fixture collapses (all "short", by the `<=` tie rule from Module 4). The test file itself says so: "The synthetic fixture's episodes are all the same length, so every episode buckets to 'short'". Only the pure `test_compute_length_bucket_splits_into_terciles` exercises real terciles.
- **A 0.6/0.2/0.2 split of 8 episodes** gives a test split of about 2 episodes (exact sizes depend on the engine's quantile rounding). `stratified_evaluate`'s length slices require `>= 3` test episodes (Module 7), so **`test_stratified_evaluate_per_task_and_length_bucket` almost certainly never executes the `per_length_bucket` branch**. Its assertions (an `overall` key exists, some `task_` key exists, slices ⊆ overall) all pass with the length branch silently skipped. The test's name claims coverage it doesn't get.
- **Two tasks** is good for task selection, but the real primary dataset has *one* task, so the code path SO-100 actually runs (single task, so `task_0` duplicates overall, Module 7) isn't the one the fixture tests.

---

## 13.4 What the tests prove well

1. **Interventions create real lineage** (`test_reduced_intervention_selects_half_and_records_lineage`). The transform run's kept and dropped IDs equal the computed ones. This is *the* claim of the project, and it's verified end to end.
2. **Weights reach the built version** (`test_rebalanced_intervention_keeps_all_episodes_and_sets_weights`): `manifest.statistics["weighted_episodes"]` is truthy, so the request survives into the engine.
3. **Templates fail as documented** (`test_rebalanced_template_fails_loudly_until_filled_in`, `test_imbalanced_template_loads_but_is_empty`).
4. **The no-overwrite rule** (`test_runner_refuses_to_overwrite_an_existing_run`).
5. **Failure modes raise specific exceptions** (`test_failure_modes.py`): unknown dataset or version raises `LookupError`, an unsupported variation raises `ValueError` with "no automatic rule", a corrupt checkpoint raises `ValueError` with "incomplete". The module docstring states the standard well: "every one of these must raise a specific, actionable error -- never degrade silently".
6. **The runner completes without torch** (`test_runner_completes_baseline_only_experiment`). It uses `model: mean_action` to exercise every non-training stage in the light venv, and asserts `"unavailable" in report_text`, so the rollout honesty rule is tested.
7. **Cross-artifact consistency, once:** `test_split_integrity_then_profile_then_stratified_eval` asserts `profile.splits == report.split_sizes`. That's a good pattern, and it's used only here.
8. **Cross-embodiment feature shapes** (smoke): the ALOHA-shaped contract gives 28-dimensional state (14 + 14) and 14-dimensional action, with normalization stats concatenated in feature order. This guards a real bug class (Module 2's note that ALOHA "generalized *after* a fix").
9. **The SmolVLA chain runs end to end** (smoke): 4 real optimizer steps, checkpoint reload, all artifacts present, loss not NaN (`final_train_loss == final_train_loss`, the classic NaN self-inequality idiom). The non-divisor `batch_size: 3` was chosen deliberately to reproduce the action-queue bug ("predict() must never reuse a cached action chunk across unrelated batches"). That's a regression test written from a real incident, which is the best kind.

---

## 13.5 What the tests miss, mapped to the gaps they would have caught

| Gap (module) | Test that would have caught it | Exists? |
|---|---|---|
| `--seed` doesn't reach `training.seed`; replicates share `experiment_id` (M11-1) | run twice with different seeds, assert `training.seed` and the `experiment_id`s differ | **no** |
| Selection never uses test (the core guarantee, M6) | spy on `Evaluator.evaluate` during `Trainer.train`, assert every `split == "val"` | **no**. It's enforced only by reading the code |
| Unknown YAML keys silently ignored (M2-1) | load a config with `learning_rat:`, expect an error | **no** |
| `model: SmolVLA` silently skips training (M2-2) | expect a validation error | **no** |
| Imbalanced with no value becomes a full run (M2-4) | assert raise or non-empty `excluded_episode_ids` | **no** |
| `frac=0.0` fails in the builder (M4-2) | call `build_intervention_dataset` with 0.0 | **no**. The existing test checks the helper and *asserts the opposite* |
| Filter-then-split moves the test set (M4-1) | assert reduced-run test IDs ⊆ baseline test IDs | **no** |
| Overlap check can't fail on real views (M5-1) | — (the fake view test is all that's possible) | n/a |
| Unassigned or empty splits pass (M5-2) | build with `val: 0.0`, expect integrity failure | **no** |
| Coverage normalization (M5-4) | two-bin bimodal counts → expect < 1.0 | **no** (only asserts `0 ≤ x ≤ 1`) |
| `_prune_checkpoints` (M6-10) | pure `tmp_path` test | **no** |
| `crash.json` on exception; signals (M6-2) | make `fit` raise, assert `crash.json` | **no** |
| NaN locks best (M6-4) | feed a NaN val metric | **no** |
| Slices reconcile with overall (M7-1) | frame-weighted slice MSE ≈ overall | **no** (stochastic models only; baselines are deterministic, so it would pass for the wrong reason) |
| Feedback branches reachable (M8-2/3) | synthetic outcomes concentrated in one episode | **no**. The integration test checks shape |
| `health_check` cross-check (M9-1) | mismatched action dimension, expect failure | **no**. The existing test *passes* a mismatched, empty checkpoint |
| `DeployedModel.load`/`predict` (M9-4) | any call | **no** |
| Report content (git `n/a`, MAE `n/a`; M10) | assert the commit appears in the report | **no** (only substring checks for name and "unavailable") |
| `mtrain compare` alignment, `evaluate` overwrite, `--apply` comments (M12) | CLI tests via `typer.testing.CliRunner` | **no CLI tests at all** |
| `run_data_fraction_sweep` (M11-10–13) | any call | **no** |
| Light venv stays light (M1-7) | import the runner, assert `"torch" not in sys.modules` | **no** |
| Registry "most recent wins" (M9-11) | register two rows, `get` returns the second | **no** |

**Pattern:** the suite tests that **each component works on the happy path and fails loudly on bad identity** (unknown versions, missing files). It doesn't test **cross-stage invariants** (seed flows through, test never selects, slices partition overall, artifacts agree with each other), and that's where every serious bug in this manual lives.

### Tests that pin wrong behaviour (characterization hazards)

- **`test_reduced_intervention_at_zero_fraction_is_honored`** asserts the helper returns `[]` for 0.0, encoding a contract the system can't honour. Its docstring argues 0.0 is a "legitimate, explicit request".
- **`test_health_check_passes_on_well_formed_checkpoint_and_contract`** passes an *empty* `model.safetensors` and metadata naming `dataset_id: "d"` against an SO-100 contract. Fixing Module 9's cross-check would turn this test red, correctly.
- **`test_invalid_deployment_contract_rejected`** (failure modes) uses `rde_checkpoint_meta.json = "{}"`, so the same presence-only stance.
- **`test_experiment_id_is_deterministic_given_same_inputs`** is true only with a pinned `created_at`, which no YAML path provides. Its name overstates the property.

### Smoke-test realities

- **It needs network on first run** (it downloads `lerobot/smolvla_base` from the Hugging Face Hub) and a torch plus lerobot install. It's skipped otherwise, "never replaced with a stub" (docstring). That's honest. It also means **no CI could run it without Hub access and a GPU or a patient CPU**.
- **The three feature-builder tests are skipped in the light venv** because `importorskip("torch")` is at module top, even though they test pure contract → shape logic. If those builders return lerobot `PolicyFeature` objects (likely, given `.shape`), lerobot is genuinely required. It's unverifiable here.
- **It imports two private engine functions** (`_policy_features_from_contract`, `_dataset_stats_from_contract`), which is Module 1's private-API coupling.
- **The end-to-end smoke asserts existence, not values.** `"mse" in test_overall["metrics"]` and files exist. With 4 steps that's appropriate (no learning is expected), but a sanity bound (for example, SmolVLA MSE finite and positive) would catch a NaN in evaluation, not just in training.

---

## 13.6 The capstone: constraint → decision map

Each row traces a non-obvious choice to the **named constraint that forced it**, with the evidence for the constraint. The second table lists the choices **no constraint explains**. Those are the ones to call arbitrary, rather than invent a justification for.

### Constraints present in this project (with evidence)

| ID | Constraint | Evidence |
|---|---|---|
| **C1** | **One laptop GPU, 8 GB VRAM; about 10 GB host RAM** | README §2 (RTX 4060 Laptop, 8 GB); `IMPLEMENTATION_PLAN.md` (about 10 GB VM RAM, engine's 6 GB cap) |
| **C2** | **No robot or simulator** | `rollout` text, README §2 point 4, `METRICS.md` §8 |
| **C3** | **Tiny datasets** (48 / 40 episodes; 5 / 4 test episodes) | README Table 1 |
| **C4** | **Data schema without a variation field; SO-100 single-task** | `profile.py` `variation_note`; `EXPERIMENTS.md` §3 |
| **C5** | **A separately owned data engine with immutable, lineage-tracked versions** (the untrusted-by-default component, from this repo's point of view) | `pyproject.toml` sibling path; `ARCHITECTURE.md` |
| **C6** | **A stochastic policy** (flow matching) **that predicts action chunks** | README §2.3; smoke-test action-queue comment; API_REFERENCE `predict_action_chunk` |
| **C7** | **Heavy, platform-specific deps** (torch with CUDA, lerobot 0.4.4) | `IMPLEMENTATION_PLAN.md` (CPU torch reinstall); two-venv tables |
| **C8** | **Pretrained base for a different embodiment** | `trainer.py` load comment; smoke cross-embodiment tests |
| **C9** | **Team of one or two researchers, no infrastructure, short runs (5–10 min)** | README §2 timings (324 / 385 / 576 s); `DEPLOYMENT.md` "avoid unnecessary infrastructure" |
| **C10** | **Project-specification rules**: never fabricate, failure ≠ bad data, don't presume the weak dimension, no silent model substitution | `EXPERIMENTS.md` §2–3, `FAILURE_ANALYSIS.md` §5, `DEPLOYMENT.md` §4, `IMPLEMENTATION_PLAN.md` |
| **C11** | **An unreliable runtime** (OOM, kills, partial writes on a laptop) | `trainer.py` crash and verify comments |
| **C12** | **Offline evaluation must not be mistaken for task success** (a corollary of C2 plus C10) | `offline.py` docstring; model-card limitations |

### Decision → constraint

| Decision (module) | Forced by | How the constraint forces it | Would flip when |
|---|---|---|---|
| Batch size 4, 1,000 steps, no grad accumulation (M2) | **C1** | 8 GB VRAM for a ~450M-param VLA plus 2–4 cameras | larger GPU, or multi-GPU |
| Validation capped at `val_batches × batch_size` = 32 frames (M6) | **C1, C9** | a full val pass (~90 s) × 10 would triple a ~5-minute run | longer runs make validation relatively cheap; use a frozen stratified subset regardless |
| `del policy; empty_cache()` before the evaluation load (M6) | **C1** | two SmolVLA copies don't fit in 8 GB | GPU with headroom |
| No optimizer state in checkpoints (M6) | **C1** (disk), inferred | Adam moments about triple checkpoint size | preemptible long runs need exact resume |
| Keep last 3 checkpoints plus best (M6) | **C1** (disk) | each save is about one model in size | cheap storage; the number 3 itself is arbitrary |
| Peak RSS and GPU recorded per run (M6) | **C1** | knowing headroom on a constrained box | — |
| Offline-only evaluation; `rollout` writes "unavailable" (M7, M12) | **C2, C10, C12** | nothing to roll out on; fabrication forbidden | a simulator or robot is connected |
| `success_rate` labelled a proxy everywhere (M7) | **C12** | an offline tolerance band isn't task success | rollouts exist |
| `DeployedModel` is a function call over `Sample`s (M9) | **C2, C9** | no consumer process or robot; avoid infrastructure | a controller needs to call it |
| Episode-level, `quantile` (exact-size) splits (M2, M5) | **C3** | independent draws at n = 48 vary ±2 episodes per split | thousands of episodes |
| Separate split seed 42 vs training seed (M2) | **C3** | the test set must stay fixed while training varies | — |
| Episode-length bucket as the imbalance variable (M4) | **C4** | the only per-episode attribute that varies on SO-100 | the schema gains a variation or condition field |
| "Visual diversity: unavailable" (M5) | **C4, C10** | no embedding model; fabrication forbidden | an image embedding model is added |
| Interventions as engine transforms; data named by version (M2, M4) | **C5** | lineage and immutability live in the engine | — (the project's core principle) |
| Re-verify splits from the materialized view (M5) | **C5** | don't trust the producing code path | — (but it checks an unrepresentable failure) |
| Read training weights back from the built version (M11) | **C5** | train on what the engine stored | — |
| Git info of *both* repos (M3) | **C5** | the engine owns `fit()` | single repository |
| Model adapter and `fit` living in the engine (M1) | **C5, C8** | features and normalization come from the engine's contract | a second model family arrives |
| `predict_action_chunk`, never `select_action`, in evaluation (engine; M9) | **C6** | the stateful action queue replays stale chunks across batches | — |
| Noise caveat on small differences (README §2.3) | **C6** | re-scoring varies up to 5% | deterministic sampling or averaging |
| Two venvs; lazy imports; baseline-model tests; `importorskip` (M1, M12, M13) | **C7** | GBs of CUDA torch for 52 tests that don't need it | CI on a GPU image anyway |
| Contract passed on fresh load, `None` on resume (M6) | **C8** | base checkpoint shapes and stats belong to another embodiment | same-embodiment pretraining |
| JSONL index plus run directories, not MLflow (M3) | **C9** | one or two users, serverless, durable files | more users, thousands of runs |
| Transaction-Script runner, no DAG (M11) | **C9** | short runs make caching and resume unnecessary | multi-hour or preemptible runs, big sweeps |
| f-string Markdown reports, fixed sections (M10) | **C9, C10** | durable docs next to artifacts; never hide a section | — |
| Validators: imbalanced or rebalanced require a declared target (M2) | **C10** | "the weak dimension must not be presumed" | — |
| `build-intervention` dry run by default; resolves from the real catalog (M12) | **C10** | the choice must follow from a real profile | — |
| Feedback explanations hedge "data defect vs limitation" (M8) | **C10** | "model failure ≠ bad data" | — |
| No fallback model in `DeployedModel` (M9) | **C10** | no silent substitution | — |
| No-overwrite run directories (M3) | **C11** | never destroy a run's checkpoints and metrics | — (but add status plus `--force` for failed runs) |
| Verify each checkpoint before it's eligible as best; `crash.json` (M6) | **C11** | partial saves and crashes on a laptop | — |
| `yaml.safe_load` everywhere (M2) | **C11/C5** (untrusted input hygiene) | YAML tags can execute code | — |

### Choices no constraint explains (call these arbitrary)

| Choice | Where | Honest description |
|---|---|---|
| Failure threshold p90, `min_failures_per_episode = 3`, concentration `> 0.5` | M8 | Arbitrary, and mutually inconsistent at these sizes (the concentration branch is unreachable) |
| Length terciles at 33.3 / 66.7, `<= t1` tie rule | M4, M7 | A conventional tercile; the tie rule is arbitrary and collapses buckets on uniform lengths |
| 10 histogram bins | M5 | A conventional default; the occupied-bin normalization is a mistake, not a choice |
| `oversample_factor = 3.0` | M2, M12 | Arbitrary; no sweep or rationale anywhere |
| `keep_last_n_checkpoints = 3` | M2, M6 | Arbitrary within C1's disk motive |
| Split seed 42, training seed 0, `val_batches = 8` | M2 | Arbitrary values (the *separation* of seeds is principled; the numbers aren't) |
| `checkpoint_every = validate_every = 100` | M2 | Arbitrary, but they must be equal for selection to work, and nothing enforces that |
| `round(frac × n)` with banker's rounding | M4 | Incidental |
| `date / name / seed` directory layout | M3 | Preference |
| 10-hex-character experiment-ID digest | M2 | Arbitrary and sufficient |
| Learning-rate 3e-5 and warmup 20 for the continuation | M2 | A reasonable heuristic ("lower LR, shorter warmup on resume"), not tuned |

### Constraints that *should* have forced a decision, but didn't

| Constraint | Decision it implies | What the code does instead |
|---|---|---|
| **C3** tiny test set | bootstrap intervals; paired comparisons on one frozen test split; multiple real seeds | single numbers; per-intervention test sets; `--seed` doesn't seed (M4, M7, M11) |
| **C6** stochastic policy | predict once and slice; average samples during validation; store predictions | slices re-infer; 32-frame single-sample validation; docs claim determinism (M6, M7, M8) |
| **C10** never presume / never fabricate | fail fast on config typos; forbid unknown keys | extra keys ignored; free-string model name silently skips training (M2, M11) |
| **C11** unreliable runtime | atomic writes; status marker; signal handling | truncate-and-write; no status; SIGTERM uncaught (M3, M6) |
| **C2 + C10** offline vs real | keep feedback decisions off the test split | the curation loop runs on test (M8) |

That last table is the most important thing you can say about this project's design: **the constraints were correctly identified, and most were honoured in the *mechanism* layer (data versions, honesty markers, memory). The statistical consequences of C3 and C6, though, were not carried through into evaluation and reporting.**

---

## 13.7 Honest gaps to own before an interviewer finds them

1. **There's no test for any cross-stage invariant**: val-only selection, seed propagation, slices partitioning overall, artifacts agreeing with each other. *Consequence:* the bugs in Modules 6, 7, 8 and 11 were all invisible to the suite.
2. **The fixture's uniform episode lengths and a 2-episode test split mean length-bucket logic is never exercised on real data**, and `test_stratified_evaluate_per_task_and_length_bucket` almost certainly skips its length branch. *Consequence:* the test's name claims coverage it doesn't get.
3. **Some tests pin wrong behaviour** (zero fraction "honoured", presence-only health checks with empty weights, "deterministic" IDs). *Consequence:* fixing the bugs will break these tests, which is correct, but they currently read as specifications.
4. **There are no tests for the CLI, the reports, the sweep, `_prune_checkpoints`, `crash.json`, `DeployedModel.load`/`predict` or registry ordering.** *Consequence:* large surfaces are unexercised.
5. **The suite can't run without the sibling engine, and the smoke test needs Hub network plus torch.** *Consequence:* no CI is possible as configured, and "52 pass" is a local, dated claim.
6. **Function-scoped fixtures rebuild ingest, QC and build per test.** *Consequence:* a slow suite (unmeasured here); `scope="session"` for read-only fixtures would help.
7. **Global `set_settings` in a fixture.** *Consequence:* hidden coupling between tests that forget the fixture.
8. **The smoke test asserts existence, not value sanity, in evaluation.** *Consequence:* a NaN MSE in evaluation would pass.
9. **The private-engine-API import in the smoke test.** *Consequence:* brittle against legitimate engine refactors.

---

## The shape to carry into the cross-cutting modules

```
 tests/conftest.py   make_lerobot_dataset(8 eps × 20 frames, 2 tasks, uniform length)
   settings(tmp, set_settings) → raw/v1 (ingest) → clean/v1 (normalize + QC) → training/v1 (0.6/0.2/0.2, seed 7)
        │ real engine objects, no mocks
 ┌──────┴───────────────────────────────────────────────────────────────────────────────┐
 unit (49): config · interventions+lineage ✓ · split (fake leak) · profile · offline eval
            runner (baseline-only ✓, no-overwrite ✓) · deployment (presence-only ✓✗) · failure modes ✓
 integration (3): view→adapter→model→Prediction→Evaluator→feedback (shape)
 smoke (4, torch+Hub): feature shapes (cross-embodiment ✓) · 4-step SmolVLA run (existence)
 └──────────────────────────────────────────────────────────────────────────────────────┘
   MISSING: cross-stage invariants · CLI · reports · sweep · prune · crash · DeployedModel.predict

 Constraints → decisions:  C1 GPU · C2 no robot · C3 tiny data · C4 schema · C5 engine · C6 stochastic
                           C7 deps · C8 embodiment · C9 team · C10 spec rules · C11 unreliable runtime
   honoured in mechanisms ✓        statistical consequences of C3/C6 not carried through ✗
```

---

## Drill — answer from memory, files closed

1. Why does the suite use a real engine instead of mocks, and what does that cost?
2. Name three bugs from this manual that a test would have caught, and the test you'd write for each.
3. *Why didn't you just* write more unit tests with mocks so the suite runs without the engine?
4. *What breaks in production:* you set up CI on a standard GitHub runner tomorrow. What happens?
5. *Interviewer framing:* "Which of your design decisions were forced by constraints, and which were just choices?"

---

# Appendix A — Model answers to the drill

**A1. Real engine vs mocks.**
"This package is mostly glue around the data engine. It decides which episodes to keep, asks the engine to build versions, reads them back, and scores through the engine's evaluator. So the risk sits almost entirely at that boundary, and mocks would hide exactly the bugs I care about. With a real engine on a synthetic dataset, I can assert that the engine's own lineage record shows the same kept and dropped episodes my code computed, which is the project's core claim. The costs: the suite can't run without the sibling engine checkout, each test rebuilds ingest, QC and build, so it's slower, and an engine change can break my tests with no warning because the engine isn't version-pinned. I'd keep the real-engine tests, scope read-only fixtures to the session for speed, and pin an engine version range."

*Why that answer lands:* it justifies the choice by where the risk is, cites the specific test it enables, and owns all three costs with mitigations.

**A2. Three bugs a test would have caught.**
"First, the seed bug: run the same config with `seed=0` and `seed=1` and assert the resolved training seed differs. Today it doesn't, so replicates are fake. Second, the zero-fraction sweep point: call `build_intervention_dataset` with fraction 0.0. The existing test only calls the helper and asserts it returns an empty list, which passes while the real path raises. Third, the health check: give it an ALOHA-shaped checkpoint and an SO-100 contract and expect `DeploymentHealthError`. Today the test suite actually passes an empty weights file with mismatched metadata. The pattern is that my tests check each component in isolation and miss invariants that span stages, so I'd add a small set of cross-stage invariant tests: seed flows through, selection only ever calls the evaluator on val, slices partition the overall predictions."

*Why that answer lands:* it gives three concrete tests tied to real bugs, explains why the existing tests missed them, and generalizes to a testing strategy.

**A3. Why not mock everything?**
"Because the bugs that matter here live in how my code and the engine agree, and a mock encodes my *assumption* about the engine, so it agrees with my code by construction. For example, a mocked builder would never reveal that splits are assigned after filtering, which is the cause of the moving test set. Where mocks *would* help is the pure logic: pruning, threshold arithmetic, report rendering, CLI argument handling. None of that needs the engine, and none of it is tested today. So the answer is both: real-engine integration tests for the boundary, and fast, engine-free unit tests for the pure functions I currently don't test at all."

*Why that answer lands:* it explains the specific epistemic weakness of mocks with an example from your own system, then identifies where they *would* add value. A nuanced answer, not a dogmatic one.

**A4. CI on a GitHub runner tomorrow.**
"Installation fails first. The engine dependency is a bare name resolved only through a uv-specific sibling-path source, so the runner would need to check out both repositories side by side and install with uv. If you fix that, the 52 light tests should run, though I've never measured how long, since fixtures rebuild a dataset per test. The smoke test would skip, because there's no torch. If you installed torch, it would try to download the SmolVLA base checkpoint from the Hugging Face Hub and run on CPU, so it would be slow and network-dependent. I'd run the light suite on every push and the smoke test nightly on a GPU runner with a cached Hub download, and add one job that imports the runner and asserts torch wasn't loaded."

*Why that answer lands:* it predicts the failure sequence in order, separates the fast and slow tiers, and proposes a realistic CI layout.

**A5. "Which decisions were forced by constraints, and which were just choices?"**
"Forced: the small batch, the 32-frame validation cap and freeing GPU memory before evaluation all come from an 8-gigabyte laptop GPU. Offline-only evaluation and the 'unavailable' rollout come from having no robot or simulator. Exact-size episode splits and the separate split seed come from having 48 episodes. Using episode length as the imbalance variable comes from SO-100 being single-task with no variation field. Interventions as versioned transforms come from the data engine owning immutable versions. Just choices: the 90th-percentile failure threshold, the 0.5 concentration cut, 3× oversampling, keeping three checkpoints, and seed 42. Those are arbitrary, and in the threshold case they're inconsistent. And the interesting part is a third list, constraints that should have forced decisions but didn't. A tiny test set and a stochastic policy should have forced bootstrap intervals, real multi-seed runs, a frozen test split across interventions and predict-once slicing. I honoured the constraints in the mechanisms and didn't carry them through to the statistics."

*Why that answer lands:* it gives a clean three-way classification with evidence, labels arbitrary choices as arbitrary, and ends with a sharp self-critique that shows design maturity. It's the single answer that best summarizes the project.

---

# Module 13 Summary — the one page to revise

**What this module is.** A 56-test suite (52 torch-free) built on a real engine and a real synthetic dataset, strong at boundary integration and loud failure modes, absent on cross-stage invariants. It's followed by the capstone map tracing each design choice to one of twelve constraints, and naming the arbitrary choices and the constraints whose statistical consequences weren't carried through.

**Five claims to make cold**
1. Real engine, no mocks: the lineage test proves interventions are real engine transforms.
2. 49 unit + 3 integration = 52 light tests (README verified), plus 4 smoke tests needing torch and Hub network.
3. The fixture's uniform lengths and 2-episode test split mean length-slice logic is effectively untested.
4. No test checks val-only selection, seed propagation, slice reconciliation, CLI, reports, sweep or pruning.
5. Constraints were honoured in mechanisms (memory, lineage, honesty) but not in statistics (C3 tiny data, C6 stochastic policy).

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Real engine fixtures | mocks | risk is at the engine boundary | no engine means no tests; slow; unpinned coupling |
| Synthetic 8 × 20, 2 tasks | real data sample | fast, deterministic, task selection possible | uniform lengths; tiny test split |
| `importorskip` smoke | stubbed SmolVLA | never fake the model | needs torch plus Hub; private-API import |
| Fake view for leak detection | none | only way to reach the failure branch | proves the check is near-tautological |
| Baseline-model runner test | skip runner in light venv | full pipeline without torch | training path untested in the light venv |
| Function-scoped fixtures | session scope | isolation | rebuild cost per test |

**Volunteer unprompted:** no cross-stage invariant tests; the length-branch test that doesn't exercise its branch; tests pinning bugs; no CLI or report tests; CI impossible as configured; the capstone's "should have forced" table.

**Numbers and names:** 56 tests (8 / 6 / 6 / 3 / 10 / 9 / 3 / 4 unit, 3 integration, 4 smoke); fixture `so100_test`, 8 episodes × 20 frames at 10 fps, 2 tasks, splits 0.6/0.2/0.2 with seed 7; smoke: 4 steps, batch 2, ckpt and val every 2, keep 1, eval batch 3 (non-divisor, deliberate), `max_frames` 20; constraints C1–C12; RTX 4060 8 GB, about 10 GB RAM, lerobot 0.4.4, torch 2.10.

**Sentences worth memorizing**
- "I test against the real engine because a mock would agree with my assumptions by construction."
- "My tests check components; my bugs live between them."
- "Every non-obvious decision traces to a constraint, except the thresholds, which are arbitrary, and I say so."
- "I honoured the constraints in the mechanisms and failed to carry them through to the statistics."
