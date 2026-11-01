# Running gaps register (feeds the Fixes & Roadmap module; excluded from the PDF)

Format: ID · module · file · gap · consequence · status (verified / unverified-needs-rde)

## From Phase 1 survey (to be re-verified in their modules)
- S1 · M11/M12 · runner.py, cli.py · `--seed` only renames run dir; training.seed unchanged; `seeds[1:]` ignored · verified
- S2 · M10 · experiment_report.py:93 vs run_dir.py:43 · git key `model_training` vs `vla_finetune` → always n/a · verified
- S3 · M2 · schema.py:150 · comment cites nonexistent `model_training.experiments.selection` · verified
- S4 · M2 · schema.py · success_tolerance, num_workers, mixed_precision never passed anywhere · verified by grep
- S5 · M2 · schema.py · pydantic models allow extra keys → YAML typos silently ignored · verified (pydantic default)
- S6 · M4/M11 · interventions.py · fraction=0.0 honored by resolver, rejected by build_intervention_dataset · verified
- S7 · M12 · cli.py evaluate · docstring says re-runs baselines; doesn't; hard-coded slices; overwrites artifacts · verified
- S8 · M9 · loader.py · health_check never compares checkpoint meta vs contract · verified
- S9 · M6 · trainer.py · selection on val_batches*batch_size = 32 frames of stochastic policy · frame choice unverified-needs-rde
- S10 · M5 · profile.py · episode_length_cv is range/mean, not CV · verified
- S11 · M4/M7 · interventions vs offline · length terciles whole-dataset vs within-split; docstring claims comparable · verified
- S12 · M3/M10/M12 · manifest.json read never written; logs/ unused · verified
- S13 · M11 · runner.py · crash outside Trainer leaves non-empty run dir, no marker, blocks rerun · verified
- S14 · M2 · schema.py · experiment_id hashes created_at → non-deterministic · verified

## Module 1
- M1-1 · trainer step lives in rde (ownership overclaim risk)
- M1-2 · not installable standalone (uv-only sibling source; pip ignores)
- M1-3 · no lockfile; floors only; engine unpinned
- M1-4 · smoke test imports rde private functions
- M1-5 · no CI
- M1-6 · no top-level __init__.py (namespace package, generic name)
- M1-7 · light-venv laziness unenforced; deployment/artifact.py eager SmolVLAPolicy import negates runner's lazy Trainer import
- M1-8 · vestigial: allow-direct-references, pytest-cov, psutil ImportError fallbacks
- M1-9 · exported-but-unwired: ModelRegistry, DeployedModel, run_data_fraction_sweep, build_model_dataset_matrix
- M1-10 · undeclared direct torch import in trainer.py / run_dir.py
- M1-11 · pyproject author/URL point to another person/repo
- WITHDRAWN · .venv-vla not gitignored — venvs live in ../, not a gap

## Module 2
- M2-1 · schema.py · no extra="forbid"; typos vanish incl. from resolved record · P0
- M2-2 · schema.py/runner.py · training.model free str; non-"smolvla" silently skips training, exit 0 · P0
- M2-3 · schema.py · num_workers, mixed_precision, success_tolerance dead; API_REFERENCE overclaims success_tolerance (= S4)
- M2-4 · schema.py/interventions.py · imbalanced with variable but no/str-typed variable_value → full-data run labelled imbalanced · P0
- M2-5 · schema.py · experiment_id/created_at user-settable despite "not user-editable" comment + API_REFERENCE
- M2-6 · schema.py · experiment_id not content-addressed; no config_hash (= S14 refined)
- M2-7 · schema.py:150 · stale `experiments.selection` citation (= S3)
- M2-8 · schema.py · split fractions unbounded; val=0 disables selection silently
- M2-9 · loader.py · base found by directory convention; no --base CLI flag though configs/smolvla/so100_pickplace.yaml says so; single base only; file is dead
- M2-10 · schema.py vs base.yaml · two disagreeing default sources
- M2-11 · baseline_extended.yaml · dated CWD-relative git-ignored resume path
- M2-12 · schema.py · flat intervention accepts kind-irrelevant fields
- M2-13 · schema.py + interventions.py · 0.5 default duplicated
- M2-14 · selection_metric free str; unknown metric → silent fallback to final
- M2-15 · oversample_factor / episode_weights unbounded (≤0 accepted)
- M2-16 · "identical test set" across baseline/extended relies on deterministic DatasetBuilder; unverified in-repo

## Module 3
- M3-1 · run_dir.py/index.py · no atomic writes (write_text truncates; buffered appends) · P1
- M3-2 · index.py · read_all has no per-line tolerance; one torn line poisons index + ModelRegistry · P1
- M3-3 · run_dir.py + runner order · failed run blocks same-day retry; no status/crash marker pre-trainer (= S13) · P1
- M3-4 · run_dir.py · make_run_dir TOCTOU race; use mkdir(exist_ok=False) · P2
- M3-5 · runner.py/run_dir.py · engine git provenance from guessed path, not rde.__file__; can be null/wrong silently · P1
- M3-6 · run_dir.py · dirty flagged but no diff saved · P2
- M3-7 · run_dir.py · driver_capability is compute capability; no driver/cudnn/transformers; vla-finetune version meaningless · P2
- M3-8 · experiment_report.py · git key mismatch → n/a (= S2) · P0 (trivial)
- M3-9 · run_dir.py · logs/ never written; no logging anywhere; engine stdout uncaptured · P2
- M3-10 · index.py · index write-only inside system; non-transactional with run; no status → failed runs/orphan versions invisible · P2
- M3-11 · default=str silently stringifies np.float32 etc. (unverified whether rde emits) · P2
- M3-12 · half the artifacts bypass RunDir · P2
- M3-13 · runs/experiments default from parents[3] → inside site-packages for wheel install (latent) · P2
- M1-11 UPDATED: README:298 credits SO-100 runs to AdityaKishore0698, ALOHA to chytra3

## Module 4
- M4-1 · interventions.py:195 · filter-then-split → each excluding intervention has its own smaller test set; imbalanced test lacks removed regime; feedback's "compare on same fixed test slice" impossible · P0 (design) · fix: split once, intervene on train only
- M4-2 · interventions.py:88-91/181 · frac=0.0 resolver returns [] but builder raises; unit test tests helper only (= S6) · P1
- M4-3 · interventions.py:97/100 · imbalanced = total removal, not under-representation (config template says under-represent) · P2 (doc/semantics)
- M4-4 · interventions.py:47 vs offline.py:171 · whole-dataset vs within-split terciles; "same as profile.py" false — profile has none (= S11) · P1
- M4-5 · interventions.py:56 · ties (<= t1) collapse buckets; uniform-length data → imbalanced long/medium silently full · P2
- M4-6 · interventions.py:91 · fraction draws not nested across fractions · P2
- M4-7 · explicit episode_ids/weights unvalidated (unknown ids, dupes, ≤0) · P2
- M4-8 · EXPERIMENTS.md §4 · "every report states rebalanced limitation" — false, not in code · P1 (doc)
- M4-9 · EXPERIMENTS.md §6 · dataset.json does not record test membership; only predictions file does · P2
- M4-10 · next_version not atomic; orphans on failure; identical requests mint new versions · P2
- M4-11 · module docstring says task_id is the one grouping key — stale · P2
- M4-12 · misleading rebalanced error for unsupported variable · P2
- M4-13 · weights recorded for val/test episodes too · P2
- M4-14 · QC filtering at build (unverified) may make dataset.json episode_ids overstate pool · unverified
- DOC (for M10): EXPERIMENTS.md §7 claims `mtrain compare` reports mean/std across seeds — no aggregation code exists

## Module 5
- M5-1 · split_integrity.py · overlap check near-tautological (split = single attribute/episode); failure branch only reachable via fake · P2 (claim downgrade)
- M5-2 · split_integrity.py:28 · ok ignores unassigned, empty splits, sizes vs config → real failures pass · P1
- M5-3 · no content-level (near-duplicate/same-session) leakage detection · P2 (limitation; README NN-tie evidence)
- M5-4 · profile.py:45-53 · normalized entropy divides by log2(#occupied bins) → evenness not coverage; bimodal 2-bin = 1.0; methodology text + module docstring overclaim · P1 · fix: log2(_HIST_BINS)
- M5-5 · profile.py:258 · episode_length_cv = range/mean (~4.5× CV at n=48); convoluted and-guard (= S10) · P2
- M5-6 · runner→profile · profile over training_version = train∪val∪test, not train split · P2
- M5-7 · profile.py · coupled to undocumented explorer dict keys · P2
- M5-8 · profile.py:112 · clamp folds out-of-range values into edge bins; NULLs dropped unreported · P2
- M5-9 · task entropy duplicated as instruction diversity · P2
- M5-10 · profile.py:181-187 · misleading "not a training version" note after exception; "unassigned" kept as 4th split when partially assigned · P2
- M5-11 · to_dict hand-listed · P2
- M5-12 · parent_version first input only · P2

## Module 6
- M6-1 · trainer.py:99 · selection on 32 val frames (likely first frames of one episode — unverified), stochastic policy, min over 10–20 → README val 0.0018 vs test 0.0239 (13×) · P0 · fix: frozen stratified val subset, avg samples (= S9)
- M6-2 · trainer.py:184 · except Exception misses KeyboardInterrupt/SIGTERM/SIGKILL; comment claims SIGTERM · P1
- M6-3 · trainer.py:228 · silent fallback best→final, no limitation recorded · P1
- M6-4 · trainer.py:163-170 · NaN first val metric locks best forever (needs isfinite) · P1
- M6-5 · cadence alignment (ckpt % val) unchecked → selection silently disabled · P2
- M6-6 · trainer.py:82-85 · resume not checked vs dataset; load before verify · P2
- M6-7 · optimizer state never saved/restored though fit reportedly accepts/returns it · P2
- M6-8 · trainer.py:208 · final duplicates step_N when aligned; unverified; unvalidated; outside try · P2
- M6-9 · verify_checkpoint presence-only; comment's kill-mid-write threat not catchable in-process · P2
- M6-10 · _prune_checkpoints untested; failed rmtree marked pruned; corrupt ckpts occupy keep slots; keep<=0 = keep all · P2
- M6-11 · crash-path del/empty_cache can't free traceback-pinned memory (comment misdescribes) · P2
- M6-12 · peak_gpu_mb = allocator peak post-load; understates provisioning · P2
- M6-13 · empty-train guard after model load · P2
- M6-14 · grad_accum step semantics unknown · unverified

## Module 7
- M7-1 · offline.py:150-185 · slices re-run stochastic inference; non-reconciling; ~3× inference on SO-100; fix: slice overall.outcomes · P1
- M7-2 · README §2.2 · "long episodes easier" from 1–2 episodes/slice — below resolution · P1 (claim downgrade)
- M7-3 · no uncertainty anywhere (bootstrap over episodes from test_outcomes.jsonl; seeds) · P1
- M7-4 · NN baseline = state-only retrieval; tie is key interpretive fact (not a bug) · framing
- M7-5 · offline.py:79-80 · per-episode "mse"/"rmse" = squared per-frame MAE (disclosed only in METRICS.md) · P2
- M7-6 · base.yaml · per_task kept for single-task SO-100 (aloha config removes as redundant) · P2 (trivial)
- M7-7 · METRICS.md §2 · claims evaluate_baselines computes ratios — it doesn't · P2 (doc)
- M7-8 · offline.py:170 · length slices silently skipped < 3 episodes, no note · P2
- M7-9 · offline.py:108 · temporal_error assumes contiguous frame_index · P2
- M7-10 · baselines unweighted in rebalanced runs · P2
- M7-11 · error_distribution omits frame count despite METRICS.md · P2
- M7-12 · success_tolerance accepted by Evaluator.evaluate (per API doc) but never passed — one-line wire-up (= S4/M2-3)
- M7-13 · EXPERIMENTS.md §6 "frozen slice" true per version only; cut points depend on test lengths · P2 (doc)

## Module 8
- M8-1 · runner→feedback · feedback runs on TEST; curation loop tunes on test (adaptive overfitting) · P0 (method) · fix: run on full val eval
- M8-2 · artifact.py:35-66 · p90 threshold always flags ~10% → reviews always non-empty → "no action" unreachable · P1 · fix: absolute/baseline-relative gate
- M8-3 · artifact.py:43 · failure_rate>0.5 unreachable (max ≈0.50 SO-100, 0.40 ALOHA) → defect branch dead · P1 · fix: base-rate-relative/binomial
- M8-4 · artifact.py:64-72 · recommendation fixed template with blank placeholders; FAILURE_ANALYSIS §6 overclaims · P2
- M8-5 · FAILURE_ANALYSIS.md §4 · claims predictions deterministic — contradicts README §2.3 stochastic · P1 (doc)
- M8-6 · FAILURE_ANALYSIS.md §3 · per-dim per-episode via episode_metrics — not implemented; "sorted by mse" — sorted by id · P2 (doc)
- M8-7 · persist=True side effect; dupes on re-run; stale after `mtrain evaluate` · P2
- M8-8 · retraining_selection computed, never consumed · P2
- M8-9 · affected_slice always whole split · P2
- M8-10 · thresholds not configurable · P2
- M8-11 · integration test checks shape only; dead branches untested · P2

## Module 9
- M9-1 · loader.py:63-89 · health_check never compares ckpt meta vs contract (action_dim, cameras, dataset) despite "any mismatch" docstring (= S8) · P1
- M9-2 · loader.py:82-87 · normalization check on contract; inference uses checkpoint preprocessor (from_checkpoint contract=None) → wrong object; real skew unchecked · P1
- M9-3 · loader.py:97 · predict takes engine Sample → replay harness, not robot deployment · P1 (framing) / roadmap
- M9-4 · DeployedModel.load/predict never executed anywhere (tests only health_check) · P1
- M9-5 · loader.py:109-110 · postprocessing_ms = wrapper overhead (postprocessor inside predict); total is a sum; clamp hides negatives; GPU sync unverified · P2
- M9-6 · registry.py · catalogue not registry (no stages/promotion); unwired (documented) · P2
- M9-7 · artifact.py:69 · metrics = test, labelled "what metrics justified selection" (DEPLOYMENT.md T2, registry docstring); val metric absent · P1
- M9-8 · artifact.py · model_version always step; pretrained_model may be resume path (unverified); model_id prefix from free tag · P2
- M9-9 · DEPLOYMENT.md §5 · says predict uses select_action; API_REFERENCE + smoke test say predict_action_chunk · P2 (doc)
- M9-10 · test_deployment.py · health-check tests pass empty safetensors + mismatched meta (encodes presence-only) · P2
- M9-11 · registry "most recent wins" untested · P2
- M9-12 · LatencyBreakdown.note names VisionStateAdapter, class uses VisionLanguageAdapter · P2 (trivial)

## Module 10
- M10-1 · experiment_report.py:93 · git(model_training) always n/a (= S2/M3-8) · P0 trivial
- M10-2 · experiment_report.py docstring · "assembled from saved artifacts" — built from in-memory ctx; no rebuild path; timestamped · P2 → add build_report_ctx(run_dir) + `mtrain report`
- M10-3 · experiment_report.py:22 · per-dimension MAE (list) dropped from report · P2
- M10-4 · experiment_report.py:112 · SmolVLA MAE hard-coded n/a · P2 trivial
- M10-5 · comparison.py · no seed aggregation (EXPERIMENTS.md §7 claims mean/std) · P1
- M10-6 · comparison.py · no shared-test-set check/column · P1
- M10-7 · cli.py compare · rich table headers from rows[0], positional cells; rich adds headerless columns → silent misalignment (verified rich source) · P1
- M10-8 · plot_results.py · selected_step 1800/800, offset 1000, LR text, ylims hard-coded; docstring "reproducible from disk" · P2
- M10-9 · build_model_dataset_matrix / plot_model_comparison / plot_data_fraction_curve uncalled; matrix pivots on unique training_version · P2
- M10-10 · figures.py:12 · matplotlib.use("Agg") at import → global side effect (notebooks) · P2
- M10-11 · model_card · "Known failure modes" = template; names unexecuted DeployedModel; raw-float metrics · P2
- M10-12 · rebalanced + best→final fallback limitations never added to report · P1
- M10-13 · figures.plot_training_curves r["loss"] unguarded → plot KeyError aborts run after training · P2
- M10-14 · manifest.json read never written; missing files → None rows silently (= S12) · P2
- M10-15 · `None` printed for selected checkpoint in baseline-only report · P2 trivial
- M10-16 · no unit tests for reports package · P1

## Module 11
- M11-1 · runner.py:73 · seed → dir name + summary only; replicates share training.seed AND experiment_id → fake replicates, model_id collision (= S1) · P0
- M11-2 · runner.py:172/191 · unknown baseline/slice names raise AFTER training · P0 (fail-fast) · fix: Literal/stage-0 check
- M11-3 · runner.py · non-transactional across run dir / engine catalog / index / engine failure records; no status · P1
- M11-4 · runner.py:~225 · record_model before feedback & before model_artifact.json written · P2
- M11-5 · evaluation.split="val" silently relabels "test" reports · P2
- M11-6 · no GPU guard (CPU SmolVLA runs proceed) · P2
- M11-7 · summary/experiments.jsonl omit NN baseline; no total run time · P2
- M11-8 · predictions run_id = absolute path · P2
- M11-9 · no stage-level resume/caching · P2 (roadmap)
- M11-10 · data_fraction.py · each fraction different test set; non-nested; "isolates quantity" docstring false · P1
- M11-11 · data_fraction.py · aborts on first failure (0.0 always fails), results lost; persist incrementally · P1
- M11-12 · data_fraction.py · name collision (_frac12), base intervention discarded, garbled docstring · P2
- M11-13 · sweep untested, no CLI · P1
- M11-14 · eval from_checkpoint ignores training.device; eval policy not released · P2
- M11-15 · same view as train & val; two-view seam unused · note
- PATCHED M1 table: run_data_fraction_sweep has no tests

## Module 12
- M12-1 · cli.py evaluate · overwrites test_overall.json/stratified.json in place, stochastic, no history (= S7) · P0 · fix: versioned rescore dir
- M12-2 · cli.py evaluate + API_REFERENCE §4.5/§9.3 · --split val writes val into test_overall.json; doc recommends it · P0
- M12-3 · cli.py evaluate · divergent re-impl: no baselines (docstring claims), slices hard-coded per_task, default batch, no update of predictions/summary/report/card/artifact/feedback · P1 · fix: factor runner stage 4
- M12-4 · cli.py build-intervention · safe_dump erases template comments, reflows folded strings · P1 · fix: ruamel / separate file
- M12-5 · cli.py:269 · imbalanced written as kind: custom → label lost · P2
- M12-6 · build-intervention imbalanced w/o --variable-value → all episodes (= M2-4) · P1
- M12-7 · _target_episodes duplicated (cli vs interventions.py) · P2
- M12-8 · resolution --version not checked vs config dataset.version · P2
- M12-9 · export omits git.json/environment.json/reports/predictions; tables/ empty; manifest.json never exists · P1
- M12-10 · validate/analyze-failures misdiagnose crashed runs · P2
- M12-11 · scripts/compare.py docstring `--runs a b c` invalid for Typer list option · P2 trivial
- M12-12 · no --base/--force/seed loop/report/sweep commands; --seed help silent · P2
- M12-13 · inspect --version default training/v1 arbitrary · P2
- M12-14 · scripts wrappers redundant with python -m, partial (7/11), scripts/test.py shadows stdlib test · P2
- M12-15 · no CLI tests · P1
- M12-16 · rollout duplicate untimestamped records · P2

## Module 13
- M13-1 · tests · no cross-stage invariant tests (val-only selection, seed propagation, slice reconciliation, artifact agreement) · P1
- M13-2 · conftest + test_evaluation_offline · uniform lengths + ~2-ep test split → per_length_bucket branch almost certainly skipped; test name overclaims · P2
- M13-3 · tests pinning wrong behaviour: zero-fraction helper, empty-safetensors health check, "deterministic" experiment_id · P2
- M13-4 · untested: CLI, reports, sweep, _prune_checkpoints, crash.json, DeployedModel.load/predict, registry ordering · P1
- M13-5 · suite needs sibling engine; smoke needs Hub + torch; no CI possible as configured (= M1-5) · P1
- M13-6 · function-scoped fixtures rebuild ingest/QC/build per test · P2
- M13-7 · global set_settings in fixture · P2
- M13-8 · smoke asserts existence not value sanity in eval · P2
- M13-9 · smoke imports private rde functions (= M1-4) · P2
- CAPSTONE written in 13.6
