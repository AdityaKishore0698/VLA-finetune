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
