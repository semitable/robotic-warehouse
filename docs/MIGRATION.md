# Migration: single RWARE fork → shared PGCT core + RWARE + Overcooked (2026-11-05)

Branch `feature/overcooked`, starting commit `122615f`. The migration was done in two separate operations:
a **pure structural move** (no content change), then the **extraction** of the PGCT-D2 algorithm into
`PGCT/pgct_core` with RWARE adapters. No algorithmic change, no hyperparameter change, no change of RNG use.
Nothing was committed, pushed or merged; the moves are staged with `git mv` (history preserved).

## 1. Path changes

| before | after |
|---|---|
| `rware/` | `RWARE/rware/` |
| `latent_team/` | `RWARE/latent_team/` |
| `examples/` | `RWARE/examples/` |
| `tests/` | `RWARE/tests/` |
| `setup.py`, `human_play.py`, `README.md` (upstream RWARE), `CITATION.cff` | `RWARE/…` |
| `runs/`, `output/` (tracked parts) | `RWARE/runs/`, `RWARE/output/` |
| `docs/img/` | `RWARE/docs/img/` (used by `RWARE/README.md`) |
| — | `PGCT/pgct_core/`, `PGCT/tests/`, `Overcooked/`, `scripts/`, `pyproject.toml`, new `README.md` |

Unchanged at the root: research records (`RESEARCH_STATE.md`, `DECISIONS.md`, `EXPERIMENTS.md`,
`CONVERSATION_LOG.md`, `HANDOFF.md`, …), `docs/` research documents, `paper/`, `presentation/`,
`MainPaper/`, `research_archive/`, `tmp/`, `AGENTS.md`, `CLAUDE.md`, `LICENSE`.
Path strings updated: `research_archive/build_archive.py` (RWARE sources), `.gitignore` (`RWARE/output/...`
patterns added next to the old ones), `.github/workflows/python-publish.yml` (builds `RWARE/`).

Windows note: the file system is case-insensitive, so `RWARE/` and the old root package `rware/` cannot
coexist. The package was renamed to a temporary name first, then moved to `RWARE/rware/`.

## 2. Import changes

| module | before | after |
|---|---|---|
| `latent_team/reward_probe_ac_v2.py` | full gate implementation | RWARE projection/mirror, `ACV2Config` (subclass of `pgct_core.RewardGateConfig`, legacy default `mirror=True`), `ACV2Controller(n, config)` = `pgct_core.RewardProbeGate` bound to the RWARE projection, unchanged `AuditedObservationEnv` and `install` |
| `latent_team/max_gae.py` | full Max-GAE implementation | imports `pgct_core.max_gae`; unchanged `ContactEnv` and `install`; `blended_advantages` keeps the trainer's `compute_gae` as default |
| `latent_team/_pgct_path.py` | — | puts `<repo>/PGCT` on `sys.path` when `pgct_core` is not installed |

Every public name used elsewhere (`ACV2Config`, `ACV2Controller`, `RewardPredictor(config, seed)`, `Ring`,
`view_size`, `mirror_view`, `mirror_action`, `features`, `response_distance`, `action_max_distance`,
`audit`, `EGO`, `DIR_LEFT/RIGHT`, `install`, `MaxGAEConfig`, `select_donors`, `donor_values`, `warm_mask`,
`uplift`, `blended_advantages`, `explained_variance`, `max_targets`, `ContactEnv`, `DONOR_MODES`, `SCALES`)
is still importable from the same RWARE module. Job names, manifests, log fields and checkpoint payloads are
unchanged.

## 3. How to run RWARE now (backward compatible)

All previous commands work unchanged **from the `RWARE` folder** (every script resolves paths relative to
its own file, `run_queue.ps1` changes into its parent folder, and `latent_team/common.py` uses
`RWARE/` as its root):

```powershell
cd RWARE
python latent_team\train_from_scratch.py --env-variant b6_corner_indiv_comm5_pen005 --modes pgct_max ippo --reward-probe-ac --ac-gate v2 --ac-distance action_max --ac-config latent_team\acv2_nomirror.json --seeds 1 --timesteps 64 --smoke-test --output-dir runs\smoke_check
```

```powershell
powershell -ExecutionPolicy Bypass -Command "& '.\RWARE\latent_team\run_main_amax_v2.ps1' -Candidate b6_corner_indiv_comm5_pen005 -Seeds 1 -Modes pgct_max,ippo -PlanOnly"
```

Results now default to `RWARE/runs/` (`C.ROOT / "runs"`), and the reference experiment is read from
`RWARE/runs/sp_20261008_120638_983201/`.

## 4. Machines with existing local results or running jobs

- `runs/` and parts of `output/` are git-ignored, so `git pull`/checkout leaves them at the **old** root
  location. After all trainings on that machine have finished, move them with
  `powershell -ExecutionPolicy Bypass -File scripts\migrate_local_layout.ps1` (dry run) and `-Apply`
  (moves only entries that do not exist yet in `RWARE/`; nothing is overwritten or deleted).
- **Do not switch to this branch while a `run_queue.ps1` queue is active** on that machine: the queue starts
  each new job with `latent_team/train_from_scratch.py` relative to its folder, which no longer exists after
  the checkout. Already-running Python processes are not affected (their modules are loaded).

## 5. Checkpoint compatibility

Checkpoints contain only tensors, numpy arrays and builtins (`agents`, `optimizers`, `config`,
`effective_env_kwargs`, `reward_probe_ac_v2` with `asdict(config)` and arrays). Moving or splitting modules
cannot break unpickling. Tested: `RWARE/tests/test_pgct_core_equivalence.py::test_existing_checkpoints_still_load`
loads every tracked agent checkpoint under `RWARE/runs` (`stage1_1ag…/final.pt` MLP, `ex5…/checkpoint_09700.pt`
CNN) into freshly built trainer networks with `strict=True`. Gate state saved by the refactored code is
byte-identical to the pre-split code (fingerprints below). No compatibility loader was needed.

## 6. Behaviour-preservation evidence (executed 2026-11-05, this PC)

Environment: conda env `marl-policy-sharing` (Python 3.11.15, torch 2.11.0+cu128, numpy 2.4.6, gymnasium
1.3.0), CPU. The documented env `RWARE_MARL` does not exist on this PC; `pyglet` is not installed here.

1. **Baseline before any change** (`tests/`, commit 122615f): 240 passed, 31 skipped, 1 failed. The skips
   needed the local reference experiment `runs/sp_20261008_120638_983201/experiment.json`, which exists only
   on the original training PC. For local testing a **stand-in** was rebuilt from the stored plan
   `output/ac_validation_20261023_final_tmp/test_from_scratch_plan_is_rand0/plan/experiment.json` (job
   `e1_pgct_s3`, a superset of the template the planner reads) and marked `"_reconstructed"`; it is git-ignored
   and was not committed. With it: **264 passed, 7 skipped, 1 failed**. The failure
   (`test_policy_probe_consensus.py::test_rware_probe_map_builder_saves_visual_manifest`) is `pyglet` missing
   on this PC (rendering), unrelated to the migration.
2. **After the structural move only** (run from `RWARE/`): **264 passed, 7 skipped, the same 1 failure**.
3. **After the extraction**: **289 passed, 8 skipped, the same 1 failure** (264 old tests + 25 new
   equivalence tests; the extra skip is a probe bank file that is not an agent checkpoint).
4. **Equivalence tests** (`RWARE/tests/test_pgct_core_equivalence.py`, exact equality, no tolerance): frozen
   pre-split copies in `RWARE/tests/legacy/` vs refactored code on identical inputs —
   gate traces over 8 updates for {pooled, action_max} × {mirror on/off} × {snapshot on/off} × {views, raw
   RWARE observations} (allocations, logs, EMA, rounds, approvals, maturity, predictor weights, state);
   config defaults/field order; mirror, features, distances; GAE; warm mask; blended advantages for raw/
   shrink/relative; donor selection for all five rules; donor critics and Max-GAE targets with resets,
   timeouts and caps; core network vs trainer network (same init, same outputs); core PPO update vs trainer
   `update_ippo` (bit-identical weights, with and without injected Max-GAE advantages); checkpoint loading;
   and a real-trainer `pgct_max`/`oracle_max` run with legacy vs new modules (identical fingerprints).
5. **Trainer-level fingerprints** (`RWARE/tests/characterize_pgct_d2.py`; candidate
   `b6_corner_indiv_comm5_pen005`, action_max, seed 1, 64-step smoke jobs; 6 modes × 3 settings: gate closed,
   oracle approval injected, permissive natural gate with real comparisons and approvals): SHA-256 of final
   weights, saved gate state and `max_gae.jsonl` rows recorded before the extraction and after it —
   **18/18 identical**. Checks inside the record: closed gate ⇒ `pgct_max` = `ippo` weights; injected ⇒
   `pgct_max` = `oracle_max` weights; natural ⇒ 10 real comparisons and 16 donor arcs.

Bit-level identity is guaranteed on one machine/torch build; across machines, rerun
`RWARE/tests/characterize_pgct_d2.py` before and after.

## 7. What did not change

The RWARE trainer, environment, `examples/`, `reward_probe_ac.py` (v1), runners, candidates, hyperparameters,
probe bank, gate thresholds, Max-GAE defaults, RNG streams, logging names and job naming. The legacy RWARE
default `ACV2Config.mirror=True` was kept for old commands; `pgct_core.RewardGateConfig.mirror` defaults to
`False` (Q_ij = E_i ∪ E_j) for new environments.

## 8. Known limitations

- The real-trainer RWARE tests that need the 1M checkpoint
  (`runs/ac_20261023_185556/.../checkpoint_01000.pt`) remain skipped on this PC.
- `RWARE/tests/legacy/` must stay as long as the equivalence test is kept; delete both together if the
  frozen reference is no longer wanted.
