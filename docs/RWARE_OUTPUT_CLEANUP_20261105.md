# RWARE generated-output cleanup (2026-11-05)

This cleanup was requested after the shared-core/Overcooked migration. It removes tracked, generated
RWARE analysis/test/paper-draft outputs that are no longer inputs to the current PGCT-D2 implementation or
the latest paper in `MainPaper/`. Source code, current experiment plans, checkpoints, and all of
`RWARE/runs/` are intentionally left untouched.

## Preserved `RWARE/output` directories

- `ac_drift_20261104`: directly cited by the latest paper and checks action-wise D2 distance/no-mirror behavior.
- `ac_early_id_20261026`: evidence for the current D2 event/action-max design.
- `ac_offline_20261026`: evidence for replacing the obsolete v1 reward gate with D2.
- `fig2_real_20261105`: source and data for the latest paper's predictor-response figure.
- `fig_data_20261105`: source and data for the latest paper's discovery/counterfactual figures.
- `paper_revision_20261105`: provenance for the latest English/Korean paper revision.
- `pgct_d2_validation_20261102`: current Max-GAE/PGCT-D2 regression evidence.
- `pgct_docs_20261102`: figure/deck generation source directly referenced by the latest paper.

## Removed directories

The following 53 directories contained 1,445 files (171,382,902 bytes / 163.44 MiB) before deletion:

```text
ac_gate_figs_20261102
ac_sparsity_20261102
ac_validation_20261023_final_tmp
ac_validation_20261023_integration_tmp
ac_validation_20261023_regression_tmp
ac_validation_20261023_saved_state_tmp
ac_validation_20261023_tmp
ac_verification_20261023
cleanup_20261015
difficulty_20261104
env_candidates_20261019
env_view_20261015
gating_diagnosis
hard_rware_validation_20261013
ijcai_method_paper_20261018
ijcai_paper_20261018
ijcai_review_20261015
latent_team
learning_review_20261013
nopen_pilot_20261022
oracle_why_20261029
paper_e1_20261018
pen005_analysis_20261023
penalty_sweep_20261022
pgct_authoring_20261102
pgct_beamer_20261021
pgct_progress_paper_20261023
pgct_report_20261022
pgct_spatial_analysis
pgct_transfer_review
pickup_stall_20261021
pilot_verdicts
probe_connect_20261019
probe_discriminative_20261023
probe_maps
probe_redesign_20261019
probe_team_check_20261015
probe_weight_20261019
request_sweep_20261023
reward_corr_check_20261019
reward_coupling_20261022
reward_probe_20261019
roadmap_20261105
scratch_review_20261015
scratch_review_20261018
status_review_20261015_131716
trajectory_probe_20261019
video_retention_20261104
video_retention_tests_20261104
video_retention_tests_b_20261104
vr_fulltests_20261104
vr_tests_c_20261104
wandb_path_diagnosis_20261104
```

These were Git-tracked files, so they remain recoverable from commit `b01ab42` or its ancestors. Historical
research notes may still mention their former paths; those references describe past work, not runnable
current dependencies.
