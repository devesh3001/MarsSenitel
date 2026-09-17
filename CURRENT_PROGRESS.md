# Latest audited status — 2026-09-17T17:26:00+05:30

Read FOLDER_AUDIT_REPORT.md for the full audit findings. Repairs applied after that audit. Canonical selection is **v3**. 17 tests pass. All original images and both ZIP manifests pass integrity checks.

**Resolved blockers:** `train.py` augment KeyError (backward compat) and Boolean CLI fixed; `package_submission.py` now includes all 14 run artifacts (v1-v8 + repeats); fusion checklist corrected to honest zero-flag outcome; geological hypotheses rewritten with Genesis Outlier framework; `verification.json` upgraded with SHA256 hashes; README/changelog reconciled to v3.

**Remaining:** Private GitHub repository not yet created; 5 organizer collaborators not yet added; team details not submitted. See SUBMISSION_CHECKLIST.md.


Read FOLDER_AUDIT_REPORT.md before using the older snapshot below. Fourteen full runs have completed 15 epochs; 12 current tests pass. All original images and both ZIP manifests pass integrity checks. v6 diagnostics are complete and it is not promoted.

**Submission blockers:** the newer notebook omits v7/v8 settings on a fresh run, the newer ZIP omits their checkpoints, v7 improvement claims exceed the evidence, fusion actually produces zero flags, and selection/documentation versions disagree. Existing source and deliverables were preserved for this audit. No cloud execution or external submission was performed.

**Next:** repair configuration and packaging contracts, reconcile measured claims, then reverify one canonical submission. See outputs/folder_audit for evidence.

---

## Earlier generated snapshot (historical; superseded above)

# Current project progress

Updated: 2026-09-17T11:58:41+05:30

This snapshot is generated from files. An incomplete checkpoint does not prove a process is currently running. Read DECISION_LOG.md for actions, evidence and the rationale for changes.

## Experiments

| Run | Training evidence | Latest epoch | Final 2,000-tree evaluation |
|---|---|---:|---|
| v1 | Completed | 15 | Completed |
| v2 | Completed | 15 | Completed |
| v3 | Completed | 15 | Completed |
| robust_seed | Completed | 15 | Completed |
| robust_split | Completed | 15 | Completed |
| v4 | Completed | 15 | Completed |
| v5 | Completed | 15 | Completed |
| improved_seed | Completed | 15 | Completed |
| improved_split | Completed | 15 | Completed |
| v6 | Checkpoint available; incomplete | 2 | Not started |

## Completed foundation

- All supplied documents and rules read; all 10,422 crops decoded and metadata joins validated.
- Source-disjoint splits, from-scratch model, resumable checkpoints and independent score threshold implemented.
- Initial threshold failure diagnosed; mixture-envelope alternative documented.
- Controlled forest-tree-count and subsample comparison completed.
- Eleven tests passed, including preprocessing invariance, connectivity, constant images and recovery after a partial checkpoint write.

## Deliverables

| Item | State |
|---|---|
| Decision log | Exists; see verification notes |
| First-stage selection | Complete |
| First-stage PDF | Exists; see verification notes |
| First-stage ZIP | Exists; see verification notes |
| Second-stage selection | Complete |
| Second-stage results | Exists; see verification notes |
| Revised PDF report | Exists; see verification notes |
| Readable Colab notebook | 26/26 code cells executed; 0 error outputs |
| Revised submission ZIP | Exists; see verification notes |

## Important boundaries

- No verified anomaly labels are available; no precision, recall, F1 or AUROC claim is justified.
- A final configuration must be selected from completed comparisons; larger models are not automatically better.
- PDF creation and visual verification are separate steps. A file existing is not proof of visual review.
- The private GitHub repository, collaborator invitations and team details have not been submitted or published.
- Per-epoch measurements are in outputs/<run>/history.csv; new training events are recorded in that run's events.jsonl.

## Current next steps

Border experiment: One final border-handling experiment; previous verified submission preserved.

The previous nine-run package remains the verified fallback. See BORDER_EXPERIMENT_PROTOCOL.md and outputs/border_experiment for the final bounded comparison. Twelve tests passed after adding near-black border handling.
