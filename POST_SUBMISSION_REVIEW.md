# Post-submission documentation review (2026-09-29 IST)

The repository baseline reviewed here is `main` at commit
`d35e30583389b908d9f094f9f50037ddcce0d560`. Confirm this SHA against
the actual submission receipt if one was recorded. This branch was created
after submission for review; its commits are not presented as deadline work.

The executed `notebooks/Mars_HiRISE_Submission.ipynb` and
`report/Mars_HiRISE_Final_Report.pdf` are the evidence for the reported runs.
No model was retrained and no score, threshold, selected candidate or PDF was
changed in this branch.

## Corrections

- v6 flagged **78** crops at threshold **0.459716**; the previous changelog
  claimed zero. Its synthetic-defect response missed the pre-specified gate.
- v7 flagged **6** crops at threshold **0.543500**; the previous changelog
  claimed 17. The 17-crop figure belongs to the canonical v3 run.
- The fixed-v3-encoder forest repeats have Spearman correlations of
  **0.994581/0.994693** and flagged-set Jaccards **0.666667/0.727273**.
  Independent training repeats have flagged-set Jaccards **0.150/0.294**
  against v3, with no crop flagged by all three runs. A high whole-dataset
  rank correlation does not establish stable candidate membership.
- All five reviewed v3 candidates are above the **99.7th percentile** of
  crop mean brightness. Their physical interpretations remain hypotheses.

## Verification and scope

The v6/v7 values were cross-checked against the saved output in the executed
submission notebook, and the v3 limits against that notebook and the final
report. Python source syntax was checked. Full PyTorch tests and retraining
were not run in the review environment because PyTorch and the image archive
were unavailable here. The technical conclusions of the submitted run are
unchanged.
