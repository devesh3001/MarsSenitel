# Final measured results

I retain v3 as the screening reference. v7 improves validation MSE by 1.68% and gradient error by 0.52%, but four of its five selected crops remain extremely bright and forest-repeat agreement weakens. v4-v6 reduce specific nuisance responses while introducing border or fill concerns; v6 fails its prespecified promotion criteria. v8 has zero image-only flags. No tested replacement establishes better hidden-anomaly detection. v3 itself remains brightness-confounded and unstable across independent runs; its flags are investigation candidates, not confirmed anomalies.

The experiments demonstrate modest reconstruction gains and controlled nuisance invariance. They do not establish a detection-accuracy improvement. I retain failed ablations and report negative outcomes.

For v4, changing initialization gives flag-set Jaccard 0.207 and full-ranking correlation 0.867; changing the source split gives 0.360 and 0.866. The three contrast runs flag 24, 11 and 10 crops, with 4 in their intersection. On 1,253 crops held out from both source partitions, rank correlation is 0.897. For raw v3, the corresponding initialization/split Jaccards remain 0.150/0.294, with zero crops flagged in all three raw runs. These are limited sensitivity checks, not detection accuracy. Initialization repeats change both encoder and forest seeds; separate forest-only repeats isolate detector randomness.

All-run measurements: outputs/comparison/experiments.csv. Population-labeled repeat comparisons: outputs/comparison/run_stability.csv. Final packaging and execution: outputs/final_verification.json (when complete).
