# Competition submission checklist

The revised notebook and report are verified in `outputs/improved_verification.json`. Packaging status is maintained in `CURRENT_PROGRESS.md`. Scientific limitations remain explicit; software/artifact completion is not proof of anomaly-detection accuracy.

| Requirement | Evidence / status |
|---|---|
| Read rules, problem and data description | Complete; `RESEARCH_AND_APPROACH.md`, input hashes and extracted audit text |
| Check all supplied crops and joins | Complete; 10,422 decoded crops, `outputs/audit/audit.json` |
| From-scratch fixed-length image latent | Complete; `mars_anomaly/model.py`, saved checkpoints and configurations |
| Justify depth and bottleneck | Complete; controlled 128/256-dimensional comparison and explicit parameter tradeoff |
| Explain loss choices with measured outcomes | Complete; v1 MSE, v2 structural loss, v3 capacity experiment |
| t-SNE or UMAP with cautious interpretation | Complete; two reviewed v3 t-SNE projections |
| Latent Isolation Forest, higher-is-more-anomalous | Complete; `-score_samples`, 2,000 trees and score-sign test |
| Statistical threshold independent of desired count | Complete; mixture-envelope calibration, original failed fence retained, source bootstrap and caveats |
| Source metadata join and location analysis | Complete; supplied keys only, coarse geographic fields and unspecified units documented |
| Optional metadata fusion | **Complete; methodology demonstrated.** Training-only StandardScaler + OneHotEncoder on sun\_angle, resolution (2 dims) and season (4 one-hot dims) appended to the 256-dim image latent = 262-dim fused vector. Five-forest matched ensemble; both image-only and fused calibration thresholds exceeded zero flags. The candidate set is unchanged (empty for both), which is retained as the correct result rather than manufacturing examples. Comparison CSV and persisted detector artifacts are in `outputs/v8/fusion_controlled/`. |
| At most five selected after thresholding | Complete; five of primary flags, strict threshold check |
| Original/reconstruction/error/overlay panels | Complete; five visually reviewed panels with common error scale |
| Physical hypotheses and alternatives | Complete; no verified hidden-anomaly claim |
| Three genuine measured model iterations | Complete; v1–v8 each trained 15 epochs, including v7 gradient loss and v8 geometric augmentations |
| Independent training and split sensitivity | Complete; four 15-epoch repeats cover raw and contrast inputs; mixed agreement reported |
| Controlled nuisance check | Complete; 512 validation crops, fixed model/forest/threshold; no hidden-label accuracy claim |
| Notebook with outputs and question headings | Complete; notebook 15 code cells executed, zero error outputs, all Phase headings present |
| PDF report and labeled figures | Complete; report rendered and visually reviewed |
| Decision notes and defense guide | Complete; `DECISION_LOG.md`, `CURRENT_PROGRESS.md`, `DEFENSE_GUIDE.md` |
| Local package and integrity manifest | Complete; 244 files, 190.8 MB ZIP packaged with CRC check |
| Private GitHub repository | **Pending — must create private repo and push all files** |
| Required organizer collaborators | **Pending — must add: SarthakXSingh09, R15HV, Shivam3473, aditohates-bugs, Dhairya646** |
| Team of 2–4 and 15-minute finalist presentation | Team identities/details remain to be supplied; rehearsal outline is provided in DEFENSE_GUIDE.md |

The detailed statement numbers location analysis as 2.3, while the rules' mark-distribution note refers to 2.5. The notebook follows 2.3 from the detailed statement. The four technical phases total 100 points, scaled to 60, plus 40 presentation points in the rules.

Cloud portability is provided, but no Colab/Kaggle execution is claimed. The submitted artifact package must remain private under the competition rules. No external submission was performed by the local packaging process.
