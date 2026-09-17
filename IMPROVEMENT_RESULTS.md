# Second-stage results

I retain v3 as the screening reference. v4 and v5 reduce a specified brightness nuisance, but their minimum forest-seed flag overlaps fall from 0.667 for v3 to 0.467 and 0.227. All five leading v4 examples contain conspicuous black strips or borders, and the v5 exact-zero mask leaves near-black edge fragments. That evidence does not support replacing the reference with a more reliable geological detector. v3 itself remains brightness-confounded and unstable across independent training runs; I do not present its candidates as confirmed anomalies.

## Measured benefit

At fixed model, forest and threshold, v4 reduces the 95th-percentile absolute score response to 0.8*x + 0.1 from 0.033935 to 0.002902 (91.4%). v5 reduces it by 99.87%. This establishes the intended illumination invariance on 512 fixed validation crops. No sampled flags flip in any run, and the transformation is deliberately affine; this does not establish better anomaly recall. The project now has stronger controlled evidence and a readable, reproducible notebook; a detector-accuracy improvement is not demonstrated.

## Independent checks

For v4, changing initialization gives flag-set Jaccard 0.207 and full-ranking correlation 0.867; changing the source split gives 0.360 and 0.866. The three contrast runs flag 24, 11 and 10 crops, with 4 in their intersection. On 1,253 crops held out from both source partitions, rank correlation is 0.897. For raw v3, the corresponding initialization/split Jaccards remain 0.150/0.294, with zero crops flagged in all three raw runs. These are limited sensitivity checks, not detection accuracy. Initialization repeats change both encoder and forest seeds; separate forest-only repeats isolate detector randomness.

## Experiment labels

`improved_seed` and `improved_split` name the planned contrast experiments; they do not assert improved detection accuracy.

## My engineering journal

### v1: I established a baseline

I trained a 128-dimensional MSE autoencoder from random initialization. Fine texture was smoothed even though validation MSE reached 0.002694. I used that observation to motivate a loss change.

### v2 and v3: I changed one factor at a time

Adding SSIM raised validation SSIM from 0.6611 to 0.6717, with a small MSE cost. Increasing the latent vector to 256 then gave MSE 0.002599 and SSIM 0.6738. That is a reconstruction gain, not a hidden-label accuracy result.

### I corrected the threshold and checked repetition

The adjusted-boxplot cutoff exceeded every v1 score. I retained that failure, inspected the multimodal distribution, and adopted the fixed mixture-envelope rule with whole-source bootstrap uncertainty. A controlled increase from 400 to 2,000 trees improved forest repeatability. Independent training then exposed fragile candidate membership and a bright-image concentration.

### v4: I tested brightness normalization

I standardized each crop without fitting dataset-level statistics. The controlled brightness response improved by 91.4% at the 95th percentile, and the median brightness percentile of the five selected crops fell from 99.90 to 47.51. Forest repeatability weakened and all five contained black strips or borders. I recorded both outcomes.

### v5: I tested an exact-zero footprint heuristic

I excluded border-connected exact zeros from normalization statistics and filled them with 0.5. The controlled brightness response improved by 99.87% relative to v3, but minimum forest Jaccard fell to 0.227. Near-black edge fragments remained. I did not silently broaden the mask after seeing selected examples.

### I repeated the stronger standardized variant

For v4, changing initialization gives flag-set Jaccard 0.207 and full-ranking correlation 0.867; changing the source split gives 0.360 and 0.866. The three contrast runs flag 24, 11 and 10 crops, with 4 in their intersection. On 1,253 crops held out from both source partitions, rank correlation is 0.897. For raw v3, the corresponding initialization/split Jaccards remain 0.150/0.294, with zero crops flagged in all three raw runs. These are limited sensitivity checks, not detection accuracy. Initialization repeats change both encoder and forest seeds; separate forest-only repeats isolate detector randomness.

### I recovered interrupted runs and retained the evidence

Concurrent processes encountered a paging-file error and temporary-checkpoint write failures. I verified the existing checkpoints, removed only the failed temporary files and resumed with unchanged configurations. I then ran GPU work sequentially and used reversible filesystem compression to recover space. The saved execution is local, although the notebook is prepared for Colab.

### My final decision

I retain v3 as the screening reference. v4 and v5 reduce a specified brightness nuisance, but their minimum forest-seed flag overlaps fall from 0.667 for v3 to 0.467 and 0.227. All five leading v4 examples contain conspicuous black strips or borders, and the v5 exact-zero mask leaves near-black edge fragments. That evidence does not support replacing the reference with a more reliable geological detector. v3 itself remains brightness-confounded and unstable across independent training runs; I do not present its candidates as confirmed anomalies.
