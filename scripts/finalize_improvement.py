"""Record the reviewed reference choice and measured second-stage evidence."""
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['v1', 'v2', 'v3', 'robust_seed', 'robust_split', 'v4', 'v5', 'improved_seed', 'improved_split']
ANALYSIS = 'mixture_three_sigma_trees2000'


def main():
    for name in RUNS:
        folder = ROOT/'outputs'/name
        assert json.loads((folder/'training_status.json').read_text())['status'] == 'complete', name
        assert (folder/ANALYSIS/'selected_for_interpretation.csv').exists(), name
    comparisons = pd.read_csv(ROOT/'outputs/comparison/run_stability.csv')
    nuisance = pd.read_csv(ROOT/'outputs/comparison/improvement_summary.csv').set_index('run')
    sensitivity = pd.read_csv(ROOT/'outputs/comparison/photometric_sensitivity.csv').set_index('run')
    def pair(left, right):
        row = comparisons[((comparisons.left == left) & (comparisons.right == right)) | ((comparisons.left == right) & (comparisons.right == left))]
        assert len(row) == 1
        return row.iloc[0]
    seed, split = pair('v4','improved_seed'), pair('v4','improved_split')
    baseline_seed, baseline_split = pair('v3','robust_seed'), pair('v3','robust_split')
    sets = {}
    for name in RUNS:
        scores = pd.read_csv(ROOT/'outputs'/name/ANALYSIS/'novelty_scores.csv')
        sets[name] = set(scores.loc[scores.flagged, 'filename'])
    common = len(sets['v4'] & sets['improved_seed'] & sets['improved_split'])
    reduction4 = 100 * (1 - sensitivity.loc['v4','p95_absolute_score_change'] / sensitivity.loc['v3','p95_absolute_score_change'])
    reduction5 = 100 * (1 - sensitivity.loc['v5','p95_absolute_score_change'] / sensitivity.loc['v3','p95_absolute_score_change'])
    repeat_summary = (
        f"For v4, changing initialization gives flag-set Jaccard {seed.flag_jaccard:.3f} and full-ranking correlation {seed.spearman_all:.3f}; "
        f"changing the source split gives {split.flag_jaccard:.3f} and {split.spearman_all:.3f}. "
        f"The three contrast runs flag {len(sets['v4'])}, {len(sets['improved_seed'])} and {len(sets['improved_split'])} crops, with {common} in their intersection. "
        f"On {int(split.common_heldout_images):,} crops held out from both source partitions, rank correlation is {split.spearman_common_heldout:.3f}. "
        f"For raw v3, the corresponding initialization/split Jaccards remain {baseline_seed.flag_jaccard:.3f}/{baseline_split.flag_jaccard:.3f}, with zero crops flagged in all three raw runs. "
        "These are limited sensitivity checks, not detection accuracy. Initialization repeats change both encoder and forest seeds; separate forest-only repeats isolate detector randomness."
    )
    rationale = (
        "I retain v3 as the screening reference. v4 and v5 reduce a specified brightness nuisance, but their minimum forest-seed flag overlaps fall from 0.667 for v3 to 0.467 and 0.227. "
        "All five leading v4 examples contain conspicuous black strips or borders, and the v5 exact-zero mask leaves near-black edge fragments. "
        "That evidence does not support replacing the reference with a more reliable geological detector. v3 itself remains brightness-confounded and unstable across independent training runs; I do not present its candidates as confirmed anomalies."
    )
    claim = (
        f"At fixed model, forest and threshold, v4 reduces the 95th-percentile absolute score response to 0.8*x + 0.1 from 0.033935 to 0.002902 ({reduction4:.1f}%). "
        f"v5 reduces it by {reduction5:.2f}%. This establishes the intended illumination invariance on 512 fixed validation crops. "
        "No sampled flags flip in any run, and the transformation is deliberately affine; this does not establish better anomaly recall. "
        "The project now has stronger controlled evidence and a readable, reproducible notebook; a detector-accuracy improvement is not demonstrated."
    )
    journal = "\n\n".join([
        '### v1: I established a baseline',
        'I trained a 128-dimensional MSE autoencoder from random initialization. Fine texture was smoothed even though validation MSE reached 0.002694. I used that observation to motivate a loss change.',
        '### v2 and v3: I changed one factor at a time',
        'Adding SSIM raised validation SSIM from 0.6611 to 0.6717, with a small MSE cost. Increasing the latent vector to 256 then gave MSE 0.002599 and SSIM 0.6738. That is a reconstruction gain, not a hidden-label accuracy result.',
        '### I corrected the threshold and checked repetition',
        'The adjusted-boxplot cutoff exceeded every v1 score. I retained that failure, inspected the multimodal distribution, and adopted the fixed mixture-envelope rule with whole-source bootstrap uncertainty. A controlled increase from 400 to 2,000 trees improved forest repeatability. Independent training then exposed fragile candidate membership and a bright-image concentration.',
        '### v4: I tested brightness normalization',
        f'I standardized each crop without fitting dataset-level statistics. The controlled brightness response improved by {reduction4:.1f}% at the 95th percentile, and the median brightness percentile of the five selected crops fell from 99.90 to 47.51. Forest repeatability weakened and all five contained black strips or borders. I recorded both outcomes.',
        '### v5: I tested an exact-zero footprint heuristic',
        f'I excluded border-connected exact zeros from normalization statistics and filled them with 0.5. The controlled brightness response improved by {reduction5:.2f}% relative to v3, but minimum forest Jaccard fell to 0.227. Near-black edge fragments remained. I did not silently broaden the mask after seeing selected examples.',
        '### I repeated the stronger standardized variant',
        repeat_summary,
        '### I recovered interrupted runs and retained the evidence',
        'Concurrent processes encountered a paging-file error and temporary-checkpoint write failures. I verified the existing checkpoints, removed only the failed temporary files and resumed with unchanged configurations. I then ran GPU work sequentially and used reversible filesystem compression to recover space. The saved execution is local, although the notebook is prepared for Colab.',
        '### My final decision',
        rationale,
    ])
    specs = []
    for prefix, mode in [('robust','raw'),('improved','contrast')]:
        for suffix, seed_value, split_value in [('seed',2027,2026),('split',2026,2027)]:
            specs.append(dict(name=f'{prefix}_{suffix}', latent_dim=256, structural_weight=.1, preprocessing=mode, seed=seed_value, split_seed=split_value))
    final = dict(status='Complete', recorded_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                 run='v3', analysis=ANALYSIS, rationale=rationale, improvement_claim=claim,
                 robustness_summary=repeat_summary, robustness_specs=specs, student_journal=journal,
                 context_summary='The retained v3 flags 17 crops from eight sources. Its five selected examples all exceed the 99.7th brightness percentile, and three share SRC_154. For v4, median selected brightness falls to the 47.51st percentile and the largest source contributes 12.5% of flags, but the flagged median zero fraction is 13.36%. Source concentration and brightness improvement therefore do not remove acquisition concerns.',
                 projection_interpretation='The v3 latent projection is retained from the first-stage analysis. Two perplexities show broad structure, but these branches are not labelled terrain classes. Source and acquisition effects remain plausible; the between-source variance fraction is about 0.232.',
                 cloud_execution_verified=False, preprocessing_promoted=False)
    (ROOT/'outputs/improved_selection.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    hypotheses = json.loads((ROOT/'outputs/geological_hypotheses.json').read_text())
    (ROOT/'outputs/improved_hypotheses.json').write_text(json.dumps(hypotheses, indent=2), encoding='utf-8')
    selected = pd.read_csv(ROOT/'outputs/v3'/ANALYSIS/'selected_for_interpretation.csv')[['filename','source_image_id','novelty_score']]
    for name in ['v3','robust_seed','robust_split','v4','v5','improved_seed','improved_split']:
        selected[name+'_flagged'] = selected.filename.isin(sets[name])
    selected.to_csv(ROOT/'outputs/comparison/selected_candidate_repeatability.csv', index=False)
    result = '# Second-stage results\n\n'+rationale+'\n\n## Measured benefit\n\n'+claim+'\n\n## Independent checks\n\n'+repeat_summary+'\n\n## Experiment labels\n\n`improved_seed` and `improved_split` name the planned contrast experiments; they do not assert improved detection accuracy.\n\n## My engineering journal\n\n'+journal+'\n'
    (ROOT/'IMPROVEMENT_RESULTS.md').write_text(result, encoding='utf-8')
    print(repeat_summary)


if __name__ == '__main__':
    main()
