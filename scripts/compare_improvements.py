"""Summarize nuisance concentration and repeatability without choosing by count."""
from pathlib import Path
import json
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]


def main():
    stats=pd.read_csv(ROOT/'outputs/audit/image_statistics.csv')
    stats['brightness_percentile']=stats['mean'].rank(pct=True)*100
    rows=[]
    for name in ['v3','v4','v5','improved_seed','improved_split']:
        directory=ROOT/'outputs'/name/'mixture_three_sigma_trees2000'
        if not (directory/'selected_for_interpretation.csv').exists():continue
        diagnostic=json.loads((directory/'diagnostics.json').read_text())
        calibration=json.loads((directory/'calibration.json').read_text())
        scores=pd.read_csv(directory/'novelty_scores.csv').merge(stats,on='filename',validate='one_to_one')
        selected=pd.read_csv(directory/'selected_for_interpretation.csv')[['filename']].merge(scores,on='filename',validate='one_to_one')
        selected.to_csv(directory/'selected_acquisition_diagnostics.csv',index=False)
        flagged=scores[scores.flagged]
        overlaps=[x['flag_jaccard'] for x in diagnostic['forest_seed_stability'] if x['flag_jaccard'] is not None]
        rows.append({'run':name,'preprocessing':diagnostic.get('preprocessing','raw'),
                     'flags':int(scores.flagged.sum()),'threshold':calibration['threshold'],
                     'threshold_interval_width':calibration['bootstrap_interval_95'][1]-calibration['bootstrap_interval_95'][0],
                     'forest_min_jaccard':min(overlaps) if overlaps else None,
                     'forest_min_spearman':min(x['spearman'] for x in diagnostic['forest_seed_stability']),
                     'median_selected_brightness_percentile':float(selected.brightness_percentile.median()) if len(selected) else None,
                     'median_flagged_crop_mean':float(flagged['mean'].median()) if len(flagged) else None,
                     'median_flagged_zero_fraction':float(flagged.zero_fraction.median()) if len(flagged) else None,
                     'flagged_source_count':flagged.source_image_id.nunique(),
                     'largest_source_share_of_flags':float(flagged.source_image_id.value_counts(normalize=True).max()) if len(flagged) else None,
                     'spearman_brightness':float(spearmanr(scores['mean'],scores.novelty_score).statistic),
                     'spearman_contrast':float(spearmanr(scores['std'],scores.novelty_score).statistic),
                     'spearman_zero_fraction':float(spearmanr(scores.zero_fraction,scores.novelty_score).statistic)})
    result=pd.DataFrame(rows)
    result.to_csv(ROOT/'outputs/comparison/improvement_summary.csv',index=False)
    print(result.to_string(index=False))


if __name__=='__main__':main()
