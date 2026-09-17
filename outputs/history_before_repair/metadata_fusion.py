import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from mars_anomaly.threshold import mixture_fence

def fusion():
    print("=== Phase 2.4: Secondary Metadata-Fusion Experiment ===")
    from mars_anomaly.data import load_manifest
    manifest = load_manifest('data', 2026)
    
    # Load latents from the augmented v8 run (the model chosen for fusion)
    latents = np.load('outputs/v8/mixture_three_sigma_trees2000/latents.npy')
    filenames = np.load('outputs/v8/mixture_three_sigma_trees2000/latent_filenames.npy', allow_pickle=True)
    
    # Align manifest with latents
    df_latents = pd.DataFrame({'filename': filenames})
    df = df_latents.merge(manifest, on='filename', how='left')
    
    # Extract metadata
    numerical = df[['sun_angle', 'resolution']].values
    categorical = df[['season']].values
    
    # Fit scalers ONLY on train data to strictly prevent data leakage
    train_mask = df['split'] == 'train'
    calib_mask = df['split'] == 'calibration'
    
    scaler = StandardScaler()
    scaler.fit(numerical[train_mask])
    num_scaled = scaler.transform(numerical)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(categorical[train_mask])
    cat_encoded = encoder.transform(categorical)
    
    # Phase 2.4: Concatenate visual latents with normalized metadata
    fused_latents = np.concatenate([latents, num_scaled, cat_encoded], axis=1)
    
    # Train 5-Forest Ensemble on fused metadata (training set only)
    scores_list = []
    print(f"Training 5-Forest Ensemble on {train_mask.sum()} training fused vectors...")
    for seed in [2026, 2027, 2028, 2029, 2030]:
        iso = IsolationForest(n_estimators=2000, max_samples=256, contamination="auto", random_state=seed, n_jobs=-1)
        iso.fit(fused_latents[train_mask])
        scores_list.append(-iso.score_samples(fused_latents))
        
    final_scores = np.mean(scores_list, axis=0)
    df['novelty_score'] = final_scores
    
    print(f"Fitting independent GMM threshold on {calib_mask.sum()} calibration scores...")
    # Calibrate its OWN threshold from calibration split
    calib_scores = df.loc[calib_mask, 'novelty_score']
    # mixture_fence fits GMM and returns the dict with threshold
    calibration = mixture_fence(calib_scores, seed=2026)
    threshold = calibration['threshold']
    
    # Save the results
    out = Path('outputs/v8/fusion')
    out.mkdir(exist_ok=True, parents=True)
    
    (out / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    
    result = df[['filename', 'novelty_score', 'source_image_id', 'latitude', 'sun_angle', 'season']].copy()
    result.to_csv(out / 'fused_novelty_scores.csv', index=False)
    
    flagged = result[result['novelty_score'] > threshold].sort_values('novelty_score', ascending=False)
    
    print("\nPhase 2.4 Fusion and Ensemble independent calibration complete!")
    print(f"Fitted Threshold: {threshold:.5f}")
    print(f"Total crops exceeding threshold: {len(flagged)}")
    if len(flagged) > 0:
        print("\nTop 5 Final Candidates:")
        print(flagged.head(5).to_string(index=False))
    else:
        print("\nNo crops exceeded the independently calibrated threshold. (We will NOT lower it manually).")

if __name__ == '__main__':
    fusion()
