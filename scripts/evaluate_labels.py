import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report
import sys
from pathlib import Path

def evaluate_labels(labels_csv, scores_csv):
    labels_path = Path(labels_csv)
    scores_path = Path(scores_csv)

    if not labels_path.exists():
        print(f"Error: Labels file {labels_csv} not found.")
        sys.exit(1)
        
    if not scores_path.exists():
        print(f"Error: Scores file {scores_csv} not found.")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    scores_df = pd.read_csv(scores_path)

    if 'filename' not in labels_df.columns or 'label' not in labels_df.columns:
        print("Error: Labels CSV must contain 'filename' and 'label' columns.")
        sys.exit(1)

    # Merge
    merged_df = scores_df.merge(labels_df, on='filename', how='inner')
    
    if len(merged_df) == 0:
        print("Error: No overlapping filenames found between labels and scores.")
        sys.exit(1)
        
    print(f"Matched {len(merged_df)} items with ground truth labels.")

    y_true = merged_df['label']
    y_scores = merged_df['novelty_score']
    
    # Optional threshold if it's in the data, else we just use the raw score for AUC metrics
    # If the user has 'flagged' boolean, we can compute hard metrics
    
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    
    print("\n--- Model Performance Metrics ---")
    print(f"ROC-AUC: {auc_roc:.4f}")
    print(f"PR-AUC:  {auc_pr:.4f}")
    
    if 'flagged' in merged_df.columns:
        y_pred = merged_df['flagged'].astype(int)
        f1 = f1_score(y_true, y_pred)
        print(f"F1 Score (using model threshold): {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate novelty scores against ground truth labels.")
    parser.add_argument('--labels', required=True, help='Path to the ground truth labels CSV (must have filename and label columns).')
    parser.add_argument('--scores', default='outputs/v7/mixture_three_sigma_trees2000/novelty_scores.csv', help='Path to the novelty_scores.csv output from the model.')
    args = parser.parse_args()
    
    evaluate_labels(args.labels, args.scores)
