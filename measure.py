import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def measure_detect(clean_path, dirty_path, det_wrong_list, output_path):
    """
    Evaluate detection results by comparing predicted wrong cells with actual differences
    between clean and dirty data.

    Parameters
    ----------
    clean_path : str
        Path to clean CSV file.
    dirty_path : str
        Path to dirty CSV file.
    det_wrong_list : list
        List of detected wrong cells, typically [(row_idx, col_name), ...].
    output_path : str
        Path to save evaluation results.
    """

    # Load clean and dirty datasets
    clean_df = pd.read_csv(clean_path, dtype=str).fillna("nan")
    dirty_df = pd.read_csv(dirty_path, dtype=str).fillna("nan")

    # Get true wrong cells (ground truth)
    true_wrong = set()
    for i in range(len(clean_df)):
        for col in clean_df.columns:
            if clean_df.at[i, col] != dirty_df.at[i, col]:
                true_wrong.add((i, col))

    # Convert predictions to set
    pred_wrong = set(det_wrong_list)

    # Compute evaluation metrics
    TP = len(true_wrong & pred_wrong)
    FP = len(pred_wrong - true_wrong)
    FN = len(true_wrong - pred_wrong)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"True Positives (TP): {TP}\n")
        f.write(f"False Positives (FP): {FP}\n")
        f.write(f"False Negatives (FN): {FN}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"[measure_detect] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
