import sys
import csv

import numpy as np


def compute_normalized_score_abiomed(csv_path, env_name):
    """
    Compute normalized scores from Abiomed evaluation results.

    Args:
        csv_path: Path to CSV file containing evaluation results
        env_name: Environment name (currently unused, kept for compatibility)

    Prints:
        Mean and standard deviation for return, ACP, and weaning scores
    """
    print("Script has started")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract metrics from CSV
    returns = [float(row['mean_return']) for row in rows]
    acps = [float(row['mean_acp']) for row in rows]
    ws1 = [float(row['mean_wean_score']) for row in rows]
    ws3 = [float(row['mean_wean_score_thr']) for row in rows]

    # Compute statistics
    mean_score = np.mean(returns)
    std_score = np.std(returns)
    mean_acp = np.mean(acps)
    std_acp = np.std(acps)
    mean_ws1 = np.mean(ws1)
    std_ws1 = np.std(ws1)
    mean_ws3 = np.mean(ws3)
    std_ws3 = np.std(ws3)

    # Print results
    print(f"Return: {mean_score:.3f} ± {std_score:.3f}")
    print(f"ACP: {mean_acp:.3f} ± {std_acp:.3f}")
    print(f"Wean Score Gradient: {mean_ws1:.3f} ± {std_ws1:.3f}")
    print(f"Wean Score Threshold: {mean_ws3:.3f} ± {std_ws3:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python normalizer.py <csv_path> <env_name>")
        sys.exit(1)
    else:
        csv_path = sys.argv[1]
        env_name = sys.argv[2]
        compute_normalized_score_abiomed(csv_path, env_name)
