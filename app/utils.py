import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(reference_file: str, production_file: str, threshold: float = 0.05):
    reference = pd.read_csv(reference_file)
    production = pd.read_csv(production_file)

    results = {}

    common_cols = list(set(reference.columns) & set(production.columns))

    for col in common_cols:
        stat, p_value = ks_2samp(reference[col], production[col])
        results[col] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": p_value < threshold
        }

    return results
