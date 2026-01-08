import matplotlib
matplotlib.use("Agg")  # OBLIGATOIRE pour Docker / Azure

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def detect_drift(reference_file, production_file, threshold=0.05, output_dir="drift_reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ✅ AUTO-GENERATE DATA IF MISSING
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"{reference_file} not found")

    if not os.path.exists(production_file):
        from app.drift_data_gen import generate_drifted_data
        generate_drifted_data(reference_file=reference_file, output_file=production_file)

    ref = pd.read_csv(reference_file)
    prod = pd.read_csv(production_file)

    results = {}

    for col in ref.columns:
        if col != "Exited" and col in prod.columns:
            stat, p = ks_2samp(ref[col].dropna(), prod[col].dropna())
            results[col] = {
                "p_value": float(p),
                "statistic": float(stat),
                "drift_detected": bool(p < threshold)
            }

    return results
