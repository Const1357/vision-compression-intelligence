#!/usr/bin/env python3
"""
Per-cluster bilinear tests where clusters are formed by RFID similarity (bins).

This avoids the degeneracy you saw when clustering by model family,
because within family RFID is constant.

Outputs:
- rfid_clusters_bilinear_models.txt
- rfid_clusters_bilinear_summary.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('analysis/results_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 3  # change to 2/3/4 depending on how many distinct RFID values you have

OUT_TXT = OUTPUT_DIR / "rfid_clusters_bilinear_models.txt"
OUT_CSV = OUTPUT_DIR / "rfid_clusters_bilinear_summary.csv"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model_type_dir in results_dir.iterdir():
        if not model_type_dir.is_dir():
            continue
        for model_dir in model_type_dir.iterdir():
            if not model_dir.is_dir():
                continue
            json_path = model_dir / f"metrics_{model_dir.name}.json"
            if not json_path.exists():
                continue

            with open(json_path, "r") as f:
                data = json.load(f)
            m = data.get("metrics", {})

            rows.append({
                "ModelType": str(data.get("model_type", model_type_dir.name)),
                "ModelSize": str(data.get("model_size", "")),
                "ImageSize": int(data.get("img_size", np.nan)) if data.get("img_size", None) is not None else np.nan,
                "FID": safe_float(m.get("model_fid", np.nan)),
                "BPP": safe_float(m.get("model_bpp", np.nan)),
                "RFID": safe_float(m.get("tokenizer_rfid", np.nan)),
                "json_path": str(json_path),
            })

    df = pd.DataFrame(rows)
    for c in ["FID", "BPP", "RFID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FID", "BPP", "RFID"]).copy()
    if df.empty:
        raise RuntimeError(f"No valid rows found under {results_dir.resolve()}")
    return df


def main():
    df = load_metrics(RESULTS_DIR)

    # If there are very few unique RFID values, binning may collapse.
    # Use rank-based qcut but fall back to unique bins if needed.
    try:
        df["RFID_cluster"] = pd.qcut(df["RFID"], q=N_BINS, duplicates="drop")
    except ValueError:
        # fallback: each unique RFID is its own cluster
        df["RFID_cluster"] = df["RFID"].astype(str)

    summary_rows = []

    with open(OUT_TXT, "w") as f:
        f.write("Per-RFID-Cluster Bilinear Model Tests\n")
        f.write("====================================\n\n")
        f.write("Models:\n")
        f.write("  BPP-only:   FID ~ BPP_c\n")
        f.write("  Additive:   FID ~ BPP_c + RFID_c\n")
        f.write("  Bilinear:   FID ~ BPP_c * RFID_c\n\n")

        for cl, g in df.groupby("RFID_cluster"):
            g = g.copy()
            n = len(g)
            f.write(f"\nCLUSTER: {cl}\n")
            f.write("-" * (9 + len(str(cl))) + "\n")
            f.write(f"n = {n}\n")

            # show points
            for _, r in g.sort_values("BPP").iterrows():
                f.write(f"  - {r['ModelType']} {r['ModelSize']} (img={r['ImageSize']}): "
                        f"BPP={r['BPP']:.6g}, RFID={r['RFID']:.6g}, FID={r['FID']:.6g}\n")
            f.write("\n")

            # Center within cluster
            g["BPP_c"] = g["BPP"] - g["BPP"].mean()
            g["RFID_c"] = g["RFID"] - g["RFID"].mean()

            # Check if RFID varies in this cluster (required for additive/bilinear)
            rfid_var = float(np.var(g["RFID"].values))
            if rfid_var == 0.0:
                f.write("RFID has zero variance in this cluster -> cannot fit additive/bilinear.\n")
                # still fit BPP-only if possible
                if n >= 3:
                    m1 = smf.ols("FID ~ BPP_c", data=g).fit()
                    f.write("\nModel: BPP-only\n")
                    f.write(m1.summary().as_text() + "\n")
                    summary_rows.append({
                        "RFID_cluster": str(cl), "Model": "BPP_only", "n": n,
                        "r2": m1.rsquared, "aic": m1.aic,
                        "p_BPP": m1.pvalues.get("BPP_c", np.nan),
                        "p_RFID": np.nan, "p_interaction": np.nan,
                        "anova_add_vs_bpp_p": np.nan, "anova_bilin_vs_add_p": np.nan
                    })
                else:
                    summary_rows.append({
                        "RFID_cluster": str(cl), "Model": "SKIPPED", "n": n,
                        "r2": np.nan, "aic": np.nan,
                        "p_BPP": np.nan, "p_RFID": np.nan, "p_interaction": np.nan,
                        "anova_add_vs_bpp_p": np.nan, "anova_bilin_vs_add_p": np.nan
                    })
                continue

            # Need enough points for each model
            m1 = smf.ols("FID ~ BPP_c", data=g).fit() if n >= 3 else None
            m2 = smf.ols("FID ~ BPP_c + RFID_c", data=g).fit() if n >= 4 else None
            m3 = smf.ols("FID ~ BPP_c * RFID_c", data=g).fit() if n >= 5 else None

            if m1 is not None:
                f.write("Model: BPP-only\n")
                f.write(m1.summary().as_text() + "\n\n")
            if m2 is not None:
                f.write("Model: Additive\n")
                f.write(m2.summary().as_text() + "\n\n")
            if m3 is not None:
                f.write("Model: Bilinear\n")
                f.write(m3.summary().as_text() + "\n\n")

            # ANOVAs
            p_add_vs_bpp = np.nan
            p_bilin_vs_add = np.nan

            if m1 is not None and m2 is not None:
                a12 = sm.stats.anova_lm(m1, m2)
                f.write("ANOVA: Additive vs BPP-only\n")
                f.write(a12.to_string() + "\n\n")
                p_add_vs_bpp = float(a12["Pr(>F)"].iloc[-1])

            if m2 is not None and m3 is not None:
                a23 = sm.stats.anova_lm(m2, m3)
                f.write("ANOVA: Bilinear vs Additive\n")
                f.write(a23.to_string() + "\n\n")
                p_bilin_vs_add = float(a23["Pr(>F)"].iloc[-1])

            # Summaries (pick the most complex model that exists for p-values)
            def pick(fit, name):
                if fit is None:
                    return {
                        "Model": name, "r2": np.nan, "aic": np.nan,
                        "p_BPP": np.nan, "p_RFID": np.nan, "p_interaction": np.nan
                    }
                return {
                    "Model": name,
                    "r2": float(fit.rsquared),
                    "aic": float(fit.aic),
                    "p_BPP": float(fit.pvalues.get("BPP_c", np.nan)),
                    "p_RFID": float(fit.pvalues.get("RFID_c", np.nan)),
                    "p_interaction": float(fit.pvalues.get("BPP_c:RFID_c", np.nan)),
                }

            # store one row per fitted model
            for name, fit in [("BPP_only", m1), ("Additive", m2), ("Bilinear", m3)]:
                row = {"RFID_cluster": str(cl), "n": n,
                       "anova_add_vs_bpp_p": p_add_vs_bpp,
                       "anova_bilin_vs_add_p": p_bilin_vs_add}
                row.update(pick(fit, name))
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_CSV, index=False)

    print(f"✔ Wrote: {OUT_TXT}")
    print(f"✔ Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()

