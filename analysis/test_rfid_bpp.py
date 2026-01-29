#!/usr/bin/env python3
"""
Test relationship between RFID and BPP.

What it does
------------
1) Loads all metrics_*.json under ../results/<ModelType>/<ModelDir>/.
2) Builds a table with BPP and RFID (+ family split for LlamaGen 256 vs 384).
3) Runs:
   - Global correlation tests (Pearson + Spearman) for BPP vs RFID
   - Global linear regression: RFID ~ BPP (OLS)
   - Cluster-wise correlation + regression per Family (if n>=3)
4) Saves:
   - rfid_bpp_table.csv
   - rfid_vs_bpp_scatter.png
   - rfid_bpp_report.txt

Run
---
python analyze_rfid_bpp.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf


# -----------------------------
# CONFIG
# -----------------------------
RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('analysis/results_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE = OUTPUT_DIR / "rfid_bpp_table.csv"
OUT_PNG = OUTPUT_DIR / "rfid_vs_bpp_scatter.png"
OUT_REPORT = OUTPUT_DIR / "rfid_bpp_report.txt"

plt.ioff()


def family_label(model_type: str, img_size: int) -> str:
    """Split LlamaGen into LlamaGen_256 / LlamaGen_384; keep others as-is."""
    if model_type == "LlamaGen":
        if int(img_size) == 256:
            return "LlamaGen_256"
        if int(img_size) == 384:
            return "LlamaGen_384"
        return f"LlamaGen_{int(img_size)}"
    return model_type


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []

    if not results_dir.exists():
        raise FileNotFoundError(f"RESULTS_DIR not found: {results_dir.resolve()}")

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
            model_type = str(data.get("model_type", model_type_dir.name))
            model_size = str(data.get("model_size", ""))
            img_size = int(data.get("img_size", np.nan)) if data.get("img_size", None) is not None else np.nan

            rows.append(
                {
                    "ModelType": model_type,
                    "Family": family_label(model_type, img_size),
                    "ModelSize": model_size,
                    "ImageSize": img_size,
                    "BPP": safe_float(m.get("model_bpp", np.nan)),
                    "RFID": safe_float(m.get("tokenizer_rfid", np.nan)),
                    "json_path": str(json_path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No metrics JSONs found under: {results_dir.resolve()}")

    # Keep only numeric valid rows
    df["BPP"] = pd.to_numeric(df["BPP"], errors="coerce")
    df["RFID"] = pd.to_numeric(df["RFID"], errors="coerce")
    df = df.dropna(subset=["BPP", "RFID"]).copy()

    return df


def corr_block(x, y, name_x="BPP", name_y="RFID"):
    """Return Pearson and Spearman results (r, p)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 3:
        return {
            "n": n,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_r": np.nan, "spearman_p": np.nan,
        }

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "n": int(n),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
    }


def write_corr(f, title, stats):
    f.write(title + "\n")
    f.write("-" * len(title) + "\n")
    f.write(f"n = {stats['n']}\n")
    if np.isfinite(stats["pearson_r"]):
        f.write(f"Pearson r = {stats['pearson_r']:.6f} (p = {stats['pearson_p']:.6g})\n")
        f.write(f"Spearman r = {stats['spearman_r']:.6f} (p = {stats['spearman_p']:.6g})\n")
    else:
        f.write("Not enough samples for correlation test (need n >= 3).\n")
    f.write("\n")


def main():
    df = load_metrics(RESULTS_DIR)

    # Save table
    df.sort_values(["Family", "BPP"]).to_csv(OUT_TABLE, index=False)

    # Global correlation tests
    global_corr = corr_block(df["BPP"].values, df["RFID"].values)

    # Global OLS: RFID ~ BPP (with centering)
    dm = df[["BPP", "RFID", "Family"]].copy()
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    ols_global = smf.ols("RFID ~ BPP_c", data=dm).fit()

    # Cluster-wise (Family) correlation + OLS if n>=3
    fam_results = []
    fam_ols_summaries = {}

    for fam, g in dm.groupby("Family"):
        st = corr_block(g["BPP"].values, g["RFID"].values)
        fam_results.append({"Family": fam, **st})

        if len(g) >= 3:
            g2 = g.copy()
            g2["BPP_c_f"] = g2["BPP"] - g2["BPP"].mean()
            ols_f = smf.ols("RFID ~ BPP_c_f", data=g2).fit()
            fam_ols_summaries[fam] = ols_f
        else:
            fam_ols_summaries[fam] = None

    fam_df = pd.DataFrame(fam_results).sort_values("pearson_p", na_position="last")

    # Plot: BPP vs RFID scatter, colored by family (no style changes beyond default colors)
    plt.figure(figsize=(8, 6))
    for fam, g in df.groupby("Family"):
        plt.scatter(g["BPP"], g["RFID"], label=fam, s=80)
        for _, r in g.iterrows():
            plt.text(float(r["BPP"]), float(r["RFID"]), str(r["ModelSize"]), fontsize=8)

    # Add global fit line (OLS slope on centered BPP is same slope)
    bpp_sorted = np.sort(df["BPP"].values.astype(float))
    slope = float(ols_global.params.get("BPP_c", np.nan))
    intercept = float(ols_global.params.get("Intercept", np.nan))
    # Since model uses BPP_c: RFID = intercept + slope*(BPP - meanBPP)
    mean_bpp = float(dm["BPP"].mean())
    y_line = intercept + slope * (bpp_sorted - mean_bpp)
    plt.plot(bpp_sorted, y_line, "--k", label=f"Global OLS (R²={ols_global.rsquared:.3f})")

    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("RFID (Tokenizer Reconstruction FID)")
    plt.title("RFID vs BPP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    # Write report
    with open(OUT_REPORT, "w") as f:
        f.write("RFID vs BPP Relationship Report\n")
        f.write("==============================\n\n")

        f.write("Data\n----\n")
        f.write(f"Loaded models: {len(df)}\n")
        f.write(f"Saved table: {OUT_TABLE.name}\n")
        f.write(f"Saved plot:  {OUT_PNG.name}\n\n")

        write_corr(f, "GLOBAL CORRELATION TESTS (BPP vs RFID)", global_corr)

        f.write("GLOBAL LINEAR MODEL (OLS)\n")
        f.write("-------------------------\n")
        f.write("Model: RFID ~ BPP_c  (BPP centered)\n\n")
        f.write(ols_global.summary().as_text() + "\n\n")

        f.write("PER-FAMILY TESTS\n")
        f.write("----------------\n")
        for fam in sorted(dm["Family"].unique()):
            g = dm[dm["Family"] == fam]
            st = fam_df[fam_df["Family"] == fam].iloc[0].to_dict()
            write_corr(f, f"Family: {fam}", st)

            if fam_ols_summaries[fam] is None:
                f.write("OLS skipped (need n >= 3 for a meaningful fit).\n\n")
            else:
                f.write("OLS: RFID ~ BPP_c_f  (BPP centered within family)\n")
                f.write(fam_ols_summaries[fam].summary().as_text() + "\n\n")

        f.write("Quick interpretation hints\n")
        f.write("--------------------------\n")
        f.write("- Pearson tests linear association; Spearman tests monotonic association.\n")
        f.write("- For n=2, correlation is not meaningful (always +/-1).\n")
        f.write("- Use the global OLS p-value on BPP_c to claim a significant global linear trend.\n")

    print(f"✔ Saved: {OUT_TABLE}")
    print(f"✔ Saved: {OUT_PNG}")
    print(f"✔ Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()

