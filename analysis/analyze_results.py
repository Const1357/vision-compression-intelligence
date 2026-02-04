import json
import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.formula.api as smf
import statsmodels.api as sm

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# ==================================================
# CONFIG
# ==================================================
RESULTS_DIR = "results"
OUTPUT_DIR = "analysis/results_analysis"

REPORT_TXT = os.path.join(OUTPUT_DIR, "report.txt")
SLOPE_REPORT_TXT = os.path.join(OUTPUT_DIR, "slope_tests.txt")
PAIRWISE_SLOPE_CSV = os.path.join(OUTPUT_DIR, "pairwise_slope_tests.csv")

FID_BPP_RFID_TXT = os.path.join(OUTPUT_DIR, "fid_bpp_rfid_models.txt")
FID_BPP_RFID_CSV = os.path.join(OUTPUT_DIR, "fid_bpp_rfid_models_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.ioff()  # never show plots

# ==================================================
# HELPERS
# ==================================================
def family_label(model_type: str, img_size: int) -> str:
    """Split LlamaGen into two families by image size; keep others as-is."""
    if model_type == "LlamaGen":
        if int(img_size) == 256:
            return "LlamaGen_256"
        if int(img_size) == 384:
            return "LlamaGen_384"
        return f"LlamaGen_{img_size}"
    return model_type


def pearson_r(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def fit_linear_bpp_to_fid(x_bpp, y_fid):
    """Fit FID = a*BPP + b and return basic stats."""
    x = np.asarray(x_bpp, dtype=float)
    y = np.asarray(y_fid, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 2:
        return {"n": n, "slope": np.nan, "intercept": np.nan, "r2": np.nan, "pearson_r": np.nan}

    X = x.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    return {
        "n": int(n),
        "slope": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "r2": float(r2_score(y, y_pred)),
        "pearson_r": pearson_r(x, y),
    }


def write_block(f, title, stats, equation_prefix="Linear fit:"):
    f.write(title + "\n")
    f.write("-" * len(title) + "\n")
    f.write(f"n = {stats.get('n', np.nan)}\n")
    if "slope" in stats and "intercept" in stats:
        f.write(f"{equation_prefix} FID = {stats['slope']:.6g} * BPP + {stats['intercept']:.6g}\n")
    f.write(f"R^2 = {stats.get('r2', np.nan):.6f}\n")
    f.write(f"Pearson r = {stats.get('pearson_r', np.nan):.6f}\n\n")


# ==================================================
# 1) LOAD METRICS JSONs
# ==================================================
def load_results(results_dir: str) -> pd.DataFrame:
    rows = []

    for model_type_dir in os.listdir(results_dir):
        type_dir = os.path.join(results_dir, model_type_dir)
        if not os.path.isdir(type_dir):
            continue

        for model_dir in os.listdir(type_dir):
            json_path = os.path.join(type_dir, model_dir, f"metrics_{model_dir}.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            m = data["metrics"]
            bpp_hist = m.get("bpp_history", [])
            bpp_n = len(bpp_hist)

            model_type = str(data["model_type"])
            img_size = int(data["img_size"])

            rows.append({
                "ModelType": model_type,
                "Family": family_label(model_type, img_size),
                "ModelSize": str(data["model_size"]),
                "ImageSize": img_size,
                "FID": float(m["model_fid"]),
                "RFID": float(m["tokenizer_rfid"]),
                "BPP": float(m["model_bpp"]),
                "BPP_std": float(np.std(bpp_hist, ddof=1)) if bpp_n >= 2 else np.nan,
                "BPP_n": int(bpp_n),
            })


    return pd.DataFrame(rows)


df = load_results(RESULTS_DIR)

print("\n=== Summary Table (sorted by BPP) ===")
print(df.sort_values("BPP").to_string(index=False))


# ==================================================
# 2) PLOTS: Global BPP vs FID + per-family + RFID 1D
# ==================================================
def plot_global_bpp_fid(df: pd.DataFrame, label_fontsize=8):
    plt.figure(figsize=(8, 6))

    colors = {
        "VAR": "tab:blue",
        "RQ-Transformer": "tab:orange",
        "LlamaGen_256": "tab:green",
        "LlamaGen_384": "tab:olive",
        "VQ-GAN": "tab:red",
    }

    # Compute global stats for the report (not necessarily plotted)
    global_stats = fit_linear_bpp_to_fid(df["BPP"].values, df["FID"].values)

    # Plot per-family points + per-family fit lines
    for fam, g in df.groupby("Family"):
        color = colors.get(fam, None)

        # Scatter
        plt.scatter(
            g["BPP"],
            g["FID"],
            label=fam,
            s=80,
            color=color,
            zorder=2,  # points under line? set line higher below
        )

        # Annotate
        for _, r in g.iterrows():
            plt.text(
                r["BPP"],
                r["FID"],
                r["ModelSize"],
                fontsize=label_fontsize,
                zorder=3
            )

        # Family-specific line (only if >=2 points)
        if len(g) >= 2:
            fam_stats = fit_linear_bpp_to_fid(g["BPP"].values, g["FID"].values)
            x_sorted = np.sort(g["BPP"].values.astype(float))
            y_fit = fam_stats["slope"] * x_sorted + fam_stats["intercept"]

            # Put line on top of points
            plt.plot(
                x_sorted,
                y_fit,
                linestyle="--",
                color=color,
                linewidth=2.0,
                alpha=0.95,
                zorder=4
            )

    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("FID")
    plt.title("Compression–Quality Relationship Across Model Families")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bpp_vs_fid_global.png"), dpi=200)
    plt.close()

    return global_stats

def plot_per_family(df: pd.DataFrame):
    fit_rows = []

    for fam, g in df.groupby("Family"):
        plt.figure(figsize=(6, 5))

        plt.errorbar(g["BPP"], g["FID"], xerr=g["BPP_std"], fmt="o", capsize=4)

        # annotate
        for _, r in g.iterrows():
            plt.text(float(r["BPP"]), float(r["FID"]), str(r["ModelSize"]), fontsize=8)

        stats = fit_linear_bpp_to_fid(g["BPP"].values, g["FID"].values)
        fit_rows.append({"scope": fam, **stats})

        X = g["BPP"].values.astype(float)
        plt.plot(X, stats["slope"] * X + stats["intercept"], "--")

        plt.title(f"{fam}: BPP vs FID")
        plt.xlabel("BPP")
        plt.ylabel("FID")
        plt.grid(True)

        plt.text(
            0.05, 0.95, f"R² = {stats['r2']:.3f}",
            transform=plt.gca().transAxes, va="top"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{fam}_bpp_vs_fid.png"), dpi=200)
        plt.close()

    return pd.DataFrame(fit_rows)


def plot_rfid_1d(df: pd.DataFrame):
    plt.figure(figsize=(10, 2.5))

    df_sorted = df.sort_values("RFID")
    y = np.zeros(len(df_sorted))

    plt.scatter(df_sorted["RFID"], y, s=80)

    for _, r in df_sorted.iterrows():
        label = f"{r['Family']} | {r['ModelSize']}"
        plt.text(
            r["RFID"], 0.02, label,
            rotation=45, ha="right", va="bottom", fontsize=8
        )

    plt.yticks([])
    plt.xlabel("RFID (Tokenizer Reconstruction FID)")
    plt.title("RFID Distribution Across Models")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rfid_1d_distribution.png"), dpi=200)
    plt.close()

def var_subset_correlation_and_plots(df: pd.DataFrame,
                                     out_prefix: str = "var_bpp_fid",
                                     label_fontsize: int = 9):
    """
    Compute Pearson(BPP, FID) inside VAR for:
      - subset3: {310M, 600M, 1B}
      - subset4: subset3 + {2B}  (outlier)
    Save 2 plots + a small txt report.
    """
    g = df[df["Family"] == "VAR"].copy()
    if g.empty:
        return None

    order3 = ["310M", "600M", "1B"]
    order4 = ["310M", "600M", "1B", "2B"]

    def pick(subset):
        return g[g["ModelSize"].isin(subset)].copy()

    def stats_block(subdf):
        return {
            "n": len(subdf),
            "pearson_r": pearson_r(subdf["BPP"].values, subdf["FID"].values),
            **fit_linear_bpp_to_fid(subdf["BPP"].values, subdf["FID"].values),
        }

    s3 = pick(order3)
    s4 = pick(order4)

    st3 = stats_block(s3) if len(s3) >= 2 else None
    st4 = stats_block(s4) if len(s4) >= 2 else None

    # -------- plots --------
    def make_plot(subdf, title, out_png, highlight_2b=False):
        plt.figure(figsize=(6.5, 5))
        plt.scatter(subdf["BPP"], subdf["FID"], s=110, zorder=3)

        for _, r in subdf.iterrows():
            plt.text(float(r["BPP"]), float(r["FID"]), str(r["ModelSize"]),
                     fontsize=label_fontsize, zorder=4)

        # fit line if >=2 points
        if len(subdf) >= 2:
            st = fit_linear_bpp_to_fid(subdf["BPP"].values, subdf["FID"].values)
            x = np.sort(subdf["BPP"].values.astype(float))
            plt.plot(x, st["slope"] * x + st["intercept"], "--", linewidth=2.0, zorder=2)
            plt.text(0.05, 0.95,
                     f"r = {st['pearson_r']:.3f}\nR² = {st['r2']:.3f}",
                     transform=plt.gca().transAxes, va="top")

        # optionally emphasize 2B
        if highlight_2b and ("2B" in subdf["ModelSize"].values):
            r2b = subdf[subdf["ModelSize"] == "2B"].iloc[0]
            plt.scatter([r2b["BPP"]], [r2b["FID"]], s=220, marker="x", zorder=5)

        plt.xlabel("BPP")
        plt.ylabel("FID")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, out_png), dpi=200)
        plt.close()

    make_plot(s3, "VAR: BPP vs FID (310M, 600M, 1B)", f"{out_prefix}_3pt.png")
    make_plot(s4, "VAR: BPP vs FID (+ 2B outlier)", f"{out_prefix}_4pt.png", highlight_2b=True)

    # -------- text report --------
    out_txt = os.path.join(OUTPUT_DIR, f"{out_prefix}_stats.txt")
    with open(out_txt, "w") as f:
        f.write("VAR subset correlation (Pearson) + linear fit\n")
        f.write("===========================================\n\n")
        if st3 is not None:
            f.write("Subset 3pt: {310M, 600M, 1B}\n")
            f.write(f"n = {st3['n']}\n")
            f.write(f"Pearson(BPP,FID) r = {st3['pearson_r']:.6f}\n")
            f.write(f"Fit: FID = {st3['slope']:.6g} * BPP + {st3['intercept']:.6g}\n")
            f.write(f"R^2 = {st3['r2']:.6f}\n\n")
        if st4 is not None:
            f.write("Subset 4pt: {310M, 600M, 1B, 2B}\n")
            f.write(f"n = {st4['n']}\n")
            f.write(f"Pearson(BPP,FID) r = {st4['pearson_r']:.6f}\n")
            f.write(f"Fit: FID = {st4['slope']:.6g} * BPP + {st4['intercept']:.6g}\n")
            f.write(f"R^2 = {st4['r2']:.6f}\n\n")

        f.write("Saved plots:\n")
        f.write(f"- {out_prefix}_3pt.png\n")
        f.write(f"- {out_prefix}_4pt.png\n")

    return {"subset3": st3, "subset4": st4, "txt": out_txt}


global_stats = plot_global_bpp_fid(df)
per_family_df = plot_per_family(df)
plot_rfid_1d(df)
var_subset_correlation_and_plots(df)


# ==================================================
# 3) ANCOVA: Are cluster slopes statistically similar?
# ==================================================
def slope_vs_rfid_across_families(df: pd.DataFrame,
                                 min_n: int = 3,
                                 out_txt: str = "family_slope_vs_rfid.txt",
                                 out_png: str = "family_slope_vs_rfid.png"):
    """
    For each Family with n >= min_n:
      - fit slope from FID ~ BPP (same as your per-family fit)
      - compute mean RFID (also report std/min/max)
    Then compute Pearson correlation across families: slope vs mean RFID.
    """
    rows = []
    for fam, g in df.groupby("Family"):
        if len(g) < min_n:
            continue
        st = fit_linear_bpp_to_fid(g["BPP"].values, g["FID"].values)
        rows.append({
            "Family": fam,
            "n": int(len(g)),
            "slope": float(st["slope"]),
            "rfid_mean": float(np.mean(g["RFID"].values.astype(float))),
            "rfid_std": float(np.std(g["RFID"].values.astype(float), ddof=1)) if len(g) >= 2 else np.nan,
            "rfid_min": float(np.min(g["RFID"].values.astype(float))),
            "rfid_max": float(np.max(g["RFID"].values.astype(float))),
        })

    fam_df = pd.DataFrame(rows)
    out_txt_path = os.path.join(OUTPUT_DIR, out_txt)

    if len(fam_df) < 2:
        with open(out_txt_path, "w") as f:
            f.write("Not enough families with n>=min_n to correlate slope vs RFID.\n")
        return fam_df, out_txt_path

    r = pearson_r(fam_df["slope"].values, fam_df["rfid_mean"].values)

    # plot
    plt.figure(figsize=(6.5, 5))
    plt.scatter(fam_df["rfid_mean"], fam_df["slope"], s=120)
    for _, row in fam_df.iterrows():
        plt.text(float(row["rfid_mean"]), float(row["slope"]), str(row["Family"]), fontsize=9)
    plt.xlabel("Family mean rFID")
    plt.ylabel("Family slope (FID ~ BPP)")
    plt.title(f"Slope vs rFID across families (n≥{min_n}), r = {r:.3f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png), dpi=200)
    plt.close()

    with open(out_txt_path, "w") as f:
        f.write("Slope (FID~BPP) vs rFID across families\n")
        f.write("======================================\n\n")
        f.write(f"Included families: n >= {min_n}\n")
        f.write(f"Pearson(slope, mean_rFID) = {r:.6f}\n\n")
        f.write("Per-family values:\n")
        f.write(fam_df.sort_values("Family").to_string(index=False))
        f.write("\n\nSaved plot:\n")
        f.write(f"- {out_png}\n")

    return fam_df, out_txt_path


def slope_similarity_tests(df: pd.DataFrame, group_col: str = "Family"):
    df_s = df[[group_col, "BPP", "FID"]].dropna().copy()
    df_s[group_col] = df_s[group_col].astype(str)
    df_s["BPP"] = df_s["BPP"].astype(float)
    df_s["FID"] = df_s["FID"].astype(float)

    # center BPP
    df_s["BPP_c"] = df_s["BPP"] - df_s["BPP"].mean()

    # full: interaction (different slopes)
    model_full = smf.ols(f"FID ~ BPP_c * C({group_col})", data=df_s).fit()
    # reduced: same slope
    model_reduced = smf.ols(f"FID ~ BPP_c + C({group_col})", data=df_s).fit()

    anova_res = sm.stats.anova_lm(model_reduced, model_full)

    groups = sorted(df_s[group_col].unique())
    baseline = groups[0] if groups else None

    # compute slopes per group under treatment coding
    base_slope = model_full.params.get("BPP_c", np.nan)
    slopes = {}
    for g in groups:
        s = base_slope
        if g != baseline:
            term = f"BPP_c:C({group_col})[T.{g}]"
            s += model_full.params.get(term, 0.0)
        slopes[g] = float(s)

    # pairwise Wald tests: slope(g1) - slope(g2) = 0
    def slope_terms(g):
        terms = [("BPP_c", 1.0)]
        if g != baseline:
            terms.append((f"BPP_c:C({group_col})[T.{g}]", 1.0))
        return terms

    def wald_slope_diff(g1, g2):
        params = list(model_full.params.index)
        R = np.zeros((1, len(params)), dtype=float)

        for name, c in slope_terms(g1):
            if name in params:
                R[0, params.index(name)] += c
        for name, c in slope_terms(g2):
            if name in params:
                R[0, params.index(name)] -= c

        return model_full.wald_test(R)

    pairwise_rows = []
    for g1, g2 in itertools.combinations(groups, 2):
        test = wald_slope_diff(g1, g2)
        stat = float(np.atleast_1d(test.statistic)[0])
        pval = float(np.atleast_1d(test.pvalue)[0])
        pairwise_rows.append({
            "group1": g1,
            "group2": g2,
            "slope1": slopes[g1],
            "slope2": slopes[g2],
            "diff": slopes[g1] - slopes[g2],
            "wald_chi2": stat,
            "pvalue": pval
        })

    pairwise_df = pd.DataFrame(pairwise_rows).sort_values("pvalue")
    pairwise_df.to_csv(PAIRWISE_SLOPE_CSV, index=False)

    with open(SLOPE_REPORT_TXT, "w") as f:
        f.write("Slope Similarity Tests Across Clusters (ANCOVA / Interaction OLS)\n")
        f.write("===============================================================\n\n")
        f.write(f"Grouping column: {group_col}\n")
        f.write(f"Total samples: {len(df_s)}\n")
        f.write(f"Clusters: {groups}\n")
        f.write(f"Baseline cluster (treatment coding): {baseline}\n\n")
        f.write("Model (full): FID ~ BPP_c * C(cluster)\n")
        f.write("Model (reduced): FID ~ BPP_c + C(cluster)\n\n")
        f.write("Global test: Do slopes differ across clusters?\n")
        f.write("------------------------------------------------\n")
        f.write(anova_res.to_string() + "\n\n")
        f.write("Estimated slopes by cluster (from full model)\n")
        f.write("--------------------------------------------\n")
        for g in groups:
            f.write(f"{g}: slope = {slopes[g]:.6g}\n")
        f.write("\n")
        f.write("Pairwise slope equality tests (Wald)\n")
        f.write("-----------------------------------\n")
        f.write(pairwise_df.to_string(index=False) + "\n\n")
        f.write("Files saved\n")
        f.write("-----------\n")
        f.write("- slope_tests.txt\n")
        f.write("- pairwise_slope_tests.csv\n")

    return anova_res, pairwise_df


# anova_res, pairwise_df = slope_similarity_tests(df, group_col="Family")


# ==================================================
# 4) MODEL FID AS FUNCTION OF BPP AND RFID
#    - BPP only
#    - Additive: BPP + RFID
#    - Interaction: BPP * RFID
#    Report R^2 + Pearson(model_pred, y)
# ==================================================
def fit_fid_bpp_rfid_models(df: pd.DataFrame):
    dm = df[["BPP", "RFID", "FID"]].dropna().copy()
    dm["BPP"] = dm["BPP"].astype(float)
    dm["RFID"] = dm["RFID"].astype(float)
    dm["FID"] = dm["FID"].astype(float)

    # Center for stability
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    dm["RFID_c"] = dm["RFID"] - dm["RFID"].mean()

    # Models
    m_bpp = smf.ols("FID ~ BPP_c", data=dm).fit()
    m_add = smf.ols("FID ~ BPP_c + RFID_c", data=dm).fit()
    m_int = smf.ols("FID ~ BPP_c * RFID_c", data=dm).fit()

    # ANOVA comparisons (nested tests)
    anova_add_vs_bpp = sm.stats.anova_lm(m_bpp, m_add)
    anova_int_vs_add = sm.stats.anova_lm(m_add, m_int)

    # Core correlations
    corr_bpp_fid = pearson_r(dm["BPP"].values, dm["FID"].values)
    corr_rfid_fid = pearson_r(dm["RFID"].values, dm["FID"].values)

    def pred_pearson(model):
        yhat = model.predict(dm)
        return pearson_r(dm["FID"].values, yhat)

    rows = [
        {
            "model": "BPP_only",
            "n": len(dm),
            "r2": float(m_bpp.rsquared),
            "pearson_y_yhat": pred_pearson(m_bpp),
            "aic": float(m_bpp.aic),
        },
        {
            "model": "Additive_BPP+RFID",
            "n": len(dm),
            "r2": float(m_add.rsquared),
            "pearson_y_yhat": pred_pearson(m_add),
            "aic": float(m_add.aic),
        },
        {
            "model": "Interaction_BPP*RFID",
            "n": len(dm),
            "r2": float(m_int.rsquared),
            "pearson_y_yhat": pred_pearson(m_int),
            "aic": float(m_int.aic),
        },
    ]
    summary_df = pd.DataFrame(rows)

    # Decide “recommended” model:
    # - Only consider interaction if it improves additive significantly (p<0.05)
    p_int = float(anova_int_vs_add["Pr(>F)"].iloc[1])
    if p_int < 0.05:
        recommended = "Interaction_BPP*RFID"
    else:
        recommended = "Additive_BPP+RFID"

    summary_df["recommended"] = (summary_df["model"] == recommended)

    summary_df.to_csv(FID_BPP_RFID_CSV, index=False)

    with open(FID_BPP_RFID_TXT, "w") as f:
        f.write("FID as function of BPP and RFID (OLS)\n")
        f.write("=====================================\n\n")
        f.write(f"n = {len(dm)}\n\n")

        f.write("Raw Pearson correlations\n")
        f.write("------------------------\n")
        f.write(f"Pearson(BPP, FID)  = {corr_bpp_fid:.6f}\n")
        f.write(f"Pearson(RFID, FID) = {corr_rfid_fid:.6f}\n\n")

        f.write("Model 1: FID ~ BPP_c\n")
        f.write(m_bpp.summary().as_text() + "\n")
        f.write(f"Pearson(y, yhat) = {pred_pearson(m_bpp):.6f}\n")
        f.write(f"R^2 = {m_bpp.rsquared:.6f}\n\n")

        f.write("Model 2: FID ~ BPP_c + RFID_c\n")
        f.write(m_add.summary().as_text() + "\n")
        f.write(f"Pearson(y, yhat) = {pred_pearson(m_add):.6f}\n")
        f.write(f"R^2 = {m_add.rsquared:.6f}\n\n")

        f.write("Model 3: FID ~ BPP_c * RFID_c\n")
        f.write(m_int.summary().as_text() + "\n")
        f.write(f"Pearson(y, yhat) = {pred_pearson(m_int):.6f}\n")
        f.write(f"R^2 = {m_int.rsquared:.6f}\n\n")

        f.write("Nested-model tests (ANOVA)\n")
        f.write("--------------------------\n")
        f.write("Additive vs BPP-only (does RFID add explanatory power?)\n")
        f.write(anova_add_vs_bpp.to_string() + "\n\n")
        f.write("Interaction vs Additive (does BPP×RFID matter?)\n")
        f.write(anova_int_vs_add.to_string() + "\n\n")

        f.write("Compact summary\n")
        f.write("---------------\n")
        f.write(summary_df.to_string(index=False) + "\n\n")

        f.write(f"Recommended model: {recommended}\n")

    return summary_df

def per_model_prediction_table(df: pd.DataFrame,
                               out_txt: str = "per_model_predictions_ascii.txt",
                               out_csv: str = "per_model_predictions.csv"):
    """
    One row per checkpoint (global).
    Columns: ModelType, ModelSize, FID, RFID, BPP,
             yhat_BPP, yhat_Add, yhat_Int
    Also includes residuals.
    """
    dm = df[["ModelType", "Family", "ModelSize", "BPP", "RFID", "FID", "BPP_std", "BPP_n"]].dropna().copy()
    dm["BPP"] = dm["BPP"].astype(float)
    dm["RFID"] = dm["RFID"].astype(float)
    dm["FID"] = dm["FID"].astype(float)

    # Centering consistent with your regression section
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    dm["RFID_c"] = dm["RFID"] - dm["RFID"].mean()

    m_bpp = smf.ols("FID ~ BPP_c", data=dm).fit()
    m_add = smf.ols("FID ~ BPP_c + RFID_c", data=dm).fit()
    m_int = smf.ols("FID ~ BPP_c * RFID_c", data=dm).fit()

    dm["yhat_BPP"] = m_bpp.predict(dm)
    dm["yhat_Add"] = m_add.predict(dm)
    dm["yhat_Int"] = m_int.predict(dm)

    dm["res_BPP"] = dm["FID"] - dm["yhat_BPP"]
    dm["res_Add"] = dm["FID"] - dm["yhat_Add"]
    dm["res_Int"] = dm["FID"] - dm["yhat_Int"]

    # 95% CI for BPP from history (if available)
    z = 1.96
    dm["BPP_se"] = np.where(dm["BPP_n"].fillna(0).astype(int) >= 2,
                            dm["BPP_std"] / np.sqrt(dm["BPP_n"].astype(float)),
                            np.nan)
    dm["BPP_ci_lo"] = dm["BPP"] - z * dm["BPP_se"]
    dm["BPP_ci_hi"] = dm["BPP"] + z * dm["BPP_se"]

    # Save CSV (full precision)
    out_csv_path = os.path.join(OUTPUT_DIR, out_csv)
    dm.to_csv(out_csv_path, index=False)

    # ASCII table (pretty printed)
    show = dm[[
        "ModelType", "ModelSize", "FID", "RFID", "BPP",
        "BPP_ci_lo", "BPP_ci_hi",
        "yhat_BPP", "yhat_Add", "yhat_Int"
    ]].copy()

    show["BPP"] = show.apply(
        lambda r: (
            f"{r['BPP']:.6f} ± {(r['BPP_ci_hi'] - r['BPP']):.6f}"
            if np.isfinite(r["BPP_ci_hi"])
            else f"{r['BPP']:.6f}"
        ),
        axis=1
    )

    fmt = {
        "FID": "{:.3f}".format,
        "RFID": "{:.3f}".format,
        "yhat_BPP": "{:.3f}".format,
        "yhat_Add": "{:.3f}".format,
        "yhat_Int": "{:.3f}".format,
    }

    for c, f in fmt.items():
        show[c] = show[c].map(f)

    # (Optional) order like your slide: by ModelType then some size ordering
    out_txt_path = os.path.join(OUTPUT_DIR, out_txt)
    with open(out_txt_path, "w") as f:
        f.write("Per-checkpoint table (global)\n")
        f.write("============================\n")
        f.write("BPP CI is 95% CI from bpp_history (if present).\n\n")
        f.write(show.to_string(index=False))
        f.write("\n")

    return dm, out_txt_path, out_csv_path

fid_bpp_rfid_summary = fit_fid_bpp_rfid_models(df)
pred_dm, pred_txt, pred_csv = per_model_prediction_table(df)

# ==================================================
# 4B) VISUALIZE BILINEAR MODEL: FID ~ BPP + RFID (+ interaction)
# ==================================================
def visualize_bilinear_effects(df: pd.DataFrame,
                               out_surface_png="fid_bpp_rfid_surface.png",
                               out_resid_png="fid_residuals_vs_rfid.png",
                               out_pred_png="fid_pred_vs_true_bilinear.png"):
    dm = df[["BPP", "RFID", "FID", "Family", "ModelSize"]].dropna().copy()
    dm["BPP"] = dm["BPP"].astype(float)
    dm["RFID"] = dm["RFID"].astype(float)
    dm["FID"] = dm["FID"].astype(float)

    # centered predictors (match your regression setup)
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    dm["RFID_c"] = dm["RFID"] - dm["RFID"].mean()

    # Fit models
    m_bpp = smf.ols("FID ~ BPP_c", data=dm).fit()
    m_int = smf.ols("FID ~ BPP_c * RFID_c", data=dm).fit()

    # --------------------------------------------------
    # (1) 2D contour projection of the bilinear model + scatter
    # --------------------------------------------------
    bpp_grid = np.linspace(dm["BPP"].min(), dm["BPP"].max(), 60)
    rfid_grid = np.linspace(dm["RFID"].min(), dm["RFID"].max(), 60)
    BBP, RRF = np.meshgrid(bpp_grid, rfid_grid)

    grid = pd.DataFrame({
        "BPP": BBP.ravel(),
        "RFID": RRF.ravel(),
    })
    grid["BPP_c"] = grid["BPP"] - dm["BPP"].mean()
    grid["RFID_c"] = grid["RFID"] - dm["RFID"].mean()

    Z = m_int.predict(grid).values.reshape(BBP.shape)

    plt.figure(figsize=(8, 6))
    cs = plt.contour(BBP, RRF, Z, levels=12)
    plt.clabel(cs, inline=True, fontsize=8)

    sc = plt.scatter(dm["BPP"], dm["RFID"], c=dm["FID"], s=80)
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("RFID (Tokenizer Reconstruction FID)")
    plt.title("Bilinear Model Surface (Contours) with Model Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_surface_png), dpi=200)
    plt.close()

    # --------------------------------------------------
    # (2) Residuals: does RFID explain what's left after BPP?
    # --------------------------------------------------
    dm["resid_bpp_only"] = m_bpp.resid

    plt.figure(figsize=(7, 5))
    plt.scatter(dm["RFID"], dm["resid_bpp_only"], s=80)
    for _, r in dm.iterrows():
        plt.text(r["RFID"], r["resid_bpp_only"], str(r["ModelSize"]), fontsize=8)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("RFID (Tokenizer Reconstruction FID)")
    plt.ylabel("Residual FID after regressing on BPP")
    plt.title("RFID Explains Residual Variation Beyond Compression")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_resid_png), dpi=200)
    plt.close()

    # --------------------------------------------------
    # (3) Predicted vs true for bilinear model (robust)
    # --------------------------------------------------
    yhat = m_int.predict(dm)  # dm already contains BPP_c and RFID_c

    plt.figure(figsize=(6, 6))
    plt.scatter(dm["FID"], yhat, s=80)

    # annotate using precomputed yhat (no row-wise predict)
    for (x_true, y_pred, label) in zip(dm["FID"].values, yhat.values, dm["ModelSize"].values):
        plt.text(float(x_true), float(y_pred), str(label), fontsize=8)

    lo = float(min(dm["FID"].min(), yhat.min()))
    hi = float(max(dm["FID"].max(), yhat.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("True FID")
    plt.ylabel("Predicted FID (BPP×RFID model)")
    plt.title("Bilinear Model Fit Quality")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_pred_png), dpi=200)
    plt.close()


# visualize_bilinear_effects(df)

def plot_bilinear_3d_additive(df: pd.DataFrame,
                              out_png: str = "bilinear_3d_surface_global_additive.png",
                              label_fontsize: int = 8,
                              grid_n: int = 25):
    """
    3D probe of FID = f(BPP, RFID) using the ADDITIVE multiple linear model:
        FID ~ BPP_c + RFID_c
    Produces a 3D scatter (observations) + fitted PLANE (model).
    """

    dm = df[["BPP", "RFID", "FID", "Family", "ModelSize"]].dropna().copy()
    dm["BPP"] = dm["BPP"].astype(float)
    dm["RFID"] = dm["RFID"].astype(float)
    dm["FID"] = dm["FID"].astype(float)

    # Centering (matches your global modeling)
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    dm["RFID_c"] = dm["RFID"] - dm["RFID"].mean()

    # Fit ADDITIVE model globally (plane)
    m_add = smf.ols("FID ~ BPP_c + RFID_c", data=dm).fit()

    # Build grid in ORIGINAL units for readability (axes show BPP/RFID)
    bpp_min, bpp_max = float(dm["BPP"].min()), float(dm["BPP"].max())
    rfid_min, rfid_max = float(dm["RFID"].min()), float(dm["RFID"].max())

    bpp_grid = np.linspace(bpp_min, bpp_max, grid_n)
    rfid_grid = np.linspace(rfid_min, rfid_max, grid_n)
    BPPg, RFIDg = np.meshgrid(bpp_grid, rfid_grid)

    grid_df = pd.DataFrame({
        "BPP": BPPg.ravel(),
        "RFID": RFIDg.ravel(),
    })
    # Apply same centering as training
    grid_df["BPP_c"] = grid_df["BPP"] - dm["BPP"].mean()
    grid_df["RFID_c"] = grid_df["RFID"] - dm["RFID"].mean()

    FID_hat = m_add.predict(grid_df).values.reshape(BPPg.shape)

    # -------------------------
    # Plot: 3D scatter + plane
    # -------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plane first (semi-transparent) so points sit on top
    ax.plot_surface(
        BPPg, RFIDg, FID_hat,
        alpha=0.35,
        rstride=1, cstride=1,
        linewidth=0.0,
        antialiased=True,
    )

    # Scatter observations
    ax.scatter(
        dm["BPP"].values,
        dm["RFID"].values,
        dm["FID"].values,
        s=60,
        depthshade=True,
    )

    # Annotate points (optional)
    for _, r in dm.iterrows():
        ax.text(
            float(r["BPP"]),
            float(r["RFID"]),
            float(r["FID"]),
            str(r["ModelSize"]),
            fontsize=label_fontsize
        )

    ax.set_xlabel("Bits Per Pixel (BPP)")
    ax.set_ylabel("Tokenizer Reconstruction FID (rFID)")
    ax.set_zlabel("FID")
    ax.set_title("3D Probe of Additive Fit: FID = α + β·BPP + γ·rFID")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png), dpi=200)
    plt.close()

    return m_add


def plot_bilinear_3d(df: pd.DataFrame,
                     out_png: str = "bilinear_3d_surface_global.png",
                     label_fontsize: int = 8,
                     grid_n: int = 25):
    """
    3D probe of FID = f(BPP, RFID) using the bilinear OLS model with centered predictors.
    Produces a 3D scatter (observations) + fitted surface (model).
    """

    dm = df[["BPP", "RFID", "FID", "Family", "ModelSize"]].dropna().copy()
    dm["BPP"] = dm["BPP"].astype(float)
    dm["RFID"] = dm["RFID"].astype(float)
    dm["FID"] = dm["FID"].astype(float)

    # Centering (must match your modeling)
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    dm["RFID_c"] = dm["RFID"] - dm["RFID"].mean()

    # Fit bilinear model globally
    m_int = smf.ols("FID ~ BPP_c * RFID_c", data=dm).fit()

    # Build grid in ORIGINAL units for readability (axes show BPP/RFID)
    bpp_min, bpp_max = float(dm["BPP"].min()), float(dm["BPP"].max())
    rfid_min, rfid_max = float(dm["RFID"].min()), float(dm["RFID"].max())

    bpp_grid = np.linspace(bpp_min, bpp_max, grid_n)
    rfid_grid = np.linspace(rfid_min, rfid_max, grid_n)
    BPPg, RFIDg = np.meshgrid(bpp_grid, rfid_grid)

    grid_df = pd.DataFrame({
        "BPP": BPPg.ravel(),
        "RFID": RFIDg.ravel(),
    })
    # Apply same centering as training
    grid_df["BPP_c"] = grid_df["BPP"] - dm["BPP"].mean()
    grid_df["RFID_c"] = grid_df["RFID"] - dm["RFID"].mean()

    FID_hat = m_int.predict(grid_df).values.reshape(BPPg.shape)

    # -------------------------
    # Plot: 3D scatter + surface
    # -------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Surface first (semi-transparent) so points sit on top
    ax.plot_surface(
        BPPg, RFIDg, FID_hat,
        alpha=0.35,
        rstride=1, cstride=1,
        linewidth=0.0,
        antialiased=True,
    )

    # Scatter observations
    ax.scatter(
        dm["BPP"].values,
        dm["RFID"].values,
        dm["FID"].values,
        s=60,
        depthshade=True,
    )

    # Annotate points (optional)
    for _, r in dm.iterrows():
        ax.text(
            float(r["BPP"]),
            float(r["RFID"]),
            float(r["FID"]),
            str(r["ModelSize"]),
            fontsize=label_fontsize
        )

    ax.set_xlabel("Bits Per Pixel (BPP)")
    ax.set_ylabel("Tokenizer Reconstruction FID (RFID)")
    ax.set_zlabel("FID")
    ax.set_title("3D Probe of Bilinear Fit: FID = f(BPP, RFID)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png), dpi=200)
    plt.close()

    return m_int

def plot_bilinear_3d_per_family(df: pd.DataFrame, min_n: int = 3, grid_n: int = 25):
    for fam, g in df.groupby("Family"):
        if len(g) < min_n:
            continue
        out = f"bilinear_3d_surface_{fam}.png"
        plot_bilinear_3d(g, out_png=out, label_fontsize=8, grid_n=grid_n)

# 3D bilinear probe (global)
m_int_global = plot_bilinear_3d(df)
m_add_global = plot_bilinear_3d_additive(df)
plot_bilinear_3d_per_family(df, min_n=3)
fam_slope_df, fam_slope_txt = slope_vs_rfid_across_families(df, min_n=3)

# ==================================================
# 5) MAIN TEXT REPORT (single place)
# ==================================================
with open(REPORT_TXT, "w") as f:
    f.write("BPP vs FID Analysis Report\n")
    f.write("==========================\n\n")

    f.write("Data source: metrics_*.json under results/<ModelType>/<ModelType>_<ModelSize>/\n")
    f.write("Note: LlamaGen is split into two families based on image size: 256 and 384.\n\n")

    write_block(f, "GLOBAL FIT (all models)", global_stats)

    for fam in sorted(df["Family"].unique()):
        g = df[df["Family"] == fam]
        stats = fit_linear_bpp_to_fid(g["BPP"].values, g["FID"].values)
        write_block(f, f"CLUSTER FIT ({fam})", stats)

    f.write("Slope similarity tests (ANCOVA)\n")
    f.write("------------------------------\n")
    f.write(f"See: {os.path.basename(SLOPE_REPORT_TXT)} and {os.path.basename(PAIRWISE_SLOPE_CSV)}\n\n")

    f.write("FID as function of BPP and RFID\n")
    f.write("------------------------------\n")
    f.write(f"See: {os.path.basename(FID_BPP_RFID_TXT)} and {os.path.basename(FID_BPP_RFID_CSV)}\n\n")

    f.write("Saved artifacts\n")
    f.write("--------------\n")
    f.write("- report.txt\n")
    f.write("- bpp_vs_fid_global.png\n")
    for fam in sorted(df["Family"].unique()):
        f.write(f"- {fam}_bpp_vs_fid.png\n")
    f.write("- rfid_1d_distribution.png\n")
    f.write(f"- {os.path.basename(SLOPE_REPORT_TXT)}\n")
    f.write(f"- {os.path.basename(PAIRWISE_SLOPE_CSV)}\n")
    f.write(f"- {os.path.basename(FID_BPP_RFID_TXT)}\n")
    f.write(f"- {os.path.basename(FID_BPP_RFID_CSV)}\n")

    f.write("- var_bpp_fid_3pt.png\n")
    f.write("- var_bpp_fid_4pt.png\n")
    f.write("- var_bpp_fid_stats.txt\n")
    f.write("- per_model_predictions_ascii.txt\n")
    f.write("- per_model_predictions.csv\n")
    f.write("- family_slope_vs_rfid.txt\n")
    f.write("- family_slope_vs_rfid.png\n")

print("\nAnalysis complete.")
print(f"Main report: {REPORT_TXT}")
print(f"ANCOVA slope tests: {SLOPE_REPORT_TXT}")
print(f"FID~(BPP,RFID) models: {FID_BPP_RFID_TXT}")
