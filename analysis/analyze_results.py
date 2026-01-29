#!/usr/bin/env python3
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
                "BPP_std": float(np.std(bpp_hist)) if len(bpp_hist) > 0 else np.nan,
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


global_stats = plot_global_bpp_fid(df)
per_family_df = plot_per_family(df)
plot_rfid_1d(df)


# ==================================================
# 3) ANCOVA: Are cluster slopes statistically similar?
# ==================================================
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


fid_bpp_rfid_summary = fit_fid_bpp_rfid_models(df)

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

print("\nAnalysis complete.")
print(f"Main report: {REPORT_TXT}")
print(f"ANCOVA slope tests: {SLOPE_REPORT_TXT}")
print(f"FID~(BPP,RFID) models: {FID_BPP_RFID_TXT}")

