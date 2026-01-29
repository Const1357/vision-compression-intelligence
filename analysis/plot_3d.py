#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import statsmodels.api as sm

# ==================================================
# CONFIG (same style as analyze_results.py)
# ==================================================
RESULTS_DIR = "results"
OUTPUT_DIR = "analysis/results_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_FAMILY_N = 3  # only generate per-family plots for families with >=3 points

# ==================================================
# HELPERS (copied/compatible with analyze_results.py)
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
            })

    df = pd.DataFrame(rows)
    # sanitize numeric
    for c in ["FID", "RFID", "BPP"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FID", "RFID", "BPP"]).reset_index(drop=True)
    return df


def fit_bilinear_ols(dm: pd.DataFrame):
    """
    Fit: FID ~ 1 + BPP_c + RFID_c + (BPP_c*RFID_c)
    using explicit design matrix (no patsy), so prediction is easy/robust.
    """
    dm = dm.copy()
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    dm["RFID_c"] = dm["RFID"] - dm["RFID"].mean()
    dm["BPPxRFID_c"] = dm["BPP_c"] * dm["RFID_c"]

    X = dm[["BPP_c", "RFID_c", "BPPxRFID_c"]]
    X = sm.add_constant(X)  # adds intercept
    y = dm["FID"]

    model = sm.OLS(y, X).fit()

    # keep centering constants for plotting/prediction grids
    center = {
        "BPP_mean": float(dm["BPP"].mean()),
        "RFID_mean": float(dm["RFID"].mean()),
    }
    return model, center


def predict_surface(model, center, bpp_grid, rfid_grid):
    """
    Predict FID on a mesh grid of BPP and RFID, respecting centering used in fit.
    """
    BPP_c = bpp_grid - center["BPP_mean"]
    RFID_c = rfid_grid - center["RFID_mean"]
    BPPxRFID_c = BPP_c * RFID_c

    Xg = np.column_stack([
        np.ones(BPP_c.size),
        BPP_c.ravel(),
        RFID_c.ravel(),
        BPPxRFID_c.ravel()
    ])
    yhat = model.predict(Xg)
    return yhat.reshape(bpp_grid.shape)


def make_interactive_3d(df: pd.DataFrame, title: str, out_html: str):
    """
    Interactive plot: scatter3d points + fitted bilinear surface.
    Saves as a self-contained HTML.
    """
    if len(df) < 4:
        print(f"[skip] {title}: need >=4 points for bilinear surface (have {len(df)})")
        return

    model, center = fit_bilinear_ols(df)

    # define surface grid ranges (pad a little)
    bpp_min, bpp_max = float(df["BPP"].min()), float(df["BPP"].max())
    rfid_min, rfid_max = float(df["RFID"].min()), float(df["RFID"].max())

    bpp_pad = 0.05 * (bpp_max - bpp_min) if bpp_max > bpp_min else 1e-4
    rfid_pad = 0.05 * (rfid_max - rfid_min) if rfid_max > rfid_min else 1e-4

    bpp_vals = np.linspace(bpp_min - bpp_pad, bpp_max + bpp_pad, 40)
    rfid_vals = np.linspace(rfid_min - rfid_pad, rfid_max + rfid_pad, 40)
    BPPg, RFIDg = np.meshgrid(bpp_vals, rfid_vals)

    FIDhat = predict_surface(model, center, BPPg, RFIDg)

    # scatter points
    scatter = go.Scatter3d(
        x=df["BPP"],
        y=df["RFID"],
        z=df["FID"],
        mode="markers+text",
        text=df["ModelSize"],
        textposition="top center",
        marker=dict(size=6),
        hovertemplate=(
            "Model=%{text}<br>"
            "BPP=%{x:.6f}<br>"
            "RFID=%{y:.3f}<br>"
            "FID=%{z:.3f}<extra></extra>"
        ),
        name="Models",
    )

    # fitted surface
    surface = go.Surface(
        x=BPPg,
        y=RFIDg,
        z=FIDhat,
        opacity=0.55,
        showscale=True,
        colorbar=dict(title="Predicted FID"),
        hovertemplate=(
            "BPP=%{x:.6f}<br>"
            "RFID=%{y:.3f}<br>"
            "Pred FID=%{z:.3f}<extra></extra>"
        ),
        name="Bilinear fit",
    )

    fig = go.Figure(data=[surface, scatter])

    # paper-friendly axis naming
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="BPP (bits/pixel)",
            yaxis_title="RFID (tokenizer reconstruction FID)",
            zaxis_title="FID (lower is better)",
        ),
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # add short model summary in title hover via annotation
    r2 = float(model.rsquared)
    fig.add_annotation(
        text=f"Bilinear OLS: R²={r2:.3f} | Fit: FID ~ 1 + BPP_c + RFID_c + BPP_c·RFID_c",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.02,
        showarrow=False,
        align="left",
        font=dict(size=12),
    )

    out_path = os.path.join(OUTPUT_DIR, out_html)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[ok] wrote {out_path}")


def fit_additive_ols(dm: pd.DataFrame):
    """
    Fit: FID ~ 1 + BPP_c + RFID_c
    using explicit design matrix (no patsy), so prediction is easy/robust.
    """
    dm = dm.copy()
    dm["BPP_c"] = dm["BPP"] - dm["BPP"].mean()
    dm["RFID_c"] = dm["RFID"] - dm["RFID"].mean()

    X = dm[["BPP_c", "RFID_c"]]
    X = sm.add_constant(X)  # adds intercept
    y = dm["FID"]

    model = sm.OLS(y, X).fit()

    # keep centering constants for plotting/prediction grids
    center = {
        "BPP_mean": float(dm["BPP"].mean()),
        "RFID_mean": float(dm["RFID"].mean()),
    }
    return model, center


def predict_surface_additive(model, center, bpp_grid, rfid_grid):
    """
    Predict FID on a mesh grid of BPP and RFID, respecting centering used in fit.
    """
    BPP_c = bpp_grid - center["BPP_mean"]
    RFID_c = rfid_grid - center["RFID_mean"]

    Xg = np.column_stack([
        np.ones(BPP_c.size),
        BPP_c.ravel(),
        RFID_c.ravel(),
    ])
    yhat = model.predict(Xg)
    return yhat.reshape(bpp_grid.shape)


def make_interactive_3d_additive(df: pd.DataFrame, title: str, out_html: str):
    """
    Interactive plot: scatter3d points + fitted additive surface.
    Saves as a self-contained HTML.
    """
    if len(df) < 3:
        print(f"[skip] {title}: need >=3 points for additive surface (have {len(df)})")
        return

    model, center = fit_additive_ols(df)

    # define surface grid ranges (pad a little)
    bpp_min, bpp_max = float(df["BPP"].min()), float(df["BPP"].max())
    rfid_min, rfid_max = float(df["RFID"].min()), float(df["RFID"].max())

    bpp_pad = 0.05 * (bpp_max - bpp_min) if bpp_max > bpp_min else 1e-4
    rfid_pad = 0.05 * (rfid_max - rfid_min) if rfid_max > rfid_min else 1e-4

    bpp_vals = np.linspace(bpp_min - bpp_pad, bpp_max + bpp_pad, 80)
    rfid_vals = np.linspace(rfid_min - rfid_pad, rfid_max + rfid_pad, 40)
    BPPg, RFIDg = np.meshgrid(bpp_vals, rfid_vals)

    FIDhat = predict_surface_additive(model, center, BPPg, RFIDg)

    # --- marker grouping: split LlamaGen by resolution ---
    def marker_group(row):
        if row["ModelType"] == "LlamaGen":
            return f"LlamaGen_{int(row['ImageSize'])}"
        return row["ModelType"]

    df = df.copy()
    df["MarkerGroup"] = df.apply(marker_group, axis=1)

    # --- marker symbol + COLOR per group (explicit, no ambiguity) ---
    group_style = {
        "VAR":               dict(symbol="circle",        color="#1f77b4"),  # blue
        "RQ-Transformer":    dict(symbol="square",        color="#ff7f0e"),  # orange
        "LlamaGen_256":      dict(symbol="diamond",       color="#2ca02c"),  # green
        "LlamaGen_384":      dict(symbol="diamond-open",  color="#d62728"),  # red
        "VQ-GAN":            dict(symbol="cross",         color="#9467bd"),  # purple
        "VQGAN (Taming Transformers)": dict(symbol="cross", color="#9467bd"),
    }

    scatters = []
    for mg, g in df.groupby("MarkerGroup"):
        style = group_style.get(mg, dict(symbol="circle", color="#7f7f7f"))

        text_position = (
            "bottom left"
            if (
                mg in {"VQ-GAN", "VQGAN (Taming Transformers)"}
            )
            else "top center"
        )

        scatters.append(
            go.Scatter3d(
                x=g["BPP"],
                y=g["RFID"],
                z=g["FID"],
                mode="markers+text",
                text=g["ModelType"] + " - " + g["ModelSize"],
                textposition=text_position,     # ← only VQGAN offset left
                textfont=dict(size=8),
                marker=dict(
                    size=4,
                    symbol=style["symbol"],
                    color=style["color"],
                ),
                hovertemplate=(
                    "Model=%{text}<br>"
                    "BPP=%{x:.6f}<br>"
                    "RFID=%{y:.3f}<br>"
                    "FID=%{z:.3f}<extra></extra>"
                ),
                name=str(mg),
                showlegend=True,
            )
        )


    surface = go.Surface(
        x=BPPg,
        y=RFIDg,
        z=FIDhat,
        opacity=0.45,
        showscale=True,
        colorbar=dict(
            title="Predicted FID",
            len=0.45,
            thickness=12,
        ),
        hovertemplate=(
            "BPP=%{x:.6f}<br>"
            "RFID=%{y:.3f}<br>"
            "Pred FID=%{z:.3f}<extra></extra>"
        ),
        name="Additive fit",
        showlegend=False,
    )

    fig = go.Figure(data=[surface] + scatters)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title="Bits per Pixel (BPP)",
                tick0=0.010,
                dtick=0.005,
            ),
            yaxis_title="Tokenizer rFID",
            zaxis_title="FID",
        ),
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1100,
        height=850,
    )

    r2 = float(model.rsquared)
    fig.add_annotation(
        text=f"Additive OLS: R²={r2:.3f} | Fit: FID ~ 1 + BPP_c + RFID_c",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.03,
        showarrow=False,
        align="left",
        font=dict(size=12),
    )

    # hardcoded
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(
            x=-1.7499375280304852,
            y=-1.4925986699023281,
            z=0.7010220350616325,
        ),
        projection=dict(type="perspective"),
    )

    fig.update_layout(scene_camera=camera)
    fig.write_image(
        f"analysis_results/fid_bpp_rfid_additive.png",
        width=1600,
        height=1200,
        scale=2,
    )




    out_path = os.path.join(OUTPUT_DIR, out_html)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[ok] wrote {out_path}")









def main():
    df = load_results(RESULTS_DIR)
    if df.empty:
        raise RuntimeError(f"No metrics JSONs found under {RESULTS_DIR}")
    
    print(df)

    # GLOBAL interactive surface
    make_interactive_3d(
        df,
        title="Bilinear probe (global): FID as a function of BPP and RFID",
        out_html="bilinear_3d_interactive_global.html",
    )

    make_interactive_3d_additive(
        df,
        title="Additive probe (global): FID as a function of BPP and RFID",
        out_html="additive_3d_interactive_global.html",
    )

    # PER-FAMILY interactive surfaces (only if >= MIN_FAMILY_N)
    for fam, g in df.groupby("Family"):
        g = g.copy()
        if len(g) < MIN_FAMILY_N:
            print(f"[skip] {fam}: n={len(g)} (<{MIN_FAMILY_N})")
            continue
        # Note: bilinear surface needs >=4 to actually fit all 4 params robustly.
        # We keep MIN_FAMILY_N=3 for inclusion checks, but the surface function requires >=4.
        make_interactive_3d(
            g,
            title=f"Bilinear probe ({fam}): FID as a function of BPP and RFID",
            out_html=f"bilinear_3d_interactive_{fam}.html",
        )


if __name__ == "__main__":
    main()

