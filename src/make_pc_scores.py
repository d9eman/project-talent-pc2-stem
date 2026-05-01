import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DOMAIN_COLS = [
    "care_arts_office_score",
    "home_domestic_score",
    "mechanical_technical_score",
    "peer_social_score",
    "sports_outdoors_score",
]


def auc_table(scores):
    tests = {
        "PC1 only": ["pc1"],
        "PC2 only": ["pc2"],
        "PC1 + PC2": ["pc1", "pc2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=36)
    rows = []
    for name, cols in tests.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        aucs = cross_val_score(model, scores[cols], scores["sex01"], cv=cv, scoring="roc_auc")
        rows.append({
            "model": name,
            "features": ",".join(cols),
            "n": len(scores),
            "cv_auc_mean": aucs.mean(),
            "cv_auc_sd": aucs.std(ddof=1),
        })
    return pd.DataFrame(rows)


def add_gci_and_groups(scores, q=0.25):
    out = scores.copy()
    out["masc_pc1_01"] = (out["pc1"] - out["pc1"].min()) / (out["pc1"].max() - out["pc1"].min())
    out["gci_pc1"] = np.where(out["sex01"] == 1, out["masc_pc1_01"], 1 - out["masc_pc1_01"])
    out["cgn_pc1"] = 0
    out["pc2_group"] = "middle_PC2"

    for sex in [0, 1]:
        mask = out["sex01"] == sex
        out.loc[mask, "cgn_pc1"] = (out.loc[mask, "gci_pc1"] <= out.loc[mask, "gci_pc1"].quantile(.20)).astype(int)
        lo = out.loc[mask, "pc2"].quantile(q)
        hi = out.loc[mask, "pc2"].quantile(1 - q)
        out.loc[mask & (out["pc2"] <= lo), "pc2_group"] = "low_PC2_social_domestic"
        out.loc[mask & (out["pc2"] >= hi), "pc2_group"] = "high_PC2_technical_physical"

    out["pc1_z"] = (out["pc1"] - out["pc1"].mean()) / out["pc1"].std(ddof=0)
    out["pc2_z"] = (out["pc2"] - out["pc2"].mean()) / out["pc2"].std(ddof=0)
    out["sex_label"] = out["sex01"].map({0: "female", 1: "male"})
    out["cgn_label"] = out["cgn_pc1"].map({0: "non_CGN", 1: "CGN"})
    return out


def plot_loadings(loadings, out_dir):
    pc12 = loadings[loadings["component"].isin(["PC1", "PC2"])].pivot(index="variable", columns="component", values="loading")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    for name, row in pc12.iterrows():
        ax.arrow(0, 0, row["PC1"], row["PC2"], length_includes_head=True, head_width=.02)
        ax.text(row["PC1"] + .01, row["PC2"] + .01, name)
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title("Domain loadings on PC1 and PC2")
    fig.tight_layout()
    fig.savefig(out_dir / "domain_pc1_pc2_loadings_biplot.png", dpi=200)
    plt.close(fig)

    pc2 = pc12["PC2"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(pc2.index, pc2.values)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("PC2 loading")
    ax.set_title("PC2 domain contrast")
    fig.tight_layout()
    fig.savefig(out_dir / "domain_pc2_loadings.png", dpi=200)
    plt.close(fig)


def plot_scree(pca, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker="o")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Five-domain PCA scree plot")
    fig.tight_layout()
    fig.savefig(out_dir / "domain_scree_plot.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain_scores", required=True)
    ap.add_argument("--out_dir", default="pc2_general_pca_results")
    ap.add_argument("--id_col", default="BY_ID_Rel")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    raw = pd.read_parquet(args.domain_scores)
    raw["sex01"] = raw["sex01"].astype(int)
    use = raw[[args.id_col, "sex01"] + DOMAIN_COLS].dropna().copy()

    X = StandardScaler().fit_transform(use[DOMAIN_COLS])
    pca = PCA(n_components=len(DOMAIN_COLS), random_state=36)
    pcs = pca.fit_transform(X)

    scores = use[[args.id_col, "sex01"]].copy()
    for i in range(pcs.shape[1]):
        scores[f"pc{i+1}"] = pcs[:, i]

    # Keep PC1 oriented so higher values are male-typed.
    if scores.groupby("sex01")["pc1"].mean().loc[1] < scores.groupby("sex01")["pc1"].mean().loc[0]:
        scores["pc1"] *= -1
        pca.components_[0] *= -1

    scores = add_gci_and_groups(scores)
    scores.to_parquet(out_dir / "domain_pca_scores.parquet", index=False)

    rows = []
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        for variable, loading in zip(DOMAIN_COLS, pca.components_[i - 1]):
            rows.append({"component": f"PC{i}", "variable": variable, "loading": loading, "explained_variance_ratio": ratio})
    loadings = pd.DataFrame(rows)
    loadings.to_csv(out_dir / "domain_pca_loadings.csv", index=False)

    corr_rows = []
    for c in DOMAIN_COLS + ["sex01", "pc1", "gci_pc1"]:
        corr_rows.append({"variable": c, "correlation_with_pc2": scores.join(use[DOMAIN_COLS])[c].corr(scores["pc2"]), "n": len(scores)})
    pd.DataFrame(corr_rows).to_csv(out_dir / "domain_pc2_correlations.csv", index=False)

    auc_table(scores.sample(min(75000, len(scores)), random_state=36)).to_csv(out_dir / "domain_sex_auc_comparison.csv", index=False)
    plot_loadings(loadings, fig_dir)
    plot_scree(pca, fig_dir)

    print(f"Wrote PCA scores, tables, and figures to {out_dir}")


if __name__ == "__main__":
    main()
