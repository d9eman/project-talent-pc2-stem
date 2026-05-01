import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_COLS = ["nonCGN_low_PC2", "nonCGN_high_PC2", "CGN_low_PC2", "CGN_high_PC2"]
GROUP_LABELS = ["non-CGN, low PC2", "non-CGN, high PC2", "CGN, low PC2", "CGN, high PC2"]
STEM = ["stem_any_ge_4", "stem_any_ge_5", "major_intent_y1_stem", "major_realized_y11_stem", "stem_job_y11"]
ENG = ["eng_any_ge_4", "eng_any_ge_5", "major_intent_y1_engineering", "major_realized_y11_engineering", "engineering_job_y11"]
LABEL = {
    "stem_any_ge_4": "HS STEM interest ≥ 4",
    "stem_any_ge_5": "HS STEM interest = 5",
    "major_intent_y1_stem": "Y1 intended STEM major",
    "major_realized_y11_stem": "Y11 realized STEM major",
    "stem_job_y11": "Y11 STEM job",
    "eng_any_ge_4": "HS engineering interest ≥ 4",
    "eng_any_ge_5": "HS engineering interest = 5",
    "major_intent_y1_engineering": "Y1 intended engineering major",
    "major_realized_y11_engineering": "Y11 realized engineering major",
    "engineering_job_y11": "Y11 engineering job",
}


def make_summary(profiles):
    keep = profiles[profiles.pc2_group.isin(["low_PC2_social_domestic", "high_PC2_technical_physical"])].copy()
    rows = []
    for sex in ["female", "male"]:
        for outcome in STEM + ENG:
            d = keep[(keep.sex == sex) & (keep.outcome == outcome)]
            vals = {}
            ns = {}
            for cgn, pc2, col in [
                ("non_CGN", "low_PC2_social_domestic", "nonCGN_low_PC2"),
                ("non_CGN", "high_PC2_technical_physical", "nonCGN_high_PC2"),
                ("CGN", "low_PC2_social_domestic", "CGN_low_PC2"),
                ("CGN", "high_PC2_technical_physical", "CGN_high_PC2"),
            ]:
                one = d[(d.cgn == cgn) & (d.pc2_group == pc2)]
                vals[col] = one.outcome_mean.iloc[0]
                ns["n_" + col] = int(one.n.iloc[0])
            rows.append({
                "sex": sex,
                "outcome": outcome,
                "outcome_label": LABEL[outcome],
                **vals,
                **ns,
                "high_minus_low_nonCGN": vals["nonCGN_high_PC2"] - vals["nonCGN_low_PC2"],
                "high_minus_low_CGN": vals["CGN_high_PC2"] - vals["CGN_low_PC2"],
                "CGN_minus_nonCGN_at_low_PC2": vals["CGN_low_PC2"] - vals["nonCGN_low_PC2"],
                "CGN_minus_nonCGN_at_high_PC2": vals["CGN_high_PC2"] - vals["nonCGN_high_PC2"],
            })
    return pd.DataFrame(rows)


def plot_pathway(summary, sex, outcomes, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(outcomes))
    d = summary[summary.sex == sex].set_index("outcome").loc[outcomes]
    for col, label in zip(GROUP_COLS, GROUP_LABELS):
        ax.plot(x, d[col] * 100, marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[o] for o in outcomes], rotation=25, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bars(summary, sex, outcomes, title, out_path, relative=False):
    d = summary[summary.sex == sex].set_index("outcome").loc[outcomes]
    vals = d[GROUP_COLS].to_numpy()
    ylabel = "Percent"
    if relative:
        vals = vals / vals[:, [0]]
        ylabel = "Relative rate (non-CGN low PC2 = 1.0)"
    else:
        vals = vals * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(outcomes))
    width = .18
    for i, label in enumerate(GROUP_LABELS):
        ax.bar(x + (i - 1.5) * width, vals[:, i], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[o] for o in outcomes], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pc2_coef(models, sex, outcomes, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(outcomes))
    d = models[(models.sex == sex) & (models.term == "pc2_z")]
    for cgn in ["non_CGN", "CGN"]:
        vals = []
        for outcome in outcomes:
            one = d[(d.cgn == cgn) & (d.outcome == outcome)]
            vals.append(one.coef.iloc[0] * 100)
        ax.plot(x, vals, marker="o", label=cgn)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[o] for o in outcomes], rotation=25, ha="right")
    ax.set_ylabel("Effect of +1 SD PC2 (percentage points)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--figures_dir", default="figures")
    args = ap.parse_args()

    results = Path(args.results_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(results / "pc2_cgn_outcome_profiles.csv")
    models = pd.read_csv(results / "pc2_within_cgn_models.csv")
    summary = make_summary(profiles)
    summary.to_csv(results / "key_pc2_pathway_table.csv", index=False)

    for sex in ["female", "male"]:
        plot_pathway(summary, sex, ENG, f"{sex.title()} engineering pathway by CGN and PC2 group", figs / f"{sex}_engineering_pathway.png")
        plot_pathway(summary, sex, STEM, f"{sex.title()} STEM pathway by CGN and PC2 group", figs / f"{sex}_stem_pathway.png")
        plot_pc2_coef(models, sex, ENG, f"{sex.title()}: +1 SD PC2 effect, engineering pathway", figs / f"{sex}_engineering_pc2_coefficients.png")
        plot_pc2_coef(models, sex, STEM, f"{sex.title()}: +1 SD PC2 effect, STEM pathway", figs / f"{sex}_stem_pc2_coefficients.png")
        plot_bars(summary, sex, ENG[:2], f"{sex.title()} early engineering interest", figs / f"{sex}_engineering_early_bar.png")
        plot_bars(summary, sex, STEM[:2], f"{sex.title()} early STEM interest", figs / f"{sex}_stem_early_bar.png")
        plot_bars(summary, sex, ENG[2:], f"{sex.title()} late engineering outcomes, zoomed", figs / f"{sex}_late_engineering_bar.png")
        plot_bars(summary, sex, ENG[2:], f"{sex.title()} late engineering outcomes, relative", figs / f"{sex}_late_engineering_relative_bar.png", relative=True)
        plot_bars(summary, sex, STEM[2:], f"{sex.title()} late STEM outcomes", figs / f"{sex}_late_stem_bar.png")

    print(f"Wrote figures to {figs}")


if __name__ == "__main__":
    main()
