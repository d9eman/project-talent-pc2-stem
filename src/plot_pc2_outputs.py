from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


STEM_PATHWAY = [
    "stem_any_ge_4",
    "stem_any_ge_5",
    "major_intent_y1_stem",
    "major_realized_y11_stem",
    "stem_job_y11",
]

ENGINEERING_PATHWAY = [
    "eng_any_ge_4",
    "eng_any_ge_5",
    "major_intent_y1_engineering",
    "major_realized_y11_engineering",
    "engineering_job_y11",
]

OUTCOME_LABELS = {
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

GROUP_ORDER = [
    ("non_CGN", "low_PC2_social_domestic"),
    ("non_CGN", "high_PC2_technical_physical"),
    ("CGN", "low_PC2_social_domestic"),
    ("CGN", "high_PC2_technical_physical"),
]

GROUP_LABELS = {
    ("non_CGN", "low_PC2_social_domestic"): "non-CGN, low PC2",
    ("non_CGN", "high_PC2_technical_physical"): "non-CGN, high PC2",
    ("CGN", "low_PC2_social_domestic"): "CGN, low PC2",
    ("CGN", "high_PC2_technical_physical"): "CGN, high PC2",
}


def load_profiles(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "pc2_cgn_outcome_profiles.csv"
    df = pd.read_csv(path)

    keep_outcomes = set(STEM_PATHWAY + ENGINEERING_PATHWAY)
    df = df[df["outcome"].isin(keep_outcomes)].copy()

    keep_pc2_groups = {"low_PC2_social_domestic", "high_PC2_technical_physical"}
    df = df[df["pc2_group"].isin(keep_pc2_groups)].copy()

    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    all_outcomes = STEM_PATHWAY + ENGINEERING_PATHWAY

    for sex in ["female", "male"]:
        for outcome in all_outcomes:
            sub = df[(df["sex"] == sex) & (df["outcome"] == outcome)]

            values = {}
            counts = {}

            for cgn, pc2_group in GROUP_ORDER:
                temp = sub[(sub["cgn"] == cgn) & (sub["pc2_group"] == pc2_group)]
                key = f"{cgn}_{pc2_group}"

                if len(temp) == 1:
                    values[key] = float(temp["outcome_mean"].iloc[0])
                    counts[key] = int(temp["n"].iloc[0])
                else:
                    values[key] = float("nan")
                    counts[key] = 0

            noncgn_low = values["non_CGN_low_PC2_social_domestic"]
            noncgn_high = values["non_CGN_high_PC2_technical_physical"]
            cgn_low = values["CGN_low_PC2_social_domestic"]
            cgn_high = values["CGN_high_PC2_technical_physical"]

            rows.append({
                "sex": sex,
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],

                "nonCGN_low_PC2": noncgn_low,
                "nonCGN_high_PC2": noncgn_high,
                "CGN_low_PC2": cgn_low,
                "CGN_high_PC2": cgn_high,

                "n_nonCGN_low_PC2": counts["non_CGN_low_PC2_social_domestic"],
                "n_nonCGN_high_PC2": counts["non_CGN_high_PC2_technical_physical"],
                "n_CGN_low_PC2": counts["CGN_low_PC2_social_domestic"],
                "n_CGN_high_PC2": counts["CGN_high_PC2_technical_physical"],

                "high_minus_low_nonCGN": noncgn_high - noncgn_low,
                "high_minus_low_CGN": cgn_high - cgn_low,
                "CGN_minus_nonCGN_at_low_PC2": cgn_low - noncgn_low,
                "CGN_minus_nonCGN_at_high_PC2": cgn_high - noncgn_high,
            })

    return pd.DataFrame(rows)


def plot_pathway(df: pd.DataFrame, sex: str, outcomes: list[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    x = list(range(len(outcomes)))

    for cgn, pc2_group in GROUP_ORDER:
        ys = []
        label = GROUP_LABELS[(cgn, pc2_group)]

        for outcome in outcomes:
            temp = df[
                (df["sex"] == sex) &
                (df["outcome"] == outcome) &
                (df["cgn"] == cgn) &
                (df["pc2_group"] == pc2_group)
            ]

            if len(temp) == 1:
                ys.append(float(temp["outcome_mean"].iloc[0]) * 100.0)
            else:
                ys.append(float("nan"))

        plt.plot(x, ys, marker="o", label=label)

    plt.xticks(x, [OUTCOME_LABELS[o] for o in outcomes], rotation=25, ha="right")
    plt.ylabel("Percent")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_within_group_models(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "pc2_within_cgn_models.csv"
    df = pd.read_csv(path)

    keep_outcomes = set(STEM_PATHWAY + ENGINEERING_PATHWAY)
    df = df[df["outcome"].isin(keep_outcomes)].copy()
    df = df[df["term"] == "pc2_z"].copy()
    df = df[df["model"] == "within_group_y_on_pc2_z"].copy()

    return df


def plot_pc2_coefficients(df: pd.DataFrame, sex: str, outcomes: list[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    x = list(range(len(outcomes)))

    for cgn in ["non_CGN", "CGN"]:
        ys = []

        for outcome in outcomes:
            temp = df[
                (df["sex"] == sex) &
                (df["cgn"] == cgn) &
                (df["outcome"] == outcome)
            ]

            if len(temp) == 1:
                # convert from proportion scale to percentage points
                ys.append(float(temp["coef"].iloc[0]) * 100.0)
            else:
                ys.append(float("nan"))

        plt.plot(x, ys, marker="o", label=cgn)

    plt.axhline(0.0)
    plt.xticks(x, [OUTCOME_LABELS[o] for o in outcomes], rotation=25, ha="right")
    plt.ylabel("Effect of +1 SD PC2 (percentage points)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_key_findings(summary: pd.DataFrame, out_path: Path) -> None:
    lines = []

    lines.append("KEY FINDINGS FROM PC2 PATHWAY TABLES")
    lines.append("=" * 50)
    lines.append("")

    for sex in ["female", "male"]:
        lines.append(sex.upper())
        lines.append("-" * 20)

        sub = summary[summary["sex"] == sex].copy()

        for outcome in [
            "eng_any_ge_4",
            "eng_any_ge_5",
            "stem_any_ge_4",
            "stem_any_ge_5",
            "major_intent_y1_stem",
            "major_realized_y11_stem",
            "stem_job_y11",
            "engineering_job_y11",
        ]:
            row = sub[sub["outcome"] == outcome]
            if len(row) != 1:
                continue

            row = row.iloc[0]

            lines.append(f"{row['outcome_label']}:")
            lines.append(f"  non-CGN low PC2:  {row['nonCGN_low_PC2']:.4f}")
            lines.append(f"  non-CGN high PC2: {row['nonCGN_high_PC2']:.4f}")
            lines.append(f"  CGN low PC2:      {row['CGN_low_PC2']:.4f}")
            lines.append(f"  CGN high PC2:     {row['CGN_high_PC2']:.4f}")
            lines.append(f"  high-low within non-CGN: {row['high_minus_low_nonCGN']:.4f}")
            lines.append(f"  high-low within CGN:     {row['high_minus_low_CGN']:.4f}")
            lines.append(f"  CGN - non-CGN at low PC2:  {row['CGN_minus_nonCGN_at_low_PC2']:.4f}")
            lines.append(f"  CGN - non-CGN at high PC2: {row['CGN_minus_nonCGN_at_high_PC2']:.4f}")
            lines.append("")

        lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="pc2_general_pca_results/cgn_pc2_mechanism_tests",
        help="Directory containing pc2_cgn_outcome_profiles.csv and related files"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = base_dir / "pathway_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(base_dir)
    summary = build_summary_table(profiles)
    summary.to_csv(out_dir / "key_pc2_pathway_table.csv", index=False)

    write_key_findings(summary, out_dir / "key_pc2_findings.txt")

    plot_pathway(
        profiles,
        sex="female",
        outcomes=ENGINEERING_PATHWAY,
        title="Female engineering pathway by CGN and PC2 group",
        out_path=out_dir / "female_engineering_pathway.png",
    )

    plot_pathway(
        profiles,
        sex="female",
        outcomes=STEM_PATHWAY,
        title="Female STEM pathway by CGN and PC2 group",
        out_path=out_dir / "female_stem_pathway.png",
    )

    plot_pathway(
        profiles,
        sex="male",
        outcomes=ENGINEERING_PATHWAY,
        title="Male engineering pathway by CGN and PC2 group",
        out_path=out_dir / "male_engineering_pathway.png",
    )

    plot_pathway(
        profiles,
        sex="male",
        outcomes=STEM_PATHWAY,
        title="Male STEM pathway by CGN and PC2 group",
        out_path=out_dir / "male_stem_pathway.png",
    )

    within_models = load_within_group_models(base_dir)

    plot_pc2_coefficients(
        within_models,
        sex="female",
        outcomes=ENGINEERING_PATHWAY,
        title="Female: effect of +1 SD PC2 within CGN groups (engineering pathway)",
        out_path=out_dir / "female_engineering_pc2_coefficients.png",
    )

    plot_pc2_coefficients(
        within_models,
        sex="female",
        outcomes=STEM_PATHWAY,
        title="Female: effect of +1 SD PC2 within CGN groups (STEM pathway)",
        out_path=out_dir / "female_stem_pc2_coefficients.png",
    )

    plot_pc2_coefficients(
        within_models,
        sex="male",
        outcomes=ENGINEERING_PATHWAY,
        title="Male: effect of +1 SD PC2 within CGN groups (engineering pathway)",
        out_path=out_dir / "male_engineering_pc2_coefficients.png",
    )

    plot_pc2_coefficients(
        within_models,
        sex="male",
        outcomes=STEM_PATHWAY,
        title="Male: effect of +1 SD PC2 within CGN groups (STEM pathway)",
        out_path=out_dir / "male_stem_pc2_coefficients.png",
    )

    print("Wrote:")
    print(f"  {out_dir / 'key_pc2_pathway_table.csv'}")
    print(f"  {out_dir / 'key_pc2_findings.txt'}")
    print(f"  {out_dir / 'female_engineering_pathway.png'}")
    print(f"  {out_dir / 'female_stem_pathway.png'}")
    print(f"  {out_dir / 'male_engineering_pathway.png'}")
    print(f"  {out_dir / 'male_stem_pathway.png'}")
    print(f"  {out_dir / 'female_engineering_pc2_coefficients.png'}")
    print(f"  {out_dir / 'female_stem_pc2_coefficients.png'}")
    print(f"  {out_dir / 'male_engineering_pc2_coefficients.png'}")
    print(f"  {out_dir / 'male_stem_pc2_coefficients.png'}")


if __name__ == "__main__":
    main()