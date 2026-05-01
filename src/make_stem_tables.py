import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

MISSING = {98: np.nan, 99: np.nan, 998: np.nan, 999: np.nan}
ENG_ITEMS = ["BY_INT005", "BY_INT045", "BY_INT048", "BY_INT068", "BY_INT090"]
STEM_ITEMS = ENG_ITEMS + ["BY_INT091", "BY_INT067", "BY_INT004", "BY_INT028"]
ENG_MAJOR = [33, 34]
STEM_MAJOR = [1, 2, 3, 4, 6, 9] + ENG_MAJOR
ENG_JOB = [240, 241, 242, 243, 244, 245, 221]
STEM_JOB = [211, 230, 232, 310, 316, 317] + ENG_JOB
OUTCOMES = [
    "eng_any_ge_4", "eng_any_ge_5", "stem_any_ge_4", "stem_any_ge_5",
    "major_intent_y1_stem", "major_intent_y1_engineering",
    "major_realized_y11_stem", "major_realized_y11_engineering",
    "stem_job_y11", "engineering_job_y11",
]


def clean(s):
    return pd.to_numeric(s, errors="coerce").replace(MISSING)


def any_ge(df, cols, cutoff):
    x = df[cols].apply(clean)
    out = pd.Series(np.nan, index=df.index)
    out[x.notna().any(axis=1)] = 0.0
    out[(x >= cutoff).any(axis=1)] = 1.0
    return out


def code_flag(s, codes):
    x = clean(s).astype("Int64")
    out = pd.Series(np.nan, index=s.index)
    out[x.notna()] = 0.0
    out[x.isin(codes)] = 1.0
    return out


def school_fe_lpm(df, y, xcols, school_col):
    d = df[[y, school_col] + xcols].dropna().copy()
    d[school_col] = d[school_col].astype(str)
    for col in [y] + xcols:
        d[col] = d[col] - d.groupby(school_col)[col].transform("mean")
    res = sm.OLS(d[y], d[xcols]).fit(cov_type="cluster", cov_kwds={"groups": d[school_col]})
    rows = []
    for i, term in enumerate(xcols):
        rows.append({"term": term, "n_used": len(d), "coef": res.params[i], "p_value": res.pvalues[i]})
    return rows


def fetch_data(scores, db_path, base_relation, y11_table, id_col, school_col):
    con = duckdb.connect(db_path)
    con.register("scores", scores)
    interests = ", ".join([f"b.{c}" for c in sorted(set(STEM_ITEMS))])
    base = con.execute(f"""
        SELECT s.*, b.{school_col}, {interests}, b.Y1_E320, b.Y11_E321
        FROM scores s
        JOIN {base_relation} b ON s.{id_col} = b.{id_col}
        WHERE b.{school_col} IS NOT NULL
    """).df()
    y11 = con.execute(f"""
        SELECT s.*, y.{school_col}, y.Y11_O202
        FROM scores s
        JOIN {y11_table} y ON s.{id_col} = y.{id_col}
        WHERE y.{school_col} IS NOT NULL AND y.Y11_QSTAT = '01'
    """).df()
    con.close()
    return base, y11


def add_outcomes(base, y11):
    base = base.copy()
    y11 = y11.copy()
    base["eng_any_ge_4"] = any_ge(base, ENG_ITEMS, 4)
    base["eng_any_ge_5"] = any_ge(base, ENG_ITEMS, 5)
    base["stem_any_ge_4"] = any_ge(base, STEM_ITEMS, 4)
    base["stem_any_ge_5"] = any_ge(base, STEM_ITEMS, 5)
    base["major_intent_y1_stem"] = code_flag(base["Y1_E320"], STEM_MAJOR)
    base["major_intent_y1_engineering"] = code_flag(base["Y1_E320"], ENG_MAJOR)
    base["major_realized_y11_stem"] = code_flag(base["Y11_E321"], STEM_MAJOR)
    base["major_realized_y11_engineering"] = code_flag(base["Y11_E321"], ENG_MAJOR)
    y11["stem_job_y11"] = code_flag(y11["Y11_O202"], STEM_JOB)
    y11["engineering_job_y11"] = code_flag(y11["Y11_O202"], ENG_JOB)
    return pd.concat([base, y11], ignore_index=True, sort=False)


def pc2_distribution(scores):
    rows = []
    for sex, sex_name in [(0, "female"), (1, "male")]:
        for cgn, cgn_name in [(0, "non_CGN"), (1, "CGN")]:
            d = scores[(scores.sex01 == sex) & (scores.cgn_pc1 == cgn)]
            rows.append({
                "sex": sex_name,
                "cgn": cgn_name,
                "n": len(d),
                "mean_pc1": d.pc1.mean(),
                "mean_pc2": d.pc2.mean(),
                "sd_pc2": d.pc2.std(ddof=0),
                "median_pc2": d.pc2.median(),
                "prop_high_pc2": (d.pc2_group == "high_PC2_technical_physical").mean(),
                "prop_low_pc2": (d.pc2_group == "low_PC2_social_domestic").mean(),
            })
    return pd.DataFrame(rows)


def profile_table(df):
    rows = []
    for outcome in OUTCOMES:
        d = df.dropna(subset=[outcome]).copy()
        for keys, g in d.groupby(["sex_label", "cgn_label", "pc2_group"], observed=True):
            sex, cgn, pc2_group = keys
            rows.append({
                "outcome": outcome,
                "sex": sex,
                "cgn": cgn,
                "pc2_group": pc2_group,
                "n": len(g),
                "outcome_mean": g[outcome].mean(),
                "mean_pc1": g.pc1.mean(),
                "mean_pc2": g.pc2.mean(),
            })
    return pd.DataFrame(rows)


def high_low_contrasts(profiles):
    rows = []
    for (outcome, sex), d in profiles.groupby(["outcome", "sex"]):
        p = d.pivot_table(index="cgn", columns="pc2_group", values="outcome_mean")
        if {"CGN", "non_CGN"}.issubset(p.index) and {"low_PC2_social_domestic", "high_PC2_technical_physical"}.issubset(p.columns):
            rows += [
                {"outcome": outcome, "sex": sex, "contrast": "among_CGN_highPC2_minus_lowPC2", "difference": p.loc["CGN", "high_PC2_technical_physical"] - p.loc["CGN", "low_PC2_social_domestic"]},
                {"outcome": outcome, "sex": sex, "contrast": "among_nonCGN_highPC2_minus_lowPC2", "difference": p.loc["non_CGN", "high_PC2_technical_physical"] - p.loc["non_CGN", "low_PC2_social_domestic"]},
                {"outcome": outcome, "sex": sex, "contrast": "highPC2_CGN_minus_nonCGN", "difference": p.loc["CGN", "high_PC2_technical_physical"] - p.loc["non_CGN", "high_PC2_technical_physical"]},
                {"outcome": outcome, "sex": sex, "contrast": "lowPC2_CGN_minus_nonCGN", "difference": p.loc["CGN", "low_PC2_social_domestic"] - p.loc["non_CGN", "low_PC2_social_domestic"]},
            ]
    return pd.DataFrame(rows)


def model_tables(df, school_col):
    within_rows, interaction_rows, atten_rows = [], [], []
    df = df.copy()
    df["cgn01"] = df["cgn_pc1"].astype(float)
    df["high_pc2"] = (df["pc2_group"] == "high_PC2_technical_physical").astype(float)
    df["cgn_x_pc2_z"] = df["cgn01"] * df["pc2_z"]
    df["cgn_x_high_pc2"] = df["cgn01"] * df["high_pc2"]

    for outcome in OUTCOMES:
        for sex, sex_name in [(0, "female"), (1, "male")]:
            dsex = df[df.sex01 == sex]
            for cgn, cgn_name in [(0, "non_CGN"), (1, "CGN")]:
                d = dsex[dsex.cgn01 == cgn]
                for r in school_fe_lpm(d, outcome, ["pc2_z"], school_col):
                    r.update({"outcome": outcome, "sex": sex_name, "cgn": cgn_name, "model": "within_group_y_on_pc2_z"})
                    within_rows.append(r)
            for xcols, name, store in [
                (["cgn01"], "M1_CGN_only", atten_rows),
                (["cgn01", "pc2_z"], "M2_CGN_plus_PC2", atten_rows),
                (["cgn01", "pc2_z", "cgn_x_pc2_z"], "CGN_x_PC2_continuous", interaction_rows),
                (["cgn01", "high_pc2", "cgn_x_high_pc2"], "CGN_x_high_PC2", interaction_rows),
            ]:
                for r in school_fe_lpm(dsex, outcome, xcols, school_col):
                    r.update({"outcome": outcome, "sex": sex_name, "model": name})
                    store.append(r)
    return pd.DataFrame(within_rows), pd.DataFrame(interaction_rows), pd.DataFrame(atten_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--db_path", default="/store/talent/db/talent_followup.db")
    ap.add_argument("--out_dir", default="pc2_general_pca_results/cgn_pc2_mechanism_tests")
    ap.add_argument("--base_relation", default="follow_up_full_view")
    ap.add_argument("--y11_table", default="follow_y11")
    ap.add_argument("--id_col", default="BY_ID_Rel")
    ap.add_argument("--school_col", default="by_school_rel")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(args.scores)
    base, y11 = fetch_data(scores, args.db_path, args.base_relation, args.y11_table, args.id_col, args.school_col)
    data = add_outcomes(base, y11)

    pc2_distribution(scores).to_csv(out_dir / "pc2_distribution_by_cgn.csv", index=False)
    profiles = profile_table(data)
    profiles.to_csv(out_dir / "pc2_cgn_outcome_profiles.csv", index=False)
    high_low_contrasts(profiles).to_csv(out_dir / "pc2_cgn_high_low_contrasts.csv", index=False)
    within, interactions, atten = model_tables(data, args.school_col)
    within.to_csv(out_dir / "pc2_within_cgn_models.csv", index=False)
    interactions.to_csv(out_dir / "pc2_cgn_interaction_models.csv", index=False)
    atten.to_csv(out_dir / "pc2_cgn_attenuation_models.csv", index=False)

    print(f"Wrote outcome tables to {out_dir}")


if __name__ == "__main__":
    main()
