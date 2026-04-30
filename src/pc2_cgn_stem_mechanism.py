#!/usr/bin/env python3
"""
pc2_cgn_stem_mechanism.py

Purpose
-------
Test whether PC2 helps explain *which kind* of CGN student moves toward or away
from STEM/engineering.

This is meant to extend the PC1/GCI/CGN framework:
  - PC1/GCI/CGN tells us whether a student is broadly gender-nonconforming
    relative to their sex.
  - PC2 tells us whether that profile is concentrated on the technical/physical
    side (+PC2) versus the social/domestic/expressive side (-PC2).

Main tests
----------
1. Are CGN girls/boys disproportionately high or low on PC2?
2. Among CGN girls, does higher PC2 predict STEM/engineering outcomes?
3. Among CGN boys, does higher PC2 buffer against the usual STEM decline?
4. In sex-separated models, does PC2 interact with CGN?
5. Does adding PC2 attenuate the CGN coefficient? (descriptive, NOT causal mediation)

Inputs
------
--scores_path:
    pc2_general_pca_results/domain_pca_scores.parquet
    Needs pc1, pc2, sex01, and either cgn_pc1 or cgn_bottom20_within_sex.

--id_source_path:
    Original parquet that has BY_ID_Rel in the same row order/index as scores_path.
    Example: SwedenFullRun_noncircular/domain_scores_raw.parquet

--db_path:
    DuckDB database with follow_up_full_view and follow_y11.

Outputs
-------
<out_dir>/pc2_distribution_by_cgn.csv
<out_dir>/pc2_cgn_outcome_profiles.csv
<out_dir>/pc2_within_cgn_models.csv
<out_dir>/pc2_cgn_interaction_models.csv
<out_dir>/pc2_cgn_attenuation_models.csv
<out_dir>/READ_FIRST_girls_cgn_pc2_stem_summary.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

COMMON_MISSING = {98: np.nan, 99: np.nan, 998: np.nan, 999: np.nan}

# -----------------------------
# STEM/engineering definitions
# -----------------------------
ENGINEERING_ITEMS = ["BY_INT005", "BY_INT045", "BY_INT048", "BY_INT068", "BY_INT090"]
STEM_ITEMS = ENGINEERING_ITEMS + ["BY_INT091", "BY_INT067", "BY_INT004", "BY_INT028"]

MAJOR_CODES = {
    "stem_other": [1, 2, 3, 4, 6, 9],
    "engineering": [33, 34],
}
CAREER_CODES = {
    "engineering": [240, 241, 242, 243, 244, 245, 221],
    "stem_other": [211, 230, 232, 310, 316, 317],
}

# Outcomes to emphasize in summaries.
FOCAL_STEM_OUTCOMES = [
    "eng_any_ge_4",
    "stem_any_ge_4",
    "eng_any_ge_5",
    "stem_any_ge_5",
    "major_intent_y1_stem",
    "major_intent_y1_engineering",
    "major_realized_y11_stem",
    "major_realized_y11_engineering",
    "stem_job_y11",
    "engineering_job_y11",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace(COMMON_MISSING)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def choose_cgn_col(df: pd.DataFrame, requested: str = "") -> str:
    candidates = [requested, "cgn_pc1", "cgn_bottom20_within_sex", "cgn", "cgn01"]
    for c in candidates:
        if c and c in df.columns:
            return c
    raise ValueError(
        "Could not find a CGN column. Tried cgn_pc1, cgn_bottom20_within_sex, cgn, cgn01. "
        "Pass --cgn_col explicitly."
    )


def attach_id_if_needed(scores: pd.DataFrame, id_source_path: str, id_col: str) -> pd.DataFrame:
    """
    Attach BY_ID_Rel to the PCA score file.

    The PCA score file usually has fewer rows than the original domain-score file because
    the PCA script drops students with missing domain scores. Pandas/Parquet normally
    preserves the original row index, so the safe attach is often:

        source.loc[scores.index, id_col]

    not a simple length-equal assignment. This function tries, in order:
      1. use id_col if it is already present in scores,
      2. attach by preserved original index if scores.index maps into id_source_path,
      3. attach by equal-length row order only when lengths match.
    """
    out = scores.copy()

    if id_col in out.columns:
        return out

    if not id_source_path:
        raise ValueError(
            f"{id_col} is not in scores_path. Provide --id_source_path to attach it by preserved index."
        )

    src = read_table(id_source_path)
    if id_col not in src.columns:
        raise ValueError(f"{id_col} not found in id_source_path={id_source_path}")

    # Case A: the PCA score file kept the original dataframe index.
    # This is expected when domain_pca_scores.parquet was written after dropna()
    # without reset_index(drop=True).
    try:
        idx = out.index
        simple_range = isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1 and idx.stop == len(out)
        if not simple_range and len(idx) > 0:
            if pd.Index(idx).isin(src.index).all():
                out[id_col] = src.loc[idx, id_col].to_numpy()
                print(
                    f"Attached {id_col} from id_source_path using preserved original index "
                    f"({len(out)} PCA rows mapped into {len(src)} source rows)."
                )
                return out
    except Exception as e:
        print(f"Preserved-index ID attach failed; trying other methods. Reason: {type(e).__name__}: {e}")

    # Case B: equal length fallback. Only safe when the two files have exactly the same rows.
    if len(src) == len(out):
        out[id_col] = src[id_col].to_numpy()
        print(f"Attached {id_col} from id_source_path by equal-length row order.")
        return out

    # Case C: hidden parquet index column fallback.
    for idx_col in ["__index_level_0__", "index", "level_0", "orig_index", "row_index"]:
        if idx_col in out.columns:
            key = pd.to_numeric(out[idx_col], errors="coerce")
            if key.notna().all() and key.astype(int).between(0, len(src) - 1).all():
                out[id_col] = src.iloc[key.astype(int).to_numpy()][id_col].to_numpy()
                print(f"Attached {id_col} from id_source_path using {idx_col}.")
                return out

    raise ValueError(
        f"id_source_path has {len(src)} rows but scores_path has {len(out)} rows, "
        f"and {id_col} is not in scores_path. The PCA score file appears to have lost the original row index.\n"
        "Best fix: recreate domain_pca_scores.parquet with BY_ID_Rel included, or run the PC2 PCA script on "
        "a domain_scores file that keeps BY_ID_Rel through the dropna step. You can also pass a scores file "
        "that already contains BY_ID_Rel."
    )

def add_pc2_groups(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Add PC2 high/low indicators within sex.

    IMPORTANT: high_pc2 is NOT a second CGN indicator. It means technical/physical
    PC2 side. low_pc2 means social/domestic/expressive PC2 side.
    """
    out = df.copy()
    out["pc1_z"] = zscore(out["pc1"])
    out["pc2_z"] = zscore(out["pc2"])
    out["pc1_x_pc2_z"] = out["pc1_z"] * out["pc2_z"]

    out["pc2_high"] = np.nan
    out["pc2_low"] = np.nan
    out["pc2_group"] = pd.Series(pd.NA, index=out.index, dtype="object")

    for sex in [0, 1]:
        mask = out["sex01"] == sex
        vals = out.loc[mask, "pc2"].dropna()
        if vals.empty:
            continue
        lo = vals.quantile(q)
        hi = vals.quantile(1 - q)
        out.loc[mask, "pc2_high"] = (out.loc[mask, "pc2"] >= hi).astype(float)
        out.loc[mask, "pc2_low"] = (out.loc[mask, "pc2"] <= lo).astype(float)
        out.loc[mask & (out["pc2"] <= lo), "pc2_group"] = "low_PC2_social_domestic"
        out.loc[mask & (out["pc2"] > lo) & (out["pc2"] < hi), "pc2_group"] = "middle_PC2"
        out.loc[mask & (out["pc2"] >= hi), "pc2_group"] = "high_PC2_technical_physical"

    out["cgn_label"] = out["cgn01"].map({0: "non_CGN", 1: "CGN"})
    out["sex_label"] = out["sex01"].map({0: "female", 1: "male"})
    out["cgn_pc2_cell"] = out["sex_label"].astype(str) + "__" + out["cgn_label"].astype(str) + "__" + out["pc2_group"].astype(str)
    return out


def within_demean(df: pd.DataFrame, cols: list[str], fe_col: str) -> pd.DataFrame:
    out = df.copy()
    out[fe_col] = out[fe_col].astype(str)
    g = out.groupby(fe_col, observed=True, sort=False)
    for c in cols:
        out[c] = out[c] - g[c].transform("mean")
    return out


def fe_ols_clustered(df: pd.DataFrame, y: str, x: list[str], fe: str, cluster: str):
    use_cols = [y, fe] + x
    if cluster != fe:
        use_cols.append(cluster)
    use_cols = list(dict.fromkeys(use_cols))
    d = df[use_cols].dropna().copy()
    if len(d) < 50:
        return None, len(d)
    if d[fe].nunique(dropna=True) < 2:
        return None, len(d)
    if any(d[c].nunique(dropna=True) < 2 for c in x):
        return None, len(d)
    if d[y].nunique(dropna=True) < 2:
        return None, len(d)

    d = within_demean(d, cols=[y] + x, fe_col=fe)
    Y = d[y].to_numpy(dtype=float)
    X = d[x].to_numpy(dtype=float)
    groups = d[cluster] if cluster in d.columns else d[fe]
    try:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        return None, len(d)
    return res, len(d)


def plain_ols_clustered(df: pd.DataFrame, y: str, x: list[str], cluster: str | None = None):
    use_cols = [y] + x + ([cluster] if cluster else [])
    use_cols = list(dict.fromkeys([c for c in use_cols if c]))
    d = df[use_cols].dropna().copy()
    if len(d) < 50:
        return None, len(d)
    if any(d[c].nunique(dropna=True) < 2 for c in x):
        return None, len(d)
    if d[y].nunique(dropna=True) < 2:
        return None, len(d)
    X = sm.add_constant(d[x].astype(float), has_constant="add")
    Y = d[y].astype(float)
    try:
        if cluster and cluster in d.columns:
            res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
        else:
            res = sm.OLS(Y, X).fit()
    except Exception:
        return None, len(d)
    return res, len(d)


def rows_from_model(res, n_used: int, outcome: str, sex_label: str, model_name: str, estimator: str, terms: Iterable[str]) -> list[dict]:
    rows = []
    if res is None:
        rows.append({
            "outcome": outcome, "sex": sex_label, "model": model_name, "estimator": estimator,
            "term": "MODEL_FAILED_OR_INSUFFICIENT_DATA", "n_used": n_used,
            "coef": np.nan, "p_value": np.nan
        })
        return rows
    for term in terms:
        if term in res.params.index:
            rows.append({
                "outcome": outcome,
                "sex": sex_label,
                "model": model_name,
                "estimator": estimator,
                "term": term,
                "n_used": n_used,
                "coef": float(res.params[term]),
                "p_value": float(res.pvalues[term]),
            })
        else:
            # FE model with numpy matrix may have integer params; handled elsewhere if needed.
            pass
    return rows


def rows_from_fe_model(res, n_used: int, outcome: str, sex_label: str, model_name: str, terms: list[str]) -> list[dict]:
    rows = []
    if res is None:
        rows.append({
            "outcome": outcome, "sex": sex_label, "model": model_name, "estimator": "school_fe_ols",
            "term": "MODEL_FAILED_OR_INSUFFICIENT_DATA", "n_used": n_used,
            "coef": np.nan, "p_value": np.nan,
            "note": "For binary outcomes, coefficient is percentage-point change in an LPM."
        })
        return rows
    for i, term in enumerate(terms):
        rows.append({
            "outcome": outcome,
            "sex": sex_label,
            "model": model_name,
            "estimator": "school_fe_ols",
            "term": term,
            "n_used": n_used,
            "coef": float(res.params[i]),
            "p_value": float(res.pvalues[i]),
            "note": "For binary outcomes, coefficient is percentage-point change in an LPM."
        })
    return rows


def any_high(df: pd.DataFrame, cols: list[str], cutoff: float) -> pd.Series:
    block = df[cols].apply(clean_numeric)
    any_obs = block.notna().any(axis=1)
    any_hi = (block >= cutoff).any(axis=1)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[any_obs] = 0.0
    out.loc[any_obs & any_hi] = 1.0
    return out


def code_group_flag(raw: pd.Series, keep_codes: list[int]) -> pd.Series:
    x = pd.to_numeric(raw, errors="coerce").replace(COMMON_MISSING).astype("Int64")
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    known = x.notna()
    out.loc[known] = 0.0
    out.loc[known & x.isin(keep_codes)] = 1.0
    return out


def build_scores_for_sql(scores: pd.DataFrame, out_dir: Path, id_col: str) -> Path:
    keep = [
        id_col, "sex01", "cgn01",
        "sex_label", "cgn_label", "cgn_pc2_cell",
        "pc1", "pc2", "pc1_z", "pc2_z", "pc1_x_pc2_z",
        "pc2_high", "pc2_low", "pc2_group",
    ]
    if "gci_pc1" in scores.columns:
        keep.append("gci_pc1")
    keep = [c for c in keep if c in scores.columns]
    path = out_dir / "_scores_with_pc2_groups.parquet"
    scores[keep].to_parquet(path, index=False)
    return path


def fetch_base_data(con, scores_tmp: Path, id_col: str, school_col: str, relation: str) -> pd.DataFrame:
    interest_cols = sorted(set(STEM_ITEMS))
    select_interest = ",\n      ".join([f"b.{c} AS {c}" for c in interest_cols])
    sql = f"""
    SELECT
      s.*,
      b.{school_col} AS {school_col},
      {select_interest},
      b.Y1_E320 AS Y1_E320,
      b.Y11_E321 AS Y11_E321
    FROM read_parquet('{scores_tmp}') s
    JOIN {relation} b
      ON s.{id_col} = b.{id_col}
    WHERE b.{school_col} IS NOT NULL
    """
    return con.execute(sql).df()


def fetch_y11_data(con, scores_tmp: Path, id_col: str, school_col: str, y11_table: str, y11_qstat_value: str) -> pd.DataFrame:
    qstat = ""
    if y11_qstat_value.strip():
        v = y11_qstat_value.strip().replace("'", "''")
        qstat = f"AND y.Y11_QSTAT = '{v}'"
    else:
        qstat = "AND y.Y11_QSTAT IS NOT NULL"
    sql = f"""
    SELECT
      s.*,
      y.{school_col} AS {school_col},
      y.Y11_QSTAT,
      y.Y11_O202,
      y.Y11_O221,
      y.Y11_O222,
      y.Y11_P201,
      y.Y11_P202
    FROM read_parquet('{scores_tmp}') s
    JOIN {y11_table} y
      ON s.{id_col} = y.{id_col}
    WHERE y.{school_col} IS NOT NULL
      {qstat}
    """
    return con.execute(sql).df()


def add_base_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in STEM_ITEMS:
        out[c] = clean_numeric(out[c])
    out["eng_any_ge_4"] = any_high(out, ENGINEERING_ITEMS, 4.0)
    out["stem_any_ge_4"] = any_high(out, STEM_ITEMS, 4.0)
    out["eng_any_ge_5"] = any_high(out, ENGINEERING_ITEMS, 5.0)
    out["stem_any_ge_5"] = any_high(out, STEM_ITEMS, 5.0)
    stem_major_codes = MAJOR_CODES["engineering"] + MAJOR_CODES["stem_other"]
    eng_major_codes = MAJOR_CODES["engineering"]
    out["major_intent_y1_stem"] = code_group_flag(out["Y1_E320"], stem_major_codes)
    out["major_intent_y1_engineering"] = code_group_flag(out["Y1_E320"], eng_major_codes)
    out["major_realized_y11_stem"] = code_group_flag(out["Y11_E321"], stem_major_codes)
    out["major_realized_y11_engineering"] = code_group_flag(out["Y11_E321"], eng_major_codes)
    return out


def add_y11_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stem_career_codes = CAREER_CODES["engineering"] + CAREER_CODES["stem_other"]
    eng_career_codes = CAREER_CODES["engineering"]
    out["stem_job_y11"] = code_group_flag(out["Y11_O202"], stem_career_codes)
    out["engineering_job_y11"] = code_group_flag(out["Y11_O202"], eng_career_codes)
    out["earnings_o221_among_earners"] = clean_numeric(out["Y11_O221"])
    out.loc[~(out["earnings_o221_among_earners"] > 0), "earnings_o221_among_earners"] = np.nan
    out["log_earnings_o222_among_earners"] = clean_numeric(out["Y11_O222"])
    out.loc[~(out["log_earnings_o222_among_earners"] > 0), "log_earnings_o222_among_earners"] = np.nan

    # Marriage variables are secondary/contextual.
    status = clean_numeric(out["Y11_P201"])
    times = clean_numeric(out["Y11_P202"])
    ever = pd.Series(np.nan, index=out.index, dtype="float64")
    ever.loc[times == 0] = 0.0
    ever.loc[(times >= 1) | status.isin([1, 2, 3, 4])] = 1.0
    out["ever_married"] = ever
    divorced = pd.Series(np.nan, index=out.index, dtype="float64")
    keep_ever = ever == 1.0
    divorced.loc[keep_ever & status.isin([1, 2, 3, 4])] = (status.loc[keep_ever & status.isin([1, 2, 3, 4])] == 3).astype(float)
    out["divorced_among_ever_married"] = divorced
    return out


def pc2_distribution_by_cgn(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sex_val, sex_label in [(0, "female"), (1, "male")]:
        dsex = scores[scores["sex01"] == sex_val].copy()
        for cgn_val, cgn_label in [(0, "non_CGN"), (1, "CGN")]:
            d = dsex[dsex["cgn01"] == cgn_val].copy()
            rows.append({
                "sex": sex_label,
                "cgn": cgn_label,
                "n": len(d),
                "mean_pc1": d["pc1"].mean(),
                "mean_pc2": d["pc2"].mean(),
                "sd_pc2": d["pc2"].std(ddof=0),
                "median_pc2": d["pc2"].median(),
                "prop_high_pc2": d["pc2_high"].mean(),
                "prop_low_pc2": d["pc2_low"].mean(),
            })
        # CGN-vs-non Cohen's d and correlation within sex
        non = dsex[dsex["cgn01"] == 0]["pc2"].dropna()
        cgn = dsex[dsex["cgn01"] == 1]["pc2"].dropna()
        pooled = np.sqrt((non.var(ddof=0) + cgn.var(ddof=0)) / 2) if len(non) and len(cgn) else np.nan
        dval = (cgn.mean() - non.mean()) / pooled if pooled and not pd.isna(pooled) else np.nan
        corr = dsex[["cgn01", "pc2"]].dropna()["cgn01"].corr(dsex[["cgn01", "pc2"]].dropna()["pc2"])
        rows.append({
            "sex": sex_label,
            "cgn": "CGN_minus_nonCGN_summary",
            "n": len(dsex),
            "mean_pc1": np.nan,
            "mean_pc2": np.nan,
            "sd_pc2": np.nan,
            "median_pc2": np.nan,
            "prop_high_pc2": np.nan,
            "prop_low_pc2": np.nan,
            "cohens_d_pc2_CGN_minus_nonCGN": dval,
            "corr_cgn_pc2": corr,
        })
    return pd.DataFrame(rows)


def outcome_profiles(df: pd.DataFrame, outcomes: list[str], school_col: str) -> pd.DataFrame:
    df = df.copy()
    # Defensive rebuild in case label columns were dropped during SQL/temp-parquet joins.
    if "sex_label" not in df.columns and "sex01" in df.columns:
        df["sex_label"] = df["sex01"].map({0: "female", 1: "male"})
    if "cgn_label" not in df.columns and "cgn01" in df.columns:
        df["cgn_label"] = df["cgn01"].map({0: "non_CGN", 1: "CGN"})
    rows = []
    group_cols = ["sex_label", "cgn_label", "pc2_group"]
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        use = df[group_cols + [outcome, "pc1", "pc2", "pc2_high", "pc2_low"]].dropna(subset=[outcome, "sex_label", "cgn_label", "pc2_group"])
        if use.empty:
            continue
        for keys, sub in use.groupby(group_cols, observed=True, sort=False):
            sex_label, cgn_label, pc2_group = keys
            rows.append({
                "outcome": outcome,
                "sex": sex_label,
                "cgn": cgn_label,
                "pc2_group": pc2_group,
                "n": len(sub),
                "outcome_mean": sub[outcome].mean(),
                "mean_pc1": sub["pc1"].mean(),
                "mean_pc2": sub["pc2"].mean(),
                "prop_pc2_high": sub["pc2_high"].mean(),
                "prop_pc2_low": sub["pc2_low"].mean(),
            })
    return pd.DataFrame(rows)


def within_cgn_models(df: pd.DataFrame, outcomes: list[str], school_col: str) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for sex_val, sex_label in [(0, "female"), (1, "male")]:
            for cgn_val, cgn_label in [(0, "non_CGN"), (1, "CGN")]:
                sub = df[(df["sex01"] == sex_val) & (df["cgn01"] == cgn_val)].copy()
                # model: y ~ pc2_z + school FE
                res, n_used = fe_ols_clustered(sub, y=outcome, x=["pc2_z"], fe=school_col, cluster=school_col)
                if res is None:
                    rows.append({
                        "outcome": outcome,
                        "sex": sex_label,
                        "cgn": cgn_label,
                        "model": "within_group_y_on_pc2_z",
                        "term": "MODEL_FAILED_OR_INSUFFICIENT_DATA",
                        "n_used": n_used,
                        "coef": np.nan,
                        "p_value": np.nan,
                    })
                else:
                    rows.append({
                        "outcome": outcome,
                        "sex": sex_label,
                        "cgn": cgn_label,
                        "model": "within_group_y_on_pc2_z",
                        "term": "pc2_z",
                        "n_used": n_used,
                        "coef": float(res.params[0]),
                        "p_value": float(res.pvalues[0]),
                        "note": "Within sex and CGN group; school FE. For binary y, coef is pp per 1 SD PC2."
                    })
    return pd.DataFrame(rows)


def interaction_and_attenuation_models(df: pd.DataFrame, outcomes: list[str], school_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_rows = []
    attenuation_rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for sex_val, sex_label in [(0, "female"), (1, "male")]:
            sub = df[df["sex01"] == sex_val].copy()
            sub["cgn_x_pc2_z"] = sub["cgn01"].astype(float) * sub["pc2_z"].astype(float)
            sub["cgn_x_pc2_high"] = sub["cgn01"].astype(float) * sub["pc2_high"].astype(float)

            # A) interaction with continuous PC2
            terms = ["cgn01", "pc2_z", "cgn_x_pc2_z"]
            res, n_used = fe_ols_clustered(sub, y=outcome, x=terms, fe=school_col, cluster=school_col)
            interaction_rows.extend(rows_from_fe_model(res, n_used, outcome, sex_label, "CGN_x_PC2_continuous", terms))

            # B) interaction with high PC2 indicator
            terms2 = ["cgn01", "pc2_high", "cgn_x_pc2_high"]
            res2, n_used2 = fe_ols_clustered(sub, y=outcome, x=terms2, fe=school_col, cluster=school_col)
            interaction_rows.extend(rows_from_fe_model(res2, n_used2, outcome, sex_label, "CGN_x_high_PC2_indicator", terms2))

            # C) attenuation: CGN only -> CGN + PC2 -> CGN + PC2 + interaction
            for model_name, terms3 in [
                ("M1_CGN_only", ["cgn01"]),
                ("M2_CGN_plus_PC2", ["cgn01", "pc2_z"]),
                ("M3_CGN_PC2_interaction", ["cgn01", "pc2_z", "cgn_x_pc2_z"]),
            ]:
                res3, n_used3 = fe_ols_clustered(sub, y=outcome, x=terms3, fe=school_col, cluster=school_col)
                attenuation_rows.extend(rows_from_fe_model(res3, n_used3, outcome, sex_label, model_name, terms3))
    return pd.DataFrame(interaction_rows), pd.DataFrame(attenuation_rows)


def high_low_contrasts(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    For each sex/outcome, compute simple contrasts:
    - Among CGN: high_PC2 minus low_PC2
    - Among non-CGN: high_PC2 minus low_PC2
    - CGN high_PC2 minus non-CGN high_PC2
    - CGN low_PC2 minus non-CGN low_PC2
    """
    rows = []
    if profiles.empty:
        return pd.DataFrame()
    for (outcome, sex), d in profiles.groupby(["outcome", "sex"], observed=True):
        def get(cgn, pc2_group):
            v = d[(d["cgn"] == cgn) & (d["pc2_group"] == pc2_group)]["outcome_mean"]
            return float(v.iloc[0]) if len(v) else np.nan
        cgn_hi = get("CGN", "high_PC2_technical_physical")
        cgn_lo = get("CGN", "low_PC2_social_domestic")
        non_hi = get("non_CGN", "high_PC2_technical_physical")
        non_lo = get("non_CGN", "low_PC2_social_domestic")
        rows.extend([
            {"outcome": outcome, "sex": sex, "contrast": "among_CGN_highPC2_minus_lowPC2", "difference": cgn_hi - cgn_lo},
            {"outcome": outcome, "sex": sex, "contrast": "among_nonCGN_highPC2_minus_lowPC2", "difference": non_hi - non_lo},
            {"outcome": outcome, "sex": sex, "contrast": "highPC2_CGN_minus_nonCGN", "difference": cgn_hi - non_hi},
            {"outcome": outcome, "sex": sex, "contrast": "lowPC2_CGN_minus_nonCGN", "difference": cgn_lo - non_lo},
        ])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_path", required=True)
    ap.add_argument("--id_source_path", default="")
    ap.add_argument("--db_path", default="/store/talent/db/talent_followup.db")
    ap.add_argument("--out_dir", default="pc2_cgn_mechanism_out")
    ap.add_argument("--id_col", default="BY_ID_Rel")
    ap.add_argument("--school_col", default="by_school_rel")
    ap.add_argument("--base_relation", default="follow_up_full_view")
    ap.add_argument("--y11_table", default="follow_y11")
    ap.add_argument("--y11_qstat_value", default="01")
    ap.add_argument("--cgn_col", default="")
    ap.add_argument("--pc2_extreme_q", type=float, default=0.25, help="Top/bottom q within sex for high/low PC2 groups.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = read_table(args.scores_path)
    scores = attach_id_if_needed(scores, args.id_source_path, args.id_col)
    cgn_col = choose_cgn_col(scores, args.cgn_col)
    scores = scores.rename(columns={cgn_col: "cgn01"}).copy()

    needed = [args.id_col, "sex01", "cgn01", "pc1", "pc2"]
    missing = [c for c in needed if c not in scores.columns]
    if missing:
        raise ValueError(f"scores_path missing required columns: {missing}")

    scores["sex01"] = clean_numeric(scores["sex01"]).astype("Int64")
    scores["cgn01"] = clean_numeric(scores["cgn01"]).astype("Int64")
    scores = scores[scores["sex01"].isin([0, 1]) & scores["cgn01"].isin([0, 1])].copy()
    scores = add_pc2_groups(scores, args.pc2_extreme_q)

    # Save PC2 distribution by CGN immediately.
    dist = pc2_distribution_by_cgn(scores)
    dist.to_csv(out_dir / "pc2_distribution_by_cgn.csv", index=False)

    scores_tmp = build_scores_for_sql(scores, out_dir, args.id_col)

    con = duckdb.connect(args.db_path)
    base = fetch_base_data(con, scores_tmp, args.id_col, args.school_col, args.base_relation)
    y11 = fetch_y11_data(con, scores_tmp, args.id_col, args.school_col, args.y11_table, args.y11_qstat_value)
    con.close()

    base = add_base_outcomes(base)
    y11 = add_y11_outcomes(y11)

    base_outcomes = [
        "eng_any_ge_4", "stem_any_ge_4", "eng_any_ge_5", "stem_any_ge_5",
        "major_intent_y1_stem", "major_intent_y1_engineering",
        "major_realized_y11_stem", "major_realized_y11_engineering",
    ] + [f"interest_{c}" for c in []]
    y11_outcomes = ["stem_job_y11", "engineering_job_y11", "earnings_o221_among_earners", "log_earnings_o222_among_earners", "ever_married", "divorced_among_ever_married"]

    # Profiles
    prof_base = outcome_profiles(base, base_outcomes, args.school_col)
    prof_y11 = outcome_profiles(y11, y11_outcomes, args.school_col)
    profiles = pd.concat([prof_base, prof_y11], ignore_index=True)
    profiles.to_csv(out_dir / "pc2_cgn_outcome_profiles.csv", index=False)

    contrasts = high_low_contrasts(profiles)
    contrasts.to_csv(out_dir / "pc2_cgn_high_low_contrasts.csv", index=False)

    # Models
    within_base = within_cgn_models(base, base_outcomes, args.school_col)
    within_y11 = within_cgn_models(y11, y11_outcomes, args.school_col)
    within = pd.concat([within_base, within_y11], ignore_index=True)
    within.to_csv(out_dir / "pc2_within_cgn_models.csv", index=False)

    inter_base, atten_base = interaction_and_attenuation_models(base, base_outcomes, args.school_col)
    inter_y11, atten_y11 = interaction_and_attenuation_models(y11, y11_outcomes, args.school_col)
    interaction = pd.concat([inter_base, inter_y11], ignore_index=True)
    attenuation = pd.concat([atten_base, atten_y11], ignore_index=True)
    interaction.to_csv(out_dir / "pc2_cgn_interaction_models.csv", index=False)
    attenuation.to_csv(out_dir / "pc2_cgn_attenuation_models.csv", index=False)

    # Summary file for main thesis question: girls + STEM outcomes.
    girls_cgn_pc2 = within[
        (within["sex"] == "female")
        & (within["cgn"] == "CGN")
        & (within["outcome"].isin(FOCAL_STEM_OUTCOMES))
        & (within["term"] == "pc2_z")
    ].copy()
    girls_cgn_pc2.to_csv(out_dir / "READ_FIRST_girls_CGN_pc2_predicts_STEM.csv", index=False)

    girls_interaction = interaction[
        (interaction["sex"] == "female")
        & (interaction["outcome"].isin(FOCAL_STEM_OUTCOMES))
        & (interaction["term"].isin(["cgn_x_pc2_z", "cgn_x_pc2_high", "pc2_z", "pc2_high", "cgn01"]))
    ].copy()
    girls_interaction.to_csv(out_dir / "READ_FIRST_girls_CGN_x_PC2_interactions.csv", index=False)

    stem_contrasts = contrasts[
        (contrasts["sex"] == "female")
        & (contrasts["outcome"].isin(FOCAL_STEM_OUTCOMES))
    ].copy()
    stem_contrasts.to_csv(out_dir / "READ_FIRST_girls_CGN_PC2_high_low_contrasts.csv", index=False)

    print("Rows in scores:", len(scores))
    print("Rows with base-year data:", len(base))
    print("Rows with Y11 data:", len(y11))
    print("CGN column used:", cgn_col)
    print("PC2 extreme quantile:", args.pc2_extreme_q)
    print("\nWrote:")
    for fn in [
        "pc2_distribution_by_cgn.csv",
        "pc2_cgn_outcome_profiles.csv",
        "pc2_cgn_high_low_contrasts.csv",
        "pc2_within_cgn_models.csv",
        "pc2_cgn_interaction_models.csv",
        "pc2_cgn_attenuation_models.csv",
        "READ_FIRST_girls_CGN_pc2_predicts_STEM.csv",
        "READ_FIRST_girls_CGN_x_PC2_interactions.csv",
        "READ_FIRST_girls_CGN_PC2_high_low_contrasts.csv",
    ]:
        print(" ", out_dir / fn)

    print("\nMain interpretation:")
    print("  1. pc2_distribution_by_cgn: Are CGN girls shifted toward high PC2?")
    print("  2. READ_FIRST_girls_CGN_pc2_predicts_STEM: Among CGN girls, does +PC2 predict STEM?")
    print("  3. READ_FIRST_girls_CGN_x_PC2_interactions: Does PC2 matter more for CGN girls than non-CGN girls?")
    print("  4. high-low contrasts: Is CGN+highPC2 the highest-STEM cell?")


if __name__ == "__main__":
    main()
