#!/usr/bin/env python3
"""
pc2_stem_outcome_suite.py

Purpose
-------
Test whether Project TALENT PC2 predicts STEM/engineering pathways after
controlling for PC1.

This is designed as the next step after running:
    project_talent_pc2_and_general_pca.py

It uses the PC scores from:
    pc2_general_pca_results/domain_pca_scores.parquet

and rebuilds the same kinds of outcomes used in the thesis/code:
  - base-year engineering/STEM interest items
  - high engineering/STEM interest cohorts, using cutoffs >=4 and >=5
  - intended STEM/engineering major
  - realized STEM/engineering major
  - STEM/engineering job from Y11 occupation codes
  - follow-through from intended STEM/engineering major to STEM/engineering job
  - earnings among earners (O221 and O222)
  - ever married and divorced among ever-married

Core question
-------------
Among girls, after controlling for PC1 (overall male-typed/female-typed axis),
does PC2 (technical/physical vs social/domestic/expressive profile) predict
STEM/engineering outcomes?

Main result to inspect
----------------------
Open:
    <out_dir>/pc2_outcome_coefficients.csv

Look for rows where:
    subset == 'female'
    model == 'PC1_plus_PC2'
    term == 'pc2_z'

For binary STEM/engineering outcomes, a positive pc2_z coefficient means:
    among girls with similar PC1, higher PC2 students are more likely to enter
    that STEM/engineering pathway.

Optional ID behavior
--------------------
The PC2 score parquet may or may not contain BY_ID_Rel. If it does not, pass
--id_source_path pointing to the original domain score parquet used to create
PC2, and this script will attach BY_ID_Rel by matching the pandas index.

Example
-------
python -u pc2_stem_outcome_suite.py \
  --scores_path pc2_general_pca_results/domain_pca_scores.parquet \
  --id_source_path SwedenFullRun_noncircular/domain_scores_raw.parquet \
  --db_path /store/talent/db/talent_followup.db \
  --out_dir pc2_general_pca_results/stem_outcome_tests
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

COMMON_MISSING = {98: np.nan, 99: np.nan, 998: np.nan, 999: np.nan}


# -------------------------------------------------------------------
# User/project mappings
# -------------------------------------------------------------------

ENGINEERING_INTEREST_ITEMS = {
    "BY_INT005": "Civil engineer",
    "BY_INT045": "Chemical engineer",
    "BY_INT048": "Aeronautical engineer",
    "BY_INT068": "Electrical engineer",
    "BY_INT090": "Mechanical engineer",
}

STEM_INTEREST_ITEMS = {
    **ENGINEERING_INTEREST_ITEMS,
    "BY_INT091": "Mathematician",
    "BY_INT067": "Biologist",
    "BY_INT004": "Chemist",
    "BY_INT028": "Research scientist",
}

DEFAULT_CAREER_CODES = pd.DataFrame([
    # engineering
    (240, "engineering", "Engineer (NEC)"),
    (241, "engineering", "Civil/Hydraulic Engineer"),
    (242, "engineering", "Electrical/Electronic Engineer"),
    (243, "engineering", "Mechanical/Automotive Engineer"),
    (244, "engineering", "Aeronautical Engineer"),
    (245, "engineering", "Chemical Engineer"),
    (221, "engineering", "Systems Analyst (Computer)"),
    # other STEM
    (211, "stem_other", "Mathematician"),
    (230, "stem_other", "Scientist or Physical Scientist"),
    (232, "stem_other", "Physicist"),
    (310, "stem_other", "Biologist/Zoologist/Botanist/Paleontologist"),
    (316, "stem_other", "Microbiologist"),
    (317, "stem_other", "Biochemist"),
], columns=["code", "group", "label"])

DEFAULT_MAJOR_CODES = pd.DataFrame([
    (1, "stem_other", "Math"),
    (2, "stem_other", "Chemistry"),
    (3, "stem_other", "Physics"),
    (4, "stem_other", "Physical science"),
    (6, "stem_other", "Biochemistry"),
    (9, "stem_other", "Biological Science"),
    (33, "engineering", "Engineering"),
    (34, "engineering", "Computer Science"),
], columns=["code", "group", "label"])


# -------------------------------------------------------------------
# I/O and cleaning helpers
# -------------------------------------------------------------------

def q(name: str) -> str:
    """DuckDB-safe quoted identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace(COMMON_MISSING)


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def load_mapping(path: str, default: pd.DataFrame) -> pd.DataFrame:
    if path and Path(path).exists():
        out = pd.read_csv(path)
    else:
        out = default.copy()
    if "code" not in out.columns or "group" not in out.columns:
        raise ValueError("Mapping files must include at least columns: code, group")
    out["code"] = pd.to_numeric(out["code"], errors="coerce").astype("Int64")
    return out.dropna(subset=["code"]).copy()


def prepare_scores(scores_path: str, id_col: str, id_source_path: str = "") -> pd.DataFrame:
    scores = read_table(scores_path).copy()

    # If parquet restored an ID index, make it a real column.
    if id_col not in scores.columns and scores.index.name == id_col:
        scores = scores.reset_index()

    # If the PC2 script did not preserve ID, attach it from the original source by index.
    if id_col not in scores.columns and id_source_path:
        src = read_table(id_source_path)
        if id_col not in src.columns and src.index.name == id_col:
            src = src.reset_index()
        if id_col not in src.columns:
            raise ValueError(f"id_source_path does not contain {id_col}")
        # Align by pandas index from the PC2 construction.
        scores[id_col] = src.loc[scores.index, id_col].to_numpy()

    if id_col not in scores.columns:
        raise ValueError(
            f"Could not find {id_col} in scores_path. "
            f"Pass --id_source_path pointing to the original domain_scores_raw.parquet, "
            f"or rerun the PC2 script after preserving {id_col}."
        )

    required = ["pc1", "pc2", "sex01"]
    missing = [c for c in required if c not in scores.columns]
    if missing:
        raise ValueError(f"scores_path is missing required columns: {missing}")

    if "cgn_pc1" not in scores.columns:
        # Fallback for older score files.
        for c in ["cgn_bottom20_within_sex", "cgn", "cgn01"]:
            if c in scores.columns:
                scores["cgn_pc1"] = scores[c]
                break
    if "gci_pc1" not in scores.columns:
        for c in ["gci", "gci_pc1"]:
            if c in scores.columns:
                scores["gci_pc1"] = scores[c]
                break

    scores["sex01"] = clean_numeric(scores["sex01"]).astype("Int64")
    scores = scores[scores["sex01"].isin([0, 1])].copy()
    scores["pc1_z"] = zscore(scores["pc1"])
    scores["pc2_z"] = zscore(scores["pc2"])
    scores["pc1_x_pc2_z"] = scores["pc1_z"] * scores["pc2_z"]
    scores["abs_pc2_z"] = scores["pc2_z"].abs()

    # Keep only columns needed downstream plus useful extras.
    keep = [id_col, "sex01", "pc1", "pc2", "pc1_z", "pc2_z", "pc1_x_pc2_z", "abs_pc2_z"]
    for c in ["gci_pc1", "cgn_pc1"]:
        if c in scores.columns:
            keep.append(c)
    keep = list(dict.fromkeys(keep))
    return scores[keep].copy()


# -------------------------------------------------------------------
# SQL fetchers
# -------------------------------------------------------------------

def fetch_base(con: duckdb.DuckDBPyConnection, scores: pd.DataFrame, args, needed_cols: Iterable[str]) -> pd.DataFrame:
    con.register("scores_df", scores)
    cols = sorted(set([c for c in needed_cols if c]))
    select_cols = ["s.*", f"b.{q(args.school_col)} AS {q(args.school_col)}"]
    select_cols += [f"b.{q(c)} AS {q(c)}" for c in cols]
    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM scores_df s
    JOIN {args.base_relation} b
      ON CAST(s.{q(args.id_col)} AS VARCHAR) = CAST(b.{q(args.id_col)} AS VARCHAR)
    WHERE b.{q(args.school_col)} IS NOT NULL
    """
    return con.execute(sql).df()


def fetch_y11(con: duckdb.DuckDBPyConnection, scores: pd.DataFrame, args, needed_cols: Iterable[str]) -> pd.DataFrame:
    con.register("scores_df", scores)
    cols = sorted(set([c for c in needed_cols if c]))
    select_cols = ["s.*", f"y.{q(args.school_col)} AS {q(args.school_col)}"]
    select_cols += [f"y.{q(c)} AS {q(c)}" for c in cols]
    where = f"WHERE y.{q(args.school_col)} IS NOT NULL"
    if args.y11_qstat_value.strip():
        v = args.y11_qstat_value.strip().replace("'", "''")
        where += f" AND CAST(y.{q(args.y11_qstat_col)} AS VARCHAR) = '{v}'"
    elif args.require_y11_qstat:
        where += f" AND y.{q(args.y11_qstat_col)} IS NOT NULL"
    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM scores_df s
    JOIN {args.y11_table} y
      ON CAST(s.{q(args.id_col)} AS VARCHAR) = CAST(y.{q(args.id_col)} AS VARCHAR)
    {where}
    """
    return con.execute(sql).df()


def fetch_joined(con: duckdb.DuckDBPyConnection, scores: pd.DataFrame, args, base_cols: Iterable[str], y11_cols: Iterable[str]) -> pd.DataFrame:
    con.register("scores_df", scores)
    bcols = sorted(set([c for c in base_cols if c]))
    ycols = sorted(set([c for c in y11_cols if c]))
    select_cols = ["s.*", f"y.{q(args.school_col)} AS {q(args.school_col)}"]
    select_cols += [f"b.{q(c)} AS {q(c)}" for c in bcols]
    select_cols += [f"y.{q(c)} AS {q(c)}" for c in ycols]
    where = f"WHERE y.{q(args.school_col)} IS NOT NULL"
    if args.y11_qstat_value.strip():
        v = args.y11_qstat_value.strip().replace("'", "''")
        where += f" AND CAST(y.{q(args.y11_qstat_col)} AS VARCHAR) = '{v}'"
    elif args.require_y11_qstat:
        where += f" AND y.{q(args.y11_qstat_col)} IS NOT NULL"
    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM scores_df s
    JOIN {args.base_relation} b
      ON CAST(s.{q(args.id_col)} AS VARCHAR) = CAST(b.{q(args.id_col)} AS VARCHAR)
    JOIN {args.y11_table} y
      ON CAST(s.{q(args.id_col)} AS VARCHAR) = CAST(y.{q(args.id_col)} AS VARCHAR)
    {where}
    """
    return con.execute(sql).df()


# -------------------------------------------------------------------
# Outcome builders
# -------------------------------------------------------------------

def any_high(df: pd.DataFrame, cols: list[str], cutoff: float) -> pd.Series:
    present_cols = [c for c in cols if c in df.columns]
    if not present_cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    block = df[present_cols].apply(clean_numeric)
    any_obs = block.notna().any(axis=1)
    any_hi = (block >= cutoff).any(axis=1)
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y.loc[any_obs] = 0.0
    y.loc[any_obs & any_hi] = 1.0
    return y


def code_group_flag(code_series: pd.Series, mapping: pd.DataFrame, group: str) -> pd.Series:
    x = pd.to_numeric(code_series, errors="coerce").replace(COMMON_MISSING).astype("Int64")
    if group == "engineering":
        codes = set(mapping.loc[mapping["group"] == "engineering", "code"].dropna().astype(int).tolist())
    elif group == "stem":
        codes = set(mapping.loc[mapping["group"].isin(["engineering", "stem_other"]), "code"].dropna().astype(int).tolist())
    else:
        raise ValueError(f"Unknown group: {group}")
    y = pd.Series(np.nan, index=x.index, dtype="float64")
    known = x.notna()
    y.loc[known] = 0.0
    y.loc[known & x.isin(list(codes))] = 1.0
    return y


def recode_ever_married(df: pd.DataFrame) -> pd.Series:
    """
    Uses the same conservative skip-logic idea from the thesis code:
    - Y11_P202 == 0 means never married.
    - Y11_P202 >= 1 or a valid Y11_P201 status means ever married.
    """
    if "Y11_P201" not in df.columns or "Y11_P202" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    status = clean_numeric(df["Y11_P201"]).where(lambda s: s.isin([1, 2, 3, 4]))
    times = clean_numeric(df["Y11_P202"])
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y.loc[times == 0] = 0.0
    y.loc[(times >= 1) | status.notna()] = 1.0
    return y


def recode_divorced_among_ever_married(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "Y11_P201" not in df.columns or "Y11_P202" not in df.columns:
        y = pd.Series(np.nan, index=df.index, dtype="float64")
        return y, pd.Series(False, index=df.index)
    ever = recode_ever_married(df)
    status = clean_numeric(df["Y11_P201"]).where(lambda s: s.isin([1, 2, 3, 4]))
    keep = ever == 1.0
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y.loc[keep & status.notna()] = (status.loc[keep & status.notna()] == 3).astype(float)
    return y, keep


def build_outcomes(base: pd.DataFrame, y11: pd.DataFrame, joined: pd.DataFrame,
                   major_codes: pd.DataFrame, career_codes: pd.DataFrame) -> list[dict]:
    outcomes: list[dict] = []

    # A) Raw interest item scores. These are continuous 1-5 style outcomes.
    for c, lab in ENGINEERING_INTEREST_ITEMS.items():
        if c in base.columns:
            outcomes.append({
                "outcome": f"interest_{c}",
                "label": lab,
                "family": "engineering_interest_item",
                "kind": "continuous",
                "df": base,
                "y": clean_numeric(base[c]),
                "keep": None,
            })
    for c, lab in STEM_INTEREST_ITEMS.items():
        if c in base.columns and c not in ENGINEERING_INTEREST_ITEMS:
            outcomes.append({
                "outcome": f"interest_{c}",
                "label": lab,
                "family": "stem_interest_item",
                "kind": "continuous",
                "df": base,
                "y": clean_numeric(base[c]),
                "keep": None,
            })

    # B) High-interest cohorts. These are the cleanest early STEM/engineering outcomes.
    eng_cols = list(ENGINEERING_INTEREST_ITEMS.keys())
    stem_cols = list(STEM_INTEREST_ITEMS.keys())
    for cutoff in [4.0, 5.0]:
        outcomes.append({
            "outcome": f"eng_any_ge_{int(cutoff)}",
            "label": f"Any engineering interest >= {int(cutoff)}",
            "family": "interest_cohort",
            "kind": "binary",
            "df": base,
            "y": any_high(base, eng_cols, cutoff),
            "keep": None,
        })
        outcomes.append({
            "outcome": f"stem_any_ge_{int(cutoff)}",
            "label": f"Any STEM interest >= {int(cutoff)}",
            "family": "interest_cohort",
            "kind": "binary",
            "df": base,
            "y": any_high(base, stem_cols, cutoff),
            "keep": None,
        })

    # C) Major outcomes.
    for col, wave_label in [("Y1_E320", "intent_y1"), ("Y11_E321", "realized_y11")]:
        if col in base.columns:
            outcomes.append({
                "outcome": f"major_{wave_label}_stem",
                "label": f"{wave_label}: STEM major",
                "family": "major",
                "kind": "binary",
                "df": base,
                "y": code_group_flag(base[col], major_codes, "stem"),
                "keep": None,
            })
            outcomes.append({
                "outcome": f"major_{wave_label}_engineering",
                "label": f"{wave_label}: Engineering/CS major",
                "family": "major",
                "kind": "binary",
                "df": base,
                "y": code_group_flag(base[col], major_codes, "engineering"),
                "keep": None,
            })

    # D) Y11 job outcomes.
    if "Y11_O202" in y11.columns:
        outcomes.append({
            "outcome": "stem_job_y11",
            "label": "STEM job at Y11",
            "family": "job",
            "kind": "binary",
            "df": y11,
            "y": code_group_flag(y11["Y11_O202"], career_codes, "stem"),
            "keep": None,
        })
        outcomes.append({
            "outcome": "engineering_job_y11",
            "label": "Engineering/CS job at Y11",
            "family": "job",
            "kind": "binary",
            "df": y11,
            "y": code_group_flag(y11["Y11_O202"], career_codes, "engineering"),
            "keep": None,
        })

    # E) Follow-through among intended major groups.
    if "Y1_E320" in joined.columns and "Y11_O202" in joined.columns:
        intended_stem = code_group_flag(joined["Y1_E320"], major_codes, "stem") == 1.0
        intended_eng = code_group_flag(joined["Y1_E320"], major_codes, "engineering") == 1.0
        outcomes.append({
            "outcome": "followthrough_intended_stem_to_stem_job",
            "label": "Follow-through: intended STEM major -> STEM job",
            "family": "followthrough",
            "kind": "binary",
            "df": joined,
            "y": code_group_flag(joined["Y11_O202"], career_codes, "stem"),
            "keep": intended_stem,
        })
        outcomes.append({
            "outcome": "followthrough_intended_eng_to_eng_job",
            "label": "Follow-through: intended Engineering/CS -> Engineering/CS job",
            "family": "followthrough",
            "kind": "binary",
            "df": joined,
            "y": code_group_flag(joined["Y11_O202"], career_codes, "engineering"),
            "keep": intended_eng,
        })

    # F) Earnings among earners.
    if "Y11_O221" in y11.columns:
        o221 = clean_numeric(y11["Y11_O221"])
        keep = o221.notna() & (o221 > 0)
        outcomes.append({
            "outcome": "earnings_o221_among_earners",
            "label": "Annual earnings O221 among earners",
            "family": "earnings",
            "kind": "continuous",
            "df": y11,
            "y": o221,
            "keep": keep,
        })
    if "Y11_O222" in y11.columns:
        o222 = clean_numeric(y11["Y11_O222"])
        keep = o222.notna() & (o222 > 0)
        outcomes.append({
            "outcome": "log_earnings_o222_among_earners",
            "label": "Log earnings O222 among earners",
            "family": "earnings",
            "kind": "continuous",
            "df": y11,
            "y": o222,
            "keep": keep,
        })

    # G) Marriage/family outcomes from the user's earlier outcome list.
    if "Y11_P201" in y11.columns and "Y11_P202" in y11.columns:
        outcomes.append({
            "outcome": "ever_married",
            "label": "Ever married",
            "family": "family",
            "kind": "binary",
            "df": y11,
            "y": recode_ever_married(y11),
            "keep": None,
        })
        divorced_y, divorced_keep = recode_divorced_among_ever_married(y11)
        outcomes.append({
            "outcome": "divorced_among_ever_married",
            "label": "Divorced among ever-married",
            "family": "family",
            "kind": "binary",
            "df": y11,
            "y": divorced_y,
            "keep": divorced_keep,
        })

    return outcomes


# -------------------------------------------------------------------
# Modeling helpers
# -------------------------------------------------------------------

def make_work_df(df: pd.DataFrame, y: pd.Series, keep: pd.Series | None, school_col: str) -> pd.DataFrame:
    need = [school_col, "sex01", "pc1", "pc2", "pc1_z", "pc2_z", "pc1_x_pc2_z", "abs_pc2_z"]
    for c in ["gci_pc1", "cgn_pc1"]:
        if c in df.columns:
            need.append(c)
    need = [c for c in need if c in df.columns]
    work = df[need].copy()
    work["y"] = y
    if keep is not None:
        work = work.loc[keep].copy()
    if "cgn_pc1" not in work.columns:
        work["cgn_pc1"] = np.nan
    return work


def subset_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {
        "female": df["sex01"] == 0,
        "male": df["sex01"] == 1,
        "all": pd.Series(True, index=df.index),
    }
    if "cgn_pc1" in df.columns:
        masks["female_cgn_only"] = (df["sex01"] == 0) & (df["cgn_pc1"] == 1)
        masks["female_non_cgn_only"] = (df["sex01"] == 0) & (df["cgn_pc1"] == 0)
    return masks


def cv_auc(df: pd.DataFrame, features: list[str]) -> tuple[float, float, int]:
    d = df[["y", *features]].dropna().copy()
    d = d[d["y"].isin([0.0, 1.0])]
    if len(d) < 100 or d["y"].nunique() < 2:
        return np.nan, np.nan, int(len(d))
    counts = d["y"].value_counts()
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return np.nan, np.nan, int(len(d))
    y = d["y"].astype(int)
    X = d[features].astype(float)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        return float(aucs.mean()), float(aucs.std()), int(len(d))
    except Exception:
        return np.nan, np.nan, int(len(d))


def fit_non_fe(df: pd.DataFrame, features: list[str], kind: str, model_label: str) -> pd.DataFrame:
    d = df[["y", *features]].dropna().copy()
    if kind == "binary":
        d = d[d["y"].isin([0.0, 1.0])]
    if len(d) < 50 or d["y"].nunique() < 2:
        return pd.DataFrame([{
            "estimator": "non_fe", "model": model_label, "term": "INSUFFICIENT_DATA",
            "n": int(len(d)), "coef": np.nan, "odds_ratio": np.nan, "p_value": np.nan,
            "note": "too little data or no y variation",
        }])

    y = d["y"].astype(float)
    X = d[features].astype(float)
    X2 = sm.add_constant(X, has_constant="add") if HAS_STATSMODELS else X

    rows = []
    if HAS_STATSMODELS:
        try:
            if kind == "binary":
                res = sm.Logit(y.astype(int), X2).fit(disp=False, maxiter=200)
                estimator = "logit"
            else:
                res = sm.OLS(y, X2).fit()
                estimator = "ols"
            for term in res.params.index:
                rows.append({
                    "estimator": estimator,
                    "model": model_label,
                    "term": term,
                    "n": int(len(d)),
                    "coef": float(res.params[term]),
                    "odds_ratio": float(np.exp(res.params[term])) if kind == "binary" else np.nan,
                    "p_value": float(res.pvalues[term]),
                    "note": "",
                })
            return pd.DataFrame(rows)
        except Exception as e:
            # fall through to sklearn coefficient-only fallback for binary, OLS fallback for continuous.
            fail_note = f"statsmodels_failed:{type(e).__name__}"
    else:
        fail_note = "statsmodels_not_available"

    if kind == "binary":
        try:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(X, y.astype(int))
            coefs = model.named_steps["logisticregression"].coef_[0]
            return pd.DataFrame([{
                "estimator": "sklearn_logit_scaled",
                "model": model_label,
                "term": features[i],
                "n": int(len(d)),
                "coef": float(coefs[i]),
                "odds_ratio": float(np.exp(coefs[i])),
                "p_value": np.nan,
                "note": fail_note,
            } for i in range(len(features))])
        except Exception as e:
            return pd.DataFrame([{
                "estimator": "non_fe", "model": model_label, "term": f"FAILED:{type(e).__name__}",
                "n": int(len(d)), "coef": np.nan, "odds_ratio": np.nan, "p_value": np.nan,
                "note": fail_note,
            }])

    # Continuous fallback without statsmodels should almost never be needed.
    return pd.DataFrame([{
        "estimator": "non_fe", "model": model_label, "term": "FAILED",
        "n": int(len(d)), "coef": np.nan, "odds_ratio": np.nan, "p_value": np.nan,
        "note": fail_note,
    }])


def within_demean(df: pd.DataFrame, cols: list[str], fe_col: str) -> pd.DataFrame:
    out = df.copy()
    out[fe_col] = out[fe_col].astype(str)
    g = out.groupby(fe_col, observed=True, sort=False)
    for c in cols:
        out[c] = out[c] - g[c].transform("mean")
    return out


def fit_fe_ols(df: pd.DataFrame, features: list[str], school_col: str, model_label: str) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame([{
            "estimator": "school_fe_ols", "model": model_label, "term": "STATSMODELS_NOT_AVAILABLE",
            "n": 0, "coef": np.nan, "odds_ratio": np.nan, "p_value": np.nan,
            "note": "",
        }])

    d = df[["y", school_col, *features]].dropna().copy()
    if len(d) < 50 or d[school_col].nunique() < 2 or d["y"].nunique() < 2:
        return pd.DataFrame([{
            "estimator": "school_fe_ols", "model": model_label, "term": "INSUFFICIENT_DATA",
            "n": int(len(d)), "coef": np.nan, "odds_ratio": np.nan, "p_value": np.nan,
            "note": "too little data, no y variation, or fewer than 2 schools",
        }])

    # Need some variation in regressors.
    if all(d[c].nunique(dropna=True) < 2 for c in features):
        return pd.DataFrame([{
            "estimator": "school_fe_ols", "model": model_label, "term": "NO_X_VARIATION",
            "n": int(len(d)), "coef": np.nan, "odds_ratio": np.nan, "p_value": np.nan,
            "note": "",
        }])

    try:
        dm = within_demean(d, cols=["y", *features], fe_col=school_col)
        Y = dm["y"].to_numpy(dtype=float)
        X = dm[features].to_numpy(dtype=float)
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": d[school_col]})
        return pd.DataFrame([{
            "estimator": "school_fe_ols",
            "model": model_label,
            "term": features[i],
            "n": int(len(d)),
            "coef": float(res.params[i]),
            "odds_ratio": np.nan,
            "p_value": float(res.pvalues[i]),
            "note": "For binary outcomes this is a linear probability model.",
        } for i in range(len(features))])
    except Exception as e:
        return pd.DataFrame([{
            "estimator": "school_fe_ols", "model": model_label, "term": f"FAILED:{type(e).__name__}",
            "n": int(len(d)), "coef": np.nan, "odds_ratio": np.nan, "p_value": np.nan,
            "note": "",
        }])


def basic_summary(df: pd.DataFrame, kind: str) -> dict:
    d = df[["y"]].dropna()
    if kind == "binary":
        d = d[d["y"].isin([0.0, 1.0])]
    return {
        "n_nonmissing": int(len(d)),
        "mean_y": float(d["y"].mean()) if len(d) else np.nan,
        "sd_y": float(d["y"].std(ddof=0)) if len(d) else np.nan,
    }


def quantile_profiles(work: pd.DataFrame, outcome: str, family: str, kind: str, out_dir: Path) -> pd.DataFrame:
    rows = []

    def add_group(sub: pd.DataFrame, subset_name: str, group_col: str):
        d = sub[["y", group_col, "pc1", "pc2", "pc1_z", "pc2_z", "cgn_pc1"]].dropna(subset=["y", group_col]).copy()
        if kind == "binary":
            d = d[d["y"].isin([0.0, 1.0])]
        if d.empty:
            return
        g = d.groupby(group_col, observed=True).agg(
            n=("y", "size"),
            outcome_mean=("y", "mean"),
            mean_pc1=("pc1", "mean"),
            mean_pc2=("pc2", "mean"),
            mean_pc1_z=("pc1_z", "mean"),
            mean_pc2_z=("pc2_z", "mean"),
            prop_cgn=("cgn_pc1", "mean"),
        ).reset_index()
        g.insert(0, "outcome", outcome)
        g.insert(1, "family", family)
        g.insert(2, "subset", subset_name)
        g.insert(3, "kind", kind)
        rows.append(g)

    female = work[work["sex01"] == 0].copy()
    if len(female) > 0 and female["pc2"].nunique(dropna=True) >= 4:
        female["pc2_quartile"] = pd.qcut(
            female["pc2"], 4,
            labels=["Q1_low_PC2", "Q2", "Q3", "Q4_high_PC2"],
            duplicates="drop",
        )
        add_group(female, "female_by_pc2_quartile", "pc2_quartile")

    cgn_female = female[female["cgn_pc1"] == 1].copy() if "cgn_pc1" in female.columns else pd.DataFrame()
    if len(cgn_female) > 0 and cgn_female["pc2"].nunique(dropna=True) >= 4:
        cgn_female["pc2_quartile"] = pd.qcut(
            cgn_female["pc2"], 4,
            labels=["Q1_low_PC2", "Q2", "Q3", "Q4_high_PC2"],
            duplicates="drop",
        )
        add_group(cgn_female, "female_cgn_by_pc2_quartile", "pc2_quartile")

    # Two-way table for girls: PC1 quartile x PC2 quartile.
    if len(female) > 0 and female["pc1"].nunique(dropna=True) >= 4 and female["pc2"].nunique(dropna=True) >= 4:
        tw = female.copy()
        tw["pc1_quartile"] = pd.qcut(
            tw["pc1"], 4,
            labels=["Q1_low_PC1", "Q2", "Q3", "Q4_high_PC1"],
            duplicates="drop",
        )
        tw["pc2_quartile"] = pd.qcut(
            tw["pc2"], 4,
            labels=["Q1_low_PC2", "Q2", "Q3", "Q4_high_PC2"],
            duplicates="drop",
        )
        d = tw[["y", "pc1_quartile", "pc2_quartile"]].dropna().copy()
        if kind == "binary":
            d = d[d["y"].isin([0.0, 1.0])]
        if not d.empty:
            two_way = d.groupby(["pc1_quartile", "pc2_quartile"], observed=True).agg(
                n=("y", "size"),
                outcome_mean=("y", "mean"),
            ).reset_index()
            two_way.insert(0, "outcome", outcome)
            two_way.insert(1, "family", family)
            two_way.to_csv(out_dir / f"female_{outcome}_pc1_by_pc2_quartiles.csv", index=False)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_path", required=True, help="domain_pca_scores.parquet from PC2 script")
    ap.add_argument("--id_source_path", default="", help="Original domain_scores_raw.parquet if scores_path lacks BY_ID_Rel")
    ap.add_argument("--db_path", default="/store/talent/db/talent_followup.db")
    ap.add_argument("--base_relation", default="follow_up_full_view")
    ap.add_argument("--y11_table", default="follow_y11")
    ap.add_argument("--id_col", default="BY_ID_Rel")
    ap.add_argument("--school_col", default="by_school_rel")
    ap.add_argument("--y11_qstat_col", default="Y11_QSTAT")
    ap.add_argument("--y11_qstat_value", default="01", help="Set empty string to only require non-null when --require_y11_qstat is used")
    ap.add_argument("--require_y11_qstat", action="store_true", default=True)
    ap.add_argument("--career_codes_csv", default="", help="Optional career mapping CSV; otherwise uses built-in codes")
    ap.add_argument("--major_codes_csv", default="", help="Optional major mapping CSV; otherwise uses built-in codes")
    ap.add_argument("--out_dir", default="pc2_stem_outcome_tests")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = prepare_scores(args.scores_path, id_col=args.id_col, id_source_path=args.id_source_path)
    career_codes = load_mapping(args.career_codes_csv, DEFAULT_CAREER_CODES)
    major_codes = load_mapping(args.major_codes_csv, DEFAULT_MAJOR_CODES)

    # Save mappings so the report can cite/reproduce exactly what was counted.
    career_codes.to_csv(out_dir / "career_codes_used.csv", index=False)
    major_codes.to_csv(out_dir / "major_codes_used.csv", index=False)

    base_cols = set(STEM_INTEREST_ITEMS.keys()) | {"Y1_E320", "Y11_E321"}
    y11_cols = {"Y11_O202", "Y11_O221", "Y11_O222", "Y11_P201", "Y11_P202", args.y11_qstat_col}

    con = duckdb.connect(args.db_path)
    base = fetch_base(con, scores, args, base_cols)
    y11 = fetch_y11(con, scores, args, y11_cols)
    joined = fetch_joined(con, scores, args, {"Y1_E320"}, {"Y11_O202", args.y11_qstat_col})
    con.close()

    print("Rows with base-year data:", len(base))
    print("Rows with Y11 data:", len(y11))
    print("Rows with joined base+Y11 data:", len(joined))

    outcomes = build_outcomes(base, y11, joined, major_codes=major_codes, career_codes=career_codes)
    pd.DataFrame([{
        "outcome": o["outcome"],
        "label": o["label"],
        "family": o["family"],
        "kind": o["kind"],
        "source_rows": len(o["df"]),
    } for o in outcomes]).to_csv(out_dir / "pc2_outcome_definitions.csv", index=False)

    model_specs = {
        "PC1_only": ["pc1_z"],
        "PC2_only": ["pc2_z"],
        "PC1_plus_PC2": ["pc1_z", "pc2_z"],
        "PC1_PC2_interaction": ["pc1_z", "pc2_z", "pc1_x_pc2_z"],
    }

    all_auc = []
    all_coef = []
    all_summary = []
    all_profiles = []

    for o in outcomes:
        outcome = o["outcome"]
        family = o["family"]
        kind = o["kind"]
        work = make_work_df(o["df"], o["y"], o["keep"], school_col=args.school_col)

        # For binary outcomes, enforce 0/1 only.
        if kind == "binary":
            work.loc[~work["y"].isin([0.0, 1.0]), "y"] = np.nan

        print(f"\n=== {outcome} ({kind}, {family}) ===")

        for subset_name, mask in subset_masks(work).items():
            sub = work.loc[mask].copy()
            summ = basic_summary(sub, kind)
            summ.update({
                "outcome": outcome,
                "label": o["label"],
                "family": family,
                "kind": kind,
                "subset": subset_name,
            })
            all_summary.append(summ)

            for model_name, features in model_specs.items():
                if kind == "binary":
                    auc_mean, auc_sd, n_auc = cv_auc(sub, features)
                    all_auc.append({
                        "outcome": outcome,
                        "family": family,
                        "subset": subset_name,
                        "model": model_name,
                        "features": ",".join(features),
                        "n_used": n_auc,
                        "cv_auc_mean": auc_mean,
                        "cv_auc_sd": auc_sd,
                    })

                coef1 = fit_non_fe(sub, features, kind=kind, model_label=model_name)
                coef1.insert(0, "outcome", outcome)
                coef1.insert(1, "family", family)
                coef1.insert(2, "kind", kind)
                coef1.insert(3, "subset", subset_name)
                all_coef.append(coef1)

                coef2 = fit_fe_ols(sub, features, school_col=args.school_col, model_label=model_name)
                coef2.insert(0, "outcome", outcome)
                coef2.insert(1, "family", family)
                coef2.insert(2, "kind", kind)
                coef2.insert(3, "subset", subset_name)
                all_coef.append(coef2)

        prof = quantile_profiles(work, outcome=outcome, family=family, kind=kind, out_dir=out_dir)
        if not prof.empty:
            all_profiles.append(prof)

    auc_df = pd.DataFrame(all_auc)
    coef_df = pd.concat(all_coef, ignore_index=True) if all_coef else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)
    profiles_df = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()

    auc_df.to_csv(out_dir / "pc2_outcome_auc_comparison.csv", index=False)
    coef_df.to_csv(out_dir / "pc2_outcome_coefficients.csv", index=False)
    summary_df.to_csv(out_dir / "pc2_outcome_summary.csv", index=False)
    profiles_df.to_csv(out_dir / "pc2_outcome_profiles_by_pc2_quartile.csv", index=False)

    # Friendly focused table: the exact rows to inspect first.
    if not coef_df.empty:
        focus = coef_df[
            (coef_df["subset"] == "female") &
            (coef_df["model"] == "PC1_plus_PC2") &
            (coef_df["term"].isin(["pc1_z", "pc2_z"])) &
            (coef_df["family"].isin(["interest_cohort", "major", "job", "followthrough"]))
        ].copy()
        focus.to_csv(out_dir / "READ_FIRST_female_pc1_pc2_stem_coefficients.csv", index=False)

    print("\nWrote:")
    print(" ", out_dir / "pc2_outcome_definitions.csv")
    print(" ", out_dir / "pc2_outcome_summary.csv")
    print(" ", out_dir / "pc2_outcome_auc_comparison.csv")
    print(" ", out_dir / "pc2_outcome_coefficients.csv")
    print(" ", out_dir / "pc2_outcome_profiles_by_pc2_quartile.csv")
    print(" ", out_dir / "READ_FIRST_female_pc1_pc2_stem_coefficients.csv")
    print("\nMain interpretation rows:")
    print("  subset == female, model == PC1_plus_PC2, term == pc2_z")
    print("  Positive pc2_z for STEM/engineering outcomes means the technical/physical PC2 profile predicts that pathway after controlling for PC1.")


if __name__ == "__main__":
    main()
