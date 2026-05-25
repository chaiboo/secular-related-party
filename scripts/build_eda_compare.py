"""
build_eda_compare.py
====================

Per-field EDA across every fraud-relevant Form 990 field, broken out by NTEE
major (religion = X plus 23 secular A-W). Outputs feed the comparison views
in index.html.

For each (table, column):
  - per-NTEE-major numerator/denominator:
      boolean  -> rate of True among non-null on canonical filings
      numeric  -> count, median, p90 on rows where field is non-null
      categorical -> distinct-value counts; top 8 values per major
  - per-year per-major trend (rate for boolean, median for numeric)
  - coverage / null-rate per major (a field that's 99% null in one sector and
    5% null in another is itself a comparison story)

Canonicalization rule:
  one row per (ein, tax_period_end), latest return_ts, tie-break by object_id.
  Same rule as p2/p3 build_data.py.

Reads:
  data/parsed/irs_xml_secular/990/*.parquet  (23 secular majors, ntee_cd A-W)
  data/parsed/irs_xml/990/*.parquet          (religion-only, ntee_cd X)

Writes:
  p3_secular_compare/data/eda_compare/per_field_summary.json
  p3_secular_compare/data/eda_compare/per_field/<table>__<column>.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEC_DIR = ROOT / "data" / "parsed" / "irs_xml_secular" / "990"
REL_DIR = ROOT / "data" / "parsed" / "irs_xml" / "990"
OUT_DIR = ROOT / "p3_secular_compare" / "data" / "eda_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIELD_DIR = OUT_DIR / "per_field"
OUT_FIELD_DIR.mkdir(exist_ok=True)

NTEE_MAJOR_LABEL = {
    "A": "Arts, Culture & Humanities",
    "B": "Education",
    "C": "Environment",
    "D": "Animal-related",
    "E": "Health Care",
    "F": "Mental Health & Crisis",
    "G": "Voluntary Health Associations",
    "H": "Medical Research",
    "I": "Crime & Legal-Related",
    "J": "Employment",
    "K": "Food, Agriculture & Nutrition",
    "L": "Housing & Shelter",
    "M": "Public Safety & Disaster Preparedness",
    "N": "Recreation & Sports",
    "O": "Youth Development",
    "P": "Human Services",
    "Q": "International, Foreign Affairs",
    "R": "Civil Rights & Civic Action",
    "S": "Community Improvement",
    "T": "Philanthropy & Voluntarism",
    "U": "Science & Technology",
    "V": "Social Science",
    "W": "Public & Societal Benefit",
    "X": "Religion-related",
}

# Field-level FK pulled from manifest.json — each row is (table, column, kind, fraud_strength).
# fraud_strength carries over from the manifest's table-level label.
FIELD_SPEC = [
    # sched_l_excess_benefit -- strongest
    ("sched_l_excess_benefit", "amt_tax_imposed",         "numeric",     "strongest"),
    ("sched_l_excess_benefit", "amt_tax_reimbursed",      "numeric",     "strongest"),
    ("sched_l_excess_benefit", "transaction_corrected",   "boolean",     "strongest"),
    ("sched_l_excess_benefit", "relationship",            "categorical", "strongest"),
    # sched_l_loans -- strong
    ("sched_l_loans", "original_principal",   "numeric",     "strong"),
    ("sched_l_loans", "balance_due",          "numeric",     "strong"),
    ("sched_l_loans", "loan_to_organization", "boolean",     "strong"),
    ("sched_l_loans", "loan_from_organization", "boolean",   "strong"),
    ("sched_l_loans", "written_agreement",    "boolean",     "strong"),
    ("sched_l_loans", "in_default",           "boolean",     "strong"),
    ("sched_l_loans", "approved_by_board",    "boolean",     "strong"),
    ("sched_l_loans", "relationship",         "categorical", "strong"),
    # sched_l_grants_to_interested -- moderate
    ("sched_l_grants_to_interested", "amount_of_assistance", "numeric",     "moderate"),
    ("sched_l_grants_to_interested", "relationship",         "categorical", "moderate"),
    ("sched_l_grants_to_interested", "type_of_assistance",   "categorical", "moderate"),
    # sched_l_business_transactions -- strong
    ("sched_l_business_transactions", "transaction_amount", "numeric",     "strong"),
    ("sched_l_business_transactions", "relationship",       "categorical", "strong"),
    ("sched_l_business_transactions", "sharing_of_revenues", "boolean",    "strong"),
    # sched_r_disregarded -- moderate
    ("sched_r_disregarded", "total_income",       "numeric",     "moderate"),
    ("sched_r_disregarded", "end_of_year_assets", "numeric",     "moderate"),
    ("sched_r_disregarded", "legal_domicile",     "categorical", "moderate"),
    # sched_r_related_tax_exempt -- moderate
    ("sched_r_related_tax_exempt", "legal_domicile",            "categorical", "moderate"),
    ("sched_r_related_tax_exempt", "exempt_code_section",       "categorical", "moderate"),
    ("sched_r_related_tax_exempt", "public_charity_status",     "categorical", "moderate"),
    ("sched_r_related_tax_exempt", "section_512b_13_controlled", "boolean",    "moderate"),
    # sched_r_related_partnership -- moderate
    ("sched_r_related_partnership", "percentage_ownership",       "numeric",     "moderate"),
    ("sched_r_related_partnership", "general_or_managing_partner", "boolean",    "moderate"),
    ("sched_r_related_partnership", "legal_domicile",             "categorical", "moderate"),
    # sched_r_related_corp_trust -- moderate
    ("sched_r_related_corp_trust", "percentage_ownership",       "numeric",     "moderate"),
    ("sched_r_related_corp_trust", "type_of_entity",             "categorical", "moderate"),
    ("sched_r_related_corp_trust", "section_512b_13_controlled", "boolean",     "moderate"),
    ("sched_r_related_corp_trust", "legal_domicile",             "categorical", "moderate"),
    # sched_r_transactions -- strongest
    ("sched_r_transactions", "transaction_amount",    "numeric",     "strongest"),
    ("sched_r_transactions", "transaction_type_code", "categorical", "strongest"),
    # sched_j_main -- strong (these are filing-level booleans, denom = canonical filings appearing in Sched J)
    ("sched_j_main", "first_class_or_charter_travel",          "boolean", "strong"),
    ("sched_j_main", "travel_companions",                       "boolean", "strong"),
    ("sched_j_main", "tax_indemnification",                     "boolean", "strong"),
    ("sched_j_main", "discretionary_spending_account",          "boolean", "strong"),
    ("sched_j_main", "housing_allowance_residence",             "boolean", "strong"),
    ("sched_j_main", "personal_services",                       "boolean", "strong"),
    ("sched_j_main", "health_or_social_club_dues",              "boolean", "strong"),
    ("sched_j_main", "compensation_committee",                  "boolean", "strong"),
    ("sched_j_main", "compensation_survey",                     "boolean", "strong"),
    ("sched_j_main", "severance_payment",                       "boolean", "strong"),
    ("sched_j_main", "supplemental_nonqualified_retirement",    "boolean", "strong"),
    ("sched_j_main", "equity_based_compensation",               "boolean", "strong"),
    ("sched_j_main", "compensation_contingent_on_revenues",     "boolean", "strong"),
    ("sched_j_main", "compensation_contingent_on_net_earnings", "boolean", "strong"),
    ("sched_j_main", "non_fixed_payments",                      "boolean", "strong"),
    # sched_j_compensation (per-officer) -- strong, numeric
    ("sched_j_compensation", "base_compensation_filing_org",       "numeric", "strong"),
    ("sched_j_compensation", "base_compensation_related_org",      "numeric", "strong"),
    ("sched_j_compensation", "bonus_filing_org",                   "numeric", "strong"),
    ("sched_j_compensation", "bonus_related_org",                  "numeric", "strong"),
    ("sched_j_compensation", "deferred_compensation_filing_org",   "numeric", "strong"),
    ("sched_j_compensation", "deferred_compensation_related_org",  "numeric", "strong"),
    ("sched_j_compensation", "nontaxable_benefits_filing_org",     "numeric", "strong"),
    ("sched_j_compensation", "total_filing_org",                   "numeric", "strong"),
    ("sched_j_compensation", "total_related_org",                  "numeric", "strong"),
    # part_vii_officers -- strong
    ("part_vii_officers", "reportable_comp_from_org",        "numeric", "strong"),
    ("part_vii_officers", "reportable_comp_from_related",    "numeric", "strong"),
    ("part_vii_officers", "other_compensation",              "numeric", "strong"),
    ("part_vii_officers", "average_hours",                   "numeric", "strong"),
    ("part_vii_officers", "individual_trustee_or_director",  "boolean", "strong"),
    ("part_vii_officers", "officer",                         "boolean", "strong"),
    ("part_vii_officers", "key_employee",                    "boolean", "strong"),
    ("part_vii_officers", "highest_compensated_employee",    "boolean", "strong"),
    ("part_vii_officers", "former",                          "boolean", "strong"),
    # part_vii_contractors -- moderate
    ("part_vii_contractors", "compensation",       "numeric",     "moderate"),
    ("part_vii_contractors", "contractor_state",   "categorical", "moderate"),
]

TABLE_LABEL = {
    "sched_l_excess_benefit": "Schedule L Pt I — Excess benefit transactions (§4958)",
    "sched_l_loans": "Schedule L Pt II — Loans to/from interested persons",
    "sched_l_grants_to_interested": "Schedule L Pt III — Grants to interested persons",
    "sched_l_business_transactions": "Schedule L Pt IV — Business transactions with interested persons",
    "sched_r_disregarded": "Schedule R Pt I — Disregarded entities",
    "sched_r_related_tax_exempt": "Schedule R Pt II — Related tax-exempt organizations",
    "sched_r_related_partnership": "Schedule R Pt III — Related taxable partnerships",
    "sched_r_related_corp_trust": "Schedule R Pt IV — Related corporations/trusts",
    "sched_r_transactions": "Schedule R Pt V — Transactions with related organizations",
    "sched_j_main": "Schedule J Pt I — Compensation policy indicators",
    "sched_j_compensation": "Schedule J Pt II — Per-officer compensation",
    "part_vii_officers": "Part VII Section A — Officers/directors/trustees/key employees",
    "part_vii_contractors": "Part VII Section B — Top 5 contractors",
}


# ---------- canonical roster ----------------------------------------------

def canonical_roster(parquet_dir: Path) -> pd.DataFrame:
    """One row per (ein, tax_period_end), latest return_ts. Carries ntee_major."""
    cols = ["object_id", "ein", "tax_period_end", "return_ts", "ntee_cd"]
    df = pd.read_parquet(parquet_dir / "main_990.parquet", columns=cols)
    df = df.sort_values(["return_ts", "object_id"], ascending=[True, True])
    df = df.drop_duplicates(subset=["ein", "tax_period_end"], keep="last")
    df["ntee_major"] = df["ntee_cd"].fillna("?").astype(str).str[0].str.upper()
    df["year"] = pd.to_datetime(df["tax_period_end"]).dt.year.astype("Int64")
    return df.reset_index(drop=True)


# ---------- per-major profilers -------------------------------------------

def profile_boolean(df: pd.DataFrame, col: str, by_major_n_filings: dict) -> dict:
    """For each NTEE major, share-of-True. Denominator is canonical filings for the
    table (count of canonical filings that have at least one row in this table)
    OR — for filing-level booleans (Sched J Pt I) — the count of canonical
    filings present in Sched J for that major. We compute both: (a) rate
    among rows where field is non-null, (b) coverage = non-null rate."""
    # Make the column properly boolean-ish: True/False/None.
    s = df[col]
    if s.dtype == object:
        s = s.where(s.notna(), other=None)
        # Strings like 'true'/'false'/'1'/'0' or '0.0'/'1.0' -> bool
        def coerce(v):
            if v is None: return None
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)):
                if np.isnan(v): return None
                return bool(v)
            t = str(v).strip().lower()
            if t in ('true', 't', '1', '1.0', 'yes'): return True
            if t in ('false', 'f', '0', '0.0', 'no'): return False
            return None
        s = s.map(coerce)
    elif np.issubdtype(s.dtype, np.floating):
        # 0.0/1.0 with NaN as null
        s = s.map(lambda v: None if pd.isna(v) else bool(v))
    elif np.issubdtype(s.dtype, np.integer):
        s = s.map(lambda v: bool(v))
    # else: already bool

    by_major = {}
    for major, sub in df.groupby("ntee_major"):
        if major not in NTEE_MAJOR_LABEL:
            continue
        ss = s.loc[sub.index]
        n_total = len(sub)
        n_nonnull = ss.notna().sum()
        n_true = (ss == True).sum()  # noqa: E712
        by_major[major] = {
            "n_rows": int(n_total),                  # rows in this table for this major (post-canonical filter)
            "n_nonnull": int(n_nonnull),
            "n_true": int(n_true),
            "rate_true_among_nonnull": float(n_true) / max(int(n_nonnull), 1),
            "coverage": float(n_nonnull) / max(int(n_total), 1),
        }
    return {"kind": "boolean", "per_major": by_major}


def profile_numeric(df: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(df[col], errors="coerce")
    by_major = {}
    for major, sub in df.groupby("ntee_major"):
        if major not in NTEE_MAJOR_LABEL:
            continue
        ss = s.loc[sub.index].dropna()
        n_nonnull = len(ss)
        n_total = len(sub)
        if n_nonnull == 0:
            by_major[major] = {
                "n_rows": int(n_total), "n_nonnull": 0,
                "coverage": 0.0,
                "median": None, "p25": None, "p75": None, "p90": None, "p95": None,
                "mean": None, "min": None, "max": None,
                "n_zero": 0, "n_negative": 0,
            }
            continue
        v = ss.values
        by_major[major] = {
            "n_rows": int(n_total),
            "n_nonnull": int(n_nonnull),
            "coverage": float(n_nonnull) / max(int(n_total), 1),
            "median": float(np.median(v)),
            "p25": float(np.quantile(v, 0.25)),
            "p75": float(np.quantile(v, 0.75)),
            "p90": float(np.quantile(v, 0.90)),
            "p95": float(np.quantile(v, 0.95)),
            "mean": float(np.mean(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "n_zero": int((v == 0).sum()),
            "n_negative": int((v < 0).sum()),
        }
    return {"kind": "numeric", "per_major": by_major}


def profile_categorical(df: pd.DataFrame, col: str, top_n: int = 10) -> dict:
    s = df[col].astype(str).where(df[col].notna(), other=None)
    by_major = {}
    # Global top values (across both corpora) - the "shared vocabulary"
    overall_top = (s.dropna().value_counts().head(top_n)).index.tolist()
    for major, sub in df.groupby("ntee_major"):
        if major not in NTEE_MAJOR_LABEL:
            continue
        ss = s.loc[sub.index].dropna()
        n_nonnull = len(ss)
        n_total = len(sub)
        vc = ss.value_counts()
        # Top within this major
        top_local = [(k, int(v), float(v) / max(n_nonnull, 1)) for k, v in vc.head(top_n).items()]
        # Counts for the shared overall vocabulary (so cross-major bars are comparable)
        shared = []
        for k in overall_top:
            cnt = int(vc.get(k, 0))
            shared.append({"value": str(k), "n": cnt, "share": float(cnt) / max(n_nonnull, 1)})
        by_major[major] = {
            "n_rows": int(n_total),
            "n_nonnull": int(n_nonnull),
            "coverage": float(n_nonnull) / max(int(n_total), 1),
            "n_distinct": int(vc.shape[0]),
            "top_local": [{"value": str(k), "n": int(n), "share": float(s)} for k, n, s in top_local],
            "shared_vocab": shared,
        }
    return {"kind": "categorical", "per_major": by_major,
            "shared_vocab_order": [str(v) for v in overall_top]}


def yearly_trend(df: pd.DataFrame, col: str, kind: str) -> dict:
    """Per-year × per-major series.
    boolean: rate-of-True among non-null
    numeric: median (and n) per year × major
    categorical: skipped (too dense for inline JSON; can be added per-field later)"""
    if kind == "categorical":
        return {}
    out = {}
    df2 = df.copy()
    df2["year"] = df2["year"].astype("Int64")
    df2 = df2[df2["year"].between(2016, 2025)]
    if kind == "boolean":
        s = df2[col]
        if s.dtype == object:
            def coerce(v):
                if v is None: return None
                if isinstance(v, bool): return v
                if isinstance(v, (int, float)):
                    if np.isnan(v): return None
                    return bool(v)
                t = str(v).strip().lower()
                if t in ('true','t','1','1.0','yes'): return True
                if t in ('false','f','0','0.0','no'): return False
                return None
            s = s.map(coerce)
        elif np.issubdtype(s.dtype, np.floating):
            s = s.map(lambda v: None if pd.isna(v) else bool(v))
        df2 = df2.assign(_v=s)
        g = df2.dropna(subset=["_v"]).groupby(["ntee_major", "year"])
        out_pts = []
        for (major, year), sub in g:
            if major not in NTEE_MAJOR_LABEL:
                continue
            n = len(sub)
            n_true = int(sub["_v"].sum())
            out_pts.append({"major": major, "year": int(year), "n": n, "n_true": n_true,
                            "rate": float(n_true) / max(n, 1)})
        return {"kind": "boolean", "points": out_pts}
    else:  # numeric
        s = pd.to_numeric(df2[col], errors="coerce")
        df2 = df2.assign(_v=s).dropna(subset=["_v"])
        g = df2.groupby(["ntee_major", "year"])
        out_pts = []
        for (major, year), sub in g:
            if major not in NTEE_MAJOR_LABEL:
                continue
            v = sub["_v"].values
            out_pts.append({"major": major, "year": int(year), "n": int(len(v)),
                            "median": float(np.median(v)),
                            "p90": float(np.quantile(v, 0.90))})
        return {"kind": "numeric", "points": out_pts}


# ---------- driver -------------------------------------------------------

def load_table_with_canonical(parquet_dir: Path, table: str,
                              canon: pd.DataFrame, extra_cols: list[str]) -> pd.DataFrame:
    """Read a table parquet, filter to canonical-filing object_ids, attach ntee_major + year."""
    path = parquet_dir / f"{table}.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols_to_load = list({"object_id", "ein", "filing_year"} | set(extra_cols))
    # Filter cols that actually exist
    sch = pd.read_parquet(path, columns=None).columns.tolist() if False else None
    # safer: just try and let pyarrow complain
    try:
        df = pd.read_parquet(path, columns=cols_to_load)
    except Exception:
        # column missing -> fall back to full load + project
        df = pd.read_parquet(path)
        df = df[[c for c in cols_to_load if c in df.columns]]
    canon_ids = set(canon["object_id"])
    df = df[df["object_id"].isin(canon_ids)].copy()
    # attach major + year from canon
    oid2 = canon.set_index("object_id")[["ntee_major", "year"]]
    df = df.join(oid2, on="object_id")
    return df


def aggregate_field(sec_dir: Path, rel_dir: Path, sec_canon: pd.DataFrame,
                    rel_canon: pd.DataFrame, table: str, col: str, kind: str) -> dict:
    sec = load_table_with_canonical(sec_dir, table, sec_canon, [col])
    rel = load_table_with_canonical(rel_dir, table, rel_canon, [col])
    if col not in sec.columns and col not in rel.columns:
        return None
    both = pd.concat([sec, rel], ignore_index=True) if not rel.empty else sec
    if col not in both.columns:
        return None

    if kind == "boolean":
        per_major_n_filings = {}  # not strictly needed; profile_boolean returns rows + nonnull
        snap = profile_boolean(both, col, per_major_n_filings)
    elif kind == "numeric":
        snap = profile_numeric(both, col)
    elif kind == "categorical":
        snap = profile_categorical(both, col)
    else:
        return None

    snap["trend"] = yearly_trend(both, col, kind)
    snap["table"] = table
    snap["column"] = col
    return snap


def main():
    print("[1/3] building canonical rosters")
    sec_canon = canonical_roster(SEC_DIR)
    rel_canon = canonical_roster(REL_DIR)
    print(f"  secular canon: {len(sec_canon):,} filings, EINs {sec_canon['ein'].nunique():,}")
    print(f"  religion canon: {len(rel_canon):,} filings, EINs {rel_canon['ein'].nunique():,}")

    # Filings-per-major (we'll need denominators for "% of canonical filings disclosing")
    sec_by_major = sec_canon.groupby("ntee_major").size().to_dict()
    rel_by_major = rel_canon.groupby("ntee_major").size().to_dict()
    n_filings_by_major = {**sec_by_major, **rel_by_major}
    print(f"  per-major filings: {sorted(n_filings_by_major.items(), key=lambda kv:-kv[1])[:5]}")

    print(f"[2/3] profiling {len(FIELD_SPEC)} fields")
    summary_rows = []
    for i, (table, col, kind, strength) in enumerate(FIELD_SPEC, 1):
        key = f"{table}__{col}"
        print(f"  [{i:>2}/{len(FIELD_SPEC)}] {key}  ({kind})")
        snap = aggregate_field(SEC_DIR, REL_DIR, sec_canon, rel_canon,
                               table, col, kind)
        if snap is None:
            print(f"     (skipped: column not found)")
            continue
        snap["kind"] = kind  # ensure carried through
        snap["fraud_strength"] = strength
        snap["table_label"] = TABLE_LABEL.get(table, table)

        # write per-field detail
        (OUT_FIELD_DIR / f"{key}.json").write_text(json.dumps(snap, separators=(",", ":")))

        # build compact summary row
        per_major_summary = {}
        for major, m in snap["per_major"].items():
            row = {"n_rows": m["n_rows"], "n_nonnull": m["n_nonnull"],
                   "coverage": m["coverage"]}
            if kind == "boolean":
                row["rate"] = m["rate_true_among_nonnull"]
                row["n_true"] = m["n_true"]
                # disclosure_rate against canonical filing denominator (for Sched J etc.)
                row["disclose_rate"] = m["n_true"] / max(n_filings_by_major.get(major, 1), 1)
            elif kind == "numeric":
                row["median"] = m["median"]
                row["p90"] = m["p90"]
                row["mean"] = m["mean"]
            elif kind == "categorical":
                row["n_distinct"] = m["n_distinct"]
                row["top1"] = m["top_local"][0] if m["top_local"] else None
            per_major_summary[major] = row

        summary_rows.append({
            "table": table,
            "column": col,
            "kind": kind,
            "fraud_strength": strength,
            "table_label": TABLE_LABEL.get(table, table),
            "per_major": per_major_summary,
            "shared_vocab_order": snap.get("shared_vocab_order", []) if kind == "categorical" else [],
        })

    print("[3/3] writing summary")
    out = {
        "ntee_majors": NTEE_MAJOR_LABEL,
        "religion_key": "X",
        "n_filings_by_major": {k: int(v) for k, v in n_filings_by_major.items()},
        "fields": summary_rows,
        "notes": (
            "Per-NTEE-major per-field EDA across every fraud-relevant Form 990 column. "
            "Canonicalization: one row per (ein, tax_period_end), latest return_ts. "
            "Religion (X) is computed from data/parsed/irs_xml; secular (A-W) from "
            "data/parsed/irs_xml_secular. Coverage = non-null rate within the table. "
            "Disclosure rate (booleans) = n_true / total canonical filings for that major. "
            "Rate-among-nonnull (booleans) = n_true / n_nonnull within the table."
        ),
    }
    (OUT_DIR / "per_field_summary.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  wrote {OUT_DIR / 'per_field_summary.json'}")
    print(f"  wrote {len(summary_rows)} per-field detail files to {OUT_FIELD_DIR}")


if __name__ == "__main__":
    main()
