"""
p3_secular_compare — data prep for the secular (non-X NTEE) corpus.

This is a thin wrapper around p2_related_party/scripts/build_data.py. We import
the canonical builder, monkey-patch its IRS_990 + OUT constants so it reads
from data/parsed/irs_xml_secular/ and writes to p3_secular_compare/data/,
then call main(). A second pass adds the NTEE-major stratification (the
'compare by subgroup' feature) and a religion-vs-secular comparison file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "p2_related_party" / "scripts"))

import build_data as bd  # noqa: E402

# --- redirect inputs and outputs --------------------------------------------

SECULAR_IRS_990 = ROOT / "data" / "parsed" / "irs_xml_secular" / "990"
SECULAR_OUT = ROOT / "p3_secular_compare" / "data"
SECULAR_COLDIR = SECULAR_OUT / "columns"
SECULAR_NETDIR = SECULAR_OUT / "network"
SECULAR_COLDIR.mkdir(parents=True, exist_ok=True)
SECULAR_NETDIR.mkdir(parents=True, exist_ok=True)

bd.IRS_990 = SECULAR_IRS_990
bd.OUT = SECULAR_OUT
bd.COLDIR = SECULAR_COLDIR
bd.NETDIR = SECULAR_NETDIR


# --- NTEE-major / subgroup stratification ---------------------------------

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
    "Y": "Mutual Benefit Organizations",
    "Z": "Unknown / Other",
}


def _attach_ntee_major(df: pd.DataFrame) -> pd.DataFrame:
    if "ntee_cd" in df.columns:
        df = df.copy()
        df["ntee_major"] = df["ntee_cd"].fillna("?").astype(str).str[0]
    return df


def build_subgroup_rates(canon: pd.DataFrame) -> dict:
    """For each related-party table that has a 'fraud strength' >= strong,
    compute the per-NTEE-major rate so the explorer can compare subgroups."""
    rates = {}
    canon = _attach_ntee_major(canon)
    if "ntee_major" not in canon.columns:
        return rates
    canon["ntee_major_label"] = canon["ntee_major"].map(NTEE_MAJOR_LABEL).fillna("Unknown / Other")
    # Filing volume per major
    rates["filings_per_major"] = (
        canon.groupby(["ntee_major", "ntee_major_label"]).size()
             .reset_index(name="n_filings")
             .sort_values("n_filings", ascending=False)
             .to_dict("records")
    )
    # Per major: %filings with sched_l_loans/biz, etc.
    by_major = {}
    object_to_major = canon.set_index("object_id")["ntee_major"].to_dict()
    n_total_by_major = canon.groupby("ntee_major").size().to_dict()

    for table, fraud_strength in [
        ("sched_l_excess_benefit", "strongest"),
        ("sched_l_loans", "strong"),
        ("sched_l_business_transactions", "strong"),
        ("sched_r_transactions", "strongest"),
        ("sched_r_related_tax_exempt", "moderate"),
        ("sched_j_main", "strong"),
    ]:
        path = SECULAR_IRS_990 / f"{table}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["object_id", "ein"])
        df = df[df.object_id.isin(canon.object_id)]
        df["ntee_major"] = df.object_id.map(object_to_major)
        # Distinct filings that have at least one row in this table per major
        with_table = df.groupby("ntee_major")["object_id"].nunique().to_dict()
        per_major = []
        for major, n_total in n_total_by_major.items():
            if major not in NTEE_MAJOR_LABEL:
                continue
            per_major.append({
                "ntee_major": major,
                "label": NTEE_MAJOR_LABEL[major],
                "n_filings": int(n_total),
                "n_with_disclosure": int(with_table.get(major, 0)),
                "rate": float(with_table.get(major, 0)) / max(n_total, 1),
            })
        by_major[table] = {
            "label": bd.TABLE_SPEC.get(table, {}).get("label", table),
            "fraud_strength": fraud_strength,
            "per_major": sorted(per_major, key=lambda r: -r["rate"]),
        }
    rates["disclosure_rates_by_major"] = by_major
    return rates


def build_religion_vs_secular() -> dict:
    """Headline rate comparison between p2 (religion) and p3 (secular)."""
    p2_roster = json.loads((ROOT / "p2_related_party" / "data" / "roster.json").read_text())
    p3_roster = json.loads((SECULAR_OUT / "roster.json").read_text())
    p2_findings = json.loads((ROOT / "p2_related_party" / "data" / "findings.json").read_text())
    p3_findings = json.loads((SECULAR_OUT / "findings.json").read_text())

    def _safe(d, *keys, default=None):
        for k in keys:
            if d is None:
                return default
            d = d.get(k) if isinstance(d, dict) else None
        return d if d is not None else default

    rows = [
        {
            "label": "In-default insider loans (Sched L Pt II)",
            "metric": "share of insider loans flagged in_default",
            "religion": _safe(p2_findings, "sched_l_loans", "n_default_true") /
                        max(_safe(p2_findings, "sched_l_loans", "n_default_known", default=1), 1),
            "secular":  _safe(p3_findings, "sched_l_loans", "n_default_true") /
                        max(_safe(p3_findings, "sched_l_loans", "n_default_known", default=1), 1),
            "lower_is_safer": True,
        },
        {
            "label": "Unwritten insider loans (Sched L Pt II)",
            "metric": "share of insider loans with no written agreement",
            # Use n_default_known as the denominator — same coverage as written_agreement field.
            "religion": _safe(p2_findings, "sched_l_loans", "n_unwritten") /
                        max(_safe(p2_findings, "sched_l_loans", "n_default_known", default=1), 1),
            "secular":  _safe(p3_findings, "sched_l_loans", "n_unwritten") /
                        max(_safe(p3_findings, "sched_l_loans", "n_default_known", default=1), 1),
            "lower_is_safer": True,
        },
        {
            "label": "§512(b)(13) controlled related orgs (Sched R Pt II + IV)",
            "metric": "share of related tax-exempts + corp/trust under filer control",
            "religion": _safe(p2_findings, "sched_r_controlled", "combined_pct"),
            "secular":  _safe(p3_findings, "sched_r_controlled", "combined_pct"),
            "lower_is_safer": False,
        },
        {
            "label": "Sentinel counterparty names (Sched L combined)",
            "metric": "share of Sched L person fields that use a sentinel non-name",
            "religion": _safe(p2_findings, "sched_l_sentinel_overall", "pct_sentinel"),
            "secular":  _safe(p3_findings, "sched_l_sentinel_overall", "pct_sentinel"),
            "lower_is_safer": True,
        },
        {
            "label": "§4958 excess-benefit admissions (Sched L Pt I)",
            "metric": "rate of filings with at least one Pt I row",
            "religion": _safe(p2_findings, "sched_l_excess_benefit", "n_rows") /
                        max(p2_roster.get("n_filings", 1), 1),
            "secular":  _safe(p3_findings, "sched_l_excess_benefit", "n_rows") /
                        max(p3_roster.get("n_filings", 1), 1),
            "lower_is_safer": True,
        },
        {
            "label": "Contingent-on-revenue officer comp (Sched J Pt I)",
            "metric": "filings disclosing comp tied to revenues",
            "religion": (_safe(p2_findings, "sched_j_contingent", "rev_true", default=0) +
                         _safe(p2_findings, "sched_j_contingent", "ne_true", default=0)) /
                        max(p2_roster.get("n_filings", 1), 1),
            "secular":  (_safe(p3_findings, "sched_j_contingent", "rev_true", default=0) +
                         _safe(p3_findings, "sched_j_contingent", "ne_true", default=0)) /
                        max(p3_roster.get("n_filings", 1), 1),
            "lower_is_safer": True,
        },
    ]
    for r in rows:
        if r["religion"] and r["secular"]:
            r["ratio"] = r["religion"] / r["secular"]
    return {
        "religion": {
            "n_filings": p2_roster.get("n_filings"),
            "n_eins": p2_roster.get("n_eins"),
            "label": "Religion-classified (NTEE-X)",
            "url": "https://chaiboo.github.io/religious-nonprofit-related-party/",
        },
        "secular": {
            "n_filings": p3_roster.get("n_filings"),
            "n_eins": p3_roster.get("n_eins"),
            "label": "Secular (NTEE A–W minus X)",
            "url": None,
        },
        "rows": rows,
    }


def main():
    # Step 1 — run the canonical p2 builder against the secular parquets
    print("=" * 60)
    print("Step 1: per-column / network / signal builds via p2 canonical")
    print("=" * 60)
    bd.main()
    print()

    # Step 2 — secular-specific NTEE-major stratification
    print("=" * 60)
    print("Step 2: NTEE-major subgroup stratification")
    print("=" * 60)
    canon = bd.build_canonical_roster()
    subgroup = build_subgroup_rates(canon)
    (SECULAR_OUT / "subgroup_rates.json").write_text(json.dumps(subgroup, indent=2))
    print(f"  {len(subgroup.get('disclosure_rates_by_major', {}))} tables stratified by NTEE major")

    # Step 3 — religion-vs-secular comparison rows
    print("=" * 60)
    print("Step 3: religion-vs-secular comparison rows")
    print("=" * 60)
    rvs = build_religion_vs_secular()
    (SECULAR_OUT / "rel_vs_sec.json").write_text(json.dumps(rvs, indent=2))
    print(f"  {len(rvs['rows'])} comparison rows written")


if __name__ == "__main__":
    main()
