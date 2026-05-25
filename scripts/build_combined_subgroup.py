"""
build_combined_subgroup.py

Compute per-NTEE-major disclosure rates for the religion (X) corpus using the
same canonicalization + counting rules as p3_secular_compare.build_subgroup_rates,
then merge the religion row into the existing secular subgroup_rates.json so the
explorer can render religion as the 24th group on a shared axis.

Outputs: p3_secular_compare/data/subgroup_with_religion.json

Hard rules honored:
  - Same canonicalization (latest return_ts per (ein, tax_period_end)) as p2/p3.
  - Same disclosure-table list & "at least one row per filing" definition.
  - Same NTEE major-label dictionary; religion gets letter "X" / label "Religion-related".
  - Verify that the resulting religion rates match findings.json / rel_vs_sec.json
    derived numbers within rounding (n_with_disclosure / n_filings).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REL_DIR = ROOT / "data" / "parsed" / "irs_xml" / "990"
OUT_DIR = ROOT / "p3_secular_compare" / "data"

# Reuse the canonicalization function from the canonical p2 builder.
sys.path.insert(0, str(ROOT / "p2_related_party" / "scripts"))
import build_data as bd  # noqa: E402

bd.IRS_990 = REL_DIR

TABLES_FRAUD_RELEVANT = [
    ("sched_l_excess_benefit", "strongest"),
    ("sched_l_loans", "strong"),
    ("sched_l_business_transactions", "strong"),
    ("sched_r_transactions", "strongest"),
    ("sched_r_related_tax_exempt", "moderate"),
    ("sched_j_main", "strong"),
]


def build_religion_row_per_table(canon: pd.DataFrame) -> dict:
    """For each fraud-relevant table, count distinct canonical filings (object_id)
    that have at least one row. Same rule as p3.build_subgroup_rates per_major loop."""
    out = {}
    n_total = len(canon)
    canon_object_ids = set(canon["object_id"])
    for table, _strength in TABLES_FRAUD_RELEVANT:
        path = REL_DIR / f"{table}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["object_id"])
        df = df[df["object_id"].isin(canon_object_ids)]
        n_with = int(df["object_id"].nunique())
        out[table] = {
            "n_filings": n_total,
            "n_with_disclosure": n_with,
            "rate": n_with / max(n_total, 1),
        }
    return out


def main():
    # 1. religion canonical roster (same fn as p2 / p3 secular)
    print("Building canonical religion roster from", REL_DIR)
    canon = bd.build_canonical_roster()
    print(f"  religion canonical filings: {len(canon)} (object_ids), {canon['ein'].nunique()} EINs")

    # 2. per-table religion stats
    print("Counting per-table disclosures for religion corpus")
    rel_rows = build_religion_row_per_table(canon)
    for t, r in rel_rows.items():
        print(f"  {t}: {r['n_with_disclosure']} / {r['n_filings']} = {r['rate']:.5f}")

    # 3. read existing secular subgroup file & merge religion in
    sec_path = OUT_DIR / "subgroup_rates.json"
    sec = json.loads(sec_path.read_text())

    # Add religion to filings_per_major
    rel_label = "Religion-related"
    filings_per_major = list(sec["filings_per_major"])
    # Drop any prior "X" entry if it slipped through
    filings_per_major = [r for r in filings_per_major if r.get("ntee_major") != "X"]
    filings_per_major.append({
        "ntee_major": "X",
        "ntee_major_label": rel_label,
        "n_filings": len(canon),
    })
    # Re-sort by n_filings desc so the rendering code can rely on a stable order
    filings_per_major.sort(key=lambda r: -r["n_filings"])

    # Add religion to each table's per_major
    new_disc = {}
    for table, tdata in sec["disclosure_rates_by_major"].items():
        per_major = [r for r in tdata["per_major"] if r.get("ntee_major") != "X"]
        if table in rel_rows:
            rel = rel_rows[table]
            per_major.append({
                "ntee_major": "X",
                "label": rel_label,
                "n_filings": rel["n_filings"],
                "n_with_disclosure": rel["n_with_disclosure"],
                "rate": rel["rate"],
            })
        per_major.sort(key=lambda r: -r["rate"])
        new_disc[table] = {
            "label": tdata["label"],
            "fraud_strength": tdata["fraud_strength"],
            "per_major": per_major,
        }

    merged = {
        "filings_per_major": filings_per_major,
        "disclosure_rates_by_major": new_disc,
        "religion_key": "X",
        "religion_label": rel_label,
        "notes": (
            "Per-NTEE-major disclosure rates for the 6 fraud-relevant Form 990 tables. "
            "Religion (X) is computed from the parallel religion corpus using identical "
            "canonicalization (latest return_ts per (ein, tax_period_end)) and identical "
            "'at least one row per canonical filing' counting. The 23 secular rows are "
            "carried over verbatim from subgroup_rates.json."
        ),
    }

    out_path = OUT_DIR / "subgroup_with_religion.json"
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"\nWrote {out_path}")
    print(f"  groups: {len(filings_per_major)}  (expect 24)")

    # 4. sanity-check the religion numbers against rel_vs_sec.json
    rvs = json.loads((OUT_DIR / "rel_vs_sec.json").read_text())
    # Excess benefit row: rate = n_rows / n_filings in rel_vs_sec
    eb = [r for r in rvs["rows"] if r["label"].startswith("§4958")]
    if eb and "sched_l_excess_benefit" in rel_rows:
        rate_here = rel_rows["sched_l_excess_benefit"]["rate"]
        rate_there = eb[0]["religion"]
        print(f"\nsanity §4958: here={rate_here:.6f} vs rel_vs_sec={rate_there:.6f} (diff={(rate_here - rate_there):.6f})")


if __name__ == "__main__":
    main()
