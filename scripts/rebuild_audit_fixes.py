"""
rebuild_audit_fixes.py
======================

One-shot rebuild that addresses the four defects from the methodology audit:

  1. Rate-construction inconsistency — uniformly use
     disclose_rate = n_distinct_filings_with_at_least_one_row / n_filings.
  2. Rate-comparability — tag every rate row with rate_kind and
     denominator_definition so the UI / interpretation layer can
     never again mix per-filing and conditional rates on a shared axis.
  3. Sentinel filter — religion-tuned token list missed the secular
     corpus's dominant placeholders (VACANT, etc.); extended list lives
     in p2_related_party/scripts/build_data.py and we just rebuild the
     affected files here.
  4. Partial 2025 — exclude 2025 from headline strip / corpus counts,
     keep it in per-field detail with a partial-year flag.

This script reads the existing JSONs, recomputes the rate sub-objects,
and writes them back. It does NOT re-run extract_secular.py (6-12h);
the parquets themselves are unchanged.

Outputs written:
    data/rel_vs_sec.json                rebuilt
    data/subgroup_with_religion.json    rebuilt
    data/subgroup_rates.json            rebuilt
    data/network/pt_vii_people.json     rebuilt
    data/network/sched_l_persons.json   rebuilt
    data/network/sched_r_tree.json      rebuilt (additive — sentinel logic doesn't touch it but we run to keep timestamps coherent)
    data/network/sankey_flows.json      rebuilt (uses _rel_bucket which references sentinel-y tokens; rebuild for parity)
    data/methodology.json               rewritten with current numbers
    data/findings.json                  partial_year flag added
    data/legal_interpretation_audit.json  NEW — kubera's worklist
    README.md                           rate numbers updated
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SECULAR_990 = ROOT / "data" / "parsed" / "irs_xml_secular" / "990"
RELIGION_990 = ROOT / "data" / "parsed" / "irs_xml" / "990"
OUT = ROOT / "p3_secular_compare" / "data"
NETDIR = OUT / "network"

sys.path.insert(0, str(ROOT / "p2_related_party" / "scripts"))
import build_data as bd  # noqa: E402
sys.path.insert(0, str(ROOT / "p3_secular_compare" / "scripts"))
import build_network_by_sector as bns  # noqa: E402

NTEE_MAJOR_LABEL = bns.NTEE_MAJOR_LABEL
VALID_MAJORS = bns.VALID_MAJORS
UNKNOWN_KEY = bns.UNKNOWN_KEY


# ---------- canonical rosters ---------------------------------------------

def _canon(irs_990: Path) -> pd.DataFrame:
    bd.IRS_990 = irs_990
    canon = bd.build_canonical_roster()
    canon["ntee_major"] = (
        canon["ntee_cd"].fillna("").astype(str).str[0].str.upper()
    )
    canon.loc[~canon["ntee_major"].isin(VALID_MAJORS), "ntee_major"] = UNKNOWN_KEY
    return canon


# ---------- per-corpus, per-table counts ---------------------------------

# Tables this audit covers. The strength label is editorial — kept here so
# the rebuilt JSONs can carry it without re-importing TABLE_SPEC.
TABLES_FRAUD_RELEVANT = [
    ("sched_l_excess_benefit", "strongest"),
    ("sched_l_loans", "strong"),
    ("sched_l_business_transactions", "strong"),
    ("sched_r_transactions", "strongest"),
    ("sched_r_related_tax_exempt", "moderate"),
    ("sched_j_main", "strong"),
]
TABLE_LABEL = {
    "sched_l_excess_benefit": "Schedule L Part I — Excess benefit transactions (§4958)",
    "sched_l_loans": "Schedule L Part II — Loans to/from interested persons",
    "sched_l_business_transactions": "Schedule L Part IV — Business transactions with interested persons",
    "sched_r_transactions": "Schedule R Part V — Transactions with related organizations",
    "sched_r_related_tax_exempt": "Schedule R Part II — Related tax-exempt organizations",
    "sched_j_main": "Schedule J Part I — Compensation policy indicators",
}


def per_table_distinct_filings(canon: pd.DataFrame, irs_990: Path,
                               include_2025: bool = True) -> dict:
    """For each fraud-relevant table, return {table -> n_distinct_filings_with_row}.

    Canonical filings = one row per (ein, tax_period_end), latest return_ts.
    A filing "has the disclosure" if at least one row in the table joins to
    that object_id."""
    keep_ids = canon["object_id"] if include_2025 else canon[canon["year"] != 2025]["object_id"]
    keep_set = set(keep_ids)
    out = {}
    for table, _ in TABLES_FRAUD_RELEVANT:
        path = irs_990 / f"{table}.parquet"
        if not path.exists():
            out[table] = {"n_distinct_filings_with_row": 0, "n_rows": 0, "n_eins": 0}
            continue
        df = pd.read_parquet(path, columns=["object_id", "ein"])
        df = df[df["object_id"].isin(keep_set)]
        out[table] = {
            "n_distinct_filings_with_row": int(df["object_id"].nunique()),
            "n_rows": int(len(df)),
            "n_eins": int(df["ein"].nunique()),
        }
    return out


def per_table_per_major(canon: pd.DataFrame, irs_990: Path,
                       include_2025: bool = True) -> dict:
    """For each table, return per-NTEE-major
       {ntee_major -> {n_filings, n_with_disclosure, rate}}."""
    sub = canon if include_2025 else canon[canon["year"] != 2025]
    n_total_by_major = sub.groupby("ntee_major").size().to_dict()
    object_to_major = sub.set_index("object_id")["ntee_major"].to_dict()
    keep_set = set(sub["object_id"])
    out = {}
    for table, _ in TABLES_FRAUD_RELEVANT:
        path = irs_990 / f"{table}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["object_id"])
        df = df[df["object_id"].isin(keep_set)]
        df["ntee_major"] = df["object_id"].map(object_to_major)
        with_table = df.groupby("ntee_major")["object_id"].nunique().to_dict()
        out[table] = {
            major: {
                "n_filings": int(n_tot),
                "n_with_disclosure": int(with_table.get(major, 0)),
                "rate": (with_table.get(major, 0) / max(n_tot, 1)),
            }
            for major, n_tot in n_total_by_major.items()
            if major in NTEE_MAJOR_LABEL
        }
    return out


# ---------- conditional rates (loans / sentinel / sched_r controlled) ----

def conditional_rates(canon: pd.DataFrame, irs_990: Path,
                      include_2025: bool = True) -> dict:
    """Compute the conditional-rate metrics used in rel_vs_sec.json.
    Each is a rate where the denominator is "rows in the relevant
    table" (NOT all filings) — so the kind is `conditional`."""
    keep = canon if include_2025 else canon[canon["year"] != 2025]
    keep_set = set(keep["object_id"])
    out = {}

    # --- sched_l_loans ---
    path = irs_990 / "sched_l_loans.parquet"
    if path.exists():
        df = pd.read_parquet(path, columns=["object_id", "in_default",
                                            "written_agreement"])
        df = df[df["object_id"].isin(keep_set)]
        n_default_known = int(df["in_default"].notna().sum())
        n_default_true = int((df["in_default"] == 1).sum())
        n_written_known = int(df["written_agreement"].notna().sum())
        n_unwritten = int((df["written_agreement"] == 0).sum())
        out["sched_l_loans"] = {
            "n_rows_total": int(len(df)),
            "n_default_known": n_default_known,
            "n_default_true": n_default_true,
            "pct_default": (n_default_true / max(n_default_known, 1)),
            "n_written_known": n_written_known,
            "n_unwritten": n_unwritten,
            "pct_unwritten": (n_unwritten / max(n_written_known, 1)),
        }

    # --- sched_l_sentinel (combined Pt II/III/IV) ---
    frames = []
    for tbl in ("sched_l_loans", "sched_l_business_transactions",
                "sched_l_grants_to_interested"):
        p = irs_990 / f"{tbl}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["object_id", "person_name"])
        d = d[d["object_id"].isin(keep_set)]
        frames.append(d)
    if frames:
        sl = pd.concat(frames, ignore_index=True)
        sl["pname"] = sl["person_name"].apply(bd.norm_name)
        sl["is_sent"] = sl["pname"].apply(bd.is_sentinel)
        out["sched_l_sentinel"] = {
            "n_rows_total": int(len(sl)),
            "n_sentinel": int(sl["is_sent"].sum()),
            "pct_sentinel": float(sl["is_sent"].sum() / max(len(sl), 1)),
        }

    # --- sched_r §512(b)(13) controlled (Pt II + Pt IV combined) ---
    parts = []
    for tbl in ("sched_r_related_tax_exempt", "sched_r_related_corp_trust"):
        p = irs_990 / f"{tbl}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["object_id", "section_512b_13_controlled"])
        d = d[d["object_id"].isin(keep_set)]
        parts.append(d)
    if parts:
        sr = pd.concat(parts, ignore_index=True)
        n_known = int(sr["section_512b_13_controlled"].notna().sum())
        n_ctrl = int((sr["section_512b_13_controlled"] == 1).sum())
        out["sched_r_controlled"] = {
            "n_rows_total": int(len(sr)),
            "n_known": n_known,
            "n_controlled": n_ctrl,
            "pct_controlled": (n_ctrl / max(n_known, 1)),
        }

    # --- sched_j contingent comp (per-filing — i.e. at least one row
    # where the boolean is true) ---
    p = irs_990 / "sched_j_main.parquet"
    if p.exists():
        d = pd.read_parquet(p, columns=["object_id",
                                        "compensation_contingent_on_revenues",
                                        "compensation_contingent_on_net_earnings"])
        d = d[d["object_id"].isin(keep_set)]
        any_contingent = d[(d["compensation_contingent_on_revenues"] == 1) |
                           (d["compensation_contingent_on_net_earnings"] == 1)]
        n_filings_total = len(keep)
        n_filings_any = int(any_contingent["object_id"].nunique())
        out["sched_j_contingent_per_filing"] = {
            "n_filings_total": n_filings_total,
            "n_filings_with_any_contingent": n_filings_any,
            "rate_per_filing": n_filings_any / max(n_filings_total, 1),
        }
    return out


# ---------- bring it all together ----------------------------------------

def main():
    parquet_mtime = datetime.fromtimestamp(
        (SECULAR_990 / "main_990.parquet").stat().st_mtime, tz=timezone.utc
    ).strftime("%Y-%m-%d")

    print("=" * 60)
    print("Step 1: canonical rosters for both corpora")
    print("=" * 60)
    sec_canon = _canon(SECULAR_990)
    rel_canon = _canon(RELIGION_990)
    print(f"  secular  canonical: {len(sec_canon):,} filings, "
          f"{sec_canon['ein'].nunique():,} EINs, "
          f"years {sec_canon['year'].min()}-{sec_canon['year'].max()}")
    print(f"  religion canonical: {len(rel_canon):,} filings, "
          f"{rel_canon['ein'].nunique():,} EINs, "
          f"years {rel_canon['year'].min()}-{rel_canon['year'].max()}")
    sec_no25 = sec_canon[sec_canon["year"] != 2025]
    rel_no25 = rel_canon[rel_canon["year"] != 2025]
    print(f"  secular  excl-2025: {len(sec_no25):,} filings, {sec_no25['ein'].nunique():,} EINs")
    print(f"  religion excl-2025: {len(rel_no25):,} filings, {rel_no25['ein'].nunique():,} EINs")

    # --- Step 2: per-table distinct filings, BOTH ways ----------------------
    print("\n" + "=" * 60)
    print("Step 2: per-table distinct-filings-with-row, both with + without 2025")
    print("=" * 60)
    sec_w25 = per_table_distinct_filings(sec_canon, SECULAR_990, include_2025=True)
    sec_no25_t = per_table_distinct_filings(sec_canon, SECULAR_990, include_2025=False)
    rel_w25 = per_table_distinct_filings(rel_canon, RELIGION_990, include_2025=True)
    rel_no25_t = per_table_distinct_filings(rel_canon, RELIGION_990, include_2025=False)
    for label, d, n in (("sec w25", sec_w25, len(sec_canon)),
                         ("sec no25", sec_no25_t, len(sec_no25)),
                         ("rel w25", rel_w25, len(rel_canon)),
                         ("rel no25", rel_no25_t, len(rel_no25))):
        for t, _ in TABLES_FRAUD_RELEVANT:
            v = d.get(t, {})
            r = v.get("n_distinct_filings_with_row", 0) / max(n, 1)
            print(f"  {label} {t}: {v.get('n_distinct_filings_with_row', 0):>7,} "
                  f"/ {n:>9,} = {r*100:7.4f}%")
        print()

    # --- Step 3: conditional rates, both ways -------------------------------
    print("=" * 60)
    print("Step 3: conditional rates (loans, sentinel, controlled, contingent)")
    print("=" * 60)
    sec_cond_w25 = conditional_rates(sec_canon, SECULAR_990, include_2025=True)
    sec_cond_no25 = conditional_rates(sec_canon, SECULAR_990, include_2025=False)
    rel_cond_w25 = conditional_rates(rel_canon, RELIGION_990, include_2025=True)
    rel_cond_no25 = conditional_rates(rel_canon, RELIGION_990, include_2025=False)
    print("  (computed)")

    # --- Step 4: rel_vs_sec.json ------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: rebuild rel_vs_sec.json")
    print("=" * 60)
    rel_vs_sec = build_rel_vs_sec(
        sec_canon, rel_canon, sec_no25, rel_no25,
        sec_w25, sec_no25_t, rel_w25, rel_no25_t,
        sec_cond_w25, sec_cond_no25, rel_cond_w25, rel_cond_no25,
        parquet_mtime,
    )
    (OUT / "rel_vs_sec.json").write_text(json.dumps(rel_vs_sec, indent=2))
    print(f"  wrote {OUT / 'rel_vs_sec.json'}")
    for r in rel_vs_sec["rows"]:
        print(f"    {r['label'][:50]:<50s}  kind={r['rate_kind']:<11s} "
              f"rel={r['religion']*100 if r['religion'] else 0:7.4f}% "
              f"sec={r['secular']*100 if r['secular'] else 0:7.4f}% "
              f"ratio={r.get('ratio', 0):5.2f}x  leader={r.get('subgroup_leader', '?')}")

    # --- Step 5: subgroup_rates.json (secular only) -----------------------
    print("\n" + "=" * 60)
    print("Step 5: rebuild subgroup_rates.json (secular)")
    print("=" * 60)
    sec_per_major_no25 = per_table_per_major(sec_canon, SECULAR_990, include_2025=False)
    sec_per_major_w25 = per_table_per_major(sec_canon, SECULAR_990, include_2025=True)
    subgroup = build_subgroup_secular(sec_canon, sec_no25,
                                       sec_per_major_no25, sec_per_major_w25,
                                       parquet_mtime)
    (OUT / "subgroup_rates.json").write_text(json.dumps(subgroup, indent=2))
    print(f"  wrote {OUT / 'subgroup_rates.json'}")

    # --- Step 6: subgroup_with_religion.json -------------------------------
    print("\n" + "=" * 60)
    print("Step 6: rebuild subgroup_with_religion.json (24 majors)")
    print("=" * 60)
    rel_per_major_no25 = per_table_per_major(rel_canon, RELIGION_990, include_2025=False)
    rel_per_major_w25 = per_table_per_major(rel_canon, RELIGION_990, include_2025=True)
    merged = build_subgroup_with_religion(
        subgroup, rel_canon, rel_no25,
        rel_per_major_no25, rel_per_major_w25, parquet_mtime
    )
    (OUT / "subgroup_with_religion.json").write_text(json.dumps(merged, indent=2))
    print(f"  wrote {OUT / 'subgroup_with_religion.json'}")

    # --- Step 7: findings.json — add partial_year flag --------------------
    print("\n" + "=" * 60)
    print("Step 7: stamp findings.json with partial_year_2025 flag")
    print("=" * 60)
    findings_path = OUT / "findings.json"
    findings = json.loads(findings_path.read_text())
    findings["_meta"] = {
        "partial_year_2025_in_aggregates": True,
        "rationale": ("Headline aggregates in this file pool 2014-2025; 2025 is a "
                      "partial cohort (~14% of a full year by row count). Use "
                      "rel_vs_sec.json for headline rates, which carry both a "
                      "2014-2024 'headline' rate and a 2014-2025 'with_2025' rate "
                      "side by side."),
        "data_snapshot_date": parquet_mtime,
    }
    # Correct the §4958 row to use distinct-filings (not raw rows).
    sec_eb = sec_w25["sched_l_excess_benefit"]
    sec_eb_no25 = sec_no25_t["sched_l_excess_benefit"]
    findings["sched_l_excess_benefit"]["n_distinct_filings_with_disclosure"] = sec_eb["n_distinct_filings_with_row"]
    findings["sched_l_excess_benefit"]["n_distinct_filings_with_disclosure_excl_2025"] = sec_eb_no25["n_distinct_filings_with_row"]
    findings["sched_l_excess_benefit"]["n_eins"] = sec_eb["n_eins"]
    findings["sched_l_excess_benefit"]["n_eins_excl_2025"] = sec_eb_no25["n_eins"]
    findings["sched_l_excess_benefit"]["n_rows"] = sec_eb["n_rows"]
    findings["sched_l_excess_benefit"]["n_rows_excl_2025"] = sec_eb_no25["n_rows"]
    findings["sched_l_excess_benefit"]["denominator_definition"] = (
        "n_distinct_filings_with_disclosure / n_filings (preferred). "
        "n_rows / n_filings is also reported but should NOT be called a 'rate'."
    )

    # Refresh sched_l_sentinel using the EXTENDED filter so findings.json
    # is internally consistent with rel_vs_sec.json post-audit.
    sec_sent = sec_cond_w25["sched_l_sentinel"]
    sec_sent_no25 = sec_cond_no25["sched_l_sentinel"]
    findings["sched_l_sentinel_overall"] = {
        "n_total": sec_sent["n_rows_total"],
        "n_sentinel": sec_sent["n_sentinel"],
        "pct_sentinel": sec_sent["pct_sentinel"],
        "n_total_excl_2025": sec_sent_no25["n_rows_total"],
        "n_sentinel_excl_2025": sec_sent_no25["n_sentinel"],
        "pct_sentinel_excl_2025": sec_sent_no25["pct_sentinel"],
        "filter_version": "extended_post_audit_2026-05",
    }

    findings_path.write_text(json.dumps(findings, indent=2))
    print(f"  wrote {findings_path}")

    # --- Step 8: rebuild network files with extended sentinel filter ------
    print("\n" + "=" * 60)
    print("Step 8: rebuild network files (sentinel filter extended)")
    print("=" * 60)
    print("  calling build_network_by_sector.main()  — combined corpora")
    bns.main()

    # --- Step 9: inspect top-50 of pt_vii_people.json -------------------
    print("\n" + "=" * 60)
    print("Step 9: post-rebuild inspection of pt_vii_people.json top-50")
    print("=" * 60)
    pv = json.loads((NETDIR / "pt_vii_people.json").read_text())
    for i, t in enumerate(pv.get("top_interlocks", [])[:50], 1):
        print(f"  {i:3d}. {t['person']!r}: {t['n_distinct_eins']} EINs")

    # --- Step 10: methodology rewrite -----------------------------------
    print("\n" + "=" * 60)
    print("Step 10: rebuild methodology.json (kills stale §4958 84-row claim)")
    print("=" * 60)
    methodology = build_methodology(
        sec_canon, rel_canon,
        sec_eb, sec_eb_no25,
        rel_w25["sched_l_excess_benefit"], rel_no25_t["sched_l_excess_benefit"],
        parquet_mtime,
    )
    (OUT / "methodology.json").write_text(json.dumps(methodology, indent=2))
    print(f"  wrote {OUT / 'methodology.json'}")

    # --- Step 11: legal_interpretation_audit.json -----------------------
    print("\n" + "=" * 60)
    print("Step 11: scan legal_interpretation.json for numeric claims")
    print("=" * 60)
    audit = build_legal_audit(rel_vs_sec, merged, sec_eb_no25, rel_no25_t["sched_l_excess_benefit"])
    (OUT / "legal_interpretation_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"  wrote {OUT / 'legal_interpretation_audit.json'}")
    print(f"  total numeric claims flagged: {len(audit['claims'])}")
    holds = sum(1 for c in audit["claims"] if c["status"] == "holds")
    revise = sum(1 for c in audit["claims"] if c["status"] == "needs_revision")
    delete = sum(1 for c in audit["claims"] if c["status"] == "should_delete")
    other = len(audit["claims"]) - holds - revise - delete
    print(f"    holds: {holds}, needs_revision: {revise}, should_delete: {delete}, unknown: {other}")


# ---------- builders ------------------------------------------------------

def build_rel_vs_sec(sec_canon, rel_canon, sec_no25, rel_no25,
                     sec_w25, sec_no25_t, rel_w25, rel_no25_t,
                     sec_cond_w25, sec_cond_no25,
                     rel_cond_w25, rel_cond_no25, parquet_mtime: str) -> dict:
    """The six head-to-head rows, tagged with rate_kind, denominator_definition,
    paired with_2025 / without_2025 numbers, absolute event counts, and
    subgroup_leader."""
    n_sec_w25 = len(sec_canon)
    n_sec_no25 = len(sec_no25)
    n_rel_w25 = len(rel_canon)
    n_rel_no25 = len(rel_no25)

    # --- subgroup_leader needs per_major rates for the per-filing tables ---
    sec_per_major_no25 = per_table_per_major(sec_canon, SECULAR_990, include_2025=False)
    rel_per_major_no25 = per_table_per_major(rel_canon, RELIGION_990, include_2025=False)
    # Religion is its own major X — merge.
    def _leader_for_table(table: str) -> dict:
        all_per = {}
        for major, v in sec_per_major_no25.get(table, {}).items():
            all_per[major] = v
        all_per["X"] = rel_per_major_no25.get(table, {}).get("X")
        # Some religion rosters may end up with `_unknown` — only keep VALID_MAJORS
        items = [(m, v) for m, v in all_per.items()
                 if v and m in NTEE_MAJOR_LABEL and m != UNKNOWN_KEY]
        if not items:
            return {"leader_major": None, "leader_label": None, "leader_rate": None}
        leader = max(items, key=lambda kv: kv[1]["rate"])
        major, v = leader
        return {
            "leader_major": major,
            "leader_label": NTEE_MAJOR_LABEL.get(major, major),
            "leader_rate": v["rate"],
            "leader_rank_of_religion": _rank_of(items, "X"),
        }

    rows = []

    # 1. In-default insider loans — CONDITIONAL
    rows.append({
        "label": "In-default insider loans (Sched L Pt II)",
        "metric": "share of insider-loan rows flagged in_default",
        "rate_kind": "conditional",
        "denominator_definition": "rows in sched_l_loans where in_default is non-null",
        "religion_headline": rel_cond_no25["sched_l_loans"]["pct_default"],
        "secular_headline": sec_cond_no25["sched_l_loans"]["pct_default"],
        "religion_with_2025": rel_cond_w25["sched_l_loans"]["pct_default"],
        "secular_with_2025": sec_cond_w25["sched_l_loans"]["pct_default"],
        "n_religion_events": rel_cond_no25["sched_l_loans"]["n_default_true"],
        "n_secular_events": sec_cond_no25["sched_l_loans"]["n_default_true"],
        "n_religion_denominator": rel_cond_no25["sched_l_loans"]["n_default_known"],
        "n_secular_denominator": sec_cond_no25["sched_l_loans"]["n_default_known"],
        "lower_is_safer": True,
        "subgroup_leader": None,  # conditional rate, no per-major leader
        "subgroup_leader_note": "Conditional rate (denominator = loan rows, not filings). NTEE-major leader board is in subgroup_with_religion.json under sched_l_loans (a per-filing rate) and is not directly comparable.",
    })
    # 2. Unwritten insider loans — CONDITIONAL
    rows.append({
        "label": "Unwritten insider loans (Sched L Pt II)",
        "metric": "share of insider-loan rows with no written agreement",
        "rate_kind": "conditional",
        "denominator_definition": "rows in sched_l_loans where written_agreement is non-null",
        "religion_headline": rel_cond_no25["sched_l_loans"]["pct_unwritten"],
        "secular_headline": sec_cond_no25["sched_l_loans"]["pct_unwritten"],
        "religion_with_2025": rel_cond_w25["sched_l_loans"]["pct_unwritten"],
        "secular_with_2025": sec_cond_w25["sched_l_loans"]["pct_unwritten"],
        "n_religion_events": rel_cond_no25["sched_l_loans"]["n_unwritten"],
        "n_secular_events": sec_cond_no25["sched_l_loans"]["n_unwritten"],
        "n_religion_denominator": rel_cond_no25["sched_l_loans"]["n_written_known"],
        "n_secular_denominator": sec_cond_no25["sched_l_loans"]["n_written_known"],
        "lower_is_safer": True,
        "subgroup_leader": None,
        "subgroup_leader_note": "Conditional rate; not in the per-major leaderboard.",
    })
    # 3. §512(b)(13) controlled — CONDITIONAL
    rows.append({
        "label": "§512(b)(13) controlled related orgs (Sched R Pt II + IV)",
        "metric": "share of related-entity rows with section_512b_13_controlled=True",
        "rate_kind": "conditional",
        "denominator_definition": "rows in sched_r_related_tax_exempt + sched_r_related_corp_trust where section_512b_13_controlled is non-null",
        "religion_headline": rel_cond_no25["sched_r_controlled"]["pct_controlled"],
        "secular_headline": sec_cond_no25["sched_r_controlled"]["pct_controlled"],
        "religion_with_2025": rel_cond_w25["sched_r_controlled"]["pct_controlled"],
        "secular_with_2025": sec_cond_w25["sched_r_controlled"]["pct_controlled"],
        "n_religion_events": rel_cond_no25["sched_r_controlled"]["n_controlled"],
        "n_secular_events": sec_cond_no25["sched_r_controlled"]["n_controlled"],
        "n_religion_denominator": rel_cond_no25["sched_r_controlled"]["n_known"],
        "n_secular_denominator": sec_cond_no25["sched_r_controlled"]["n_known"],
        "lower_is_safer": False,
        "subgroup_leader": None,
        "subgroup_leader_note": "Conditional rate; per-filing version of this metric is sched_r_related_tax_exempt in the per-major leaderboard.",
    })
    # 4. Sentinel counterparty names — CONDITIONAL (now post-extended-filter)
    rows.append({
        "label": "Sentinel counterparty names (Sched L combined)",
        "metric": "share of Sched L person-name fields that resolve to a sentinel non-name",
        "rate_kind": "conditional",
        "denominator_definition": "rows in sched_l_loans + sched_l_grants_to_interested + sched_l_business_transactions",
        "religion_headline": rel_cond_no25["sched_l_sentinel"]["pct_sentinel"],
        "secular_headline": sec_cond_no25["sched_l_sentinel"]["pct_sentinel"],
        "religion_with_2025": rel_cond_w25["sched_l_sentinel"]["pct_sentinel"],
        "secular_with_2025": sec_cond_w25["sched_l_sentinel"]["pct_sentinel"],
        "n_religion_events": rel_cond_no25["sched_l_sentinel"]["n_sentinel"],
        "n_secular_events": sec_cond_no25["sched_l_sentinel"]["n_sentinel"],
        "n_religion_denominator": rel_cond_no25["sched_l_sentinel"]["n_rows_total"],
        "n_secular_denominator": sec_cond_no25["sched_l_sentinel"]["n_rows_total"],
        "lower_is_safer": True,
        "subgroup_leader": None,
        "subgroup_leader_note": "Conditional rate; the per-sector breakdown lives in sched_l_persons.json -> sentinel_rates_by_sector.",
        "sentinel_filter_note": "Counts use the extended sentinel filter (post-audit) — VACANT, OPEN, TBD, VOLUNTEER, etc. now classified as sentinels.",
    })
    # 5. §4958 excess-benefit — PER-FILING
    leader_4958 = _leader_for_table("sched_l_excess_benefit")
    rate_rel_4958_no25 = rel_no25_t["sched_l_excess_benefit"]["n_distinct_filings_with_row"] / max(n_rel_no25, 1)
    rate_sec_4958_no25 = sec_no25_t["sched_l_excess_benefit"]["n_distinct_filings_with_row"] / max(n_sec_no25, 1)
    rate_rel_4958_w25 = rel_w25["sched_l_excess_benefit"]["n_distinct_filings_with_row"] / max(n_rel_w25, 1)
    rate_sec_4958_w25 = sec_w25["sched_l_excess_benefit"]["n_distinct_filings_with_row"] / max(n_sec_w25, 1)
    rows.append({
        "label": "§4958 excess-benefit admissions (Sched L Pt I)",
        "metric": "share of canonical filings with at least one Pt I row",
        "rate_kind": "per_filing",
        "denominator_definition": "n_distinct_filings_with_at_least_one_row / n_canonical_filings",
        "religion_headline": rate_rel_4958_no25,
        "secular_headline": rate_sec_4958_no25,
        "religion_with_2025": rate_rel_4958_w25,
        "secular_with_2025": rate_sec_4958_w25,
        "n_religion_events": rel_no25_t["sched_l_excess_benefit"]["n_distinct_filings_with_row"],
        "n_secular_events": sec_no25_t["sched_l_excess_benefit"]["n_distinct_filings_with_row"],
        "n_religion_denominator": n_rel_no25,
        "n_secular_denominator": n_sec_no25,
        "lower_is_safer": True,
        "subgroup_leader": leader_4958,
    })
    # 6. Contingent-on-revenue/net-earnings officer comp — PER-FILING
    rate_rel_cont_no25 = rel_cond_no25["sched_j_contingent_per_filing"]["rate_per_filing"]
    rate_sec_cont_no25 = sec_cond_no25["sched_j_contingent_per_filing"]["rate_per_filing"]
    rate_rel_cont_w25 = rel_cond_w25["sched_j_contingent_per_filing"]["rate_per_filing"]
    rate_sec_cont_w25 = sec_cond_w25["sched_j_contingent_per_filing"]["rate_per_filing"]
    rows.append({
        "label": "Contingent-on-revenue/net-earnings officer comp (Sched J Pt I)",
        "metric": "share of canonical filings disclosing at least one contingent-comp boolean = True",
        "rate_kind": "per_filing",
        "denominator_definition": "n_canonical_filings (any sched_j_main row with revenues OR net-earnings contingent = 1)",
        "religion_headline": rate_rel_cont_no25,
        "secular_headline": rate_sec_cont_no25,
        "religion_with_2025": rate_rel_cont_w25,
        "secular_with_2025": rate_sec_cont_w25,
        "n_religion_events": rel_cond_no25["sched_j_contingent_per_filing"]["n_filings_with_any_contingent"],
        "n_secular_events": sec_cond_no25["sched_j_contingent_per_filing"]["n_filings_with_any_contingent"],
        "n_religion_denominator": n_rel_no25,
        "n_secular_denominator": n_sec_no25,
        "lower_is_safer": True,
        "subgroup_leader": None,
        "subgroup_leader_note": "Per-filing rate but sched_j_main is in the per-major leaderboard as a 'has at least one row' rate, not a 'has contingent comp' rate; the two are not the same.",
    })

    for r in rows:
        if r["religion_headline"] and r["secular_headline"]:
            r["ratio"] = r["religion_headline"] / r["secular_headline"]
            r["religion"] = r["religion_headline"]  # legacy keys
            r["secular"] = r["secular_headline"]
        else:
            r["religion"] = r["religion_headline"]
            r["secular"] = r["secular_headline"]
            r["ratio"] = None

    return {
        "religion": {
            "n_filings": n_rel_w25,
            "n_eins": int(rel_canon["ein"].nunique()),
            "n_filings_excl_2025": n_rel_no25,
            "n_eins_excl_2025": int(rel_no25["ein"].nunique()),
            "label": "Religion-classified (NTEE-X)",
            "url": "https://chaiboo.github.io/religious-nonprofit-related-party/",
        },
        "secular": {
            "n_filings": n_sec_w25,
            "n_eins": int(sec_canon["ein"].nunique()),
            "n_filings_excl_2025": n_sec_no25,
            "n_eins_excl_2025": int(sec_no25["ein"].nunique()),
            "label": "Secular (NTEE A–W minus X)",
            "url": None,
        },
        "rows": rows,
        "_meta": {
            "partial_year_2025_excluded_from_headline": True,
            "headline_year_range": "2014-2024",
            "with_2025_year_range": "2014-2025",
            "rate_construction_for_per_filing": "n_distinct_filings_with_at_least_one_row / n_canonical_filings",
            "rate_construction_for_conditional": "see denominator_definition per row",
            "data_snapshot_date": parquet_mtime,
            "sentinel_filter_version": "extended_post_audit_2026-05",
            "note": ("Each row carries rate_kind ('per_filing' vs 'conditional'). "
                     "The UI must not place rows of different rate_kind on a "
                     "shared axis without explicit segregation."),
        },
    }


def _rank_of(items, key):
    """Return 1-based rank of `key` among items sorted by rate desc."""
    sorted_items = sorted(items, key=lambda kv: -kv[1]["rate"])
    for i, (k, _) in enumerate(sorted_items, 1):
        if k == key:
            return i
    return None


def build_subgroup_secular(sec_canon, sec_no25, per_major_no25, per_major_w25,
                            parquet_mtime: str) -> dict:
    """subgroup_rates.json — secular-only, 23 majors. Headline uses
    excl-2025; with_2025 carried alongside."""
    n_total_by_major_no25 = sec_no25.groupby("ntee_major").size().to_dict()
    filings_per_major = sorted(
        [{"ntee_major": m, "ntee_major_label": NTEE_MAJOR_LABEL.get(m, m),
          "n_filings": int(n)}
         for m, n in n_total_by_major_no25.items()
         if m in NTEE_MAJOR_LABEL and m != "X" and m != UNKNOWN_KEY],
        key=lambda r: -r["n_filings"],
    )
    n_total_by_major_w25 = sec_canon.groupby("ntee_major").size().to_dict()
    filings_per_major_w25 = sorted(
        [{"ntee_major": m, "ntee_major_label": NTEE_MAJOR_LABEL.get(m, m),
          "n_filings": int(n)}
         for m, n in n_total_by_major_w25.items()
         if m in NTEE_MAJOR_LABEL and m != "X" and m != UNKNOWN_KEY],
        key=lambda r: -r["n_filings"],
    )

    by_major = {}
    for table, strength in TABLES_FRAUD_RELEVANT:
        per_no25 = per_major_no25.get(table, {})
        per_w25 = per_major_w25.get(table, {})
        rows = []
        for m in [r["ntee_major"] for r in filings_per_major]:
            v_no25 = per_no25.get(m, {})
            v_w25 = per_w25.get(m, {})
            rows.append({
                "ntee_major": m,
                "label": NTEE_MAJOR_LABEL.get(m, m),
                "n_filings": v_no25.get("n_filings", 0),
                "n_with_disclosure": v_no25.get("n_with_disclosure", 0),
                "rate": v_no25.get("rate", 0.0),
                "n_filings_with_2025": v_w25.get("n_filings", 0),
                "n_with_disclosure_with_2025": v_w25.get("n_with_disclosure", 0),
                "rate_with_2025": v_w25.get("rate", 0.0),
            })
        rows.sort(key=lambda r: -r["rate"])
        by_major[table] = {
            "label": TABLE_LABEL[table],
            "fraud_strength": strength,
            "rate_kind": "per_filing",
            "denominator_definition": "n_distinct_filings_with_at_least_one_row / n_canonical_filings_in_major",
            "headline_year_range": "2014-2024",
            "per_major": rows,
        }
    return {
        "filings_per_major": filings_per_major,
        "filings_per_major_with_2025": filings_per_major_w25,
        "disclosure_rates_by_major": by_major,
        "_meta": {
            "rate_construction": "n_distinct_filings_with_at_least_one_row / n_canonical_filings_in_major",
            "rate_kind": "per_filing",
            "partial_year_2025_excluded_from_headline": True,
            "headline_year_range": "2014-2024",
            "with_2025_year_range": "2014-2025",
            "data_snapshot_date": parquet_mtime,
            "notes": "Secular-only (23 majors A-W minus X). Religion-merged version is subgroup_with_religion.json.",
        },
    }


def build_subgroup_with_religion(subgroup_secular, rel_canon, rel_no25,
                                  rel_per_major_no25, rel_per_major_w25,
                                  parquet_mtime: str) -> dict:
    """24-major version: secular + religion (X) row, same denominator
    construction. Headline excludes 2025; with_2025 carried alongside."""
    n_rel_no25 = len(rel_no25)
    n_rel_w25 = len(rel_canon)

    filings_per_major = list(subgroup_secular["filings_per_major"])
    filings_per_major = [r for r in filings_per_major if r["ntee_major"] != "X"]
    filings_per_major.append({
        "ntee_major": "X",
        "ntee_major_label": "Religion-related",
        "n_filings": n_rel_no25,
    })
    filings_per_major.sort(key=lambda r: -r["n_filings"])

    filings_per_major_w25 = list(subgroup_secular["filings_per_major_with_2025"])
    filings_per_major_w25 = [r for r in filings_per_major_w25 if r["ntee_major"] != "X"]
    filings_per_major_w25.append({
        "ntee_major": "X",
        "ntee_major_label": "Religion-related",
        "n_filings": n_rel_w25,
    })
    filings_per_major_w25.sort(key=lambda r: -r["n_filings"])

    new_disc = {}
    for table, tdata in subgroup_secular["disclosure_rates_by_major"].items():
        per_major = [r for r in tdata["per_major"] if r["ntee_major"] != "X"]
        rel_no25_t = rel_per_major_no25.get(table, {}).get("X", {})
        rel_w25_t = rel_per_major_w25.get(table, {}).get("X", {})
        per_major.append({
            "ntee_major": "X",
            "label": "Religion-related",
            "n_filings": rel_no25_t.get("n_filings", n_rel_no25),
            "n_with_disclosure": rel_no25_t.get("n_with_disclosure", 0),
            "rate": rel_no25_t.get("rate", 0.0),
            "n_filings_with_2025": rel_w25_t.get("n_filings", n_rel_w25),
            "n_with_disclosure_with_2025": rel_w25_t.get("n_with_disclosure", 0),
            "rate_with_2025": rel_w25_t.get("rate", 0.0),
        })
        per_major.sort(key=lambda r: -r["rate"])
        new_disc[table] = {
            "label": tdata["label"],
            "fraud_strength": tdata["fraud_strength"],
            "rate_kind": tdata["rate_kind"],
            "denominator_definition": tdata["denominator_definition"],
            "headline_year_range": tdata["headline_year_range"],
            "per_major": per_major,
        }
    return {
        "filings_per_major": filings_per_major,
        "filings_per_major_with_2025": filings_per_major_w25,
        "disclosure_rates_by_major": new_disc,
        "religion_key": "X",
        "religion_label": "Religion-related",
        "_meta": {
            "rate_construction": "n_distinct_filings_with_at_least_one_row / n_canonical_filings_in_major",
            "rate_kind": "per_filing",
            "partial_year_2025_excluded_from_headline": True,
            "headline_year_range": "2014-2024",
            "with_2025_year_range": "2014-2025",
            "data_snapshot_date": parquet_mtime,
            "notes": ("Per-NTEE-major disclosure rates for the 6 fraud-relevant Form 990 tables. "
                      "Religion (X) is computed from the parallel religion corpus using identical "
                      "canonicalization (latest return_ts per (ein, tax_period_end)) and identical "
                      "'at least one row per canonical filing' counting. 24 NTEE majors (A-W + X)."),
        },
    }


def build_methodology(sec_canon, rel_canon,
                       sec_eb, sec_eb_no25, rel_eb, rel_eb_no25,
                       parquet_mtime: str) -> list:
    """methodology.json — interpretive caveats, refreshed with actual numbers."""
    return [
        {
            "title": "Rate construction (post-audit)",
            "bullets": [
                "Every per-filing disclosure rate in this tool uses one denominator: "
                "<code>n_distinct_filings_with_at_least_one_row / n_canonical_filings</code>.",
                "This is &ldquo;what share of canonical filings disclosed this at all&rdquo;. "
                "<code>n_rows / n_filings</code> is NOT used as a rate (it over-weights orgs "
                "that disclose many rows in one filing).",
                "Conditional rates (in-default loans, unwritten loans, &sect;512(b)(13) controlled, "
                "sentinel names) have a different denominator (rows in the relevant table) and "
                "are explicitly tagged <code>rate_kind=conditional</code> in <code>rel_vs_sec.json</code>. "
                "The UI must not place rows of different rate_kind on the same axis.",
            ],
        },
        {
            "title": "Partial-year 2025 handling (post-audit)",
            "bullets": [
                f"2025 is a partial cohort &mdash; the snapshot is from {parquet_mtime}.",
                "Headline rates and corpus counts exclude 2025. The <em>with_2025</em> "
                "versions are carried alongside in every rates JSON for transparency.",
                "Per-field detail (columns/, summary.json) still includes 2025 with the "
                "year clearly visible &mdash; throwing away 2025 entirely would lose information.",
            ],
        },
        {
            "title": "Sentinel filter (extended post-audit)",
            "bullets": [
                "The religion-tuned sentinel filter caught <code>SUBSTANTIAL CONTRIBUTOR / NONE / VARIOUS / BOARD MEMBER / OFFICER</code> etc. "
                "but missed the secular corpus's dominant role-placeholders.",
                "Extended list now also catches: <code>VACANT, OPEN, POSITION OPEN, TBD, TBA, VOLUNTEER, RESIGNED, FORMER, RETIRED, "
                "DECEASED, INTERIM, ACTING, SEE ABOVE, SEE ATTACHED, AS NEEDED, NOT APPLICABLE, NOT REQUIRED, UNKNOWN, BOARD OF DIRECTORS</code>.",
                "A heuristic gate also flags single-word common role/surname stand-ins "
                "(<code>PRESIDENT, EXECUTIVE, MANAGER, FOUNDER</code> etc. as single uppercase tokens) "
                "and short all-letter initials (&le;3 chars).",
                "VACANT alone was #1 in the pre-audit <code>pt_vii_people.json</code> at 344 EINs.",
            ],
        },
        {
            "title": "Churches that don't file are invisible",
            "bullets": [
                "&sect;6033(a)(3)(A)(i) exempts churches from filing Form 990.",
                "The religion subset here is a self-selecting voluntary-filer cohort: ECFA members, federal-grant recipients under the Single Audit Act, religious nonprofits that opted in.",
                "Any rate or ratio you compute is filer-vs-filer, not population-vs-population &mdash; the denominator is what chose to file.",
            ],
        },
        {
            "title": "Group rulings cloak denominational structures",
            "bullets": [
                "Catholic dioceses, SBC affiliates, UMC/ELCA conferences typically file under a single group exemption (Rev. Proc. 80-27 / 2018-40).",
                "Sub-entities don't appear individually &mdash; fewer than 30 of 197 US Catholic dioceses file Form 990 directly.",
                "The visible related-party landscape is biased toward independent / non-denominational filers.",
            ],
        },
        {
            "title": "Sched R is a disclosure tree, not a peer network",
            "bullets": [
                "Only a small share of disclosed related tax-exempt EINs are themselves filers in this corpus &mdash; the rest are non-filing affiliates the filer is reporting on.",
                "Read the network as <em>disclosed corporate families</em>, not <em>a network of peer nonprofits</em>.",
            ],
        },
        {
            "title": "§4958 admissions are a floor, not a population",
            "bullets": [
                "Schedule L Part I (excess-benefit transactions) is a self-incrimination box &mdash; reporting a non-zero entry is a tax admission.",
                f"Religion corpus (2014-2024, excl partial-2025): {rel_eb_no25['n_distinct_filings_with_row']} canonical filings with at least one Pt I row, at {rel_eb_no25['n_eins']} distinct EINs, {rel_eb_no25['n_rows']} raw rows.",
                f"Secular corpus (2014-2024, excl partial-2025): {sec_eb_no25['n_distinct_filings_with_row']} canonical filings with at least one Pt I row, at {sec_eb_no25['n_eins']} distinct EINs, {sec_eb_no25['n_rows']} raw rows.",
                "The actual population of excess-benefit transactions is unknowable from Form 990 alone &mdash; this is the disclosed floor, not an estimate of behavior.",
            ],
        },
        {
            "title": "Sentinels are kept separate from real names",
            "bullets": [
                "Person-name and relationship columns include filer dodges plus role-placeholders.",
                "These are real disclosures with the counterparty obscured &mdash; not missing data.",
                "Surfaced as their own bucket in the Sched L persons tab, never merged with real names.",
            ],
        },
        {
            "title": "The 2019 dip is COVID, not a behavioral collapse",
            "bullets": [
                "Every related-party table shows a ~45% trough at tax-period-end 2019.",
                "Those filings actually landed in calendar 2020 under IRS extension orders; the tax_period field stamps them as 2019.",
                "Treat 2019 as not-a-comparable-year when reading any time series in the room.",
            ],
        },
        {
            "title": "Pt VII person-name interlocks are candidates, not confirmed identities",
            "bullets": [
                "<code>pt_vii_people.json</code>'s top-interlock list is dominated by common name collisions "
                "(<em>DAVID SMITH</em>, <em>MICHAEL JOHNSON</em> et al.) once placeholder sentinels are filtered.",
                "Each high-degree person is a candidate for further investigation, not a confirmed cross-org affiliation. "
                "Same-name collisions are systematic in a 187k-EIN secular corpus.",
            ],
        },
        {
            "title": "Corpus scope",
            "bullets": [
                "Form 990 (full) filings only &mdash; 990-EZ filers don't file Sched L Pt IV or Sched R, and 990-PF runs the different &sect;4941 self-dealing regime.",
                f"Secular: {len(sec_canon[sec_canon['year'] != 2025]):,} filings (2014-2024 headline), {sec_canon['ein'].nunique():,} EINs.",
                f"Religion (NTEE-X subset): {len(rel_canon[rel_canon['year'] != 2025]):,} filings (2014-2024 headline), {rel_canon['ein'].nunique():,} EINs.",
                f"Data snapshot: {parquet_mtime}.",
            ],
        },
    ]


def build_legal_audit(rel_vs_sec: dict, subgroup: dict, sec_eb_no25, rel_eb_no25) -> dict:
    """Walk legal_interpretation.json, find every numeric claim, and tag
    whether it still holds against the rebuilt rates."""
    legal = json.loads((OUT / "legal_interpretation.json").read_text())

    # Build a tiny lookup of current-truth values
    rvs_by_label = {r["label"]: r for r in rel_vs_sec["rows"]}
    eb_row = rvs_by_label.get("§4958 excess-benefit admissions (Sched L Pt I)", {})
    eb_rate_rel = eb_row.get("religion_headline")
    eb_rate_sec = eb_row.get("secular_headline")
    eb_ratio = eb_row.get("ratio")
    leader_4958 = eb_row.get("subgroup_leader", {}) or {}
    leader_4958_major = leader_4958.get("leader_major")
    leader_4958_rate = leader_4958.get("leader_rate")
    leader_4958_rank = leader_4958.get("leader_rank_of_religion")

    claims = []

    def _flag(field_id: str, key: str, snippet: str, status: str, current_value: str, note: str):
        claims.append({
            "claim_text": snippet,
            "field_id": field_id,
            "key": key,
            "current_value": current_value,
            "status": status,
            "note": note,
        })

    # --- Scan field-level claims ---
    # Patterns we care about: percentages, "Nx", row/EIN counts, dollar amounts, "n=N"
    pat_pct = re.compile(r"(\d+(?:\.\d+)?)%")
    pat_ratio = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(?:the\s+)?(?:secular|religion|rel)", re.I)
    pat_rows = re.compile(r"(\d+)\s+rows?", re.I)
    pat_eins = re.compile(r"(\d+)\s+(?:distinct\s+)?EINs?", re.I)
    pat_neq = re.compile(r"n\s*=\s*(\d+)", re.I)
    pat_dollars = re.compile(r"\$([\d,]+(?:\.\d+)?)", re.I)
    pat_filings = re.compile(r"([\d,]+)\s+filings", re.I)

    for field_id, fdata in legal["fields"].items():
        for key, val in fdata.items():
            if not isinstance(val, str):
                continue
            # 4958-specific claims
            if "0.095" in val or "0.038" in val or "2.5×" in val or "2.5x" in val.lower():
                # Old §4958 numbers (using n_rows/n_filings)
                current = f"new §4958 per-filing rates: religion={eb_rate_rel*100 if eb_rate_rel else 0:.4f}%, secular={eb_rate_sec*100 if eb_rate_sec else 0:.4f}%, ratio={eb_ratio if eb_ratio else 0:.2f}x"
                _flag(
                    field_id, key,
                    snippet=val[:200] + ("..." if len(val) > 200 else ""),
                    status="needs_revision",
                    current_value=current,
                    note="Pre-audit numbers used n_rows / n_filings (0.0947% / 0.0380% / 2.5x). Post-audit uses n_distinct_filings_with_row / n_filings.",
                )
            # Mental Health & Crisis (F) leading 0.108%
            if "0.108%" in val or "F actually leads" in val or "Mental Health & Crisis" in val and "0.10" in val:
                current = f"new leader: {leader_4958_major} ({NTEE_MAJOR_LABEL.get(leader_4958_major, '?')}) at {leader_4958_rate*100 if leader_4958_rate else 0:.4f}%; religion ranks #{leader_4958_rank}"
                _flag(
                    field_id, key,
                    snippet=val[:200] + ("..." if len(val) > 200 else ""),
                    status="holds" if leader_4958_major == "F" else "needs_revision",
                    current_value=current,
                    note="Pre-audit had F leading at 0.108%. Re-check under post-audit denominator.",
                )
            # "84 rows total" / "86 rows" / similar stale §4958 counts
            for m in pat_rows.finditer(val):
                n = int(m.group(1))
                if n in (84, 86):
                    _flag(
                        field_id, key,
                        snippet=val[max(0, m.start()-50):min(len(val), m.end()+80)],
                        status="needs_revision",
                        current_value=f"religion §4958: {rel_eb_no25['n_rows']} rows excl-2025; secular §4958: {sec_eb_no25['n_rows']} rows excl-2025",
                        note=f"Stale {n}-row claim. Replace with current religion (and/or secular) counts.",
                    )
            # 27 distinct EINs (stale)
            for m in pat_eins.finditer(val):
                n = int(m.group(1))
                if n == 27:
                    _flag(
                        field_id, key,
                        snippet=val[max(0, m.start()-50):min(len(val), m.end()+80)],
                        status="needs_revision",
                        current_value=f"religion §4958: {rel_eb_no25['n_eins']} distinct EINs excl-2025; secular §4958: {sec_eb_no25['n_eins']} distinct EINs excl-2025",
                        note="Stale '27 distinct EINs' claim. Replace.",
                    )
    # --- Comparison-pair claims ---
    for pair in legal.get("comparison_pairs", []):
        for key in ("rationale", "interpretive_warning", "what_to_conclude", "what_not_to_conclude"):
            val = pair.get(key, "")
            if not isinstance(val, str):
                continue
            for m in pat_filings.finditer(val):
                # filings counts are likely still close; flag for verification
                _flag(
                    pair.get("pair_id", "<no_id>"), key,
                    snippet=val[max(0, m.start()-30):min(len(val), m.end()+40)],
                    status="verify",
                    current_value="N/A (verify against subgroup_with_religion.filings_per_major)",
                    note="Filings-count claim; should be verified against filings_per_major in subgroup_with_religion.json.",
                )

    return {
        "_meta": {
            "purpose": "Worklist for kubera (Phase B): every numeric claim in legal_interpretation.json that may now be contradicted by the rebuilt data.",
            "generated_by": "p3_secular_compare/scripts/rebuild_audit_fixes.py",
            "do_not_use_as_authoritative_truth": "This file flags claims, it does not rewrite them. Phase B should rewrite legal_interpretation.json based on these flags.",
        },
        "status_legend": {
            "holds": "claim still consistent with post-audit data",
            "needs_revision": "claim is contradicted or numerically wrong under post-audit denominator",
            "should_delete": "claim depended on a metric that no longer exists or was always wrong",
            "verify": "claim should be cross-checked against current rebuilt data; cannot be auto-flagged",
        },
        "claims": claims,
    }


if __name__ == "__main__":
    main()
