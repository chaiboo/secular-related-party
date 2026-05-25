"""
build_network_by_sector.py
==========================

Rebuilds the four network JSONs at p3_secular_compare/data/network/ so every
row / edge / node carries a `ntee_major` letter (A-W secular, X religion).

The four files:
  * sankey_flows.json         (Sched R Pt V + Sched L flow data)
  * sched_r_tree.json         (Sched R Pt II corporate-family edges)
  * pt_vii_people.json        (cross-org officer interlock candidates)
  * sched_l_persons.json      (interested-person edges + sentinel bucket)

This script reads BOTH the secular parquets (data/parsed/irs_xml_secular/990/)
and the religion parquets (data/parsed/irs_xml/990/), tags every record with
its filer-EIN's NTEE major (from main_990.ntee_cd[0]), and emits per-sector
aggregate blocks for the explorer.

Hard rules honored:
  * Amendment-deduplication via build_canonical_roster() (same as p2/p3).
  * Existing JSON keys preserved — additive only, never destructive.
  * No composite scores. Per-sector raw counts + the existing headline metric.
  * If ntee_cd is missing or the first letter is not A-X, the row is tagged
    "_unknown" and reported in the status; never dropped silently.
  * Records tagged by the FILING EIN's sector. Related-entity sectors are NOT
    inferred — most related entities are not themselves 990 filers, so their
    sector is unknown.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SECULAR_990 = ROOT / "data" / "parsed" / "irs_xml_secular" / "990"
RELIGION_990 = ROOT / "data" / "parsed" / "irs_xml" / "990"
OUT = ROOT / "p3_secular_compare" / "data"
NETDIR = OUT / "network"

# Pull canonicalization, sentinel, txn-code helpers from the p2 canonical builder
sys.path.insert(0, str(ROOT / "p2_related_party" / "scripts"))
import build_data as bd  # noqa: E402

VALID_MAJORS = list("ABCDEFGHIJKLMNOPQRSTUVWX")  # 24 majors (A-W + X)
UNKNOWN_KEY = "_unknown"

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
    UNKNOWN_KEY: "Unknown / unmapped",
}


# ---------- canonical rosters from both corpora --------------------------

def _canon_from(irs_990: Path) -> pd.DataFrame:
    """Amendment-deduped roster, with ntee_major attached."""
    bd.IRS_990 = irs_990  # repoint the canonicaliser at this corpus
    canon = bd.build_canonical_roster()
    canon["ntee_major"] = (
        canon["ntee_cd"].fillna("").astype(str).str[0].str.upper()
    )
    canon.loc[~canon["ntee_major"].isin(VALID_MAJORS), "ntee_major"] = UNKNOWN_KEY
    return canon


def build_combined_canon() -> tuple[pd.DataFrame, dict[Path, pd.DataFrame]]:
    """Combined canonical roster, plus a per-corpus dict so loaders know which
    parquet directory each canonical filing came from."""
    sec = _canon_from(SECULAR_990)
    sec["_src"] = "secular"
    rel = _canon_from(RELIGION_990)
    rel["_src"] = "religion"
    combined = pd.concat([sec, rel], ignore_index=True)
    return combined, {SECULAR_990: sec, RELIGION_990: rel}


# ---------- helpers --------------------------------------------------------

def _load_table(table: str, columns: list[str], canon_by_corpus: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    """Read `table.parquet` from BOTH corpora, filter to canonical object_ids,
    attach ntee_major + year + foundation_group + business_name."""
    frames = []
    for corpus_dir, canon in canon_by_corpus.items():
        path = corpus_dir / f"{table}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=columns)
        keep = canon[["object_id", "ein", "year", "foundation_group",
                      "is_church_status", "business_name", "ntee_major"]]
        df = df.merge(keep, on="object_id", how="inner", suffixes=("", "__main"))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns + ["ein", "year", "foundation_group",
                                               "is_church_status", "business_name",
                                               "ntee_major"])
    return pd.concat(frames, ignore_index=True)


def _sector_bucket(ntee_major: str | None) -> str:
    if ntee_major is None or not isinstance(ntee_major, str) or ntee_major not in VALID_MAJORS:
        return UNKNOWN_KEY
    return ntee_major


def _quantile_pair(series: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (0.0, 0.0)
    return (float(s.median()), float(s.quantile(0.9)))


# ---------- (1) sankey_flows.json ----------------------------------------

def build_sankey_flows(canon_by_corpus: dict[Path, pd.DataFrame]) -> dict:
    out = {}

    # --- Sched R Pt V money flows ---
    txn = _load_table(
        "sched_r_transactions",
        ["object_id", "ein", "related_org_name", "transaction_type_code",
         "transaction_amount"],
        canon_by_corpus,
    )
    txn["txn_code"] = txn["transaction_type_code"].apply(bd.canon_txn_code)
    txn["amt"] = pd.to_numeric(txn["transaction_amount"], errors="coerce").fillna(0)
    # Attach §512(b)(13) controlled flag via the related entity name
    rosters = []
    for corpus_dir, _ in canon_by_corpus.items():
        for tbl in ("sched_r_related_tax_exempt.parquet",
                    "sched_r_related_corp_trust.parquet"):
            p = corpus_dir / tbl
            if not p.exists():
                continue
            r = pd.read_parquet(p, columns=["object_id", "entity_name",
                                            "section_512b_13_controlled"])
            rosters.append(r)
    roster = pd.concat(rosters, ignore_index=True) if rosters else pd.DataFrame(
        columns=["object_id", "entity_name", "section_512b_13_controlled"]
    )
    roster["entity_norm"] = roster["entity_name"].apply(bd.norm_name)
    roster_lookup = (roster.dropna(subset=["entity_norm"])
                     .drop_duplicates(["object_id", "entity_norm"], keep="last")
                     [["object_id", "entity_norm", "section_512b_13_controlled"]])
    txn["related_norm"] = txn["related_org_name"].apply(bd.norm_name)
    merged = txn.merge(
        roster_lookup,
        left_on=["object_id", "related_norm"],
        right_on=["object_id", "entity_norm"],
        how="left",
    )

    def _ctrl_label(v):
        if pd.isna(v):
            return "Controlled status not on Sched R"
        return "§512(b)(13)-controlled" if int(v) == 1 else "Related, not controlled"

    merged["ctrl_label"] = merged["section_512b_13_controlled"].apply(_ctrl_label)
    # Filter the SANKEY VISUALIZATION dataset to canonical foundation groups +
    # canonical txn codes (matching the p2 canonical builder's behaviour). The
    # by_sector aggregates below use ALL rows, including the dropped ones.
    fg_label = {
        "public_charity": "Public charity",
        "supporting_org": "Supporting organization",
        "church": "Church-status filer",
        "private_foundation": "Private foundation",
    }
    short_txn = {
        "A": "A · receipt of interest/rent/royalty",
        "B": "B · gift TO related org",
        "C": "C · gift FROM related org",
        "D": "D · loan TO related org",
        "E": "E · loan FROM related org",
        "F": "F · dividends received",
        "G": "G · asset sale TO related",
        "H": "H · asset purchase FROM related",
        "I": "I · asset exchange",
        "J": "J · lease facilities TO related",
        "K": "K · lease facilities FROM related",
        "L": "L · services performed FOR related",
        "M": "M · services performed BY related",
        "N": "N · shared facilities/staff",
        "O": "O · shared paid employees",
        "P": "P · expense reimbursement TO related",
        "Q": "Q · expense reimbursement FROM related",
        "R": "R · other transfer TO related",
        "S": "S · other transfer FROM related",
    }
    merged["txn_long"] = merged["txn_code"].map(short_txn)
    fg_filter = merged["foundation_group"].isin(fg_label.keys())
    code_filter = merged["txn_code"].isin(short_txn.keys())
    canon_sankey = merged[fg_filter & code_filter].copy()
    canon_sankey["fg_label"] = canon_sankey["foundation_group"].map(fg_label)

    # Per-sector sankey datasets (only sectors with at least one row contribute)
    # Two top-level aggregates that mirror the p2 sankey schema, plus a per-sector breakdown.
    def _by_dollar_count(df):
        return (df.groupby(["fg_label", "txn_long", "ctrl_label"], as_index=False)
                  .agg(amt=("amt", "sum"), n=("amt", "size")))

    def _flows_to_sankey(df, value_col):
        nodes, node_idx = [], {}

        def add(label, col_idx):
            key = (label, col_idx)
            if key not in node_idx:
                node_idx[key] = len(nodes)
                nodes.append({"name": label, "col": col_idx})
            return node_idx[key]

        edges_a = df.groupby(["fg_label", "txn_long"], as_index=False)[value_col].sum()
        edges_b = df.groupby(["txn_long", "ctrl_label"], as_index=False)[value_col].sum()
        links = []
        for _, r in edges_a.iterrows():
            if r[value_col] <= 0:
                continue
            links.append({"source": add(r["fg_label"], 0),
                          "target": add(r["txn_long"], 1),
                          "value": float(r[value_col])})
        for _, r in edges_b.iterrows():
            if r[value_col] <= 0:
                continue
            links.append({"source": add(r["txn_long"], 1),
                          "target": add(r["ctrl_label"], 2),
                          "value": float(r[value_col])})
        return {"nodes": nodes, "links": links}

    overall = _by_dollar_count(canon_sankey)
    sr_v_by_sector = {}
    for sector, grp in canon_sankey.groupby("ntee_major", dropna=False):
        sec_key = _sector_bucket(sector)
        agg = _by_dollar_count(grp)
        sr_v_by_sector[sec_key] = {
            "label": NTEE_MAJOR_LABEL.get(sec_key, sec_key),
            "n_rows": int(len(grp)),
            "n_distinct_ribbons": int(len(agg)),
            "total_dollars": float(grp["amt"].sum()),
            "by_dollar": _flows_to_sankey(agg, "amt"),
            "by_count": _flows_to_sankey(agg, "n"),
        }

    out["sched_r_v"] = {
        "by_dollar": _flows_to_sankey(overall, "amt"),
        "by_count": _flows_to_sankey(overall, "n"),
        "total_dollars": float(canon_sankey["amt"].sum()),
        "total_rows": int(len(canon_sankey)),
        "txn_codebook": bd._TXN_TYPE_DESC,
        "method_note": (
            "Each transaction flows from the filer's foundation group, through "
            "its transaction-type code, to whether the related counterparty is "
            "§512(b)(13)-controlled. Width is either dollar volume or row count. "
            "Sankey only shows the four canonical foundation groups + 19 canonical "
            "transaction codes; the by_sector aggregates count those same rows."
        ),
        "by_sector": sr_v_by_sector,
    }

    # --- Sched L money flows ---
    parts = []
    for tbl_name, lbl, amt_cols in [
        ("sched_l_loans", "Loans (Pt II)", ["balance_due", "original_principal"]),
        ("sched_l_grants_to_interested", "Grants (Pt III)", ["amount_of_assistance"]),
        ("sched_l_business_transactions", "Business transactions (Pt IV)", ["transaction_amount"]),
    ]:
        cols = ["object_id", "ein", "person_name", "relationship"] + amt_cols
        d = _load_table(tbl_name, cols, canon_by_corpus)
        d["sl_part"] = lbl
        amt = None
        for c in amt_cols:
            cur = pd.to_numeric(d[c], errors="coerce")
            amt = cur if amt is None else amt.fillna(cur)
        d["amt"] = amt.fillna(0)
        parts.append(d[["object_id", "ein", "person_name", "relationship",
                        "sl_part", "amt", "foundation_group", "ntee_major"]])
    sl = pd.concat(parts, ignore_index=True)

    def _rel_bucket(r):
        if r is None or pd.isna(r):
            return "Not specified"
        u = str(r).strip().upper()
        if not u:
            return "Not specified"
        if "FAMILY" in u: return "Family member"
        if "BOARD" in u or "DIRECTOR" in u or "TRUSTEE" in u: return "Board / director / trustee"
        if "OFFICER" in u or "PRESIDENT" in u or "CEO" in u or "EXECUTIVE" in u or "CFO" in u: return "Officer"
        if "KEY EMPLOYEE" in u or "EMPLOYEE" in u or "STAFF" in u: return "Employee / key employee"
        if "SUBSTANTIAL" in u or "CONTRIBUTOR" in u or "DONOR" in u: return "Substantial contributor"
        if "BUSINESS" in u or "OWNED" in u or "AFFILIAT" in u or "PARTNER" in u or "LLC" in u or "CORP" in u: return "Business affiliate"
        if u in {"SEE PART V", "SEE STATEMENT", "SEE ATTACHED", "SEE SCHEDULE O"}: return "Sentinel / dodge"
        return "Other / specific"

    fg_label = {
        "public_charity": "Public charity",
        "supporting_org": "Supporting organization",
        "church": "Church-status filer",
        "private_foundation": "Private foundation",
    }
    sl["rel_bucket"] = sl["relationship"].apply(_rel_bucket)
    sl["fg_label"] = sl["foundation_group"].map(fg_label).fillna("Other filer")

    def _sl_flows(df, value_col):
        nodes, node_idx = [], {}

        def add(label, col_idx):
            key = (label, col_idx)
            if key not in node_idx:
                node_idx[key] = len(nodes)
                nodes.append({"name": label, "col": col_idx})
            return node_idx[key]

        edges_a = df.groupby(["fg_label", "sl_part"], as_index=False)[value_col].sum()
        edges_b = df.groupby(["sl_part", "rel_bucket"], as_index=False)[value_col].sum()
        links = []
        for _, r in edges_a.iterrows():
            if r[value_col] <= 0:
                continue
            links.append({"source": add(r["fg_label"], 0),
                          "target": add(r["sl_part"], 1),
                          "value": float(r[value_col])})
        for _, r in edges_b.iterrows():
            if r[value_col] <= 0:
                continue
            links.append({"source": add(r["sl_part"], 1),
                          "target": add(r["rel_bucket"], 2),
                          "value": float(r[value_col])})
        return {"nodes": nodes, "links": links}

    by_amt = (sl[sl["amt"] > 0]
              .groupby(["fg_label", "sl_part", "rel_bucket"], as_index=False)
              .agg(amt=("amt", "sum"), n=("amt", "size")))
    by_n = (sl.groupby(["fg_label", "sl_part", "rel_bucket"], as_index=False)
              .agg(amt=("amt", "sum"), n=("amt", "size")))

    sl_by_sector = {}
    for sector, grp in sl.groupby("ntee_major", dropna=False):
        sec_key = _sector_bucket(sector)
        grp_pos = grp[grp["amt"] > 0]
        agg_amt = (grp_pos.groupby(["fg_label", "sl_part", "rel_bucket"], as_index=False)
                          .agg(amt=("amt", "sum"), n=("amt", "size")))
        agg_n = (grp.groupby(["fg_label", "sl_part", "rel_bucket"], as_index=False)
                    .agg(amt=("amt", "sum"), n=("amt", "size")))
        sl_by_sector[sec_key] = {
            "label": NTEE_MAJOR_LABEL.get(sec_key, sec_key),
            "n_rows": int(len(grp)),
            "n_distinct_ribbons": int(len(agg_n)),
            "total_dollars": float(grp["amt"].sum()),
            "by_dollar": _sl_flows(agg_amt, "amt"),
            "by_count": _sl_flows(agg_n, "n"),
        }

    out["sched_l"] = {
        "by_dollar": _sl_flows(by_amt, "amt"),
        "by_count": _sl_flows(by_n, "n"),
        "total_dollars": float(sl["amt"].sum()),
        "total_rows": int(len(sl)),
        "method_note": (
            "Each insider transaction flows from the filer's foundation group, "
            "through which Schedule L part it lives in, to the bucketed counterparty "
            "relationship. Width is either dollar volume or row count. Pt I "
            "excess-benefit transactions excluded — Pt I dollars are tax-on-excess-benefit, "
            "not transaction amount."
        ),
        "by_sector": sl_by_sector,
    }

    # Top-level _unknown reporting
    out["_unknown_counts"] = {
        "sched_r_v_rows": int((canon_sankey["ntee_major"] == UNKNOWN_KEY).sum()),
        "sched_l_rows": int((sl["ntee_major"] == UNKNOWN_KEY).sum()),
    }
    return out


# ---------- (2) sched_r_tree.json ----------------------------------------

def build_sched_r_tree(canon_by_corpus: dict[Path, pd.DataFrame]) -> dict:
    df = _load_table(
        "sched_r_related_tax_exempt",
        ["object_id", "ein", "entity_name", "entity_ein",
         "primary_activity", "exempt_code_section"],
        canon_by_corpus,
    )
    df["entity_ein"] = df["entity_ein"].apply(
        lambda x: None if pd.isna(x) or str(x).strip() == "" else str(x).strip()
    )
    df["entity_norm"] = df["entity_name"].apply(bd.norm_name)
    df["entity_key"] = df["entity_ein"].fillna(df["entity_norm"])

    # Edges aggregated at (ein, entity_key) — same as p2 canonical, but the
    # aggregation also collapses ntee_major (one per filer EIN, so safe).
    edges = (df.dropna(subset=["entity_key"])
               .groupby(["ein", "entity_key"], as_index=False)
               .agg(filer_name=("business_name", "first"),
                    entity_name=("entity_name", "first"),
                    entity_ein=("entity_ein", "first"),
                    primary_activity=("primary_activity", "first"),
                    years_disclosed=("year", "nunique"),
                    ntee_major=("ntee_major", "first")))
    deg = edges.groupby("entity_key").size().reset_index(name="cited_by_n_filers")
    edges = edges.merge(deg, on="entity_key")

    top_entities = (edges.drop_duplicates("entity_key")
                         .sort_values("cited_by_n_filers", ascending=False)
                         .head(200)
                         [["entity_key", "entity_name", "entity_ein",
                           "primary_activity", "cited_by_n_filers"]])

    # by_sector: count of distinct related entities per filer EIN, distribution per sector
    per_filer_per_sector = (edges.groupby(["ntee_major", "ein"])
                                  .size()
                                  .reset_index(name="n_entities_disclosed"))
    by_sector = {}
    for sector in sorted(edges["ntee_major"].dropna().unique()):
        sec_key = _sector_bucket(sector)
        sub_edges = edges[edges["ntee_major"] == sector]
        sub_per_filer = per_filer_per_sector[per_filer_per_sector["ntee_major"] == sector]
        if len(sub_per_filer) == 0:
            continue
        med, p90 = _quantile_pair(sub_per_filer["n_entities_disclosed"])
        by_sector[sec_key] = {
            "label": NTEE_MAJOR_LABEL.get(sec_key, sec_key),
            "n_edges": int(len(sub_edges)),
            "n_distinct_filers": int(sub_edges["ein"].nunique()),
            "n_distinct_entities": int(sub_edges["entity_key"].nunique()),
            "median_entities_per_filer": med,
            "p90_entities_per_filer": p90,
            "max_entities_per_filer": int(sub_per_filer["n_entities_disclosed"].max()),
        }

    return {
        "n_edges": int(len(edges)),
        "n_distinct_filers": int(edges["ein"].nunique()),
        "n_distinct_entities": int(edges["entity_key"].nunique()),
        "reciprocity": {
            "entities_cited_by_ge_2_filers": int((deg["cited_by_n_filers"] >= 2).sum()),
            "entities_cited_by_ge_5_filers": int((deg["cited_by_n_filers"] >= 5).sum()),
            "entities_cited_by_ge_10_filers": int((deg["cited_by_n_filers"] >= 10).sum()),
        },
        "top_entities": top_entities.to_dict("records"),
        "edges_for_top_entities": (
            edges[edges["entity_key"].isin(top_entities["entity_key"])]
            [["ein", "filer_name", "entity_key", "entity_name", "primary_activity",
              "years_disclosed", "cited_by_n_filers", "ntee_major"]]
            .to_dict("records")
        ),
        "by_sector": by_sector,
        "_unknown_count": int((edges["ntee_major"] == UNKNOWN_KEY).sum()),
    }


# ---------- (3) pt_vii_people.json --------------------------------------

def build_pt_vii_people(canon_by_corpus: dict[Path, pd.DataFrame]) -> dict:
    df = _load_table(
        "part_vii_officers",
        ["object_id", "ein", "person_name", "title",
         "reportable_comp_from_org", "reportable_comp_from_related"],
        canon_by_corpus,
    )
    df["pname"] = df["person_name"].apply(bd.norm_name)
    df = df[df["pname"].notna()]
    df = df[~df["pname"].apply(bd.is_sentinel)]
    df = df[df["pname"].str.len() >= 6]

    # Person -> distinct EINs (overall)
    pe = df.groupby("pname")["ein"].nunique().reset_index(name="n_distinct_eins")
    pe_year = (df.groupby(["pname", "year"])["ein"].nunique()
                 .groupby("pname").max().reset_index(name="n_eins_in_one_year"))
    pe = pe.merge(pe_year, on="pname")
    multi = pe[pe["n_distinct_eins"] >= 3].sort_values("n_distinct_eins", ascending=False)

    detail = []
    for _, r in multi.head(200).iterrows():
        sub = df[df["pname"] == r["pname"]]
        eins = (sub.groupby("ein")
                   .agg(filer_name=("business_name", "first"),
                        title=("title", lambda x: bd.norm_name(x.iloc[0]) if len(x) else None),
                        years=("year", "nunique"),
                        max_comp_from_org=("reportable_comp_from_org", "max"),
                        max_comp_from_related=("reportable_comp_from_related", "max"),
                        ntee_major=("ntee_major", "first"))
                   .reset_index())
        # The dominant sector for this person (used for chip filtering of the top-interlocks list)
        sector_counts = sub["ntee_major"].value_counts()
        detail.append({
            "person": r["pname"],
            "n_distinct_eins": int(r["n_distinct_eins"]),
            "n_eins_in_one_year": int(r["n_eins_in_one_year"]),
            "orgs": eins.to_dict("records"),
            "sectors": [
                {"ntee_major": _sector_bucket(s), "n_orgs": int(c)}
                for s, c in sector_counts.items()
            ],
        })

    # by_sector: interlock candidates per filer, distribution per sector
    # Define an "interlock candidate" at an EIN as a person who appears at ≥2 distinct EINs
    # and is on that EIN's officer roster.
    multi_persons = set(pe[pe["n_distinct_eins"] >= 2]["pname"])
    multi_df = df[df["pname"].isin(multi_persons)]
    per_filer = multi_df.groupby(["ntee_major", "ein"])["pname"].nunique().reset_index(
        name="n_interlock_candidates"
    )
    by_sector = {}
    for sector in sorted(df["ntee_major"].dropna().unique()):
        sec_key = _sector_bucket(sector)
        sub_per_filer = per_filer[per_filer["ntee_major"] == sector]
        sub_df = df[df["ntee_major"] == sector]
        n_persons_total = int(sub_df["pname"].nunique())
        n_persons_ge2 = int(sub_df.groupby("pname")["ein"].nunique().pipe(lambda s: (s >= 2).sum()))
        n_persons_ge3 = int(sub_df.groupby("pname")["ein"].nunique().pipe(lambda s: (s >= 3).sum()))
        n_persons_ge5 = int(sub_df.groupby("pname")["ein"].nunique().pipe(lambda s: (s >= 5).sum()))
        if len(sub_per_filer):
            med, p90 = _quantile_pair(sub_per_filer["n_interlock_candidates"])
            max_v = int(sub_per_filer["n_interlock_candidates"].max())
        else:
            med, p90, max_v = 0.0, 0.0, 0
        by_sector[sec_key] = {
            "label": NTEE_MAJOR_LABEL.get(sec_key, sec_key),
            "n_persons": n_persons_total,
            "n_persons_ge_2_orgs": n_persons_ge2,
            "n_persons_ge_3_orgs": n_persons_ge3,
            "n_persons_ge_5_orgs": n_persons_ge5,
            "median_interlock_candidates_per_filer": med,
            "p90_interlock_candidates_per_filer": p90,
            "max_interlock_candidates_per_filer": max_v,
        }

    return {
        "n_persons_ge_2_orgs": int((pe["n_distinct_eins"] >= 2).sum()),
        "n_persons_ge_3_orgs": int((pe["n_distinct_eins"] >= 3).sum()),
        "n_persons_ge_5_orgs": int((pe["n_distinct_eins"] >= 5).sum()),
        "top_interlocks": detail,
        "method_note": "Officers are upper-stripped, sentinel non-names removed, "
                       "min 6 characters required. Same-name collisions remain possible — "
                       "view each interlock as a candidate, not a confirmed identity match. "
                       "Each org entry carries the filing EIN's NTEE major; interlocks "
                       "spanning two sectors are visible by reading the orgs[] sector column.",
        "by_sector": by_sector,
        "_unknown_count": int((df["ntee_major"] == UNKNOWN_KEY).sum()),
    }


# ---------- (4) sched_l_persons.json ------------------------------------

def build_sched_l_persons(canon_by_corpus: dict[Path, pd.DataFrame]) -> dict:
    frames = []
    for part, lbl in [("sched_l_loans", "loan"),
                      ("sched_l_grants_to_interested", "grant"),
                      ("sched_l_business_transactions", "biz_txn")]:
        amt_cols = []
        if part == "sched_l_loans":
            amt_cols = ["original_principal", "balance_due"]
        elif part == "sched_l_grants_to_interested":
            amt_cols = ["amount_of_assistance"]
        elif part == "sched_l_business_transactions":
            amt_cols = ["transaction_amount"]
        cols = ["object_id", "ein", "person_name", "relationship"] + amt_cols
        df = _load_table(part, cols, canon_by_corpus)
        df["sched_l_part"] = lbl
        if "balance_due" in df.columns:
            df["amt"] = df["balance_due"].fillna(df.get("original_principal"))
        elif "amount_of_assistance" in df.columns:
            df["amt"] = df["amount_of_assistance"]
        elif "transaction_amount" in df.columns:
            df["amt"] = df["transaction_amount"]
        keep_cols = ["object_id", "ein", "business_name", "year", "sched_l_part",
                     "person_name", "relationship", "amt", "ntee_major"]
        frames.append(df[keep_cols])
    all_sl = pd.concat(frames, ignore_index=True)
    all_sl["pname"] = all_sl["person_name"].apply(bd.norm_name)
    all_sl["is_sentinel"] = all_sl["pname"].apply(bd.is_sentinel)

    # --- sentinel rates by sector ---
    # n_person_fields = total person-name field uses, n_sentinel = those that resolved to a sentinel
    sentinel_rates_by_sector = {}
    for sector in sorted(all_sl["ntee_major"].dropna().unique()):
        sec_key = _sector_bucket(sector)
        sub = all_sl[all_sl["ntee_major"] == sector]
        n_fields = int(len(sub))
        n_sent = int(sub["is_sentinel"].sum())
        # Per-token breakdown (only on sentinel rows)
        token_counts = (sub[sub["is_sentinel"]]
                        .groupby("pname", dropna=True)
                        .size()
                        .sort_values(ascending=False))
        by_token = {str(k): int(v) for k, v in token_counts.head(30).items()}
        sentinel_rates_by_sector[sec_key] = {
            "label": NTEE_MAJOR_LABEL.get(sec_key, sec_key),
            "n_person_fields": n_fields,
            "n_sentinel": n_sent,
            "rate": (n_sent / n_fields) if n_fields else 0.0,
            "by_token": by_token,
        }

    # Overall sentinel-token bucket (preserves the existing sentinel_examples shape)
    sentinel_overall = (all_sl[all_sl["is_sentinel"]]
                        .groupby("pname")
                        .size()
                        .reset_index(name="n")
                        .sort_values("n", ascending=False)
                        .head(30))

    # --- real-name edges with sector attached ---
    real = all_sl[~all_sl["is_sentinel"]].dropna(subset=["pname"])
    real_edges = (real.groupby(["ein", "pname"])
                      .agg(filer_name=("business_name", "first"),
                           parts=("sched_l_part", lambda s: sorted(set(s))),
                           n_rows=("amt", "size"),
                           sum_amt=("amt", "sum"),
                           years=("year", lambda y: sorted(set(int(v) for v in y.dropna()))),
                           ntee_major=("ntee_major", "first"))
                      .reset_index())
    multi_filer = real_edges.groupby("pname").size().reset_index(name="n_filers")
    multi_filer = (multi_filer[multi_filer["n_filers"] >= 2]
                   .sort_values("n_filers", ascending=False))

    # by_sector: counts of edges/persons/filers per sector
    by_sector = {}
    for sector in sorted(real_edges["ntee_major"].dropna().unique()):
        sec_key = _sector_bucket(sector)
        sub = real_edges[real_edges["ntee_major"] == sector]
        by_sector[sec_key] = {
            "label": NTEE_MAJOR_LABEL.get(sec_key, sec_key),
            "n_edges": int(len(sub)),
            "n_persons_real": int(sub["pname"].nunique()),
            "n_filers_with_real_person": int(sub["ein"].nunique()),
        }

    return {
        "n_edges": int(len(real_edges)),
        "n_persons_real": int(real_edges["pname"].nunique()),
        "n_filers_with_real_person": int(real_edges["ein"].nunique()),
        "n_persons_at_ge_2_filers": int(len(multi_filer)),
        "sentinel_examples": sentinel_overall.to_dict("records"),
        "cross_filer_persons": multi_filer.head(100).to_dict("records"),
        "method_note": (
            "Persons normalized via upper+strip. Same-name collisions possible. "
            "Each edge carries the filing EIN's NTEE major; cross-sector edges are "
            "those where the same person appears under two different majors."
        ),
        "sentinel_rates_by_sector": sentinel_rates_by_sector,
        "by_sector": by_sector,
        "_unknown_counts": {
            "edges": int((real_edges["ntee_major"] == UNKNOWN_KEY).sum()),
            "all_person_fields": int((all_sl["ntee_major"] == UNKNOWN_KEY).sum()),
        },
    }


# ---------- main ---------------------------------------------------------

def main():
    print("Building combined canonical roster (secular + religion)...")
    combined, by_corpus = build_combined_canon()
    print(f"  combined: {len(combined):,} canonical filings, "
          f"{combined['ein'].nunique():,} EINs")
    print("  by sector:")
    for sec, n in combined["ntee_major"].value_counts().items():
        print(f"    {sec}: {n:,}")
    n_unknown = int((combined["ntee_major"] == UNKNOWN_KEY).sum())
    print(f"  _unknown filings: {n_unknown}")

    # File sizes before
    sizes_before = {}
    for fname in ["sankey_flows", "sched_r_tree", "pt_vii_people", "sched_l_persons"]:
        p = NETDIR / f"{fname}.json"
        sizes_before[fname] = p.stat().st_size if p.exists() else 0

    print("\nBuilding sankey_flows.json...")
    sk = build_sankey_flows(by_corpus)
    (NETDIR / "sankey_flows.json").write_text(json.dumps(sk, indent=0))
    print(f"  sched_r_v: total ${sk['sched_r_v']['total_dollars']:,.0f} across {sk['sched_r_v']['total_rows']:,} txns; "
          f"{len(sk['sched_r_v']['by_sector'])} sectors with rows")
    print(f"  sched_l:   total ${sk['sched_l']['total_dollars']:,.0f} across {sk['sched_l']['total_rows']:,} rows; "
          f"{len(sk['sched_l']['by_sector'])} sectors with rows")
    print(f"  _unknown counts: {sk['_unknown_counts']}")

    print("\nBuilding sched_r_tree.json...")
    sr = build_sched_r_tree(by_corpus)
    (NETDIR / "sched_r_tree.json").write_text(json.dumps(sr, indent=0))
    print(f"  {sr['n_edges']:,} edges, {sr['n_distinct_filers']:,} filers, "
          f"{sr['n_distinct_entities']:,} entities; {len(sr['by_sector'])} sectors")
    print(f"  _unknown edges: {sr['_unknown_count']}")

    print("\nBuilding pt_vii_people.json...")
    pv = build_pt_vii_people(by_corpus)
    (NETDIR / "pt_vii_people.json").write_text(json.dumps(pv, indent=0))
    print(f"  ge2_orgs={pv['n_persons_ge_2_orgs']:,}, ge3={pv['n_persons_ge_3_orgs']:,}, "
          f"ge5={pv['n_persons_ge_5_orgs']:,}; {len(pv['by_sector'])} sectors")
    print(f"  _unknown rows: {pv['_unknown_count']}")

    print("\nBuilding sched_l_persons.json...")
    sl = build_sched_l_persons(by_corpus)
    (NETDIR / "sched_l_persons.json").write_text(json.dumps(sl, indent=0))
    print(f"  {sl['n_edges']:,} edges, {sl['n_persons_at_ge_2_filers']:,} persons at ≥2 filers; "
          f"{len(sl['sentinel_rates_by_sector'])} sectors w/ sentinel data")
    print(f"  _unknown counts: {sl['_unknown_counts']}")

    # File sizes after
    print("\nFile sizes (KB before -> after):")
    for fname in ["sankey_flows", "sched_r_tree", "pt_vii_people", "sched_l_persons"]:
        p = NETDIR / f"{fname}.json"
        before_kb = sizes_before[fname] / 1024
        after_kb = p.stat().st_size / 1024
        print(f"  {fname:20s}: {before_kb:9.1f} KB -> {after_kb:9.1f} KB "
              f"(+{after_kb - before_kb:+.1f} KB)")

    # Sanity check: per-sector aggregates should sum to total
    print("\nSum-check (per-sector aggregates vs top-level totals):")
    sum_sr_v_rows = sum(s["n_rows"] for s in sk["sched_r_v"]["by_sector"].values())
    sum_sl_rows = sum(s["n_rows"] for s in sk["sched_l"]["by_sector"].values())
    sum_sr_edges = sum(s["n_edges"] for s in sr["by_sector"].values())
    sum_sl_edges = sum(s["n_edges"] for s in sl["by_sector"].values())
    print(f"  sched_r_v: per-sector sum={sum_sr_v_rows:,} vs top-level={sk['sched_r_v']['total_rows']:,} "
          f"{'OK' if sum_sr_v_rows == sk['sched_r_v']['total_rows'] else 'MISMATCH'}")
    print(f"  sched_l:   per-sector sum={sum_sl_rows:,} vs top-level={sk['sched_l']['total_rows']:,} "
          f"{'OK' if sum_sl_rows == sk['sched_l']['total_rows'] else 'MISMATCH'}")
    print(f"  sched_r_tree edges: per-sector sum={sum_sr_edges:,} vs top-level={sr['n_edges']:,} "
          f"{'OK' if sum_sr_edges == sr['n_edges'] else 'MISMATCH'}")
    print(f"  sched_l edges:      per-sector sum={sum_sl_edges:,} vs top-level={sl['n_edges']:,} "
          f"{'OK' if sum_sl_edges == sl['n_edges'] else 'MISMATCH'}")

    # Spot-check NTEE major correctness on a few records
    print("\nSpot-check ntee_major correctness:")
    sample = combined[["ein", "ntee_cd", "ntee_major", "_src"]].drop_duplicates("ein").sample(
        n=10, random_state=0
    )
    for _, r in sample.iterrows():
        first = (str(r["ntee_cd"])[0] if pd.notna(r["ntee_cd"]) else "?").upper()
        ok = "OK" if r["ntee_major"] == first or (first not in VALID_MAJORS and r["ntee_major"] == UNKNOWN_KEY) else "FAIL"
        print(f"  EIN {r['ein']} ntee_cd={r['ntee_cd']!r:10s} -> ntee_major={r['ntee_major']} (src={r['_src']}) [{ok}]")


if __name__ == "__main__":
    main()
