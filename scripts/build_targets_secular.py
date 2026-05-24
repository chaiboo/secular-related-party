"""
Build the IRS filing target list for the SECULAR (non-X NTEE) comparison corpus.

Mirrors scripts/extract/build_targets.py but inverts the NTEE filter — every
NTEE major group EXCEPT X (religion-related). Output goes to
data/parsed/irs_xml_secular/targets.parquet so the secular extraction reads
from a parallel filesystem path and does not collide with the p2 NTEE-X corpus.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEGACY_PROC = ROOT / "data" / "processed" / "p1"
PARSED_DIR = ROOT / "data" / "parsed" / "irs_xml_secular"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

KEEP_RETURN_TYPES = {"990", "990O", "990EO", "990PR", "990EZ", "990PF"}


def main() -> None:
    idx = pd.read_parquet(LEGACY_PROC / "irs_990_index_all_years.parquet")
    print(f"Loaded full IRS index: {len(idx):,} rows")

    bmf = pd.read_parquet(ROOT / "data" / "parsed" / "bmf" / "bmf_full.parquet")
    bmf["EIN"] = bmf["EIN"].astype(str).str.zfill(9)
    bmf["ntee_major"] = bmf["NTEE_CD"].fillna("?").str[0]
    # SECULAR = every NTEE major present in BMF EXCEPT X. Also drop blank/unknown majors
    # since those are mostly admin records, miscoded, or pre-NTEE-system filers.
    valid_majors = set("ABCDEFGHIJKLMNOPQRSTUVW")  # NTEE-CC majors A-W minus X (religion)
    bmf = bmf[bmf["ntee_major"].isin(valid_majors)].copy()
    print(f"BMF non-X (NTEE majors A-W minus X): {len(bmf):,} orgs ({bmf['EIN'].nunique():,} unique EINs)")
    print()
    print("Top NTEE majors in secular BMF:")
    print(bmf["ntee_major"].value_counts().head(15).to_string())

    # Foundation grouping — reuse the p2 helper
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "scripts" / "eda"))
    from _helpers import foundation_to_group as _fgrp  # type: ignore
    bmf["foundation_group"] = [
        _fgrp(f, s) for f, s in zip(bmf["FOUNDATION"], bmf["SUBSECTION"])
    ]

    bmf_slim = bmf[["EIN", "NAME", "STATE", "SUBSECTION", "FOUNDATION", "NTEE_CD",
                    "RULING", "ntee_major", "foundation_group"]].rename(columns={
        "NAME": "bmf_name",
        "STATE": "bmf_state",
    })
    bmf_slim["NTEE_SUBCODE"] = bmf_slim["NTEE_CD"].str[:3]
    # IS_CHURCH_STATUS = False for the entire secular corpus by construction.
    # IS_FILING_RELIGIOUS_NONPROFIT = False as well — kept for column-shape parity with p2.
    bmf_slim["IS_CHURCH_STATUS"] = False
    bmf_slim["IS_FILING_RELIGIOUS_NONPROFIT"] = False

    idx["EIN"] = idx["EIN"].astype(str).str.zfill(9)
    idx = idx[idx["RETURN_TYPE"].isin(KEEP_RETURN_TYPES)].copy()

    merged = idx.merge(bmf_slim, on="EIN", how="inner")
    print(f"\nSecular filings (any form variant): {len(merged):,}")
    print(f"Distinct EINs: {merged['EIN'].nunique():,}")

    print("\nForm variant × filing year:")
    pivot = merged.groupby(["filing_year", "RETURN_TYPE"]).size().unstack(fill_value=0).sort_index()
    print(pivot)

    out = PARSED_DIR / "targets.parquet"
    merged.to_parquet(out, index=False)
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
