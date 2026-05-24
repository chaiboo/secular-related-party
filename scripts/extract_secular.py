"""
Run the IRS XML extractor over the secular (non-X NTEE) targets, writing to
a parallel parquet directory at data/parsed/irs_xml_secular/ so the religion
corpus parquets at data/parsed/irs_xml/ are untouched.

This monkey-patches the shared scripts/extract/extract_irs_xml.py module's
path constants before invoking its main(). It's a thin wrapper — no logic
divergence — so any future fix to the canonical extractor flows here for free.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "extract"))

import extract_irs_xml as ext  # noqa: E402

# Redirect everything to the secular parallel filesystem
SECULAR_PARSED = ROOT / "data" / "parsed" / "irs_xml_secular"
SECULAR_PARSED.mkdir(parents=True, exist_ok=True)
ext.PARSED_DIR = SECULAR_PARSED
ext.PARTIAL = SECULAR_PARSED / "_partial"
ext.PARTIAL.mkdir(parents=True, exist_ok=True)
ext.LOG_PATH = ROOT / "data" / "irs_xml_extract_secular.log"
ext.DONE_PATH = SECULAR_PARSED / "zips_done.txt"
ext.ZIP_CACHE = Path("/tmp/p3_zip_cache_secular")
ext.ZIP_CACHE.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ext.main()
