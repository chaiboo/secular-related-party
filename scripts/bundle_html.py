"""Thin wrapper around p2_related_party/scripts/bundle_html.py with paths
redirected to the p3 project. Inlines every data/*.json into index.html."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "p2_related_party" / "scripts"))

import bundle_html as bh  # noqa: E402

P3 = ROOT / "p3_secular_compare"
bh.P2 = P3  # the canonical bundler still names its var P2; we redirect to P3
bh.HTML = P3 / "index.html"
bh.DATA = P3 / "data"

if __name__ == "__main__":
    bh.main()
