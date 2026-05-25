"""
bundle_html.py
==============

Inlines every data file into the single <script id="DATA" type="application/json">
blob in index.html. The HTML reads ONE blob at boot via JSON.parse on that tag's
text content; sub-views index into top-level keys (codebook, legal, summary,
trends, detail, rel_vs_sec, fraud_signals, methodology, findings,
subgroup_with_religion).

Network files (sched_r_tree, pt_vii_people, sched_l_persons, sankey_flows) are
NOT inlined — they're fetched lazily by the page when the user enters the
relevant mode, because they're large (8 MB / 5 MB / etc).

Re-runnable. Overwrites the DATA blob in place.

Reads:
  data/codebook.json                                   -> DATA.codebook
  data/legal_interpretation.json                       -> DATA.legal
  data/eda_compare/per_field_summary.json              -> DATA.summary
  data/eda_compare/trends.json                         -> DATA.trends
  data/rel_vs_sec.json                                 -> DATA.rel_vs_sec
  data/fraud_signals.json                              -> DATA.fraud_signals
  data/methodology.json                                -> DATA.methodology
  data/findings.json                                   -> DATA.findings
  data/subgroup_with_religion.json                     -> DATA.subgroup_with_religion
  (merged) data/eda_compare/per_field/*.json           -> DATA.detail

Writes: index.html (in place)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "index.html"

SOURCES = [
    ("codebook",               ROOT / "data" / "codebook.json"),
    ("legal",                  ROOT / "data" / "legal_interpretation.json"),
    ("summary",                ROOT / "data" / "eda_compare" / "per_field_summary.json"),
    ("trends",                 ROOT / "data" / "eda_compare" / "trends.json"),
    ("rel_vs_sec",             ROOT / "data" / "rel_vs_sec.json"),
    ("fraud_signals",          ROOT / "data" / "fraud_signals.json"),
    ("methodology",            ROOT / "data" / "methodology.json"),
    ("findings",               ROOT / "data" / "findings.json"),
    ("subgroup_with_religion", ROOT / "data" / "subgroup_with_religion.json"),
]

PER_FIELD_DIR = ROOT / "data" / "eda_compare" / "per_field"

DATA_TAG = re.compile(
    r'(<script\s+id="DATA"\s+type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def merge_per_field() -> dict:
    merged = {}
    for fp in sorted(PER_FIELD_DIR.glob("*.json")):
        merged[fp.stem] = json.loads(fp.read_text())
    return merged


def sanitize(compact: str) -> str:
    compact = re.sub(r"\bNaN\b", "null", compact)
    compact = re.sub(r"\bInfinity\b", "null", compact)
    compact = re.sub(r"\b-Infinity\b", "null", compact)
    return compact


def main():
    blob = {}
    for key, path in SOURCES:
        if not path.exists():
            print(f"  MISSING  {path} — skipping {key}")
            continue
        blob[key] = json.loads(path.read_text())
        print(f"  loaded   {key:<24} from {path.relative_to(ROOT)}")

    if PER_FIELD_DIR.exists():
        blob["detail"] = merge_per_field()
        print(f"  merged   {'detail':<24} from {len(blob['detail'])} files in {PER_FIELD_DIR.relative_to(ROOT)}")
    else:
        print(f"  MISSING  {PER_FIELD_DIR} — skipping detail")

    payload = sanitize(json.dumps(blob, separators=(",", ":"), ensure_ascii=False))

    html = HTML.read_text()
    if not DATA_TAG.search(html):
        raise SystemExit("FATAL: no <script id=\"DATA\" type=\"application/json\"> tag in index.html")

    html = DATA_TAG.sub(lambda m: m.group(1) + payload + m.group(3), html)
    HTML.write_text(html)

    print(f"\n  payload size: {len(payload)/1024:.1f} KB")
    print(f"  final index.html: {HTML.stat().st_size/1024:.1f} KB")
    print(f"  top-level keys: {list(blob.keys())}")


if __name__ == "__main__":
    main()
