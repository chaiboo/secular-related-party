"""
classify_comparisons.py
=======================

Reads per_field_summary.json and classifies each field by how religion sits
in the cross-NTEE distribution. The four buckets:

  - distinct_high       : religion is at or above the 90th percentile across
                          the 23 secular majors on the comparison metric.
                          (Religion is genuinely high relative to the field as
                          a whole.)
  - distinct_low        : religion is at or below the 10th percentile across
                          secular majors. (Religion is genuinely low.)
  - looks_like_<sector> : religion sits within +/- 15% of one specific sector
                          and is not at either tail; that sector is closest
                          neighbour.
  - inside_pack         : religion sits between p25 and p75 of secular and
                          has no notably close neighbour.

The comparison metric is:
  - boolean : disclose_rate (n_true / canonical filings for that major)
              OR rate-among-nonnull if disclose_rate is dominated by coverage
              gaps. We pick disclose_rate as the primary because it matches
              how the headline aggregates were computed in rel_vs_sec.json.
  - numeric : median value (no zero-handling; let the medians speak)
  - categorical : skipped here (categorical comparisons surface in the
                  vocab-share view, not here)

Also computes:
  - religion_rank_of_24 (1 = highest religion-vs-others)
  - secular_min, p25, p50, p75, p90, max for the metric (24-major distribution)
  - closest_sector by absolute distance

Output: eda_compare/comparison_classification.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "eda_compare" / "per_field_summary.json"
OUT_PATH = ROOT / "data" / "eda_compare" / "comparison_classification.json"


def metric_for_field(kind: str, m: dict) -> float | None:
    if m is None:
        return None
    if kind == "boolean":
        # primary metric: disclose_rate; if 0 and nonnull also 0, fall back to coverage
        v = m.get("disclose_rate")
        if v is None or (v == 0 and m.get("n_nonnull", 0) == 0):
            return None
        return v
    if kind == "numeric":
        return m.get("median")
    return None


def main():
    src = json.loads(IN_PATH.read_text())
    out_fields = []
    for f in src["fields"]:
        kind = f["kind"]
        if kind not in ("boolean", "numeric"):
            continue
        per_major = f["per_major"]
        rel_metric = metric_for_field(kind, per_major.get("X"))
        # Build secular distribution (exclude X, exclude majors with no data)
        secular_pts = []
        for major, m in per_major.items():
            if major == "X":
                continue
            v = metric_for_field(kind, m)
            if v is None:
                continue
            secular_pts.append((major, v))
        if rel_metric is None or len(secular_pts) < 5:
            continue
        secular_vals = np.array([v for _, v in secular_pts])
        secular_majors = [m for m, _ in secular_pts]
        rel = rel_metric
        # Rank: 1 = highest including religion
        all_vals = np.append(secular_vals, rel)
        all_majors = secular_majors + ["X"]
        order = np.argsort(-all_vals)
        ranks = {all_majors[i]: int(np.where(order == idx)[0][0]) + 1
                 for idx, i in enumerate(range(len(all_majors)))}
        # Quantiles of secular distribution
        q = {
            "min": float(np.min(secular_vals)),
            "p10": float(np.quantile(secular_vals, 0.10)),
            "p25": float(np.quantile(secular_vals, 0.25)),
            "p50": float(np.quantile(secular_vals, 0.50)),
            "p75": float(np.quantile(secular_vals, 0.75)),
            "p90": float(np.quantile(secular_vals, 0.90)),
            "max": float(np.max(secular_vals)),
        }
        # Closest sector by absolute distance
        diffs = np.abs(secular_vals - rel)
        closest_i = int(np.argmin(diffs))
        closest = {
            "major": secular_majors[closest_i],
            "value": float(secular_vals[closest_i]),
            "rel_diff_pct": (
                float(abs(secular_vals[closest_i] - rel) / max(abs(rel), 1e-12))
                if rel != 0 else None
            ),
        }
        # Degenerate-medians guard: if rel == 0 AND p90_secular == 0, the metric
        # is uninformative (e.g. most officers don't get bonuses, so median is 0
        # everywhere). We mark these "degenerate" rather than distinct.
        all_zero = (rel == 0 and q["p90"] == 0)
        if all_zero:
            bucket = "degenerate"
        elif rel >= q["p90"] and rel > 0:
            bucket = "distinct_high"
        elif rel <= q["p10"] and q["max"] > 0:
            bucket = "distinct_low"
        elif closest["rel_diff_pct"] is not None and closest["rel_diff_pct"] < 0.15:
            bucket = f"looks_like_{closest['major']}"
        else:
            bucket = "inside_pack"

        out_fields.append({
            "table": f["table"],
            "column": f["column"],
            "kind": kind,
            "fraud_strength": f["fraud_strength"],
            "table_label": f["table_label"],
            "religion_metric": rel,
            "religion_rank_of_24": ranks["X"],
            "secular_quantiles": q,
            "closest_secular": closest,
            "bucket": bucket,
        })

    # Sort: distinct_high first (most surprising), then distinct_low, then looks_like, then inside_pack, then degenerate
    order_key = {"distinct_high": 0, "distinct_low": 1, "inside_pack": 3, "degenerate": 4}
    def k(row):
        b = row["bucket"]
        if b in order_key:
            return (order_key[b], -row["religion_metric"])
        if b.startswith("looks_like_"):
            return (2, row["religion_rank_of_24"])
        return (5, row["religion_rank_of_24"])
    out_fields.sort(key=k)

    OUT_PATH.write_text(json.dumps({
        "n_classified": len(out_fields),
        "fields": out_fields,
        "notes": (
            "Religion-vs-23-secular classification for each boolean and numeric "
            "field. Booleans use 'disclose_rate' (n_true / canonical filings) as the "
            "comparison metric; numerics use median. Buckets: distinct_high (X at "
            "or above secular p90), distinct_low (X at or below p10), looks_like_X "
            "(X within 15% of one specific sector), inside_pack (X between p25 and "
            "p75 with no notably close neighbour). Order: distinct_high first."
        ),
    }, indent=2))
    print(f"  wrote {OUT_PATH}")
    print(f"  classified {len(out_fields)} fields")
    buckets = {}
    for r in out_fields:
        b = r["bucket"].split("_")[0] if r["bucket"].startswith("looks") else r["bucket"]
        buckets[b] = buckets.get(b, 0) + 1
    print("  bucket counts:", buckets)


if __name__ == "__main__":
    main()
