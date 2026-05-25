# Related-party disclosure at secular nonprofits — the comparison view

Companion to the religion-classified tool. Reads the same four Form 990 boxes (Schedule L, Schedule R, Schedule J, Part VII) across **187,133 secular nonprofits** (every NTEE major group A–W except X, which is religion) and **1,160,678 canonical filings** in the 2014–2024 headline range (1,200,910 filings if the partial-2025 cohort is included).

**Live tool:** https://chaiboo.github.io/secular-related-party/
**Religion companion:** https://chaiboo.github.io/religious-nonprofit-related-party/

## What this is

Same four disclosure boxes, same scope rules, same parser as the religion tool — only the NTEE filter differs. This tool exists so you can read the secular distribution on its own AND compare head-to-head with religion to see what's distinctive about the religion-classified subset.

It does *not* cover: 990-EZ filers, 990-PF filers, group-ruling subordinates, or churches that decline to file under §6033(a)(3)(A)(i) (the church-exemption hole stays open in both tools).

## What's distinctive in this tool

Two tabs that the religion tool doesn't have:

- **Religion vs secular** — six rates the regulatory literature flags as fraud-relevant, computed identically on both corpora, paired-dot rows split by rate kind. **The six rows are not directly comparable to each other** — `§4958` and `contingent comp` are per-filing rates, the other four are conditional rates (denominator = rows in the relevant table, not filings). They live on different axes in the UI. Headlines (2014–2024 excl partial-2025):
  - **§4958 excess-benefit admissions** (per-filing): religion **0.058%**, secular **0.026%** — religion **2.19×** secular. But Mental Health & Crisis (F) actually leads the per-major leaderboard at **0.111%**; religion ranks **#2** of 24 NTEE majors.
  - **Unwritten insider loans** (conditional, share of loan rows w/o written agreement): religion **39.9%**, secular **31.8%** — religion 1.26× secular.
  - **§512(b)(13)-controlled related orgs** (conditional, share of related-entity rows): secular **39.3%**, religion **31.2%** — secular 1.26× religion.
  - **Sentinel counterparty names on Sched L** (conditional, share of person-name fields resolving to a placeholder, post-extended-filter): secular **9.3%**, religion **6.1%** — secular 1.52× religion.
  - **Contingent-on-revenue/net-earnings officer comp** (per-filing): secular **0.49%**, religion **0.06%** — secular **7.86×** religion (biggest gap).
  - **In-default insider loans** (conditional): essentially tied at 0.67% vs 0.77%.

- **By NTEE subgroup** — within secular, every disclosure rate stratified across 23 NTEE majors (24 once religion is folded in as X). Lets you see whether the secular pattern is uniform (it isn't) or driven by certain sectors (e.g., Human Services, Health Care, Education).

The other tabs (Column browser, Money flows, Disclosed corporate families, Officer interlocks, Sched L persons, Fraud signals, Methodology) mirror the religion tool.

## Rate construction (post-audit)

- **Per-filing rates** use one denominator: `n_distinct_filings_with_at_least_one_row / n_canonical_filings`. This is "what share of canonical filings disclosed this at all" — not `n_rows / n_filings`, which over-weights orgs that disclose many rows in one filing.
- **Conditional rates** (in-default loans, unwritten loans, §512(b)(13) controlled, sentinel names) have a different denominator (rows in the relevant table). Every row in `rel_vs_sec.json` is tagged with `rate_kind` and `denominator_definition` so the UI can keep them on separate axes.
- **2025 is excluded from headline rates** — the snapshot ends mid-cycle. `with_2025` versions are carried alongside in every rates JSON for transparency.

## Sentinel filter (extended post-audit)

The religion-tuned sentinel list missed the secular corpus's dominant role-placeholders. The extended list now also catches `VACANT, OPEN, POSITION OPEN, TBD, TBA, VOLUNTEER, RESIGNED, FORMER, RETIRED, DECEASED, INTERIM, ACTING, SEE ABOVE, SEE ATTACHED, AS NEEDED, NOT APPLICABLE, NOT REQUIRED, UNKNOWN, BOARD OF DIRECTORS`, plus a heuristic gate for single-word common role/surname stand-ins. `VACANT` alone was #1 in the pre-audit `pt_vii_people.json` at 344 EINs.

## Strength labels in the literature

- **strongest** — disclosure itself is the admission ([IRC §4958](https://www.law.cornell.edu/uscode/text/26/4958), [IRC §512(b)(13)](https://www.law.cornell.edu/uscode/text/26/512))
- **strong** — named as a fraud indicator in primary regulatory literature ([Grassley Senate Finance review of six media ministries, Jan 6 2011](https://www.finance.senate.gov/ranking-members-news/grassley-releases-review-of-tax-issues-raised-by-media-based-ministries); [ECFA Standard 6](https://www.ecfa.org/standards.aspx); [Treas. Reg. §53.4958-6](https://www.law.cornell.edu/cfr/text/26/53.4958-6))
- **moderate** — secondary literature, or structural rather than transactional

Editorial overlay, not a property of the data and not a composite risk score.

## Files

```
index.html                Single-file tool, all data inlined (~11.5 MB)
data/
  manifest.json           Schedule / column inventory
  roster.json             Corpus-level counts
  summary.json            Per-table row counts and yearly volume
  codebook.json           Plain-English definition per field
  fraud_signals.json      Schedule-by-schedule reference content
  methodology.json        Interpretive caveats (post-audit refresh)
  findings.json           Headline numbers, with partial_year_2025 flag
  bool_cogrid.json        Boolean co-occurrence matrices
  rel_vs_sec.json         Six head-to-head rates, rate_kind-tagged
  subgroup_rates.json     Per-NTEE-major disclosure rates within secular
  subgroup_with_religion.json  24-major version with religion folded in as X
  legal_interpretation.json    Field-by-field legal/regulatory framing
  legal_interpretation_audit.json  Worklist of numeric claims to revise post-audit
  columns/                Per-column distribution + yearly trend
  network/
    sched_r_tree.json     Disclosed corporate-family edges
    pt_vii_people.json    Cross-org officer interlock candidates (extended sentinel filter)
    sched_l_persons.json  Interested-person edges + sentinel bucket (extended)
    sankey_flows.json     Sched R Pt V + Sched L Sankey data
scripts/
  build_targets_secular.py     Builds the non-X target list from IRS index + BMF
  extract_secular.py           Runs the canonical extractor against secular targets
  build_data.py                Builds the JSON dataset + the comparison + subgroup data
  build_combined_subgroup.py   Merges religion into the 24-major subgroup file
  build_network_by_sector.py   Rebuilds the four network JSONs with sector tags
  rebuild_audit_fixes.py       One-shot rebuild for the post-audit data fixes
  classify_comparisons.py      EDA-compare classification
  build_eda_compare.py         Per-field comparison datasets
  bundle_html.py               Inlines every data file into index.html
```

## Build pipeline

```bash
python scripts/build_targets_secular.py   # one-time: builds the target list
python scripts/extract_secular.py         # ~6-12 hours, re-parses TEOS zips for non-X
python scripts/build_data.py              # rebuilds data/*.json from the parsed parquets
python scripts/rebuild_audit_fixes.py     # applies post-audit rate/sentinel fixes
python scripts/bundle_html.py             # re-inlines data/ into index.html
```

## Data sources

- IRS Tax Exempt Organizations Search (TEOS) bulk e-file 990 XML, 2014–2025
- IRS Business Master File (BMF) for the NTEE major-group filter
- Same parser as the religion tool — all eleven previously-NULL fraud-relevant fields are populated here too

Data snapshot: 2026-05-24.

## License

MIT.
