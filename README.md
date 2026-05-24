# Related-party disclosure at secular nonprofits — the comparison view

Companion to the religion-classified tool. Reads the same four Form 990 boxes (Schedule L, Schedule R, Schedule J, Part VII) across **187,133 secular nonprofits** (every NTEE major group A–W except X, which is religion) and **1,200,910 filings**, 2014–2025.

**Live tool:** https://chaiboo.github.io/secular-related-party/
**Religion companion:** https://chaiboo.github.io/religious-nonprofit-related-party/

## What this is

Same four disclosure boxes, same scope rules, same parser as the religion tool — only the NTEE filter differs. This tool exists so you can read the secular distribution on its own AND compare head-to-head with religion to see what's distinctive about the religion-classified subset.

It does *not* cover: 990-EZ filers, 990-PF filers, group-ruling subordinates, or churches that decline to file under §6033(a)(3)(A)(i) (the church-exemption hole stays open in both tools).

## What's distinctive in this tool

Two tabs that the religion tool doesn't have:

- **Religion vs secular** — six rates the regulatory literature flags as fraud-relevant, computed identically on both corpora, displayed as paired-dot rows on a shared axis with the ratio called out. Headlines from this view:
  - §4958 excess-benefit admissions: religion 2.5× the secular rate
  - Unwritten insider loans: religion 1.25× secular
  - §512(b)(13)-controlled related orgs: secular 1.25× religion
  - Sentinel counterparty names on Sched L: secular 1.51× religion
  - Comp tied to revenue / net earnings: secular **7.7×** religion (this is the biggest gap)
  - In-default loans: essentially tied

- **By NTEE subgroup** — within secular, every disclosure rate stratified across 23 NTEE majors. Lets you see whether the secular pattern is uniform (it isn't) or driven by certain sectors (e.g., Human Services, Health Care, Education).

The other tabs (Column browser, Money flows, Disclosed corporate families, Officer interlocks, Sched L persons, Fraud signals, Methodology) mirror the religion tool.

## Strength labels in the literature

- **strongest** — disclosure itself is the admission ([IRC §4958](https://www.law.cornell.edu/uscode/text/26/4958), [IRC §512(b)(13)](https://www.law.cornell.edu/uscode/text/26/512))
- **strong** — named as a fraud indicator in primary regulatory literature ([Grassley Senate Finance review of six media ministries, Jan 6 2011](https://www.finance.senate.gov/ranking-members-news/grassley-releases-review-of-tax-issues-raised-by-media-based-ministries); [ECFA Standard 6](https://www.ecfa.org/standards.aspx); [Treas. Reg. §53.4958-6](https://www.law.cornell.edu/cfr/text/26/53.4958-6))
- **moderate** — secondary literature, or structural rather than transactional

Editorial overlay, not a property of the data and not a composite risk score.

## Files

```
index.html                Single-file tool, all data inlined (~11.5 MB — bigger because the corpus is ~14× larger)
data/
  manifest.json           Schedule / column inventory
  roster.json             Corpus-level counts
  summary.json            Per-table row counts and yearly volume
  codebook.json           Plain-English definition per field
  fraud_signals.json      Schedule-by-schedule reference content
  methodology.json        Interpretive caveats
  findings.json           Headline numbers shown on the landing
  bool_cogrid.json        Boolean co-occurrence matrices
  rel_vs_sec.json         Six head-to-head comparison rates (vs religion tool)
  subgroup_rates.json     Per-NTEE-major disclosure rates within secular
  columns/                Per-column distribution + yearly trend
  network/
    sched_r_tree.json     Disclosed corporate-family edges (~7.5 MB)
    pt_vii_people.json    Cross-org officer interlock candidates (~3.9 MB)
    sched_l_persons.json  Interested-person edges + sentinel bucket
    sankey_flows.json     Sched R Pt V + Sched L Sankey data
scripts/
  build_targets_secular.py  Builds the non-X target list from IRS index + BMF
  extract_secular.py        Runs the canonical extractor against secular targets
  build_data.py             Builds the JSON dataset + the comparison + subgroup data
  bundle_html.py            Inlines every data file into index.html
```

## Build pipeline

```bash
python scripts/build_targets_secular.py   # one-time: builds the target list
python scripts/extract_secular.py         # ~6-12 hours, re-parses TEOS zips for non-X
python scripts/build_data.py              # rebuilds data/*.json from the parsed parquets
python scripts/bundle_html.py             # re-inlines data/ into index.html
```

## Data sources

- IRS Tax Exempt Organizations Search (TEOS) bulk e-file 990 XML, 2014–2025
- IRS Business Master File (BMF) for the NTEE major-group filter
- Same parser as the religion tool — all eleven previously-NULL fraud-relevant fields are populated here too

## License

MIT.
