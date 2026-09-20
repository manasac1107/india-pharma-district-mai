# District-Level Pharma Market Attractiveness Index (India)

A reproducible pipeline that builds a district-level **Pharma Market Attractiveness
Index (MAI)** for India from NFHS-4/5, HMIS, state GDP, hospital-directory, and Census
2011 population data — built for a case competition. Covers data cleaning, an
entropy/judgment-weighted composite index, K-Means market segmentation, a
forward-looking momentum layer, and a sales-force allocation model.

Every non-obvious modeling decision, bug found, and fix applied is documented in
[`02_clean_data/results/Model_Results_Summary.md`](02_clean_data/results/Model_Results_Summary.md)
— that file is the methodology reference. This README explains how the repo is laid out
and how to reproduce it end to end.

## What this is

- **Unit of analysis:** Indian district (706 districts, reconciled across NFHS boundaries,
  post-2011 district splits, and state renaming/bifurcation).
- **Output:** four composite indices per district (Overall / Chronic / Acute / Chronic+Acute
  Overlap MAI), a forward-looking Momentum score and 3-year projection, a 4-segment
  K-Means clustering, MAI-based tiers, and a worked sales-force allocation example.
- **Framing:** a "demand" framing is used throughout — higher chronic/acute disease
  prevalence counts *toward* attractiveness (larger addressable patient pool), not
  against it. See Section 1 of `Model_Results_Summary.md` for the reasoning and the
  alternative framing that was considered and rejected.

## Data sources

| File | Source | Notes |
|---|---|---|
| `India.csv` | NFHS-4 vs NFHS-5 district factsheet | Long format, 105 indicators; 32 selected |
| `hospital_directory.csv` | National hospital directory | 30,273 rows; deduped, cleaned |
| `state-wise-decadal-change-of-states-share-in-national-gdp.xlsx` | State GDP share by decade | Duplicate-row and CAGR bugs found and fixed |
| `key-health-management-performance-indicators-of-all-the-states-in-india.xlsx` | HMIS | State-level; some indicators excluded as population-biased absolute counts |
| `01_raw_data/census_2011_district_population.csv` | [India Census 2011 district file](https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/master/india-districts-census-2011.csv) | Added later to fix a population/market-size gap |

**Explicitly excluded** (and why): `datafile.csv` (redundant subset of `India.csv`),
drinking-water/population-projection files (India-wide only, no district variation),
Cost Inflation Index (no currency values to deflate), Symptom-Disease dataset
(non-authoritative/synthetic).

## Pipeline stages

The pipeline was built iteratively across two work sessions, so the `results/` folder
contains some superseded intermediate scripts (kept for the audit trail — nothing was
deleted so every fix is traceable). **Use the "canonical script" column below to
reproduce the pipeline**; ignore same-named scripts without a version suffix once a
`_v2`/`_v3`/`_final`-style successor exists.

### Stage 1 — Data cleaning (`02_clean_data/`)

| Step | Canonical script | Output |
|---|---|---|
| District name crosswalk (NFHS ↔ hospital directory, fuzzy-matched + manually reviewed) | `step1_crosswalk.py` → `step1b_apply_reviews.py` | `district_crosswalk_v2.csv` |
| Dedupe hospital directory (merge known duplicate district spellings) | `step1c_dedupe_hospital_directory.py` | `hospital_directory_deduped.csv` |
| Clean NFHS health panel (strip whitespace, drop placeholder rows, pivot wide, select 32 indicators) | `step2_clean_india.py` | `clean_district_health_panel.csv` |
| Clean state GDP trend (fix duplicate-row and CAGR-anchor bugs, compute CAGR) | `step3_clean_gdp.py` | `clean_state_gdp_trend.csv` |
| Clean HMIS panel (splice renamed OPD series, filter childhood indicators) | `step4_clean_hmis.py` | `clean_state_hmis_panel.csv` |
| Clean hospital directory (fix state/district swap, compute facility counts) | `step5_clean_hospitals.py` | `clean_district_hospital_counts.csv` |
| Merge into master district panel | `step6_merge_master.py` | `master_district_panel.csv` |
| Flag NFHS-4→5 trend-capable indicators; null out fake-zero placeholders | `step_trend_capability.py` | `indicator_trend_capability.csv` |
| Exploratory data analysis | `step7_eda.py` | `eda_charts/`, `EDA_Findings_Summary.md` |

### Stage 2 — Index construction & modeling (`02_clean_data/results/`)

| Step | Canonical script | Output |
|---|---|---|
| Normalize variables, decide direction (demand framing) | `step1_normalize.py` | `direction_decisions.csv` |
| Broadcast state-level GDP/HMIS data to districts | `step2_broadcast_state.py` | `master_district_panel_broadcast.csv` |
| Build the 36-variable scoring set + entropy/judgment weights | `step3_normalize_full_and_weights.py` | `entropy_weights.csv`, `judgment_weights.csv`, `normalized_district_panel_full.csv` |
| Compare entropy vs. judgment weighting | `step3b_overall_mai_compare.py` | `overall_mai_entropy_vs_judgment.csv` |
| Construct Chronic / Acute / Overlap MAI | `step4_four_indices.py` | `four_indices.csv` |
| Momentum Index + Future_MAI_3yr | `step5_momentum_future.py` | `four_indices_with_momentum.csv` |
| Census 2011 population: build parent-district crosswalk for post-2011 splits | `build_population_mapping.py` | `population_parent_mapping.csv`, `population_spelling_fixes.csv` |
| Apportion population to split districts (hospital-weighted, with 3 fallback rules — see below) | `step_fix_double_count.py` | `master_district_panel_with_population.csv`, `population_apportionment_detail.csv` |
| Add a 15% population floor to Overall MAI (fixes small-district statistical-noise dominance) | `step_population_floor_fix.py` | refreshed `entropy_weights.csv`, `overall_mai_entropy_vs_judgment.csv` |
| K-Means clustering (4 segments, entropy-weighted inputs) | `step6_kmeans_v3_weighted.py` | `clustered_districts.csv` |
| Validation: sensitivity, equal-weight baseline, Top/Bottom 10 | `step7_validation_v2.py` | `sensitivity_analysis.csv` |
| Recompute PatientPool + Acute MAI with real population | `step_recompute_with_population_v2.py` | `patientpool_before_after_comparison.csv` |
| Tiering, sales-force allocation, final assembly | `step8_business_layer.py` → `step_final_assembly.py` | **`final_district_mai_scores.csv`** |
| Final charts | `step9_charts.py` | `results_charts/*.png` |

Superseded/exploratory scripts kept for the audit trail:
`step6_kmeans.py`, `step6_kmeans_v2.py` (pre-population-fix clustering attempts),
`step7_validation.py` (pre-population-fix validation), `step_add_population_to_overall.py`
(the "population as a 37th entropy-weighted variable" attempt — didn't work, see
`Model_Results_Summary.md` §11), `step_population_apportion.py` and
`step_recompute_with_population.py` (pre-double-counting-fix versions).

## Key methodology decisions

Full detail and reasoning for every decision below is in
[`Model_Results_Summary.md`](02_clean_data/results/Model_Results_Summary.md).

- **Demand framing:** disease/risk-factor prevalence counts toward attractiveness, not
  against it (§1).
- **Trend-capability constraint:** only 12 of 32 NFHS indicators have real data in both
  survey rounds; the other 20 (cancer screening, blood sugar, blood pressure detail,
  tobacco, alcohol, literacy) are NFHS-5-only and never have a change/trend score
  computed for them (§0).
- **HMIS absolute-count exclusion:** 13 HMIS broadcast indicators are raw case counts
  with no population denominator, which mechanically favored large-population states
  (an early run had Uttar Pradesh sweeping the Top 10) — excluded from scoring (§3).
- **Population floor:** entropy weighting could not give Census population a meaningful
  organic weight (0.05%), so a manual 15% floor was added to Overall MAI after
  small-population districts (e.g. Nicobar, pop. 36,842) were found ranking above major
  metros (§11).
- **Population apportionment for post-2011 district splits:** hospital-count-weighted
  with three fallback rules to equal-split — missing hospital data, implausible weight
  ratios (e.g. an urban parent district retaining nearly all weight over a rural split-off
  by hospital density alone), and a double-counting bug where a parent-namesake district
  and its split-off children were both getting counted (§10–11).
- **PatientPool proxy:** `Population_2011 × mean(normalized chronic + acute-prevalence
  composite)` — used only for the sales-force worked example, explicitly not a validated
  patient count.

## How to reproduce

```bash
pip install -r requirements.txt
```

Run the Stage 1 scripts in `02_clean_data/` in the order listed in the table above, then
the Stage 2 scripts in `02_clean_data/results/` in order. Each script reads its inputs
from relative paths (`../` for one directory up), so run them **with `02_clean_data/` or
`02_clean_data/results/` as the working directory**, matching the table's column.

## Repository structure

```
├── README.md
├── requirements.txt
├── *.csv / *.xlsx                        # raw source files
├── 01_raw_data/
│   └── census_2011_district_population.csv
└── 02_clean_data/
    ├── step*.py                          # Stage 1 cleaning scripts
    ├── clean_*.csv, master_district_panel.csv, indicator_trend_capability.csv
    ├── EDA_Findings_Summary.md
    ├── eda_charts/
    └── results/
        ├── step*.py                      # Stage 2 modeling scripts
        ├── final_district_mai_scores.csv # <- the final scored table
        ├── Model_Results_Summary.md      # <- full methodology writeup
        └── results_charts/               # final PNG charts
```

## Known limitations

See `Model_Results_Summary.md` §12 for the full list, including: 58 districts (8.2%)
have no hospital-directory match and are mean-imputed; Ladakh has no data in the GDP,
HMIS, or Census 2011 sources at all (post-dates all three); 118 districts' population is
an apportioned approximation, not directly measured; the 15% population-floor weight is
a manual override, not an optimized or externally validated constant; and K-Means
clustering uses index-derived weights, so cluster membership is not a fully independent
view of the raw data.
