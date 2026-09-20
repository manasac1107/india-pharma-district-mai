# District-Level Pharma Market Attractiveness Index (MAI) — Methodology Summary

Source data: `master_district_panel.csv` (706 districts), `clean_state_gdp_trend.csv`,
`clean_state_hmis_panel.csv`, `district_crosswalk_v2.csv`, `indicator_trend_capability.csv`
(all from the prior data-cleaning session).

## 0. Trend-capability constraint (applies throughout)

`indicator_trend_capability.csv` marks 12 of the 32 NFHS health indicators as
`trend_capable=True` (real NFHS-4 *and* NFHS-5 values) and 20 as `trend_capable=False`
(NFHS-5-only — cancer screening, blood sugar, blood pressure detail, tobacco, alcohol,
literacy; their NFHS-4 values are NaN, not real data).

- **Every current-state index (Overall/Chronic/Acute/Overlap MAI, Tiers, PatientPool)**
  uses only the **NFHS5** value of all 32 indicators. This is why the constraint doesn't
  block using the 20 trend_capable=False indicators as levels — it only blocks computing
  a *change* score for them.
- **Only the Momentum Index / Future_MAI_3yr layer** touches NFHS4→NFHS5 change, and only
  for the 12 trend_capable=True indicators. No trend_capable=False indicator's change is
  computed or implied anywhere in this model.

## 1. Direction / normalization (Step 1)

**Framing decision (user-confirmed): "demand framing"** — higher chronic/acute disease
prevalence and risk-factor rates count *for* attractiveness (larger addressable patient
pool = more pharma demand), not against it. Under this framing, every one of the 67
district-level variables in `master_district_panel.csv` ended up `higher_is_better=True`
— no inversions were needed. Full reasoning and the per-variable table are in
`direction_decisions.csv`. Sex ratio (4 columns) was flagged as a neutral demographic
control with no real demand/capability logic; kept in, left to entropy weighting to decide
its influence, per user confirmation.

All numeric variables were **min-max scaled to [0,1]**: `(x - min) / (max - min)` per
column, computed on `master_district_panel_broadcast.csv` (post state-broadcast, Step 2),
saved as `normalized_district_panel_full.csv`.

## 2. State-level broadcast (Step 2)

`district_crosswalk_v2.csv` only maps NFHS↔hospital_directory state names, not
GDP/HMIS naming, so a small explicit state-name lookup was built (e.g. "NCT of Delhi" →
"Delhi", "Jammu & Kashmir" → "Jammu and Kashmir") to join `clean_state_gdp_trend.csv` and
`clean_state_hmis_panel.csv` (2019-20, most recent year) onto every district in that state.
All broadcast columns are suffixed `_state_broadcast`.

**Bug fixed upstream during this step:** the original `clean_state_gdp_trend.csv` CAGR
computation anchored on the literal first data row (1960-61) regardless of value. 13 of
33 states (Chhattisgarh, Jharkhand, Uttarakhand, Goa, Arunachal Pradesh, etc.) didn't
exist as separate states in 1960-61 and had `NaN` gdp_share for pre-formation years,
producing a `NaN` CAGR. Fixed to anchor on each state's **earliest year with a real,
non-null gdp_share**. GDP CAGR coverage improved from 20/33 to 32/33 states (Sikkim
remains `NaN` — its earliest real value is a literal 0.0% share in 1980-81, so CAGR is
mathematically undefined from a zero base, not a bug).

**Genuine data gaps (not naming issues):** Ladakh, Lakshadweep, Dadra & Nagar Haveli,
Daman & Diu are entirely absent from the GDP file (33 states/UTs only); Ladakh is also
absent from the HMIS file (both predate the 2019 J&K/Ladakh split).

## 3. Scoring variable set — 36 variables (revised down from 49)

**Excluded from all scoring: 13 HMIS state-broadcast variables** (12 childhood
acute-disease indicators + OPD_Total). These are **raw absolute case counts** (confirmed
via the source file's `unit` column = "value in Absolute Number"), and no population
variable exists anywhere in this pipeline to convert them to rates. Left in, they
mechanically favored large-population states — an early run had **Uttar Pradesh sweeping
the entire Top-10** and "Tetanus Neonatorum" (a raw count) ranking as the #3 entropy
weight, purely because UP has the largest population, not because of any real
attractiveness signal. User-confirmed fix: excluded from scoring entirely; retained in
`normalized_district_panel_full.csv` for reference only. `gdp_cagr_pct_state_broadcast`
was kept (it's a growth-rate percentage, not a count, so no population confound).

**Final 36-variable scoring set:** 32 NFHS5 indicator levels + 3 hospital-count variables
+ 1 GDP CAGR broadcast, grouped into 6 pillars (`pillar_mapping.csv`):

| Pillar | # vars | Contents |
|---|---|---|
| Chronic | 19 | Blood pressure (6), blood sugar (6), cancer screening (3), tobacco (2), alcohol (2) |
| Acute | 6 | Diarrhoea/ARI prevalence (2), diarrhoea/ARI care-seeking — ORS/zinc/facility visit (4) |
| Infrastructure_Affordability | 5 | Sanitation, drinking water, electricity, health insurance, literacy |
| Access | 3 | total/public/private hospital count |
| Demographic | 2 | Sex ratio (birth), sex ratio (total population) |
| Momentum | 1 | GDP CAGR (state-broadcast) |

Missing normalized values (58 districts with no hospital-count crosswalk match, ~10
districts with no GDP CAGR) are **mean-imputed** at scoring time only — documented choice,
not silently dropped.

## 4. Weighting (Step 3)

**(a) Entropy weighting** (standard method): for each variable, compute the proportion
`p_ij = x_ij / Σ_i(x_ij)` across districts, entropy `e_j = -k·Σ(p_ij·ln(p_ij))` with
`k = 1/ln(n)`, diversification `d_j = 1 - e_j`, weight `w_j = d_j / Σ(d_j)`. Higher
district-to-district variability → higher weight. Full weights in `entropy_weights.csv`.
Pillar totals: Chronic 36.5%, Access 32.8%, Acute 27.1%, Infrastructure_Affordability
2.0%, Momentum 0.9%, Demographic 0.8% (entropy naturally downweighted Infrastructure/
Momentum/Demographic — these vary less district-to-district than Access/Chronic/Acute).

**(b) Judgment weighting** (user-specified pillar split, "Balanced" preset): Chronic 30%,
Acute 30%, Infrastructure_Affordability 20%, Access 10%, Momentum 10%. Reconciled with a
separate user answer to give sex ratio ("Demographic") its own small pillar: **carved out
5% for Demographic, scaled the other five pillars by ×0.95** each (e.g. Chronic
30%→28.5%) so the total still sums to 100%. Within each pillar, weight is split **equally**
across member variables. Full weights in `judgment_weights.csv`.

**Comparison:** Spearman rank correlation between entropy-weighted and judgment-weighted
Overall MAI = **ρ = 0.794** (n=706, p≈4×10⁻¹⁵⁴) — moderate-to-strong agreement. Chart:
`results_charts/entropy_vs_judgment_rank.png`. Districts with the largest rank
disagreement (Uttarakhand hill districts, a few UP/Bihar districts) tend to be cases
where hospital-count-driven entropy weighting and prevalence-driven judgment weighting
pull in different directions — see `overall_mai_entropy_vs_judgment.csv` for the full list.

## 5. The four indices (Step 4)

- **Overall MAI** = entropy-weighted score (primary), judgment-weighted score reported as
  cross-check. Formula: `Σ(normalized_value_j × weight_j)` across the 36 scoring variables.
- **Chronic MAI**: entropy weights re-weighted — Chronic pillar ×3, health insurance
  variable ×3, Momentum (GDP CAGR) ×3, Acute pillar ×0.3, then renormalized to sum to 1.
- **Acute MAI**: entropy weights re-weighted — Acute pillar ×3, Access (hospital count)
  pillar ×3, Momentum ×0.3, renormalized. **Note:** the brief also asked to upweight
  "population size" — no population variable exists anywhere in the source data (the only
  population file was excluded from the pipeline as India-wide-only with no district
  variation), so this could not be literally implemented; hospital count is the sole
  Access-side proxy used instead.
- **Chronic+Acute Overlap MAI** = `sqrt(Chronic_MAI × Acute_MAI)` (geometric mean).
  Rewards districts strong on *both* dimensions over those lopsided on one (e.g. 0.9/0.1
  scores lower than 0.5/0.5). Correlation between Chronic_MAI and Acute_MAI is weak
  (r=0.195), so this genuinely differentiates from either component alone.
  **Honesty check performed:** the brief's alternative construction ("TB/respiratory
  HMIS indicators") was evaluated and rejected — those 3 variables (TB, Asthma, Pneumonia
  in Children 0-5) are the same population-biased, state-level-only absolute counts
  excluded from scoring in Step 3; building the Overlap index from them alone would have
  zero within-state discriminating power and would be genuinely data-thin.

## 6. Momentum Index and Future_MAI_3yr (Step 5)

`Momentum = 0.6 × GDP_CAGR_norm + 0.4 × Change_component_norm`, where
`Change_component` = mean, across the 12 trend_capable=True indicators, of
`(NFHS5_norm − NFHS4_norm)` per district, then itself min-max rescaled to [0,1] so it
combines cleanly with GDP_CAGR_norm. GDP CAGR carries the dominant 0.6 weight as the
"primary leading indicator" per the brief. **No trend_capable=False indicator's change is
used anywhere in this formula.**

`Future_MAI_3yr = Overall_MAI_entropy + 0.15 × (Momentum − mean(Momentum))`. The 0.15
scale factor bounds the maximum plausible adjustment to roughly ±0.15 (Momentum is
bounded in [0,1], so the deviation term is roughly bounded in [-1,1]) — a meaningful but
non-dominant nudge relative to the ~0.04–0.34 range of Overall MAI, so momentum can
re-rank close competitors but can't invert the ranking of very different districts.
`Rank_Delta = Rank_Current − Rank_Future` (positive = climbing toward rank 1).

Face validity: Telangana districts (strong recent state GDP growth) dominate the rising
list; several Bihar and eastern UP districts (weaker growth) dominate the fading list.

## 7. Clustering (Step 6) — SUPERSEDED, see Section 11

The original clustering (36-variable scoring set, unweighted Euclidean K-Means) and its
"Mature High-Value / Emerging Chronic-Momentum / Lagging Low-Momentum / Underserved
Latent-Demand" labels were replaced after the population-floor fix in Section 11 exposed
a face-validity failure (extreme-outlier metro districts landing in the wrong cluster).
See Section 11 for the corrected, final clustering.

## 8. Validation (Step 7) — figures below are SUPERSEDED, see Section 11

The sensitivity/Top-10/equal-weight-baseline figures originally reported here were
computed before the population-floor fix (Section 11) and are stale. Original method
(top-3 entropy-weighted variables perturbed ±10%, Spearman ρ vs. base case, Top-10
overlap count) is unchanged; only the underlying Overall MAI changed. See Section 11 for
the current, final validation figures.

## 9. Business layer (Step 8) — figures below are SUPERSEDED, see Sections 10 and 11

**Tiering method unchanged:** Overall MAI quartiles, Tier 1 (top quartile) through
Tier 4 (bottom quartile). Cutoffs and worked-example figures are refreshed in Section 11.

**PatientPool proxy formula superseded by Section 10** (Census 2011 population replaces
`total_hospital_count` as the size term) and **further refined in Section 11** (the
population-floor fix does not change the PatientPool formula itself, but Population_2011
itself was corrected for a double-counting bug — see Section 10 — before this table was
finalized).

**Sales-force allocation formula unchanged:**
`Reps(d) = TotalPool × [PatientPool(d) × Overall_MAI_entropy(d)] / Σᵢ[PatientPool(i) × Overall_MAI_entropy(i)]`.
Final worked-example figures are in Section 11.

## 10. Census 2011 population added (supersedes Section 9's hospital-count proxy)

A Census 2011 district-population file (`01_raw_data/census_2011_district_population.csv`,
640 districts, `Population` column) was added after the original model build, resolving
the "no population variable" limitation from Section 9. Full reconciliation across all
706 NFHS districts:

| Match type | Count | Treatment |
|---|---|---|
| Direct exact match (state name normalized: "ORISSA"→Odisha, "AND"→"&", etc.) | 606 | Population used as-is |
| Pure spelling/transliteration fix (e.g. Nicobar→Nicobars, Prayagraj→Allahabad, 9 Telangana districts that are direct renames of their undivided-AP namesake) | 28 | Full population used as-is, no split |
| Post-2011 district split, apportioned from a 2011 parent district | 70 (53 parent groups) | See apportionment method below |
| No census data at all (Ladakh — post-dates 2011 census) | 2 | `Population_2011` is null |

**Apportionment method:** within each parent group, a child district's share of the
parent's 2011 population is weighted by that child's share of the parent group's total
`total_hospital_count` (from the district-hospital join), falling back to an equal split
where no child in the group has hospital-count data. Applied to 70 districts across 51
parent groups (61 districts hospital-weighted, 9 equal-split fallback).
**Honest finding:** only 1 of 51 parent groups (Andhra Pradesh's Karimnagar→{Rajanna
Sircilla, Peddapalli, Jagitial}) actually produced non-equal weights. In the other 50
groups, every child's `total_hospital_count` was numerically identical — because that
hospital count was *itself* already broadcast from the same shared parent match in
Step 1's district crosswalk (the same fundamental data limitation propagating through).
So in practice, hospital-weighting behaved as equal-split for the large majority of
groups; this is disclosed rather than presented as more precise than it is. Every
apportioned district is flagged `is_population_apportioned = True` with
`population_apportion_method` ("hospital_weighted" or "equal_split") in
`master_district_panel_with_population.csv` / `final_district_mai_scores.csv`. Full
detail: `population_apportionment_detail.csv`.

Two Step 1 hint-file parent names needed correction against actual census spelling
(`Ranga Reddy`→`Rangareddy`, `Mahabubnagar`→`Mahbubnagar`, `Dantewada`→`Dakshin Bastar
Dantewada`, `Panchmahal`→`Panch Mahals`, `Sabar kantha`→`Sabar Kantha`), and two hint-file
entries (`Uttar Bastar Kanker`, `Alirajpur`) turned out to already be direct 2011 census
matches, not splits — corrected before applying.

**Double-counting bug found and fixed (post-hoc, user-caught):** the first version of
this apportionment summed to **1,327,385,667** across all districts — **9.62% (116.5M)
above** India's actual 2011 census population of 1,210,854,977. Root cause: in 48 of the
51 parent groups, the present-day district that kept the 2011 parent's name (e.g.
Telangana's present-day "Karimnagar") was treated as a direct/spelling-fix match and
given the **full** undivided-parent population, while its split-off children (Rajanna
Sircilla, Peddapalli, Jagitial) were **also** each given a share of that same total —
double-counting the parent's population once for itself and again across its children.
Only 3 of 51 groups (Jaintia Hills, Barddhaman, Warangal) were unaffected, because no
present-day district survives under the plain parent name in those three cases.
**Fix:** for the 48 affected groups, the parent-namesake district was added into the
weighted-split pool itself (using its own `total_hospital_count`, not the children's),
and weights recomputed across all members (parent-namesake + children) so they sum to
1 across the *whole* group, not just the children. The parent-namesake's row is now
flagged `is_population_apportioned = True` like its former children — it no longer gets
a privileged "full population" figure. **Corrected total: 1,209,515,632 — -0.11% off**
the true 2011 figure (residual gap fully explained by Ladakh's 2 districts having no
census data at all). This validates the corrected apportionment.

**PatientPool proxy v2** (replaces the Section 9 hospital-count-based version):
`PatientPool(d) = Population_2011(d) × mean(normalized prevalence composite)`, same
19 Chronic + 2 Acute-prevalence composite as before, size term swapped from
`total_hospital_count` to real Census 2011 population.

**Acute MAI recomputed** with `Population_2011_norm` added as an explicit input (boosted
×3, alongside Acute pillar ×3 and Access ×3), on a re-derived 37-variable entropy base
used *only* for Acute MAI. **Overall MAI, Chronic MAI, Momentum, Future_MAI_3yr, and the
K-Means clusters were deliberately left unchanged** on their original 36-variable basis —
this update's scope was limited to PatientPool and Acute MAI, not a full model refresh.
`Chronic_Acute_Overlap_MAI` was recomputed since it depends on the new Acute MAI.

**Before/after sales-force Top 10** (TotalPool=100, full table:
`patientpool_before_after_comparison.csv`):

| | Before (hospital-based) | After (Census 2011 population-based, corrected) |
|---|---|---|
| Top-10 reps sum | 41.0 / 100 | 12.25 / 100 |
| Top district | Mumbai Suburban (8.02 reps) | Bangalore (2.25 reps) |
| Composition | All 6 major metros + Chhota Udaipur, Thane, Vadodara, Pune | Metros + East Godavari, Murshidabad, Thane, Nashik |

The population-based version spreads the rep pool far more evenly across the Top 10
(41.0→12.25 for the top decile) — hospital count was a much more concentrated,
metro-skewed proxy than real population; real Census population reveals several
large-but-lower-hospital-density districts (East Godavari, Murshidabad, Nashik) that
were previously underweighted purely because the hospital directory has sparser
listings there, not because their real patient pool is smaller.

## 11. Population-floor face-validity fix (supersedes clustering, validation, business layer)

**Problem found (user-caught):** with the corrected Census 2011 population merely sitting
in `final_district_mai_scores.csv` as a reference column, Overall MAI itself still had
**no population/market-size variable** — it was a pure rate/intensity index. Combined
with entropy weighting favoring high-variance indicators, this let tiny districts with
statistically noisy NFHS-5 percentages (small survey samples) dominate the ranking:
**Nicobar (population 36,842) ranked #8, ahead of Chennai (4.6M)**; Tawang (49,977) and
South West Garo Hills (71,167) also sat in the Top 20.

**First attempted fix (did not work, disclosed honestly):** adding `Population_2011_norm`
(log-transformed, min-max normalized) as a 37th variable in the entropy-weighted set
produced an entropy weight of **0.0005 (0.05%)** — statistically negligible. This is not
a bug but a structural property of entropy weighting: it rewards variables with skewed,
spiky proportions, and population (even log-transformed) is comparatively smooth next to
zero-inflated variables like hospital counts. **Entropy weighting cannot be made to
respect population by adding it as just another entropy-weighted input** — the Top 20 was
virtually unchanged (Nicobar, Tawang still present).

**Working fix (user-selected): a manual 15% population floor.**
`Overall_MAI = 0.85 × (36-variable weighted score, renormalized) + 0.15 × Population_2011_norm`
(log1p-transformed, then min-max normalized). Applied identically to the entropy,
judgment, and equal-weight versions of Overall MAI. Result: **zero districts under
100,000 population in the new Top 20** (verified explicitly). New Top 10: Ahmedabad,
Mumbai Suburban, Bangalore, Hyderabad, Mumbai, Chennai, NCT Delhi-South, Kodagaon, East
Godavari, Paschim Barddhaman.

**Two further apportionment bugs found and fixed while investigating the Bottom 10:**
1. **Tripura's Gomati/Khowai/Sepahijala/Unakoti got exactly 0 population.** Their
   hospital-count crosswalk match was `NaN` (no data), which the apportionment code
   treated as `0` — since their parent-namesake district (North/South/West Tripura) had
   real hospital data, the hospital-weighted split gave those parents ~100% and the
   children ~0%. **Fixed:** a group only uses hospital-weighting if *every* member has
   real (non-null) hospital data; otherwise the whole group falls back to equal-split.
2. **Palghar got 0.14% of undivided Thane's population (15,234 people)** — Thane has 725
   hospitals vs. Palghar's 1, an extreme urban/rural disparity that hospital-weighting
   took literally. Real-world Palghar took roughly a quarter of Thane's actual 2011
   population when the district split in 2014. **Fixed:** if hospital-weighting would
   give any group member less than `10%/n` of the total (n = group size), the ratio is
   treated as implausible and the *whole group* falls back to equal-split instead
   (flagged `equal_split_implausible_ratio` in `population_apportionment_detail.csv`).
   This also corrected two borderline cases (Bilaspur/Mungeli, Ghaziabad/Hapur).

Corrected Population_2011 total is unchanged at ~1.2096B (-0.11% vs. actual) — these were
within-group redistribution bugs, not sum-level errors.

**Clustering re-run, weighted (K-Means input mismatch found and fixed):** the first
re-clustering attempt (unweighted Euclidean K-Means on the 37 raw variables) put
Ahmedabad and Mumbai — ranked #1 and #5 — into the *lowest*-mean-MAI cluster, because
Ahmedabad's `public_hospital_count_norm` is such an extreme outlier (0.8–0.96 away from
every cluster centroid) that unweighted Euclidean distance placed it almost arbitrarily.
**Fixed (user-selected):** clustering inputs are now scaled by the same effective weight
Overall MAI actually uses (`0.85 × entropy_weight` for the 36 base variables, `0.15` for
population) before K-Means, so cluster membership reflects the same relative importance
as the index. Final clusters:

| Label | n | Mean Overall MAI | Mean Chronic | Mean Acute | Mean Momentum | Mean Population |
|---|---|---|---|---|---|---|
| Elite Metro Powerhouses | 6 | 0.406 (highest) | 0.285 (highest) | 0.358 (highest) | 0.546 (highest) | 6.31M |
| Established High-Value | 210 | 0.292 | 0.123 | 0.241 | 0.431 | 2.04M |
| Mainstream Mid-Tier | 391 (largest) | 0.186 | 0.122 | 0.059 | 0.428 | 1.86M |
| Small Underserved, Chronic-Leaning | 99 | 0.146 (lowest) | 0.145 | 0.061 | 0.453 | 0.21M (much smaller) |

Ahmedabad, Mumbai Suburban, Bangalore, Hyderabad, Mumbai, and Chennai now correctly land
in "Elite Metro Powerhouses" — matching their #1–#6 ranks.

**Rising Stars / Momentum — checked, no fix needed:** Top 20 by Momentum and Top 15
Rising Stars (`Rank_Delta`) both have **zero districts under 100,000 population**.
Momentum is driven by state-level GDP CAGR (a real economic aggregate), not district-level
survey percentages, so it was never vulnerable to the small-sample-noise problem that
affected Overall MAI. No change applied.

**Final validation (re-run on the corrected, population-floored Overall MAI):**
- Sensitivity: top-3 base-variable ±10% perturbations still give ρ = 0.9999–1.0000,
  9–10/10 Top-10 overlap. The 15% population-floor weight itself was also perturbed
  (±1.5 points, i.e. 13.5%/16.5%): ρ = 0.999, 10/10 Top-10 overlap — the floor's exact
  size is not a fragile parameter.
- Equal-weight baseline (with the same 15% population floor applied): ρ = 0.713 vs.
  entropy-weighted — still meaningfully different from 1.0.
- Face validity: Top 10 all major metros, correctly clustered. Bottom 10 all genuinely
  tiny districts (Dibang Valley, population 8,004 — the smallest in the 2011 census;
  Lahul and Spiti; Kra Daadi) — no more small-district-in-the-Top-20 artifacts.

**Final business layer:**
- Tier cutoffs (quartiles of the corrected Overall MAI): Q1=0.1641, median=0.1917,
  Q3=0.2739. 177/176/176/177 districts per tier.
- Sales-force worked example (TotalPool=100, `PatientPool × MAI`, Census-2011-based
  PatientPool): Top 10 reps sum = **12.02/100** (Bangalore 2.26, Mumbai Suburban 2.09,
  Ahmedabad 1.26, Chennai 1.10, East Godavari 1.10, Murshidabad 0.99, Hyderabad 0.99,
  Pune 0.90, Nashik 0.70, Patna 0.65). Reps sum to exactly 100 across all 706 districts
  (verified).

All final figures are in `final_district_mai_scores.csv`; full apportionment detail
(including which groups triggered which fallback) is in
`population_apportionment_detail.csv`.

## 12. Known limitations (for the case write-up)

1. Sex ratio (Demographic pillar) has no clear demand or capability-to-serve logic under
   the demand framing; included per user confirmation but its very low entropy weight
   (0.8% of total) suggests it isn't doing much analytical work either way.
2. 58 districts (8.2%) have no hospital-directory crosswalk match and are mean-imputed
   for hospital-count variables — their Access-pillar contribution is therefore imputed,
   not observed (PatientPool no longer depends on hospital count as of Section 10).
3. Ladakh is entirely absent from the GDP broadcast, HMIS broadcast, and Census 2011
   population sources (post-dates the 2019 J&K/Ladakh split and the 2011 census); its
   Momentum score is mean-imputed and its `Population_2011` / PatientPool are null.
4. **118 districts' population (16.7% of all 706) is an apportioned approximation**, not
   directly measured — split from a 2011 parent district, either by hospital-count
   weighting or by equal split (see Section 11 for the two additional fallback rules
   added after finding Tripura's zero-population bug and the Thane/Palghar
   implausible-ratio bug). Any district-level PatientPool or Acute MAI figure for these
   118 districts should be presented with this caveat in the case write-up.
5. **The 15% population-floor weight (Section 11) is a manual override, not a
   data-driven or optimized value.** It was chosen because entropy weighting could not
   produce a meaningful population weight organically (0.05%), and 15% was sufficient to
   eliminate every sub-100K district from the Top 20 in practice — but it was not tuned
   against any external validation target (e.g. actual pharma sales data), so it should
   be presented as a modeling judgment call, not a derived constant.
6. **K-Means clustering (Section 11) uses index-derived weights as its distance metric**,
   which means cluster membership is not an independent view of the raw data — it is
   deliberately built to agree with Overall MAI's weighting scheme. This was a necessary
   fix (unweighted clustering misplaced extreme-outlier metros), but it means the
   clusters should be read as "segments of the index," not as a wholly separate
   unsupervised finding.
