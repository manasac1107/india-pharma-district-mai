import pandas as pd
import numpy as np

pop = pd.read_csv("master_district_panel_with_population.csv")
final_prev = pd.read_csv("final_district_mai_scores.csv")
norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]
pillar_map = pd.read_csv("pillar_mapping.csv").set_index("variable")["pillar"]

# --- rebuild a clean baseline (pre-population, hospital-based v1) from the preserved
#     v1 columns, discarding the double-counted v2 columns from the previous run ---
final = final_prev[[
    "State", "District", "Overall_MAI_entropy", "Overall_MAI_judgment", "Overall_MAI_equal_weight",
    "Rank_entropy", "Rank_judgment", "Rank_Diff_abs", "Chronic_MAI", "Momentum", "Future_MAI_3yr",
    "Rank_Current", "Rank_Future", "Rank_Delta", "Cluster_Label", "Tier",
]].copy()
final["Acute_MAI_v1_hospital_based"] = final_prev["Acute_MAI_v1_hospital_based"]
final["PatientPool_proxy_v1_hospital_based"] = final_prev["PatientPool_proxy_v1_hospital_based"]
final["Reps_allocated_v1_hospital_based"] = final_prev["Reps_allocated_v1_hospital_based"]

# --- corrected Population_2011 (double-counting fix applied) ---
final = final.merge(pop[["State", "District", "Population_2011", "is_population_apportioned",
                          "population_apportion_method"]], on=["State", "District"], how="left")

# refresh master_district_panel.csv with the corrected population columns
master = pd.read_csv("../master_district_panel.csv")
master = master.drop(columns=[c for c in ["Population_2011", "is_population_apportioned", "population_apportion_method"] if c in master.columns])
master = master.merge(pop[["State", "District", "Population_2011", "is_population_apportioned", "population_apportion_method"]],
                       on=["State", "District"], how="left")
master.to_csv("../master_district_panel.csv", index=False)
print(f"Refreshed master_district_panel.csv with corrected Population_2011 ({master['Population_2011'].notna().sum()}/{len(master)} non-null)")

norm = norm.drop(columns=["Population_2011", "Population_2011_norm"], errors="ignore")
norm = norm.merge(pop[["State", "District", "Population_2011"]], on=["State", "District"], how="left")
log_pop = np.log1p(norm["Population_2011"])
lo_lp, hi_lp = log_pop.min(), log_pop.max()
norm["Population_2011_norm"] = (log_pop - lo_lp) / (hi_lp - lo_lp)

SCORING_VARS = entropy_w.index.tolist()
X37 = norm[SCORING_VARS + ["Population_2011_norm"]].copy()
X37_imputed = X37.fillna(X37.mean())
n = len(X37_imputed)
eps = 1e-12
P = X37_imputed / (X37_imputed.sum(axis=0) + eps)
P = P.clip(lower=eps)
k = 1 / np.log(n)
entropy37 = -k * (P * np.log(P)).sum(axis=0)
diversification37 = 1 - entropy37
entropy_w37 = diversification37 / diversification37.sum()
print(f"Population_2011_norm entropy weight (37-var pool): {entropy_w37['Population_2011_norm']:.4f}")

acute_boosts_37 = [
    (lambda v: v == "Population_2011_norm", 3.0),
    (lambda v: v in pillar_map.index and pillar_map[v] == "Acute", 3.0),
    (lambda v: v in pillar_map.index and pillar_map[v] == "Access", 3.0),
    (lambda v: v in pillar_map.index and pillar_map[v] == "Momentum", 0.3),
]
acute_weights_37 = entropy_w37.copy()
for match_fn, mult in acute_boosts_37:
    mask = acute_weights_37.index.map(match_fn)
    acute_weights_37.loc[mask] = acute_weights_37.loc[mask] * mult
acute_weights_37 = acute_weights_37 / acute_weights_37.sum()
acute_weights_37.to_csv("acute_mai_weights_with_population.csv", header=["weight"])

final["Acute_MAI"] = X37_imputed[acute_weights_37.index].values @ acute_weights_37.values
final["Chronic_Acute_Overlap_MAI"] = np.sqrt(final["Chronic_MAI"].clip(lower=0) * final["Acute_MAI"].clip(lower=0))

# --- PatientPool v2: corrected Population_2011 x prevalence composite ---
chronic_vars = pillar_map[pillar_map == "Chronic"].index.tolist()
acute_prevalence_vars = [v for v in pillar_map[pillar_map == "Acute"].index if "Prevalence of" in v]
prevalence_vars = chronic_vars + acute_prevalence_vars
prevalence_composite = norm[prevalence_vars].mean(axis=1, skipna=True)
population_filled = norm["Population_2011"].fillna(0)
final["PatientPool_proxy"] = population_filled.values * prevalence_composite.values
final["PatientPool_x_MAI"] = final["PatientPool_proxy"] * final["Overall_MAI_entropy"]

TOTAL_POOL = 100
final["Reps_allocated"] = TOTAL_POOL * final["PatientPool_x_MAI"] / final["PatientPool_x_MAI"].sum()

final.to_csv("final_district_mai_scores.csv", index=False)
print(f"\nSaved corrected final_district_mai_scores.csv: {final.shape}")

# ============================================================
# Before/after Top-10 sales-force comparison: v1 (hospital-based, always correct) vs
# v2 (Census 2011 population-based, NOW double-counting-corrected)
# ============================================================
final["PatientPool_x_MAI_v1"] = final["PatientPool_proxy_v1_hospital_based"] * final["Overall_MAI_entropy"]

before_top10 = final.sort_values("PatientPool_x_MAI_v1", ascending=False).head(10)[
    ["State", "District", "PatientPool_proxy_v1_hospital_based", "Reps_allocated_v1_hospital_based"]
]
after_top10 = final.sort_values("PatientPool_x_MAI", ascending=False).head(10)[
    ["State", "District", "PatientPool_proxy", "Reps_allocated", "Population_2011", "is_population_apportioned"]
]

pd.set_option("display.width", 160)
print("\n=== BEFORE (hospital-count-based PatientPool, v1) Top 10 ===")
print(before_top10.round(2).to_string(index=False))
print(f"Reps sum (before, top10): {before_top10['Reps_allocated_v1_hospital_based'].sum():.2f}")

print("\n=== AFTER (Census 2011 population-based, double-count CORRECTED) Top 10 ===")
print(after_top10.round(2).to_string(index=False))
print(f"Reps sum (after, top10): {after_top10['Reps_allocated'].sum():.2f}")

compare_cols = ["State", "District", "PatientPool_proxy_v1_hospital_based", "PatientPool_proxy",
                "Reps_allocated_v1_hospital_based", "Reps_allocated", "Population_2011", "is_population_apportioned"]
final[compare_cols].rename(columns={"PatientPool_proxy": "PatientPool_v2", "Reps_allocated": "Reps_v2"}).to_csv(
    "patientpool_before_after_comparison.csv", index=False
)
print("\nSaved patientpool_before_after_comparison.csv (corrected)")
