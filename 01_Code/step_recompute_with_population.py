import pandas as pd
import numpy as np

pop = pd.read_csv("master_district_panel_with_population.csv")
final_old = pd.read_csv("final_district_mai_scores.csv")
norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]
pillar_map = pd.read_csv("pillar_mapping.csv").set_index("variable")["pillar"]

# ============================================================
# Add Population_2011 to master_district_panel.csv (the actual source file, per the
# original request), and merge into the working panel for scoring.
# ============================================================
master = pd.read_csv("../master_district_panel.csv")
master = master.merge(pop[["State", "District", "Population_2011", "is_population_apportioned", "population_apportion_method"]],
                       on=["State", "District"], how="left")
master.to_csv("../master_district_panel.csv", index=False)
print(f"Added Population_2011 to master_district_panel.csv ({master['Population_2011'].notna().sum()}/{len(master)} non-null)")

norm = norm.merge(pop[["State", "District", "Population_2011"]], on=["State", "District"], how="left")
lo, hi = norm["Population_2011"].min(), norm["Population_2011"].max()
norm["Population_2011_norm"] = (norm["Population_2011"] - lo) / (hi - lo)

SCORING_VARS = entropy_w.index.tolist()

# ============================================================
# Recompute entropy weight for the 37-variable set (36 existing + Population), used ONLY
# for Acute MAI (per the scoped request -- Overall MAI / Chronic MAI / Momentum / clusters
# are left untouched on their original 36-variable basis, since only Acute MAI and
# PatientPool were asked to be recomputed).
# ============================================================
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

print(f"\nPopulation_2011_norm entropy weight (in 37-var pool): {entropy_w37['Population_2011_norm']:.4f}")

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

Acute_MAI_new = X37_imputed[acute_weights_37.index].values @ acute_weights_37.values

# ============================================================
# PatientPool proxy v2: Population_2011(d) x mean(normalized chronic + acute-prevalence
# composite) -- same prevalence composite as before, size term swapped from
# total_hospital_count to real Census 2011 population.
# ============================================================
chronic_vars = pillar_map[pillar_map == "Chronic"].index.tolist()
acute_prevalence_vars = [v for v in pillar_map[pillar_map == "Acute"].index if "Prevalence of" in v]
prevalence_vars = chronic_vars + acute_prevalence_vars
prevalence_composite = norm[prevalence_vars].mean(axis=1, skipna=True)

population_filled = norm["Population_2011"].fillna(0)  # Ladakh only (2 districts)
PatientPool_v2 = population_filled.values * prevalence_composite.values

# ============================================================
# Rebuild final table
# ============================================================
final = final_old.copy()
final["Acute_MAI_v1_hospital_based"] = final["Acute_MAI"]
final["Acute_MAI"] = Acute_MAI_new
final["Chronic_Acute_Overlap_MAI"] = np.sqrt(final["Chronic_MAI"].clip(lower=0) * final["Acute_MAI"].clip(lower=0))

final["Population_2011"] = pop["Population_2011"].values
final["is_population_apportioned"] = pop["is_population_apportioned"].values
final["PatientPool_proxy_v1_hospital_based"] = final["PatientPool_proxy"]
final["PatientPool_proxy"] = PatientPool_v2
final["PatientPool_x_MAI"] = final["PatientPool_proxy"] * final["Overall_MAI_entropy"]

TOTAL_POOL = 100
final["Reps_allocated_v1_hospital_based"] = final["Reps_allocated"]
final["Reps_allocated"] = TOTAL_POOL * final["PatientPool_x_MAI"] / final["PatientPool_x_MAI"].sum()

final.to_csv("final_district_mai_scores.csv", index=False)
print(f"\nSaved updated final_district_mai_scores.csv: {final.shape}")

# ============================================================
# Before/after Top-10 sales-force comparison
# ============================================================
before_top10 = final_old.sort_values("PatientPool_x_MAI", ascending=False).head(10)[
    ["State", "District", "PatientPool_proxy", "Reps_allocated"]
].rename(columns={"PatientPool_proxy": "PatientPool_v1_hospital", "Reps_allocated": "Reps_v1_hospital"})

after_top10 = final.sort_values("PatientPool_x_MAI", ascending=False).head(10)[
    ["State", "District", "PatientPool_proxy", "Reps_allocated", "Population_2011", "is_population_apportioned"]
].rename(columns={"PatientPool_proxy": "PatientPool_v2_population", "Reps_allocated": "Reps_v2_population"})

print("\n=== BEFORE (hospital-count-based PatientPool) Top 10 ===")
pd.set_option("display.width", 160)
print(before_top10.round(2).to_string(index=False))
print(f"Reps sum (before, top10): {before_top10['Reps_v1_hospital'].sum():.2f}")

print("\n=== AFTER (Census 2011 population-based PatientPool) Top 10 ===")
print(after_top10.round(2).to_string(index=False))
print(f"Reps sum (after, top10): {after_top10['Reps_v2_population'].sum():.2f}")

merged_compare = final[["State", "District", "PatientPool_proxy_v1_hospital_based", "PatientPool_proxy",
                         "Reps_allocated_v1_hospital_based", "Reps_allocated", "Population_2011", "is_population_apportioned"]]
merged_compare = merged_compare.rename(columns={"PatientPool_proxy": "PatientPool_v2", "Reps_allocated": "Reps_v2"})
merged_compare.to_csv("patientpool_before_after_comparison.csv", index=False)
print("\nSaved patientpool_before_after_comparison.csv (full 706-district before/after table)")
