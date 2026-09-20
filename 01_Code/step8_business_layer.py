import pandas as pd
import numpy as np

final = pd.read_csv("four_indices_final.csv")
norm = pd.read_csv("normalized_district_panel_full.csv")
pillar_map = pd.read_csv("pillar_mapping.csv")
m = pd.read_csv("master_district_panel_broadcast.csv")

# ============================================================
# Tiering: Overall MAI quartiles
# ============================================================
q1, q2, q3 = final["Overall_MAI_entropy"].quantile([0.25, 0.5, 0.75])
print(f"Overall MAI quartile cutoffs: Q1={q1:.4f}, Q2(median)={q2:.4f}, Q3={q3:.4f}")

def tier_of(x):
    if x >= q3:
        return "Tier 1"
    elif x >= q2:
        return "Tier 2"
    elif x >= q1:
        return "Tier 3"
    else:
        return "Tier 4"

final["Tier"] = final["Overall_MAI_entropy"].apply(tier_of)
print("\nTier counts:")
print(final["Tier"].value_counts().sort_index().to_string())
print("(Tier 1 = top quartile by Overall MAI, i.e. highest attractiveness; Tier 4 = bottom quartile)")

# ============================================================
# PatientPool proxy (user-confirmed): total_hospital_count x mean(normalized chronic +
# acute PREVALENCE indicators). "Prevalence" here means true disease/risk-factor rate
# variables only -- the 19 Chronic-pillar indicators (blood pressure, blood sugar, cancer
# screening uptake, tobacco/alcohol use) plus the 2 acute disease-prevalence indicators
# (diarrhoea prevalence, ARI prevalence). Acute care-seeking variables (ORS/zinc/facility
# visit) are deliberately excluded from this composite -- they measure treatment-seeking
# behavior, not disease burden, so including them would double-count healthcare access
# that's already captured via the Access pillar.
# CAVEAT (repeated from the modeling discussion): total_hospital_count is a catchment-
# activity proxy, NOT a real population count. No true population variable exists in any
# source file used in this pipeline. This PatientPool is illustrative of the allocation
# METHOD, not a validated true patient count -- flagged explicitly for the case write-up.
# ============================================================
chronic_vars = pillar_map.loc[pillar_map["pillar"] == "Chronic", "variable"].tolist()
acute_prevalence_vars = [
    v for v in pillar_map.loc[pillar_map["pillar"] == "Acute", "variable"]
    if "Prevalence of" in v
]
print(f"\nPrevalence composite built from {len(chronic_vars)} Chronic + {len(acute_prevalence_vars)} Acute-prevalence variables")
print("Acute-prevalence variables used:", acute_prevalence_vars)

prevalence_vars = chronic_vars + acute_prevalence_vars
prevalence_composite = norm[prevalence_vars].mean(axis=1, skipna=True)

hospital_count = m["total_hospital_count"].fillna(0)
final["PatientPool_proxy"] = hospital_count.values * prevalence_composite.values

# ============================================================
# Reps(d) = TotalPool * [PatientPool(d) * MAI(d)] / sum_i[PatientPool(i) * MAI(i)]
# ============================================================
TOTAL_POOL = 100
final["PatientPool_x_MAI"] = final["PatientPool_proxy"] * final["Overall_MAI_entropy"]
final["Reps_allocated"] = TOTAL_POOL * final["PatientPool_x_MAI"] / final["PatientPool_x_MAI"].sum()

final.to_csv("four_indices_business.csv", index=False)

print(f"\n=== Worked example: {TOTAL_POOL}-rep pool, Top 10 districts by PatientPool x MAI ===")
top10 = final.sort_values("PatientPool_x_MAI", ascending=False).head(10)
pd.set_option("display.width", 160)
print(top10[["State", "District", "Tier", "PatientPool_proxy", "Overall_MAI_entropy", "Reps_allocated"]].round(3).to_string(index=False))
print(f"Reps allocated to these 10 districts: {top10['Reps_allocated'].sum():.2f} / {TOTAL_POOL}")

print(f"\n=== Bottom 5 districts by PatientPool x MAI ===")
bottom5 = final.sort_values("PatientPool_x_MAI", ascending=True).head(5)
print(bottom5[["State", "District", "Tier", "PatientPool_proxy", "Overall_MAI_entropy", "Reps_allocated"]].round(3).to_string(index=False))
print(f"Reps allocated to these 5 districts: {bottom5['Reps_allocated'].sum():.4f} / {TOTAL_POOL}")

print(f"\nTotal reps allocated across all {len(final)} districts: {final['Reps_allocated'].sum():.2f} (should equal {TOTAL_POOL})")
