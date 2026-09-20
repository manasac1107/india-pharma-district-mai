import pandas as pd
import numpy as np

four = pd.read_csv("four_indices_final.csv")
patientpool = pd.read_csv("patientpool_before_after_comparison.csv")

# --- refresh Future_MAI_3yr / Rank_Current / Rank_Future / Rank_Delta using the
#     corrected (population-floor) Overall_MAI_entropy ---
mean_momentum = four["Momentum"].mean()
ADJUSTMENT_SCALE = 0.15
four["Future_MAI_3yr"] = four["Overall_MAI_entropy"] + ADJUSTMENT_SCALE * (four["Momentum"] - mean_momentum)
four["Rank_Current"] = four["Overall_MAI_entropy"].rank(ascending=False, method="min").astype(int)
four["Rank_Future"] = four["Future_MAI_3yr"].rank(ascending=False, method="min").astype(int)
four["Rank_Delta"] = four["Rank_Current"] - four["Rank_Future"]

# --- Tiering: quartiles of the corrected Overall_MAI_entropy ---
q1, q2, q3 = four["Overall_MAI_entropy"].quantile([0.25, 0.5, 0.75])
def tier_of(x):
    if x >= q3: return "Tier 1"
    elif x >= q2: return "Tier 2"
    elif x >= q1: return "Tier 3"
    else: return "Tier 4"
four["Tier"] = four["Overall_MAI_entropy"].apply(tier_of)
print(f"Tier cutoffs: Q1={q1:.4f}, median={q2:.4f}, Q3={q3:.4f}")
print(four["Tier"].value_counts().sort_index().to_string())

# --- PatientPool / Reps (already corrected, population-based) ---
four = four.merge(
    patientpool[["State", "District", "PatientPool_v2", "Reps_v2"]],
    on=["State", "District"], how="left"
).rename(columns={"PatientPool_v2": "PatientPool_proxy", "Reps_v2": "Reps_allocated"})
four["PatientPool_x_MAI"] = four["PatientPool_proxy"] * four["Overall_MAI_entropy"]

final_cols = [
    "State", "District",
    "Overall_MAI_entropy", "Overall_MAI_judgment", "Overall_MAI_equal_weight",
    "Rank_entropy", "Rank_judgment", "Rank_Diff_abs",
    "Chronic_MAI", "Acute_MAI", "Chronic_Acute_Overlap_MAI",
    "Momentum", "Future_MAI_3yr", "Rank_Current", "Rank_Future", "Rank_Delta",
    "Cluster_Label", "Tier",
    "Population_2011",
    "PatientPool_proxy", "PatientPool_x_MAI", "Reps_allocated",
]
final = four[final_cols].copy()
final.to_csv("final_district_mai_scores.csv", index=False)
print(f"\nSaved final_district_mai_scores.csv: {final.shape}")
print(f"Reps sum check: {final['Reps_allocated'].sum():.4f} (should be 100)")

print("\n=== FINAL Top 10 by Overall MAI ===")
pd.set_option("display.width", 160)
print(final.sort_values("Overall_MAI_entropy", ascending=False).head(10)[
    ["State", "District", "Overall_MAI_entropy", "Population_2011", "Cluster_Label", "Tier", "Reps_allocated"]
].round(3).to_string(index=False))
