import pandas as pd
import numpy as np
from scipy.stats import spearmanr

norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w36 = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]
overall = pd.read_csv("overall_mai_entropy_vs_judgment.csv")
four = pd.read_csv("clustered_districts.csv")

BASE_WEIGHT, POP_WEIGHT = 0.85, 0.15
SCORING_VARS_36 = entropy_w36.index.tolist()
X36 = norm[SCORING_VARS_36].copy()
X36_imputed = X36.fillna(X36.mean())
pop_norm = norm["Population_2011_norm"].fillna(norm["Population_2011_norm"].mean())

base_scores = overall.set_index(["State", "District"])["Overall_MAI_entropy"]
base_scores = base_scores.reindex(list(zip(norm["State"], norm["District"]))).values

base_top10_keys = set(zip(
    four.sort_values("Overall_MAI_entropy", ascending=False).head(10)["State"],
    four.sort_values("Overall_MAI_entropy", ascending=False).head(10)["District"],
))

# ============================================================
# 7a. Sensitivity: perturb top-3 of the 36 BASE variables (+/-10%), AND separately
# perturb the population-floor weight itself (0.15 -> 0.135 / 0.165), since that's now a
# material design parameter of the corrected formula.
# ============================================================
top3 = entropy_w36.sort_values(ascending=False).head(3)
print("Top 3 entropy-weighted base variables:")
print(top3.to_string())

results = []
for var in top3.index:
    for pct in [0.10, -0.10]:
        w = entropy_w36.copy()
        w[var] = w[var] * (1 + pct)
        w = w / w.sum()
        base_score = X36_imputed[SCORING_VARS_36].values @ w[SCORING_VARS_36].values
        scores = BASE_WEIGHT * base_score + POP_WEIGHT * pop_norm.values
        rho, _ = spearmanr(base_scores, scores)
        tmp = norm[["State", "District"]].copy()
        tmp["s"] = scores
        p_top10 = set(zip(tmp.sort_values("s", ascending=False).head(10)["State"], tmp.sort_values("s", ascending=False).head(10)["District"]))
        results.append({"perturbation_type": "base_variable", "variable": var, "perturbation": f"{pct:+.0%}",
                         "spearman_rho_vs_base": round(rho, 4), "top10_overlap_count": len(base_top10_keys & p_top10)})

for new_pop_w in [0.135, 0.165]:
    new_base_w = 1 - new_pop_w
    scores = new_base_w * (X36_imputed[SCORING_VARS_36].values @ entropy_w36[SCORING_VARS_36].values) + new_pop_w * pop_norm.values
    rho, _ = spearmanr(base_scores, scores)
    tmp = norm[["State", "District"]].copy()
    tmp["s"] = scores
    p_top10 = set(zip(tmp.sort_values("s", ascending=False).head(10)["State"], tmp.sort_values("s", ascending=False).head(10)["District"]))
    results.append({"perturbation_type": "population_floor_weight", "variable": "Population_2011_norm",
                     "perturbation": f"{new_pop_w:.0%} (base case 15%)",
                     "spearman_rho_vs_base": round(rho, 4), "top10_overlap_count": len(base_top10_keys & p_top10)})

sens_df = pd.DataFrame(results)
sens_df.to_csv("sensitivity_analysis.csv", index=False)
print("\n=== Sensitivity analysis (with population floor) ===")
pd.set_option("display.width", 160)
print(sens_df.to_string(index=False))

# ============================================================
# 7b. Top 10 / Bottom 10
# ============================================================
print("\n=== Top 10 districts by Overall MAI (with population floor) ===")
print(four.sort_values("Overall_MAI_entropy", ascending=False).head(10)[
    ["State", "District", "Overall_MAI_entropy", "Population_2011", "Cluster_Label"]
].to_string(index=False))

print("\n=== Bottom 10 districts by Overall MAI ===")
print(four.sort_values("Overall_MAI_entropy", ascending=True).head(10)[
    ["State", "District", "Overall_MAI_entropy", "Population_2011", "Cluster_Label"]
].to_string(index=False))

# ============================================================
# 7c. Equal-weight baseline (also with 15% population floor applied identically)
# ============================================================
equal_w36 = pd.Series(1 / 36, index=SCORING_VARS_36)
equal_scores = BASE_WEIGHT * (X36_imputed[SCORING_VARS_36].values @ equal_w36[SCORING_VARS_36].values) + POP_WEIGHT * pop_norm.values
rho_equal, p_equal = spearmanr(base_scores, equal_scores)
print(f"\nSpearman rho (entropy+pop vs equal-weight+pop): {rho_equal:.4f}")

four["Overall_MAI_equal_weight"] = overall.set_index(["State","District"])["Overall_MAI_equal_weight"].reindex(
    list(zip(four["State"], four["District"]))
).values
four.to_csv("four_indices_final.csv", index=False)
print("\nSaved four_indices_final.csv (refreshed) and sensitivity_analysis.csv")
