import pandas as pd
import numpy as np
from scipy.stats import spearmanr

norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]
four = pd.read_csv("clustered_districts.csv")

SCORING_VARS = entropy_w.index.tolist()
X = norm[SCORING_VARS].copy()
X_imputed = X.fillna(X.mean())

base_scores = four["Overall_MAI_entropy"]
base_top10 = set(four.sort_values("Overall_MAI_entropy", ascending=False).head(10)["District"] + "_" + four.sort_values("Overall_MAI_entropy", ascending=False).head(10)["State"])

# ============================================================
# 7a. Sensitivity analysis: perturb each of the top-3 entropy-weighted variables by +/-10%
# ============================================================
top3 = entropy_w.sort_values(ascending=False).head(3)
print("Top 3 entropy-weighted variables (perturbation targets):")
print(top3.to_string())

results = []
for var in top3.index:
    for pct in [0.10, -0.10]:
        w = entropy_w.copy()
        w[var] = w[var] * (1 + pct)
        w = w / w.sum()  # renormalize to sum to 1
        scores = X_imputed[SCORING_VARS].values @ w[SCORING_VARS].values
        rho, _ = spearmanr(base_scores, scores)

        tmp = four.copy()
        tmp["perturbed_score"] = scores
        perturbed_top10 = set(
            tmp.sort_values("perturbed_score", ascending=False).head(10)["District"] + "_" +
            tmp.sort_values("perturbed_score", ascending=False).head(10)["State"]
        )
        overlap = len(base_top10 & perturbed_top10)

        results.append({
            "variable": var, "perturbation": f"{pct:+.0%}",
            "spearman_rho_vs_base": round(rho, 4),
            "top10_overlap_count": overlap,
        })

sens_df = pd.DataFrame(results)
sens_df.to_csv("sensitivity_analysis.csv", index=False)
print("\n=== Sensitivity analysis: +/-10% perturbation of top-3 weights ===")
print(sens_df.to_string(index=False))
print("\n(top10_overlap_count = how many of the base-case Top-10 districts remain in the")
print(" perturbed-case Top-10, out of 10)")

# ============================================================
# 7b. Top 10 / Bottom 10 districts by Overall MAI (face-validity check)
# ============================================================
print("\n=== Top 10 districts by Overall MAI (entropy-weighted) ===")
top10_tbl = four.sort_values("Overall_MAI_entropy", ascending=False).head(10)[
    ["State", "District", "Overall_MAI_entropy", "Chronic_MAI", "Acute_MAI", "Momentum", "Cluster_Label"]
]
print(top10_tbl.to_string(index=False))

print("\n=== Bottom 10 districts by Overall MAI (entropy-weighted) ===")
bottom10_tbl = four.sort_values("Overall_MAI_entropy", ascending=True).head(10)[
    ["State", "District", "Overall_MAI_entropy", "Chronic_MAI", "Acute_MAI", "Momentum", "Cluster_Label"]
]
print(bottom10_tbl.to_string(index=False))

# ============================================================
# 7c. Equal-weight baseline comparison
# ============================================================
equal_w = pd.Series(1 / len(SCORING_VARS), index=SCORING_VARS)
equal_scores = X_imputed[SCORING_VARS].values @ equal_w[SCORING_VARS].values
rho_equal, p_equal = spearmanr(base_scores, equal_scores)
print(f"\n=== Equal-weight baseline comparison ===")
print(f"Spearman rho (entropy-weighted vs. equal-weighted Overall MAI): {rho_equal:.4f}, p = {p_equal:.2e}")

four["Overall_MAI_equal_weight"] = equal_scores
four.to_csv("four_indices_final.csv", index=False)
print("\nSaved sensitivity_analysis.csv and four_indices_final.csv")
