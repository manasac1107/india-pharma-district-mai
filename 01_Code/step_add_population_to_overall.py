import pandas as pd
import numpy as np

norm = pd.read_csv("normalized_district_panel_full.csv")
old_entropy_w = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]
old_pillar_map = pd.read_csv("pillar_mapping.csv").set_index("variable")["pillar"]
pop = pd.read_csv("master_district_panel_with_population.csv")

norm = norm.merge(pop[["State", "District", "Population_2011"]], on=["State", "District"], how="left")

# --- log-transform (population is heavily right-skewed: ~37K to ~11M) then min-max ---
log_pop = np.log1p(norm["Population_2011"])
lo, hi = log_pop.min(), log_pop.max()
norm["Population_2011_norm"] = (log_pop - lo) / (hi - lo)
norm.to_csv("normalized_district_panel_full.csv", index=False)

OLD_SCORING_VARS = old_entropy_w.index.tolist()
SCORING_VARS = OLD_SCORING_VARS + ["Population_2011_norm"]

# --- pillar mapping: new "Market_Size" pillar for population, judgment weight carved
#     out at 10%, other 6 pillars (Chronic/Acute/Infra-Afford/Access/Momentum/Demographic)
#     scaled by x0.90 so everything still sums to 100%. Documented, flagged for review. ---
pillar_map = old_pillar_map.copy()
pillar_map["Population_2011_norm"] = "Market_Size"
pillar_map.to_csv("pillar_mapping.csv", header=["pillar"])

# ============================================================
# Recompute canonical entropy weights on the 37-variable set (this REPLACES the old
# 36-var entropy_weights.csv used by Overall/Chronic/Acute MAI, judgment weights,
# clustering, and sensitivity -- population is now a first-class scoring variable
# everywhere downstream, not just Acute MAI's separate side-table.
# ============================================================
X = norm[SCORING_VARS].copy()
X_imputed = X.fillna(X.mean())
n = len(X_imputed)
eps = 1e-12
P = X_imputed / (X_imputed.sum(axis=0) + eps)
P = P.clip(lower=eps)
k = 1 / np.log(n)
entropy = -k * (P * np.log(P)).sum(axis=0)
diversification = 1 - entropy
entropy_weights = diversification / diversification.sum()

entropy_weights_df = pd.DataFrame({
    "variable": SCORING_VARS,
    "pillar": pillar_map.loc[SCORING_VARS].values,
    "entropy": entropy.values,
    "diversification": diversification.values,
    "entropy_weight": entropy_weights.values,
}).sort_values("entropy_weight", ascending=False)
entropy_weights_df.to_csv("entropy_weights.csv", index=False)

print("Population_2011_norm entropy weight (37-var canonical):",
      round(entropy_weights_df.set_index("variable").loc["Population_2011_norm", "entropy_weight"], 4))
print("\nTop 10 entropy weights (37-var canonical):")
pd.set_option("display.width", 160)
print(entropy_weights_df.head(10).to_string(index=False))

# ============================================================
# Judgment weights: carve 10% for Market_Size, scale other 6 pillars by x0.90
# ============================================================
PILLAR_WEIGHTS = {
    "Chronic": 0.30 * 0.95 * 0.90,
    "Acute": 0.30 * 0.95 * 0.90,
    "Infrastructure_Affordability": 0.20 * 0.95 * 0.90,
    "Access": 0.10 * 0.95 * 0.90,
    "Momentum": 0.10 * 0.95 * 0.90,
    "Demographic": 0.05 * 0.90,
    "Market_Size": 0.10,
}
assert abs(sum(PILLAR_WEIGHTS.values()) - 1.0) < 1e-9

pillar_counts = pillar_map.loc[SCORING_VARS].value_counts().to_dict()
judgment_weights_df = pd.DataFrame({"variable": SCORING_VARS, "pillar": pillar_map.loc[SCORING_VARS].values})
judgment_weights_df["pillar_weight"] = judgment_weights_df["pillar"].map(PILLAR_WEIGHTS)
judgment_weights_df["n_in_pillar"] = judgment_weights_df["pillar"].map(pillar_counts)
judgment_weights_df["judgment_weight"] = judgment_weights_df["pillar_weight"] / judgment_weights_df["n_in_pillar"]
judgment_weights_df = judgment_weights_df.drop(columns=["pillar_weight", "n_in_pillar"])
judgment_weights_df.to_csv("judgment_weights.csv", index=False)
print("\nJudgment weights by pillar:")
print(judgment_weights_df.groupby("pillar")["judgment_weight"].sum().round(4).to_string())

# ============================================================
# Recompute Overall MAI (entropy, judgment, equal-weight)
# ============================================================
entropy_w = entropy_weights_df.set_index("variable")["entropy_weight"]
judgment_w = judgment_weights_df.set_index("variable")["judgment_weight"]
equal_w = pd.Series(1 / len(SCORING_VARS), index=SCORING_VARS)

out = norm[["State", "District"]].copy()
out["Overall_MAI_entropy"] = X_imputed[SCORING_VARS].values @ entropy_w[SCORING_VARS].values
out["Overall_MAI_judgment"] = X_imputed[SCORING_VARS].values @ judgment_w[SCORING_VARS].values
out["Overall_MAI_equal_weight"] = X_imputed[SCORING_VARS].values @ equal_w[SCORING_VARS].values
out["Rank_entropy"] = out["Overall_MAI_entropy"].rank(ascending=False, method="min").astype(int)
out["Rank_judgment"] = out["Overall_MAI_judgment"].rank(ascending=False, method="min").astype(int)
out["Rank_Diff_abs"] = (out["Rank_entropy"] - out["Rank_judgment"]).abs()

out.to_csv("overall_mai_entropy_vs_judgment.csv", index=False)

print("\n=== NEW Top 20 by Overall MAI (entropy-weighted, with population) ===")
top20 = out.merge(norm[["State", "District", "Population_2011"]], on=["State", "District"]).sort_values(
    "Overall_MAI_entropy", ascending=False
).head(20)
print(top20[["State", "District", "Overall_MAI_entropy", "Population_2011"]].round(3).to_string(index=False))

from scipy.stats import spearmanr
rho, p = spearmanr(out["Overall_MAI_entropy"], out["Overall_MAI_judgment"])
print(f"\nSpearman rho (entropy vs judgment, new): {rho:.4f}")
