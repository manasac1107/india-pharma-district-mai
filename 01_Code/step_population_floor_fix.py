import pandas as pd
import numpy as np
from scipy.stats import spearmanr

norm = pd.read_csv("normalized_district_panel_full.csv")

# always refresh Population_2011/_norm from the corrected source rather than trusting a
# possibly-stale cached column in normalized_district_panel_full.csv
pop = pd.read_csv("master_district_panel_with_population.csv")
norm = norm.drop(columns=["Population_2011", "Population_2011_norm"], errors="ignore")
norm = norm.merge(pop[["State", "District", "Population_2011"]], on=["State", "District"], how="left")
log_pop = np.log1p(norm["Population_2011"])
lo_lp, hi_lp = log_pop.min(), log_pop.max()
norm["Population_2011_norm"] = (log_pop - lo_lp) / (hi_lp - lo_lp)
norm.to_csv("normalized_district_panel_full.csv", index=False)

# ============================================================
# Rebuild the 36-variable base (health/hospital/GDP-CAGR only, NO population) fresh and
# deterministic, then blend with a 15% population floor:
#   Overall_MAI = 0.85 x (36-var weighted score, weights renormalized to sum 1 among the
#   36) + 0.15 x Population_2011_norm
# Applied identically to entropy, judgment, and equal-weight versions, per the user's
# choice, so the population floor is not an entropy-only quirk.
# ============================================================
trend_cap = pd.read_csv("../indicator_trend_capability.csv")
base_indicators = trend_cap["indicator"].tolist()
nfhs5_vars = [f"{i}_NFHS5_norm" for i in base_indicators]
hospital_vars = ["total_hospital_count_norm", "public_hospital_count_norm", "private_hospital_count_norm"]
broadcast_vars = ["gdp_cagr_pct_state_broadcast_norm"]
SCORING_VARS_36 = nfhs5_vars + hospital_vars + broadcast_vars
assert len(SCORING_VARS_36) == 36

def pillar_of(var):
    if var in hospital_vars:
        return "Access"
    if var == "gdp_cagr_pct_state_broadcast_norm":
        return "Momentum"
    low = var.lower()
    if any(k in low for k in ["sex ratio"]):
        return "Demographic"
    if any(k in low for k in ["blood sugar", "blood pressure", "cancer", "tobacco", "alcohol"]):
        return "Chronic"
    if any(k in low for k in ["diarrhoea", "ari", "acute respiratory"]):
        return "Acute"
    if any(k in low for k in ["sanitation", "drinkingwater", "electricity", "insurance", "literate"]):
        return "Infrastructure_Affordability"
    return "UNMAPPED"

pillar_map = pd.Series({v: pillar_of(v) for v in SCORING_VARS_36})
assert (pillar_map == "UNMAPPED").sum() == 0

X36 = norm[SCORING_VARS_36].copy()
X36_imputed = X36.fillna(X36.mean())
n = len(X36_imputed)
eps = 1e-12
P = X36_imputed / (X36_imputed.sum(axis=0) + eps)
P = P.clip(lower=eps)
k = 1 / np.log(n)
entropy = -k * (P * np.log(P)).sum(axis=0)
diversification = 1 - entropy
entropy_w36 = diversification / diversification.sum()

entropy_weights_df = pd.DataFrame({
    "variable": SCORING_VARS_36, "pillar": pillar_map[SCORING_VARS_36].values,
    "entropy": entropy.values, "diversification": diversification.values,
    "entropy_weight": entropy_w36.values,
}).sort_values("entropy_weight", ascending=False)
entropy_weights_df.to_csv("entropy_weights.csv", index=False)
pillar_map.rename("pillar").to_frame().reset_index().rename(columns={"index": "variable"}).to_csv("pillar_mapping.csv", index=False)

# --- judgment weights (36-var, original scheme incl. 5% Demographic carve-out) ---
PILLAR_WEIGHTS = {
    "Chronic": 0.30 * 0.95, "Acute": 0.30 * 0.95, "Infrastructure_Affordability": 0.20 * 0.95,
    "Access": 0.10 * 0.95, "Momentum": 0.10 * 0.95, "Demographic": 0.05,
}
pillar_counts = pillar_map.value_counts().to_dict()
judgment_w36 = pillar_map.map(PILLAR_WEIGHTS) / pillar_map.map(pillar_counts)
judgment_weights_df = pd.DataFrame({
    "variable": SCORING_VARS_36, "pillar": pillar_map[SCORING_VARS_36].values,
    "judgment_weight": judgment_w36[SCORING_VARS_36].values,
})
judgment_weights_df.to_csv("judgment_weights.csv", index=False)

equal_w36 = pd.Series(1 / 36, index=SCORING_VARS_36)

pop_norm = norm.set_index(["State", "District"])["Population_2011_norm"] if "Population_2011_norm" in norm.columns else None
if pop_norm is None:
    raise RuntimeError("Population_2011_norm missing -- run step_add_population_to_overall.py's normalization step first")

POP_WEIGHT = 0.15
BASE_WEIGHT = 0.85

score_36_entropy = X36_imputed[SCORING_VARS_36].values @ entropy_w36[SCORING_VARS_36].values
score_36_judgment = X36_imputed[SCORING_VARS_36].values @ judgment_w36[SCORING_VARS_36].values
score_36_equal = X36_imputed[SCORING_VARS_36].values @ equal_w36[SCORING_VARS_36].values

out = norm[["State", "District", "Population_2011", "Population_2011_norm"]].copy()
pop_norm_filled = out["Population_2011_norm"].fillna(out["Population_2011_norm"].mean())

out["Overall_MAI_entropy"] = BASE_WEIGHT * score_36_entropy + POP_WEIGHT * pop_norm_filled.values
out["Overall_MAI_judgment"] = BASE_WEIGHT * score_36_judgment + POP_WEIGHT * pop_norm_filled.values
out["Overall_MAI_equal_weight"] = BASE_WEIGHT * score_36_equal + POP_WEIGHT * pop_norm_filled.values

out["Rank_entropy"] = out["Overall_MAI_entropy"].rank(ascending=False, method="min").astype(int)
out["Rank_judgment"] = out["Overall_MAI_judgment"].rank(ascending=False, method="min").astype(int)
out["Rank_Diff_abs"] = (out["Rank_entropy"] - out["Rank_judgment"]).abs()
out.to_csv("overall_mai_entropy_vs_judgment.csv", index=False)

rho, p = spearmanr(out["Overall_MAI_entropy"], out["Overall_MAI_judgment"])
print(f"Formula: Overall_MAI = {BASE_WEIGHT} x (36-var weighted score) + {POP_WEIGHT} x Population_2011_norm (log-transformed, min-max normalized)")
print(f"Spearman rho (entropy vs judgment, with population floor): {rho:.4f}")

print("\n=== NEW Top 20 by Overall MAI (entropy-weighted, WITH 15% population floor) ===")
top20 = out.sort_values("Overall_MAI_entropy", ascending=False).head(20)
pd.set_option("display.width", 160)
print(top20[["State", "District", "Overall_MAI_entropy", "Population_2011"]].round(4).to_string(index=False))

n_under_100k = (top20["Population_2011"] < 100000).sum()
print(f"\nDistricts with population < 100,000 in new Top 20: {n_under_100k}")
