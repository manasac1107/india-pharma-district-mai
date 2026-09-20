import pandas as pd
import numpy as np

m = pd.read_csv("master_district_panel_broadcast.csv")
id_cols = ["State", "District"]
numeric_cols = [c for c in m.columns if c not in id_cols]
for c in numeric_cols:
    m[c] = pd.to_numeric(m[c], errors="coerce")

# --- full normalization (district-level + state-broadcast), all higher_is_better per
#     confirmed demand framing, consistent with Step 1 ---
norm = m[id_cols].copy()
for c in numeric_cols:
    lo, hi = m[c].min(), m[c].max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        norm[c + "_norm"] = np.nan
    else:
        norm[c + "_norm"] = (m[c] - lo) / (hi - lo)
norm.to_csv("normalized_district_panel_full.csv", index=False)
print(f"Saved normalized_district_panel_full.csv: {norm.shape}")

# --- define the 49-variable scoring set: NFHS5 level for all 32 indicators (never NFHS4
#     for the 20 trend_capable=False ones, per the standing constraint), + 3 hospital
#     count vars + 14 state-broadcast vars ---
trend_cap = pd.read_csv("../indicator_trend_capability.csv")
base_indicators = trend_cap["indicator"].tolist()
nfhs5_vars = [f"{i}_NFHS5_norm" for i in base_indicators]
hospital_vars = ["total_hospital_count_norm", "public_hospital_count_norm", "private_hospital_count_norm"]

# EXCLUDED from scoring: the 12 HMIS childhood-disease indicators + OPD_Total are raw
# absolute case counts (confirmed via the source file's "unit" column = "value in
# Absolute Number"), and no population variable exists anywhere in this pipeline to
# convert them to rates. Left in as-is, they mechanically favor large-population states
# (Uttar Pradesh swept the Top-10 entirely in an earlier run because of this) -- a size
# confound, not a real attractiveness signal. User confirmed: exclude from all scoring,
# keep gdp_cagr_pct_state_broadcast (a growth-rate percentage, not a count).
broadcast_vars = ["gdp_cagr_pct_state_broadcast_norm"]
EXCLUDED_ABSOLUTE_COUNT_BROADCASTS = [
    c + "_norm" for c in numeric_cols
    if c.endswith("_state_broadcast") and c != "gdp_cagr_pct_state_broadcast"
]
print(f"\nExcluded {len(EXCLUDED_ABSOLUTE_COUNT_BROADCASTS)} absolute-count HMIS broadcast variables from scoring "
      "(population-size confound, no population data to rate-adjust them):")
for c in EXCLUDED_ABSOLUTE_COUNT_BROADCASTS:
    print(f"    - {c}")

SCORING_VARS = nfhs5_vars + hospital_vars + broadcast_vars
missing = [v for v in SCORING_VARS if v not in norm.columns]
assert not missing, f"missing columns: {missing}"
print(f"\nScoring variable set: {len(SCORING_VARS)} variables "
      f"({len(nfhs5_vars)} NFHS5 indicator levels + {len(hospital_vars)} hospital + {len(broadcast_vars)} broadcast [GDP CAGR only])")

# --- pillar mapping (used for judgment weighting, and for Step 4 pillar indices) ---
CHRONIC_KEYWORDS = ["blood sugar", "blood pressure", "cancer", "tobacco", "alcohol"]
ACUTE_DISTRICT_KEYWORDS = ["diarrhoea", "ari", "acute respiratory"]
INFRA_AFFORD_KEYWORDS = ["sanitation", "drinkingwater", "electricity", "insurance", "literate"]
SEX_RATIO_KEYWORDS = ["sex ratio"]

def pillar_of(var):
    if var in hospital_vars:
        return "Access"
    if var == "gdp_cagr_pct_state_broadcast_norm":
        return "Momentum"
    if var.endswith("_state_broadcast_norm") and "OPD_Total" in var:
        return "Infrastructure_Affordability"
    if var.endswith("_state_broadcast_norm"):  # HMIS childhood acute-disease indicators
        return "Acute"
    low = var.lower()
    if any(k in low for k in SEX_RATIO_KEYWORDS):
        return "Demographic"
    if any(k in low for k in CHRONIC_KEYWORDS):
        return "Chronic"
    if any(k in low for k in ACUTE_DISTRICT_KEYWORDS):
        return "Acute"
    if any(k in low for k in INFRA_AFFORD_KEYWORDS):
        return "Infrastructure_Affordability"
    return "UNMAPPED"

pillar_map = pd.DataFrame({"variable": SCORING_VARS})
pillar_map["pillar"] = pillar_map["variable"].apply(pillar_of)
unmapped = pillar_map[pillar_map["pillar"] == "UNMAPPED"]
if len(unmapped):
    print("\nWARNING: unmapped variables:")
    print(unmapped)
pillar_map.to_csv("pillar_mapping.csv", index=False)
print("\nPillar variable counts:")
print(pillar_map["pillar"].value_counts().to_string())

# ============================================================
# Entropy weighting (Step 3a)
# ============================================================
X = norm[SCORING_VARS].copy()
# mean-impute missing normalized values for weighting/scoring purposes only (documented
# choice -- 58 districts lack hospital counts, ~9-13 lack GDP CAGR/HMIS broadcast, etc.)
X_imputed = X.fillna(X.mean())

n = len(X_imputed)
eps = 1e-12
P = X_imputed / (X_imputed.sum(axis=0) + eps)  # proportion within each variable, across districts
P = P.clip(lower=eps)  # avoid log(0)
k = 1 / np.log(n)
entropy = -k * (P * np.log(P)).sum(axis=0)
diversification = 1 - entropy
entropy_weights = diversification / diversification.sum()

entropy_weights_df = pd.DataFrame({
    "variable": SCORING_VARS,
    "pillar": pillar_map.set_index("variable").loc[SCORING_VARS, "pillar"].values,
    "entropy": entropy.values,
    "diversification": diversification.values,
    "entropy_weight": entropy_weights.values,
}).sort_values("entropy_weight", ascending=False)
entropy_weights_df.to_csv("entropy_weights.csv", index=False)

print("\n=== Top 10 entropy-weighted variables ===")
pd.set_option("display.max_colwidth", 65)
pd.set_option("display.width", 160)
print(entropy_weights_df.head(10).to_string(index=False))
print(f"\nSum of entropy weights: {entropy_weights_df['entropy_weight'].sum():.4f}")

# ============================================================
# Judgment weighting (Step 3b) -- pillar splits reconciled from user answers:
# Chronic 30%, Acute 30%, Infrastructure_Affordability 20%, Access 10%, Momentum 10%
# (the "Balanced" preset), each scaled by 0.95 to carve out 5% for a new Demographic
# pillar (sex ratio) that wasn't in the original 5-pillar preset -- reconciling the
# "include sex ratio as its own small pillar" answer with the balanced-split answer.
# Within each pillar, weight is split equally across that pillar's member variables.
# ============================================================
PILLAR_WEIGHTS = {
    "Chronic": 0.30 * 0.95,
    "Acute": 0.30 * 0.95,
    "Infrastructure_Affordability": 0.20 * 0.95,
    "Access": 0.10 * 0.95,
    "Momentum": 0.10 * 0.95,
    "Demographic": 0.05,
}
assert abs(sum(PILLAR_WEIGHTS.values()) - 1.0) < 1e-9

pillar_counts = pillar_map["pillar"].value_counts().to_dict()
judgment_weights_df = pillar_map.copy()
judgment_weights_df["pillar_weight"] = judgment_weights_df["pillar"].map(PILLAR_WEIGHTS)
judgment_weights_df["n_in_pillar"] = judgment_weights_df["pillar"].map(pillar_counts)
judgment_weights_df["judgment_weight"] = judgment_weights_df["pillar_weight"] / judgment_weights_df["n_in_pillar"]
judgment_weights_df = judgment_weights_df.drop(columns=["pillar_weight", "n_in_pillar"])
judgment_weights_df.to_csv("judgment_weights.csv", index=False)

print("\n=== Judgment weights by pillar ===")
print(judgment_weights_df.groupby("pillar")["judgment_weight"].sum().round(4).to_string())
print(f"\nSum of judgment weights: {judgment_weights_df['judgment_weight'].sum():.4f}")
