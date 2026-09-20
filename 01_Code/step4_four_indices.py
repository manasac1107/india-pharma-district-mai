import pandas as pd
import numpy as np
from scipy.stats import pearsonr

norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]
pillar_map = pd.read_csv("pillar_mapping.csv").set_index("variable")["pillar"]
overall = pd.read_csv("overall_mai_entropy_vs_judgment.csv")

SCORING_VARS = entropy_w.index.tolist()
X = norm[SCORING_VARS].copy()
X_imputed = X.fillna(X.mean())

INSURANCE_VAR = "Households with any usual member covered under a health insurance/financing scheme (%)_NFHS5_norm"
assert INSURANCE_VAR in SCORING_VARS

def reweight(base_weights, boosts, renormalize=True):
    """boosts: dict of {variable_or_pillar_match_fn: multiplier}. Applies multipliers to
    base_weights (entropy weights), then renormalizes to sum to 1."""
    w = base_weights.copy()
    for match_fn, mult in boosts:
        mask = w.index.map(match_fn)
        w.loc[mask] = w.loc[mask] * mult
    if renormalize:
        w = w / w.sum()
    return w

# ============================================================
# Chronic MAI: upweight Chronic pillar, insurance, GDP momentum; downweight Acute pillar
# ============================================================
chronic_boosts = [
    (lambda v: pillar_map[v] == "Chronic", 3.0),
    (lambda v: v == INSURANCE_VAR, 3.0),
    (lambda v: pillar_map[v] == "Momentum", 3.0),
    (lambda v: pillar_map[v] == "Acute", 0.3),
]
chronic_weights = reweight(entropy_w, chronic_boosts)
chronic_weights.to_csv("chronic_mai_weights.csv", header=["weight"])

overall["Chronic_MAI"] = X_imputed[SCORING_VARS].values @ chronic_weights[SCORING_VARS].values

# ============================================================
# Acute MAI: upweight Acute pillar, hospital count (Access); downweight GDP momentum.
# NOTE: no population-size variable exists anywhere in the source data (flagged --
# hospital_directory has no population field, and the only India-wide population file
# was excluded from this pipeline as India-wide-only with no district variation), so
# "population size" from the Step 4 instructions cannot literally be included. Hospital
# count is used as the sole Access-side upweight.
# ============================================================
acute_boosts = [
    (lambda v: pillar_map[v] == "Acute", 3.0),
    (lambda v: pillar_map[v] == "Access", 3.0),
    (lambda v: pillar_map[v] == "Momentum", 0.3),
]
acute_weights = reweight(entropy_w, acute_boosts)
acute_weights.to_csv("acute_mai_weights.csv", header=["weight"])

overall["Acute_MAI"] = X_imputed[SCORING_VARS].values @ acute_weights[SCORING_VARS].values

# ============================================================
# Chronic+Acute Overlap MAI
# ============================================================
r, p = pearsonr(overall["Chronic_MAI"], overall["Acute_MAI"])
print(f"Correlation between Chronic_MAI and Acute_MAI: r = {r:.3f}, p = {p:.2e}")

# Honesty check on the "TB/respiratory HMIS indicator" alternative construction: these
# variables exist, but are STATE-level broadcasts (13 unique state values repeated across
# ~20 districts each), not district-differentiated -- an index built only from these would
# have essentially no within-state discriminating power, i.e. genuinely data-thin.
tb_resp_vars = [c for c in norm.columns if any(k in c for k in ["Tuberculosis", "Asthma", "Pneumonia"])]
print(f"\nTB/respiratory-relevant HMIS broadcast variables available: {len(tb_resp_vars)}")
for v in tb_resp_vars:
    print(f"    - {v}")
print("These are state-level broadcasts (same value repeated across every district in a "
      "state) -- using them alone as the Overlap index would be data-thin (no district-level "
      "variation within a state). Using them as a pure standalone index is NOT recommended.")

# Primary construction: geometric mean of Chronic_MAI and Acute_MAI. Both components are
# already bounded in [0,1] (weighted average of [0,1]-scaled variables, weights sum to 1),
# so the geometric mean is also in [0,1], and rewards districts that are strong on BOTH
# dimensions rather than lopsided on one -- a district scoring 0.9/0.1 scores lower than
# one scoring 0.5/0.5, which is the intended "overlap" behavior.
overall["Chronic_Acute_Overlap_MAI"] = np.sqrt(
    overall["Chronic_MAI"].clip(lower=0) * overall["Acute_MAI"].clip(lower=0)
)

overall.to_csv("four_indices.csv", index=False)
print(f"\nSaved four_indices.csv: {overall.shape}")
print(overall[["Overall_MAI_entropy", "Chronic_MAI", "Acute_MAI", "Chronic_Acute_Overlap_MAI"]].describe().to_string())
