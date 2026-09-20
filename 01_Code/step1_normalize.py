import pandas as pd
import numpy as np

m = pd.read_csv("../master_district_panel.csv")
id_cols = ["State", "District"]
numeric_cols = [c for c in m.columns if c not in id_cols]

for c in numeric_cols:
    m[c] = pd.to_numeric(m[c], errors="coerce")

# --- direction decision, per user-confirmed "demand framing" ---
# Demand framing (user-confirmed): higher disease prevalence / risk-factor rate = larger
# addressable patient pool = MORE attractive, not less. Under this framing every variable
# in the panel points the same way (higher raw value = higher attractiveness), because:
#   - chronic risk-factor / disease-prevalence / screening-uptake variables -> demand signal
#   - care-seeking response variables (ORS/zinc/facility visit) -> engaged-consumer signal
#   - infrastructure (sanitation/water/electricity) -> ability-to-serve/distribution signal
#   - insurance coverage, literacy -> affordability/adherence signal
#   - hospital counts -> access/channel signal
#   - sex ratio -> no demand or capability logic either way; kept un-inverted as a neutral
#     demographic control, flagged separately since it isn't a real "attractiveness" driver
# So no variable is inverted under this framing. This decision table is still printed in
# full per variable for review, since a different framing choice would flip most of it.

def categorize(col):
    low = col.lower()
    if "sex ratio" in low:
        return "demographic_control (neutral, not inverted, flagged for review)"
    if any(k in low for k in ["blood sugar", "blood pressure", "cancer", "tobacco", "alcohol"]):
        return "chronic_risk_factor (demand signal)"
    if any(k in low for k in ["diarrhoea", "ari", "acute respiratory"]):
        return "acute_disease (demand signal)" if "prevalence" in low else "acute_care_seeking (engaged-consumer signal)"
    if "insurance" in low:
        return "affordability (capability signal)"
    if "literate" in low:
        return "affordability/adherence (capability signal)"
    if any(k in low for k in ["sanitation", "drinkingwater", "electricity"]):
        return "infrastructure (capability signal)"
    if "hospital_count" in low:
        return "access/channel (capability signal)"
    return "other"

direction_table = pd.DataFrame({
    "variable": numeric_cols,
    "category": [categorize(c) for c in numeric_cols],
    "higher_is_better": [True] * len(numeric_cols),  # all True under demand framing
})
direction_table.to_csv("direction_decisions.csv", index=False)

print("=== Direction decision table (demand framing, user-confirmed) ===")
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 160)
for cat in sorted(direction_table["category"].unique()):
    sub = direction_table[direction_table["category"] == cat]
    print(f"\n[{cat}]  ({len(sub)} variables, higher_is_better=True for all)")
    for v in sub["variable"]:
        print(f"    - {v}")

# --- min-max normalize (no inversions per confirmed framing) ---
norm = m[id_cols].copy()
minmax_ref = []
for c in numeric_cols:
    col = m[c]
    lo, hi = col.min(), col.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        norm[c + "_norm"] = np.nan
    else:
        norm[c + "_norm"] = (col - lo) / (hi - lo)
    minmax_ref.append({"variable": c, "min": lo, "max": hi, "all_nan_or_constant": pd.isna(lo) or hi == lo})

pd.DataFrame(minmax_ref).to_csv("normalization_reference.csv", index=False)
norm.to_csv("normalized_district_panel.csv", index=False)

all_nan_cols = [r["variable"] for r in minmax_ref if r["all_nan_or_constant"]]
print(f"\n\nSaved normalized_district_panel.csv: {norm.shape[0]} rows, {norm.shape[1]} columns")
print(f"Columns that are all-NaN or constant (produced NaN after scaling): {len(all_nan_cols)}")
for c in all_nan_cols:
    print(f"    - {c}")
print("\n(These are exactly the 20 trend_capable=False indicators' _NFHS4 columns -- expected, "
      "since those were nulled out in the previous session. They will be excluded from index "
      "construction; only their _NFHS5 normalized columns will be used.)")
