import pandas as pd
import numpy as np

norm = pd.read_csv("normalized_district_panel_full.csv")
trend_cap = pd.read_csv("../indicator_trend_capability.csv")
four = pd.read_csv("four_indices.csv")

trend_capable_indicators = trend_cap.loc[trend_cap["trend_capable"], "indicator"].tolist()
assert len(trend_capable_indicators) == 12

# ============================================================
# Momentum Index
# Formula: Momentum = 0.6 * GDP_CAGR_norm + 0.4 * Change_component_norm
#   - GDP_CAGR_norm: min-max normalized state GDP-share CAGR (broadcast to district) --
#     the primary leading indicator per the brief, given the dominant 0.6 weight.
#   - Change_component: mean, across the 12 trend_capable=True indicators, of
#     (NFHS5_norm - NFHS4_norm) per district -- i.e. average direction-adjusted
#     improvement from NFHS-4 to NFHS-5. This raw mean-change (naturally in [-1, 1]) is
#     then itself min-max rescaled to [0, 1] across districts ("Change_component_norm")
#     so it combines cleanly with GDP_CAGR_norm on a comparable scale.
#   - trend_capable=False indicators (cancer, blood sugar, BP, alcohol, tobacco, literacy)
#     are excluded from the change component entirely -- their NFHS4 values don't exist,
#     so no change score is computed or implied for them, per the standing constraint.
# ============================================================
gdp_cagr_norm_col = "gdp_cagr_pct_state_broadcast_norm"

change_cols = []
for ind in trend_capable_indicators:
    c4, c5 = f"{ind}_NFHS4_norm", f"{ind}_NFHS5_norm"
    assert c4 in norm.columns and c5 in norm.columns
    change_col = f"{ind}__change"
    norm[change_col] = norm[c5] - norm[c4]
    change_cols.append(change_col)

raw_change_mean = norm[change_cols].mean(axis=1, skipna=True)  # nanmean across the 12
lo, hi = raw_change_mean.min(), raw_change_mean.max()
change_component_norm = (raw_change_mean - lo) / (hi - lo)

gdp_cagr_norm = norm[gdp_cagr_norm_col]
gdp_cagr_norm_imputed = gdp_cagr_norm.fillna(gdp_cagr_norm.mean())
change_component_norm_imputed = change_component_norm.fillna(change_component_norm.mean())

print(f"Districts missing GDP CAGR (imputed with mean): {gdp_cagr_norm.isna().sum()}")
print(f"Districts missing raw_change_mean entirely (imputed with mean): {raw_change_mean.isna().sum()}")

momentum = 0.6 * gdp_cagr_norm_imputed + 0.4 * change_component_norm_imputed

four["Momentum"] = momentum.values
print(f"\nMomentum summary:\n{four['Momentum'].describe().to_string()}")

# ============================================================
# Future_MAI_3yr = Overall_MAI + 0.15 * (Momentum - mean(Momentum))
# Reasoning: a district's forward-looking score is nudged up or down from its current
# Overall MAI based on how far above/below the national-average Momentum it sits. The
# 0.15 scale factor caps the maximum plausible swing: Momentum is bounded in [0,1], so
# (Momentum - mean(Momentum)) is bounded roughly in [-1, 1], meaning the adjustment can
# shift Future_MAI_3yr by at most ~+/-0.15 -- a meaningful but not dominant nudge relative
# to typical Overall_MAI values (~0.04-0.50 range), so momentum can re-rank close
# competitors but can't singlehandedly invert the base ranking of very different districts.
# ============================================================
mean_momentum = four["Momentum"].mean()
ADJUSTMENT_SCALE = 0.15
four["Future_MAI_3yr"] = four["Overall_MAI_entropy"] + ADJUSTMENT_SCALE * (four["Momentum"] - mean_momentum)

four["Rank_Current"] = four["Overall_MAI_entropy"].rank(ascending=False, method="min").astype(int)
four["Rank_Future"] = four["Future_MAI_3yr"].rank(ascending=False, method="min").astype(int)
# positive Rank_Delta = climbing (moving toward rank 1, i.e. Future rank number is smaller)
four["Rank_Delta"] = four["Rank_Current"] - four["Rank_Future"]

four.to_csv("four_indices_with_momentum.csv", index=False)

print(f"\nFormula used: Future_MAI_3yr = Overall_MAI_entropy + {ADJUSTMENT_SCALE} * (Momentum - mean(Momentum))")
print(f"mean(Momentum) = {mean_momentum:.4f}")

print("\nTop 10 rising stars (largest positive Rank_Delta):")
pd.set_option("display.width", 160)
print(four.sort_values("Rank_Delta", ascending=False).head(10)[
    ["State", "District", "Rank_Current", "Rank_Future", "Rank_Delta", "Momentum"]
].to_string(index=False))

print("\nTop 10 fading (largest negative Rank_Delta):")
print(four.sort_values("Rank_Delta").head(10)[
    ["State", "District", "Rank_Current", "Rank_Future", "Rank_Delta", "Momentum"]
].to_string(index=False))
