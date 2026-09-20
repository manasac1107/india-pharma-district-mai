import pandas as pd
import numpy as np

gdp = pd.read_excel("../state-wise-decadal-change-of-states-share-in-national-gdp.xlsx")

BIFURCATED_STATES = ["Andhra Pradesh", "Bihar", "Madhya Pradesh", "Uttar Pradesh"]

# drop the "Together"-notes row for these four states in EVERY fiscal_year, not just
# 2023-24: the same duplicate-row pattern (one 'Together' row, one blank-notes row with
# an identical gdp_share value) exists in every year in this file, confirmed with the
# user after finding pre-2023-24 years summed to 128-136% instead of ~100%.
is_combined_row = (
    gdp["state"].isin(BIFURCATED_STATES)
    & gdp["notes"].astype(str).str.contains("Together", case=False, na=False)
)
print(f"Dropping {is_combined_row.sum()} combined-notes row(s) across all years for {BIFURCATED_STATES}")
gdp_clean = gdp[~is_combined_row].copy()

# --- verify: gdp_share sums to ~100 (+/-5) per fiscal_year ---
sums = gdp_clean.groupby("fiscal_year")["gdp_share"].sum().sort_index()
print("\ngdp_share sum per fiscal_year (after fix):")
print(sums.to_string())
bad = sums[(sums < 95) | (sums > 105)]
if len(bad):
    print("\nWARNING: fiscal years outside 100 +/- 5:")
    print(bad)
else:
    print("\nAll fiscal years within 100 +/- 5 -- fix verified.")

# --- CAGR per state: earliest to latest available year ---
def year_start(fy):
    return int(str(fy).split("-")[0])

gdp_clean["year_start"] = gdp_clean["fiscal_year"].apply(year_start)

def compute_cagr(g):
    # anchor on the earliest year with a real (non-NaN) gdp_share, not the earliest row --
    # several states (Chhattisgarh, Jharkhand, Uttarakhand, Goa, Arunachal Pradesh, etc.)
    # didn't exist as separate states/UTs in 1960-61 and have NaN gdp_share for the years
    # before their formation, which silently produced a NaN CAGR for 13/33 states if the
    # first row was used blindly regardless of value.
    g = g.sort_values("year_start")
    g_real = g[g["gdp_share"].notna()]
    if len(g_real) < 2:
        first = g.iloc[0]
        last = g.iloc[-1]
        return pd.Series({
            "earliest_year": first["fiscal_year"], "earliest_gdp_share": first["gdp_share"],
            "latest_year": last["fiscal_year"], "latest_gdp_share": last["gdp_share"],
            "cagr_pct": np.nan,
        })
    first = g_real.iloc[0]
    last = g_real.iloc[-1]
    n_years = last["year_start"] - first["year_start"]
    if n_years <= 0 or first["gdp_share"] <= 0:
        return pd.Series({
            "earliest_year": first["fiscal_year"], "earliest_gdp_share": first["gdp_share"],
            "latest_year": last["fiscal_year"], "latest_gdp_share": last["gdp_share"],
            "cagr_pct": np.nan,
        })
    cagr = (last["gdp_share"] / first["gdp_share"]) ** (1 / n_years) - 1
    return pd.Series({
        "earliest_year": first["fiscal_year"], "earliest_gdp_share": first["gdp_share"],
        "latest_year": last["fiscal_year"], "latest_gdp_share": last["gdp_share"],
        "cagr_pct": round(cagr * 100, 3),
    })

trend = gdp_clean.groupby("state", group_keys=True).apply(compute_cagr, include_groups=False).reset_index()

out = gdp_clean.drop(columns=["year_start"]).merge(trend, on="state", how="left")
out.to_csv("clean_state_gdp_trend.csv", index=False)
print(f"\nSaved clean_state_gdp_trend.csv: {out.shape[0]} rows, {out.shape[1]} columns ({out['state'].nunique()} states)")
