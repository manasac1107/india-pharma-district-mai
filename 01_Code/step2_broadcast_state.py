import pandas as pd
import numpy as np

m = pd.read_csv("../master_district_panel.csv")
gdp = pd.read_csv("../clean_state_gdp_trend.csv")
hmis = pd.read_csv("../clean_state_hmis_panel.csv")

# district_crosswalk_v2.csv only maps NFHS state/district names to hospital_directory
# names -- it has no GDP-state or HMIS-state columns, so it can't directly resolve those
# joins. Built a small explicit state-name mapping instead (NFHS naming -> GDP/HMIS
# naming), the same kind of manual correction the crosswalk itself relies on for its
# low-confidence rows. Confirmed by direct set comparison, not fuzzy-matched, since the
# mismatches are exact known spelling variants (& vs and, Delhi vs NCT of Delhi, etc.)
NFHS_TO_OTHER = {
    "Andaman & Nicobar Island": "Andaman and Nicobar Islands",
    "NCT of Delhi": "Delhi",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "Daman & Diu": "Daman and Diu",
}

m["_state_join"] = m["State"].replace(NFHS_TO_OTHER)

# --- GDP CAGR broadcast ---
gdp_lookup = gdp[["state", "cagr_pct"]].drop_duplicates("state").rename(
    columns={"cagr_pct": "gdp_cagr_pct_state_broadcast"}
)
m = m.merge(gdp_lookup, left_on="_state_join", right_on="state", how="left").drop(columns=["state"])

no_gdp = sorted(m.loc[m["gdp_cagr_pct_state_broadcast"].isna(), "State"].unique())
print(f"States with no GDP CAGR match ({len(no_gdp)}): {no_gdp}")
print("  -> Ladakh/Lakshadweep/Dadra & Nagar Haveli/Daman & Diu are genuinely absent from")
print("     the GDP file itself (33 states/UTs only) -- not a naming problem, a real gap.")
print("  -> Sikkim has a real 0.0 gdp_share in its earliest available year (1980-81), so a")
print("     CAGR is mathematically undefined (can't compute a growth rate from a zero base).")

# --- HMIS broadcast: most recent year (2019-20) OPD_Total + childhood acute-disease indicators ---
LATEST_YEAR = "2019-20"
hmis_latest = hmis[hmis["fiscal_year"] == LATEST_YEAR].drop(columns=["fiscal_year"])
hmis_cols = [c for c in hmis_latest.columns if c != "state"]
hmis_latest = hmis_latest.rename(columns={c: f"{c}_state_broadcast" for c in hmis_cols})

m = m.merge(hmis_latest, left_on="_state_join", right_on="state", how="left").drop(columns=["state"])

no_hmis = sorted(m.loc[m[f"OPD_Total_state_broadcast"].isna(), "State"].unique())
print(f"\nStates with no HMIS {LATEST_YEAR} match ({len(no_hmis)}): {no_hmis}")
print("  -> Ladakh is genuinely absent from the HMIS file (post-dates the J&K/Ladakh split).")

m = m.drop(columns=["_state_join"])
m.to_csv("master_district_panel_broadcast.csv", index=False)

broadcast_cols = [c for c in m.columns if c.endswith("_state_broadcast")]
print(f"\nSaved master_district_panel_broadcast.csv: {m.shape[0]} rows, {m.shape[1]} columns")
print(f"{len(broadcast_cols)} state-broadcast columns added:")
for c in broadcast_cols:
    print(f"    - {c}")
