import pandas as pd
import numpy as np

census = pd.read_csv("../../01_raw_data/census_2011_district_population.csv")
census["State name"] = census["State name"].str.strip().str.title()
STATE_FIX = {"Orissa":"Odisha","Pondicherry":"Puducherry","Nct Of Delhi":"NCT of Delhi",
             "Andaman And Nicobar Islands":"Andaman & Nicobar Island","Jammu And Kashmir":"Jammu & Kashmir",
             "Dadra And Nagar Haveli":"Dadra & Nagar Haveli","Daman And Diu":"Daman & Diu"}
census["State name"] = census["State name"].replace(STATE_FIX)
census_pop = census.set_index(["State name", "District name"])["Population"]

m = pd.read_csv("../master_district_panel.csv")
nfhs_pairs = m[["State", "District"]].drop_duplicates().reset_index(drop=True)
hosp_lookup = m.set_index(["State", "District"])["total_hospital_count"]

sf = pd.read_csv("population_spelling_fixes.csv")
pm = pd.read_csv("population_parent_mapping.csv")

result = nfhs_pairs.copy()
result["Population_2011"] = np.nan
result["is_population_apportioned"] = False
result["population_apportion_method"] = None

# --- direct exact matches ---
for i, r in result.iterrows():
    key = (r["State"], r["District"])
    if key in census_pop.index:
        result.loc[i, "Population_2011"] = census_pop[key]

# --- spelling fixes: full population, no split ---
for _, r in sf.iterrows():
    key = (r["nfhs_state"], r["nfhs_district"])
    census_key = (r["census_state"], r["census_district"])
    mask = (result["State"] == key[0]) & (result["District"] == key[1])
    result.loc[mask, "Population_2011"] = census_pop.get(census_key, np.nan)

# --- apportioned (parent-split) districts: hospital-count-weighted, equal-split fallback ---
apportion_log = []
for (pstate, pdist), grp in pm.groupby(["parent_census_state", "parent_census_district"]):
    parent_pop = census_pop.get((pstate, pdist), np.nan)
    children = list(zip(grp["nfhs_state"], grp["nfhs_district"]))
    hosp_counts = [hosp_lookup.get(c, np.nan) for c in children]
    hosp_counts_filled = [0 if (pd.isna(h)) else h for h in hosp_counts]
    total_hosp = sum(hosp_counts_filled)

    if total_hosp > 0:
        weights = [h / total_hosp for h in hosp_counts_filled]
        method = "hospital_weighted"
    else:
        weights = [1 / len(children)] * len(children)
        method = "equal_split"

    for (cstate, cdist), w, h in zip(children, weights, hosp_counts):
        child_pop = parent_pop * w if pd.notna(parent_pop) else np.nan
        mask = (result["State"] == cstate) & (result["District"] == cdist)
        result.loc[mask, "Population_2011"] = child_pop
        result.loc[mask, "is_population_apportioned"] = True
        result.loc[mask, "population_apportion_method"] = method
        apportion_log.append({
            "parent_state": pstate, "parent_district": pdist, "parent_population": parent_pop,
            "child_state": cstate, "child_district": cdist, "child_hospital_count": h,
            "weight": round(w, 4), "method": method, "apportioned_population": round(child_pop, 0) if pd.notna(child_pop) else None,
        })

log_df = pd.DataFrame(apportion_log)
log_df.to_csv("population_apportionment_detail.csv", index=False)

n_hosp_weighted = (log_df["method"] == "hospital_weighted").sum()
n_equal = (log_df["method"] == "equal_split").sum()
print(f"Apportioned districts: {len(log_df)} total")
print(f"  -- hospital-count-weighted: {n_hosp_weighted}")
print(f"  -- equal-split fallback (no hospital data in group): {n_equal}")

n_groups_hosp = log_df.groupby(["parent_state","parent_district"])["method"].first()
print(f"\nParent groups using hospital-weighting: {(n_groups_hosp=='hospital_weighted').sum()} / {len(n_groups_hosp)}")

result.to_csv("master_district_panel_with_population.csv", index=False)
print(f"\nSaved master_district_panel_with_population.csv: {result.shape}")
print(f"Population_2011 non-null: {result['Population_2011'].notna().sum()} / {len(result)}")
print(f"Districts with no population data at all (Ladakh): {result['Population_2011'].isna().sum()}")

# sanity print for Telangana Warangal group (5 children, largest group)
print("\nSanity check -- Warangal group apportionment:")
print(log_df[log_df["parent_district"]=="Warangal"][["child_district","child_hospital_count","weight","apportioned_population"]].to_string(index=False))
