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
# master_district_panel.csv already has Population_2011 etc merged in from the previous
# step -- drop those stale columns and rebuild cleanly to avoid confusion.
m = m.drop(columns=["Population_2011", "is_population_apportioned", "population_apportion_method"])
nfhs_pairs = m[["State", "District"]].drop_duplicates().reset_index(drop=True)
hosp_lookup = m.set_index(["State", "District"])["total_hospital_count"]

sf = pd.read_csv("population_spelling_fixes.csv")
pm = pd.read_csv("population_parent_mapping.csv")
dup_groups = pd.read_csv("double_counted_parent_groups.csv")
dup_group_keys = set(zip(dup_groups["parent_census_state"], dup_groups["parent_census_district"]))
dup_parent_nfhs_key = {
    (r["parent_census_state"], r["parent_census_district"]): eval(r["nfhs_parent_key"])
    for _, r in dup_groups.iterrows()
}

result = nfhs_pairs.copy()
result["Population_2011"] = np.nan
result["is_population_apportioned"] = False
result["population_apportion_method"] = None

# --- direct exact matches (fill first; will be overwritten below for the 48 double-counted parents) ---
for i, r in result.iterrows():
    key = (r["State"], r["District"])
    if key in census_pop.index:
        result.loc[i, "Population_2011"] = census_pop[key]

# --- spelling fixes: full population, no split (also overwritten below where flagged as double-counted) ---
for _, r in sf.iterrows():
    key = (r["nfhs_state"], r["nfhs_district"])
    census_key = (r["census_state"], r["census_district"])
    mask = (result["State"] == key[0]) & (result["District"] == key[1])
    result.loc[mask, "Population_2011"] = census_pop.get(census_key, np.nan)

# --- apportioned (parent-split) districts, CORRECTED: where the parent-namesake district
# also survives as a present-day NFHS district (48 of 51 groups), it is added into the
# weighted-split pool itself (using its OWN hospital count) instead of separately
# receiving the full undivided 2011 population. This eliminates the double-count the user
# flagged -- e.g. Telangana's present-day Karimnagar previously got the FULL undivided
# Karimnagar population AND its split-off children (Rajanna Sircilla, Peddapalli,
# Jagitial) also each got a share of that same total.
# ============================================================
apportion_log = []
for (pstate, pdist), grp in pm.groupby(["parent_census_state", "parent_census_district"]):
    parent_pop = census_pop.get((pstate, pdist), np.nan)
    children = list(zip(grp["nfhs_state"], grp["nfhs_district"]))

    is_double_counted = (pstate, pdist) in dup_group_keys
    if is_double_counted:
        parent_nfhs_key = dup_parent_nfhs_key[(pstate, pdist)]
        members = children + [parent_nfhs_key]
    else:
        members = children

    hosp_counts = [hosp_lookup.get(mem, np.nan) for mem in members]
    # only use hospital-weighting if EVERY member has real (non-NaN) hospital data --
    # treating a missing crosswalk match as "0 activity" silently zeroed out real
    # population for Tripura's Gomati/Khowai/Sepahijala/Unakoti (all NaN hospital count,
    # while their parent-namesake district had real data and absorbed 100% of the
    # weight). Falling back to equal-split for the WHOLE group avoids ever assigning a
    # literal zero population to a district that has real people, just missing hospital
    # data.
    all_have_data = all(pd.notna(h) for h in hosp_counts)
    total_hosp = sum(h for h in hosp_counts if pd.notna(h))

    if all_have_data and total_hosp > 0:
        weights = [h / total_hosp for h in hosp_counts]
        # sanity bound: hospital density and population share are not always
        # proportional (e.g. Thane retained 725 hospitals vs Palghar's 1 after their
        # 2014 split, but Palghar did NOT retain only 0.14% of undivided Thane's real
        # population -- rural/tribal areas are systematically hospital-sparse relative
        # to their population). If hospital-weighting would give any member less than
        # 10%/n of the total, that ratio is implausible -- fall back to equal-split for
        # the whole group instead, per user confirmation.
        floor = 0.10 / len(members)
        if min(weights) < floor:
            weights = [1 / len(members)] * len(members)
            method = "equal_split_implausible_ratio"
        else:
            method = "hospital_weighted"
    else:
        weights = [1 / len(members)] * len(members)
        method = "equal_split"

    for mem, w, h in zip(members, weights, hosp_counts):
        mem_pop = parent_pop * w if pd.notna(parent_pop) else np.nan
        mask = (result["State"] == mem[0]) & (result["District"] == mem[1])
        result.loc[mask, "Population_2011"] = mem_pop
        result.loc[mask, "is_population_apportioned"] = True
        result.loc[mask, "population_apportion_method"] = method
        apportion_log.append({
            "parent_state": pstate, "parent_district": pdist, "parent_population": parent_pop,
            "member_state": mem[0], "member_district": mem[1],
            "is_parent_namesake": is_double_counted and mem == parent_nfhs_key,
            "member_hospital_count": h, "weight": round(w, 4),
            "method": method, "apportioned_population": round(mem_pop, 0) if pd.notna(mem_pop) else None,
        })

log_df = pd.DataFrame(apportion_log)
log_df.to_csv("population_apportionment_detail.csv", index=False)

result.to_csv("master_district_panel_with_population.csv", index=False)

total_pop = result["Population_2011"].sum()
print(f"Corrected sum of Population_2011: {total_pop:,.0f}")
print(f"Actual India 2011 census population: 1,210,854,977")
print(f"Difference: {total_pop - 1210854977:,.0f} ({(total_pop/1210854977-1)*100:.3f}%)")
print(f"\nNon-null districts: {result['Population_2011'].notna().sum()} / {len(result)}")
print(f"Apportioned districts (incl. reclassified parent-namesakes): {result['is_population_apportioned'].sum()}")

print("\nSanity check -- Karimnagar group (now includes the parent-namesake as a weighted member):")
print(log_df[log_df["parent_district"]=="Karimnagar"][["member_district","is_parent_namesake","member_hospital_count","weight","apportioned_population"]].to_string(index=False))
