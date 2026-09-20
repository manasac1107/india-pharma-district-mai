import pandas as pd

# ============================================================
# SPELLING_FIXES: same 2011 district, just a different transliteration/spacing in NFHS
# vs census -- full population used directly, NO apportionment.
# ============================================================
SPELLING_FIXES = {
    ("Andaman & Nicobar Island", "Nicobar"): "Nicobars",
    ("Andaman & Nicobar Island", "North & Middle Andaman"): "North  AND Middle Andaman",
    ("Dadra & Nagar Haveli", "Dadra & Nagar Haveli"): "Dadra AND Nagar Haveli",
    ("Gujarat", "Ahmedabad"): "Ahmadabad",
    ("Gujarat", "Banaskantha"): "Banas Kantha",
    ("Gujarat", "Sabarkantha"): "Sabar Kantha",
    ("Gujarat", "Dahod"): "Dohad",
    ("Gujarat", "Panchmahal"): "Panch Mahals",
    ("Himachal Pradesh", "Lahul and Spiti"): "Lahul AND Spiti",
    ("Maharashtra", "Ahmednagar"): "Ahmadnagar",
    ("Punjab", "Gurudaspur"): "Gurdaspur",
    ("Rajasthan", "Jhunjhunu"): "Jhunjhunun",
    ("Sikkim", "North District"): "North  District",
    ("Uttarakhand", "Haridwar"): "Hardwar",
    ("Uttarakhand", "Pauri Garhwal"): "Garhwal",
    ("West Bengal", "Darjeeling"): "Darjiling",
    ("Chhattisgarh", "Dantewada"): "Dakshin Bastar Dantewada",
    ("Puducherry", "Puducherry"): "PONDICHERRY",
    ("Uttar Pradesh", "Prayagraj"): "Allahabad",
    # Telangana direct renames -- same district as undivided Andhra Pradesh, state
    # renamed after 2014 bifurcation, no further district-level split
    ("Telangana", "Adilabad"): ("Andhra Pradesh", "Adilabad"),
    ("Telangana", "Hyderabad"): ("Andhra Pradesh", "Hyderabad"),
    ("Telangana", "Karimnagar"): ("Andhra Pradesh", "Karimnagar"),
    ("Telangana", "Khammam"): ("Andhra Pradesh", "Khammam"),
    ("Telangana", "Mahabubnagar"): ("Andhra Pradesh", "Mahabubnagar"),
    ("Telangana", "Medak"): ("Andhra Pradesh", "Medak"),
    ("Telangana", "Nalgonda"): ("Andhra Pradesh", "Nalgonda"),
    ("Telangana", "Nizamabad"): ("Andhra Pradesh", "Nizamabad"),
    ("Telangana", "Ranga Reddy"): ("Andhra Pradesh", "Ranga Reddy"),
}

# ============================================================
# PARENT_MAP: genuine post-2011 district splits requiring population apportionment.
# Source column notes whether reused from Step 1's null_with_optional_parent_mapping.csv
# (as instructed) or newly proposed for this census join (flagged for review).
# Telangana parents reused from Step 1 are looked up under Andhra Pradesh in census
# (Telangana didn't exist as a state in 2011).
# ============================================================
step1_hints = pd.read_csv("../files/null_with_optional_parent_mapping.csv")
PARENT_MAP = {}
for _, r in step1_hints.iterrows():
    state = r["nfhs_state"]
    census_state = "Andhra Pradesh" if state == "Telangana" else state
    PARENT_MAP[(state, r["nfhs_district"])] = {
        "parent_census_state": census_state,
        "parent_census_district": r["suggested_parent_hospital_dir_district"],
        "source": "Step1_reused",
    }

NEW_PARENTS = {
    ("Telangana", "Jagitial"): ("Andhra Pradesh", "Karimnagar"),
    ("Telangana", "Warangal Rural"): ("Andhra Pradesh", "Warangal"),
    ("Telangana", "Warangal Urban"): ("Andhra Pradesh", "Warangal"),
    ("Assam", "Biswanath"): ("Assam", "Sonitpur"),
    ("Assam", "Charaideo"): ("Assam", "Sivasagar"),
    ("Assam", "Hojai"): ("Assam", "Nagaon"),
    ("Assam", "Majuli"): ("Assam", "Jorhat"),
    ("Assam", "South Salmara Mancachar"): ("Assam", "Dhubri"),
    ("Assam", "West Karbi Anglong"): ("Assam", "Karbi Anglong"),
    ("Arunachal Pradesh", "Kra Daadi"): ("Arunachal Pradesh", "Kurung Kumey"),
    ("Arunachal Pradesh", "Longding"): ("Arunachal Pradesh", "Tirap"),
    ("Arunachal Pradesh", "Namsai"): ("Arunachal Pradesh", "Lohit"),
    ("Arunachal Pradesh", "Siang"): ("Arunachal Pradesh", "East Siang"),
    ("Chhattisgarh", "Mungeli"): ("Chhattisgarh", "Bilaspur"),
    ("Gujarat", "Morbi"): ("Gujarat", "Rajkot"),
    ("Meghalaya", "East Jaintia Hills"): ("Meghalaya", "Jaintia Hills"),
    ("Meghalaya", "West Jaintia Hills"): ("Meghalaya", "Jaintia Hills"),
    ("Meghalaya", "North Garo Hills"): ("Meghalaya", "West Garo Hills"),
    ("Meghalaya", "South West Garo Hills"): ("Meghalaya", "South Garo Hills"),
    ("Meghalaya", "South West Khasi Hills"): ("Meghalaya", "West Khasi Hills"),
    ("NCT of Delhi", "Shahdara"): ("NCT of Delhi", "East"),
    ("NCT of Delhi", "South East"): ("NCT of Delhi", "South"),
    ("Punjab", "Fazilka"): ("Punjab", "Firozpur"),
    ("Punjab", "Pathankot"): ("Punjab", "Gurdaspur"),
    ("Tripura", "Khowai"): ("Tripura", "West Tripura"),
    ("Tripura", "Sepahijala"): ("Tripura", "West Tripura"),
    ("Tripura", "Gomati"): ("Tripura", "South Tripura"),
    ("Tripura", "Unakoti"): ("Tripura", "North Tripura"),
    ("Uttar Pradesh", "Amethi"): ("Uttar Pradesh", "Sultanpur"),
    ("Uttar Pradesh", "Hapur"): ("Uttar Pradesh", "Ghaziabad"),
    ("Uttar Pradesh", "Sambhal"): ("Uttar Pradesh", "Moradabad"),
    ("Uttar Pradesh", "Shamli"): ("Uttar Pradesh", "Muzaffarnagar"),
    ("Maharashtra", "Palghar"): ("Maharashtra", "Thane"),
    ("West Bengal", "Paschim Barddhaman"): ("West Bengal", "Barddhaman"),
    ("West Bengal", "Purba Barddhaman"): ("West Bengal", "Barddhaman"),
}
for k, (cs, cd) in NEW_PARENTS.items():
    PARENT_MAP[k] = {"parent_census_state": cs, "parent_census_district": cd, "source": "newly_proposed"}

# ============================================================
# Unresolvable (no census state at all, post-dates census entirely): Ladakh
# ============================================================
NO_DATA = [("Ladakh", "Kargil"), ("Ladakh", "Leh (Ladakh)")]

pm_df = pd.DataFrame([
    {"nfhs_state": k[0], "nfhs_district": k[1], **v} for k, v in PARENT_MAP.items()
])
pm_df.to_csv("population_parent_mapping.csv", index=False)

grouping = pm_df.groupby(["parent_census_state", "parent_census_district", "source"]).agg(
    n_children=("nfhs_district", "count"),
    children=("nfhs_district", lambda s: ", ".join(s)),
).reset_index().sort_values(["parent_census_state", "parent_census_district"])
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
print(f"Total districts needing apportionment: {len(pm_df)}")
print(f"Total spelling-fix districts (no apportionment): {len(SPELLING_FIXES)}")
print(f"Total no-data (Ladakh): {len(NO_DATA)}")
print(f"\n=== Parent -> children groupings ({len(grouping)} parent groups) ===")
print(grouping.to_string(index=False))
