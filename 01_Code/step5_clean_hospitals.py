import pandas as pd

hosp = pd.read_csv("../hospital_directory_deduped.csv", low_memory=False)
hosp["State"] = hosp["State"].astype(str).str.strip()
hosp["District"] = hosp["District"].astype(str).str.strip()

VALID_STATES = {
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli", "Daman and Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
}

bad_state_mask = ~hosp["State"].isin(VALID_STATES)
print(f"Rows with invalid State value: {bad_state_mask.sum()}")
print(hosp.loc[bad_state_mask, ["Hospital_Name", "Address_Original_First_Line", "State", "District", "Pincode"]].to_string())

# known issue: "North Twenty Four Parganas" leaked into State (should be West Bengal);
# address ("Dum Dum, North Dum Dum", pincode 700074) confirms the true district is
# North 24 Parganas -- fix both columns rather than dropping, since the correct values
# are recoverable from the address.
leaked_district_mask = hosp["State"] == "North Twenty Four Parganas"
hosp.loc[leaked_district_mask, "District"] = "North 24 Parganas"
hosp.loc[leaked_district_mask, "State"] = "West Bengal"

# re-check for any remaining invalid states and drop those (can't recover them)
still_bad = ~hosp["State"].isin(VALID_STATES)
print(f"\nRemaining invalid-State rows after fix: {still_bad.sum()}")
if still_bad.sum():
    print(hosp.loc[still_bad, ["Hospital_Name", "State", "District"]].to_string())
    hosp = hosp[~still_bad].copy()

# --- facility-count proxy: total rows per (State, District) ---
total_counts = hosp.groupby(["State", "District"]).size().rename("total_hospital_count")

# --- category split (ignore Hospital_Category == "0" placeholder) ---
cat = hosp[hosp["Hospital_Category"].isin(["Private", "Public/ Government"])]
cat_counts = (
    cat.groupby(["State", "District", "Hospital_Category"]).size().unstack(fill_value=0)
)
cat_counts = cat_counts.rename(columns={"Private": "private_hospital_count", "Public/ Government": "public_hospital_count"})

out = pd.concat([total_counts, cat_counts], axis=1).fillna(0).reset_index()
for col in ["total_hospital_count", "public_hospital_count", "private_hospital_count"]:
    if col not in out.columns:
        out[col] = 0
    out[col] = out[col].astype(int)

out = out[["State", "District", "total_hospital_count", "public_hospital_count", "private_hospital_count"]]
out.to_csv("clean_district_hospital_counts.csv", index=False)
print(f"\nSaved clean_district_hospital_counts.csv: {out.shape[0]} rows ({out['State'].nunique()} states/UTs)")
print(f"Total hospitals counted: {out['total_hospital_count'].sum()} (raw file had {len(hosp)} rows after invalid-state cleanup)")
