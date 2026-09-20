import pandas as pd
import numpy as np

BASE = ".."
FILES = "files"

cw = pd.read_csv(f"{BASE}/district_crosswalk.csv")

manual = pd.read_csv(f"{FILES}/manual_overrides.csv")
parent = pd.read_csv(f"{FILES}/null_with_optional_parent_mapping.csv")
pure_null = pd.read_csv(f"{FILES}/pure_null_list.csv")

cw = cw.set_index(["nfhs_state", "nfhs_district"])

# 1. Lower auto-accept threshold to 80 -- no data change needed, this only affects
#    which rows would be flagged for review; match_confidence values are untouched.

# 2. Manual overrides -> hospital_dir_state/district = given values, confidence = 100
manual_idx = manual.set_index(["nfhs_state", "nfhs_district"])
for key, row in manual_idx.iterrows():
    cw.loc[key, "hospital_dir_state"] = row["hospital_dir_state"]
    cw.loc[key, "hospital_dir_district"] = row["hospital_dir_district"]
    cw.loc[key, "match_confidence"] = 100.0

# 3. Parent-broadcast mapping -> hospital_dir_district = suggested parent, flag is_parent_broadcast
cw["is_parent_broadcast"] = False
parent_idx = parent.set_index(["nfhs_state", "nfhs_district"])
for key, row in parent_idx.iterrows():
    cw.loc[key, "hospital_dir_district"] = row["suggested_parent_hospital_dir_district"]
    cw.loc[key, "is_parent_broadcast"] = True

# 4. Pure null list -> hospital_dir_district = NaN, no force match
pure_null_idx = pure_null.set_index(["nfhs_state", "nfhs_district"])
for key in pure_null_idx.index:
    cw.loc[key, "hospital_dir_district"] = np.nan

cw = cw.reset_index()
cw.to_csv(f"{FILES}/../district_crosswalk_v2.csv", index=False)

total = len(cw)
populated = cw["hospital_dir_district"].notna().sum()
blank = cw["hospital_dir_district"].isna().sum()

print(f"Total rows: {total}")
print(f"hospital_dir_district populated: {populated}")
print(f"hospital_dir_district blank/NaN: {blank}")
print(f"is_parent_broadcast=True rows: {(cw['is_parent_broadcast']==True).sum()}")
print(f"manual_override rows applied: {len(manual_idx)}")
print(f"pure_null rows applied: {len(pure_null_idx)}")
