import pandas as pd

health = pd.read_csv("clean_district_health_panel.csv")
hosp_counts = pd.read_csv("clean_district_hospital_counts.csv")
crosswalk = pd.read_csv("district_crosswalk_v2.csv")

print(f"health panel: {health.shape}")
print(f"hospital counts: {hosp_counts.shape}")
print(f"crosswalk: {crosswalk.shape}")

# attach hospital_dir_state/district to each NFHS (State, District) via crosswalk
base = health.merge(
    crosswalk[["nfhs_state", "nfhs_district", "hospital_dir_state", "hospital_dir_district"]],
    left_on=["State", "District"],
    right_on=["nfhs_state", "nfhs_district"],
    how="left",
)

merged = base.merge(
    hosp_counts,
    left_on=["hospital_dir_state", "hospital_dir_district"],
    right_on=["State", "District"],
    how="left",
    suffixes=("", "_hospdir"),
)

merged = merged.drop(columns=["nfhs_state", "nfhs_district", "hospital_dir_state", "hospital_dir_district", "State_hospdir", "District_hospdir"])

hospital_cols = ["total_hospital_count", "public_hospital_count", "private_hospital_count"]
health_cols = [c for c in health.columns if c not in ("State", "District")]

has_health = merged[health_cols].notna().any(axis=1)
has_hosp = merged[hospital_cols].notna().all(axis=1)

n_complete = (has_health & has_hosp).sum()
n_partial = len(merged) - n_complete

print(f"\nDistricts with complete data (health + hospital counts): {n_complete}")
print(f"Districts with partial data (missing one or both sources): {n_partial}")
print(f"  -- of the partial ones, missing hospital counts entirely: {(has_health & ~has_hosp).sum()}")
print(f"  -- of the partial ones, missing all health indicators: {(~has_health & has_hosp).sum()}")

merged.to_csv("master_district_panel.csv", index=False)
print(f"\nSaved master_district_panel.csv: {merged.shape[0]} rows, {merged.shape[1]} columns")
