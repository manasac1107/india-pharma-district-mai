import pandas as pd

hosp = pd.read_csv("../hospital_directory.csv", low_memory=False)
hosp["State"] = hosp["State"].astype(str).str.strip()
hosp["District"] = hosp["District"].astype(str).str.strip()

# canonical spelling <- variant spelling, scoped by state to avoid cross-state collisions
RENAMES = {
    ("Jharkhand", "khunti"): "Khunti",
    ("Bihar", "Muzzffarpur"): "Muzaffarpur",
    ("West Bengal", "24 Pargs (N)"): "North 24 Parganas",
}

before_counts = {}
for (st, variant), canonical in RENAMES.items():
    before_counts[(st, variant)] = len(hosp[(hosp["State"] == st) & (hosp["District"] == variant)])
    mask = (hosp["State"] == st) & (hosp["District"] == variant)
    hosp.loc[mask, "District"] = canonical

hosp.to_csv("../hospital_directory_deduped.csv", index=False)

print("Duplicate-district merge summary:")
for (st, variant), canonical in RENAMES.items():
    n = before_counts[(st, variant)]
    total_after = len(hosp[(hosp["State"] == st) & (hosp["District"] == canonical)])
    print(f"  {st}: '{variant}' ({n} rows) merged into '{canonical}' -> {total_after} total rows now under canonical name")
print(f"\nSaved: hospital_directory_deduped.csv ({len(hosp)} rows, unchanged row count -- renamed District values only)")
