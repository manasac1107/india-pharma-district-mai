import pandas as pd

hmis = pd.read_excel("../key-health-management-performance-indicators-of-all-the-states-in-india.xlsx")
hmis["state"] = hmis["state"].astype(str).str.strip()

# --- splice OPD (Number) [pre-2017-18] and OPD (Allopathic) [2017-18 onward] into OPD_Total ---
opd = hmis[hmis["indicator"].isin(["OPD (Number)", "OPD (Allopathic)"])].copy()
overlap_years = set(opd[opd["indicator"] == "OPD (Number)"]["fiscal_year"]) & set(
    opd[opd["indicator"] == "OPD (Allopathic)"]["fiscal_year"]
)
print(f"OPD series overlap years (should be empty): {overlap_years}")
opd["indicator"] = "OPD_Total"
opd = opd.drop_duplicates(subset=["state", "fiscal_year"])  # safety, should be no-op given no overlap

# --- Children 0-5 raw case-count indicators (not "Percent...") ---
c05_mask = hmis["indicator"].str.contains("Children 0-5 Years of Age", na=False) & ~hmis[
    "indicator"
].str.startswith("Percent")
c05 = hmis[c05_mask].copy()
print(f"\nChildren 0-5 raw case-count indicators kept ({c05['indicator'].nunique()}):")
for i in sorted(c05["indicator"].unique()):
    print(f"    - {i}")
print(f"Years covered: {sorted(c05['fiscal_year'].unique())}")

keep = pd.concat([opd, c05], ignore_index=True)

wide = keep.pivot_table(
    index=["state", "fiscal_year"], columns="indicator", values="value", aggfunc="first"
).reset_index()

wide.to_csv("clean_state_hmis_panel.csv", index=False)
print(f"\nSaved clean_state_hmis_panel.csv: {wide.shape[0]} rows, {wide.shape[1]} columns")
print(f"OPD_Total non-null years: {sorted(wide.loc[wide['OPD_Total'].notna(), 'fiscal_year'].unique())}")
