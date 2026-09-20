import pandas as pd
import numpy as np

SINGLE_TIMEPOINT_INDICATORS = [
    "Ever undergone a breast examination for breast cancer (%)",
    "Ever undergone a screening test for cervical cancer (%)",
    "Ever undergone an oral cavity examination for oral cancer (%)",
    "Female Blood sugar level  high (141-160 mg/dl) (%)",
    "Female Blood sugar level  high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%)",
    "Female Blood sugar level  very high (>160 mg/dl) (%)",
    "MaleBlood sugar level  high (141-160 mg/dl) (%)",
    "Male Blood sugar level  high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%)",
    "Male Blood sugar level  very high (>160 mg/dl) (%)",
    "Female Elevated blood pressure or taking medicine to control blood pressure (%)",
    "Female Mildly elevated blood pressure (Systolic 140-159 mm of Hg and/or Diastolic 90-99 mm of Hg) (%)",
    "Female Moderately or severely elevated blood pressure (%)",
    "Male Elevated blood pressure or taking medicine to control blood pressure (%)",
    "Male Mildly elevated blood pressure (Systolic 140-159 mm of Hg and/or Diastolic 90-99 mm of Hg) (%)",
    "Male Moderately or severely elevated blood pressure (%)",
    "Men age 15 years and above who consume alcohol (%)",
    "Women age 15 years and above who consume alcohol (%)",
    "Men age 15 years and above who use any kind of tobacco (%)",
    "Women age 15 years and above who use any kind of tobacco (%)",
    "Women who are literate (%)",
]

panel = pd.read_csv("clean_district_health_panel.csv")
master = pd.read_csv("master_district_panel.csv")

base_indicators = sorted(set(c[:-6] for c in panel.columns if c.endswith("_NFHS4")))
assert set(SINGLE_TIMEPOINT_INDICATORS).issubset(set(base_indicators)), "mismatch vs panel columns"
trend_capable_indicators = [i for i in base_indicators if i not in SINGLE_TIMEPOINT_INDICATORS]
assert len(SINGLE_TIMEPOINT_INDICATORS) == 20 and len(trend_capable_indicators) == 12

# --- indicator-level metadata file ---
meta = pd.DataFrame(
    {"indicator": base_indicators}
).assign(trend_capable=lambda d: ~d["indicator"].isin(SINGLE_TIMEPOINT_INDICATORS))
meta = meta.sort_values(["trend_capable", "indicator"], ascending=[False, True])
meta.to_csv("indicator_trend_capability.csv", index=False)
print(f"Saved indicator_trend_capability.csv: {len(meta)} indicators ({meta['trend_capable'].sum()} trend_capable=True, {(~meta['trend_capable']).sum()} False)")

# --- null out NFHS4 columns for single-timepoint indicators in both wide files ---
def null_out(df, label):
    nfhs4_cols_to_null = [f"{i}_NFHS4" for i in SINGLE_TIMEPOINT_INDICATORS if f"{i}_NFHS4" in df.columns]
    before = df[nfhs4_cols_to_null].notna().sum().sum()
    df[nfhs4_cols_to_null] = np.nan
    after = df[nfhs4_cols_to_null].notna().sum().sum()
    print(f"\n{label}: nulled {len(nfhs4_cols_to_null)} NFHS4 columns")
    print(f"  non-null NFHS4 values in those columns: {before} -> {after}")
    return df

panel = null_out(panel, "clean_district_health_panel.csv")
master = null_out(master, "master_district_panel.csv")

panel.to_csv("clean_district_health_panel.csv", index=False)
master.to_csv("master_district_panel.csv", index=False)
print("\nSaved updated clean_district_health_panel.csv and master_district_panel.csv")
