import pandas as pd
import re

india = pd.read_csv("../India.csv", low_memory=False)

# --- known fix: trailing whitespace on State/District ---
india["State"] = india["State"].astype(str).str.strip()
india["District"] = india["District"].astype(str).str.strip()

# --- drop J&K placeholder row ---
before = len(india)
india = india[~india["District"].str.contains("Data Not Available", case=False, na=False)].copy()
print(f"Dropped {before - len(india)} 'Data Not Available' row(s)")

# --- category keyword matching (case-insensitive substring on Indicator name) ---
CATEGORIES = {
    "chronic_blood_sugar_hypertension": ["blood sugar", "hypertension", "blood pressure"],
    "cancer": ["cancer"],
    "tobacco_alcohol": ["tobacco", "alcohol"],
    "acute_disease": ["diarrhoea", "diarrhea", "acute respiratory", "ari"],
    "health_insurance": ["health insurance"],
    # "literacy" keyword literally matches nothing (source spells it "literate");
    # user confirmed to include the literal literacy indicator by name instead.
    "literacy": ["literacy", "who are literate"],
    "sex_ratio": ["sex ratio"],
    # "drinking water" keyword literally matches nothing (source spells it "drinkingwater",
    # no space); user confirmed to include it under infrastructure.
    "infrastructure": ["electricity", "sanitation", "drinking water", "drinkingwater"],
}

all_indicators = sorted(india["Indicator"].unique())
matched_indicators = {}
selected = set()

print("\n=== Indicator matches per category (literal substring match on keywords given) ===")
for cat, keywords in CATEGORIES.items():
    matches = []
    for ind in all_indicators:
        low = ind.lower()
        if any(kw.lower() in low for kw in keywords):
            matches.append(ind)
    matched_indicators[cat] = matches
    selected.update(matches)
    print(f"\n[{cat}]  keywords={keywords}  ({len(matches)} matched)")
    for m in matches:
        print(f"    - {m}")

print(f"\nTotal unique indicators selected: {len(selected)} / {len(all_indicators)}")

# --- pivot long -> wide ---
sub = india[india["Indicator"].isin(selected)].copy()
wide = sub.pivot_table(
    index=["State", "District"],
    columns="Indicator",
    values=["NFHS 4", "NFHS 5"],
    aggfunc="first",
)
wide.columns = [f"{ind}_{'NFHS4' if src == 'NFHS 4' else 'NFHS5'}" for src, ind in wide.columns]
wide = wide.reset_index()

wide.to_csv("clean_district_health_panel.csv", index=False)
print(f"\nSaved clean_district_health_panel.csv: {wide.shape[0]} rows (districts), {wide.shape[1]} columns")
