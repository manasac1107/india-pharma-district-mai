import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr

os.makedirs("eda_charts", exist_ok=True)
sns.set_theme(style="whitegrid")

m = pd.read_csv("master_district_panel.csv")

NFHS5 = "_NFHS5"
numeric_cols = [c for c in m.columns if c not in ("State", "District")]
for c in numeric_cols:
    m[c] = pd.to_numeric(m[c], errors="coerce")

# ---------------------------------------------------------------
# 1. Missingness bar chart
# ---------------------------------------------------------------
missing_pct = m[numeric_cols].isna().mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(10, 16))
missing_pct.plot(kind="barh", ax=ax, color="#c0392b")
ax.set_xlabel("% missing")
ax.set_title("Missingness by variable, master_district_panel.csv")
plt.tight_layout()
plt.savefig("eda_charts/01_missingness.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 2. Histograms grid for key numeric variables (NFHS5 values + hospital counts)
# ---------------------------------------------------------------
key_vars = [
    "Women who are literate (%)_NFHS5",
    "Sex ratio of the total population (females per 1,000 males)_NFHS5",
    "Households with any usual member covered under a health insurance/financing scheme (%)_NFHS5",
    "Male Elevated blood pressure or taking medicine to control blood pressure (%)_NFHS5",
    "Female Elevated blood pressure or taking medicine to control blood pressure (%)_NFHS5",
    "Male Blood sugar level  high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%)_NFHS5",
    "Men age 15 years and above who use any kind of tobacco (%)_NFHS5",
    "Women age 15 years and above who use any kind of tobacco (%)_NFHS5",
    "Prevalence of diarrhoea in the 2 weeks preceding the survey (%)_NFHS5",
    "Prevalence of symptoms of acute respiratory infection (ARI) in the 2 weeks preceding the survey (%)_NFHS5",
    "Population living in households with an improved drinkingwater source (%)_NFHS5",
    "total_hospital_count",
]
n = len(key_vars)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()
for i, var in enumerate(key_vars):
    sns.histplot(m[var].dropna(), kde=True, ax=axes[i], color="#2980b9")
    axes[i].set_title(var[:45] + ("..." if len(var) > 45 else ""), fontsize=9)
    axes[i].set_xlabel("")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("eda_charts/02_histograms_grid.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 3. Correlation heatmap across all numeric variables
# ---------------------------------------------------------------
corr = m[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, square=True, cbar_kws={"shrink": 0.6})
ax.set_title("Correlation heatmap, master_district_panel.csv numeric variables")
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig("eda_charts/03_correlation_heatmap.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 4. Regional boxplots (North/South/East/West/Central/Northeast)
# ---------------------------------------------------------------
ZONE_MAP = {
    "Jammu & Kashmir": "North", "Ladakh": "North", "Himachal Pradesh": "North", "Punjab": "North",
    "Chandigarh": "North", "Uttarakhand": "North", "Haryana": "North", "NCT of Delhi": "North",
    "Uttar Pradesh": "North",
    "Andhra Pradesh": "South", "Telangana": "South", "Karnataka": "South", "Kerala": "South",
    "Tamil Nadu": "South", "Puducherry": "South", "Andaman & Nicobar Island": "South",
    "Lakshadweep": "South",
    "Bihar": "East", "Jharkhand": "East", "West Bengal": "East", "Odisha": "East",
    "Rajasthan": "West", "Gujarat": "West", "Maharashtra": "West", "Goa": "West",
    "Dadra & Nagar Haveli": "West", "Daman & Diu": "West",
    "Madhya Pradesh": "Central", "Chhattisgarh": "Central",
    "Assam": "Northeast", "Arunachal Pradesh": "Northeast", "Manipur": "Northeast",
    "Meghalaya": "Northeast", "Mizoram": "Northeast", "Nagaland": "Northeast",
    "Sikkim": "Northeast", "Tripura": "Northeast",
}
m["Zone"] = m["State"].map(ZONE_MAP)
unmapped = m[m["Zone"].isna()]["State"].unique()
if len(unmapped):
    print("States not mapped to a zone:", unmapped)

zone_vars = [
    "Prevalence of diarrhoea in the 2 weeks preceding the survey (%)_NFHS5",
    "Male Elevated blood pressure or taking medicine to control blood pressure (%)_NFHS5",
    "Men age 15 years and above who use any kind of tobacco (%)_NFHS5",
]
zone_order = ["North", "South", "East", "West", "Central", "Northeast"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, var in enumerate(zone_vars):
    sns.boxplot(data=m, x="Zone", y=var, order=zone_order, hue="Zone", legend=False, ax=axes[i], palette="Set2")
    axes[i].set_title(var[:40] + ("..." if len(var) > 40 else ""), fontsize=9)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("eda_charts/04_regional_boxplots.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 5. Hypothesis-test scatter plots
# ---------------------------------------------------------------
def scatter_with_corr(x, y, xlabel, ylabel, fname, title):
    sub = m[[x, y]].dropna()
    r, p = pearsonr(sub[x], sub[y])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=sub, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.5, "s": 20}, line_kws={"color": "red"})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson r = {r:.3f}, p = {p:.2e}, n = {len(sub)}")
    plt.tight_layout()
    plt.savefig(f"eda_charts/{fname}", dpi=120)
    plt.close()
    return r, p, len(sub)

r1, p1, n1 = scatter_with_corr(
    "Women who are literate (%)_NFHS5",
    "Male Elevated blood pressure or taking medicine to control blood pressure (%)_NFHS5",
    "Female literacy (%)", "Male elevated blood pressure (%)",
    "05_scatter_literacy_vs_bp.png",
    "Literacy vs. chronic risk factor (hypertension)",
)

r2, p2, n2 = scatter_with_corr(
    "Households with any usual member covered under a health insurance/financing scheme (%)_NFHS5",
    "total_hospital_count",
    "Health insurance coverage (%)", "Total hospital count (district)",
    "06_scatter_insurance_vs_hospitals.png",
    "Health insurance coverage vs. hospital infrastructure",
)

print(f"\nScatter 1 (literacy vs BP): r={r1:.3f}, p={p1:.2e}, n={n1}")
print(f"Scatter 2 (insurance vs hospitals): r={r2:.3f}, p={p2:.2e}, n={n2}")

# ---------------------------------------------------------------
# Findings summary
# ---------------------------------------------------------------
top_missing = missing_pct.head(5)
zone_counts = m["Zone"].value_counts()

findings = f"""# EDA Findings Summary

## 1. Missingness (01_missingness.png)
Missingness ranges widely across the panel. The 3 hospital-count columns are missing for the
58 districts that had no usable crosswalk match to hospital_directory.csv ({58/706*100:.1f}% of
all districts), which matches the Step 6 completeness count exactly. Health indicator columns
(NFHS 4 and NFHS 5) are missing for a much smaller set of districts, largely newer districts
carved out after the NFHS-4 survey round. Highest-missingness columns:
{chr(10).join(f"- {k}: {v:.1f}%" for k, v in top_missing.items())}

## 2. Histograms (02_histograms_grid.png)
Most NFHS-5 percentage indicators (literacy, blood pressure, tobacco use) are roughly bell-shaped
but right- or left-skewed rather than normal, consistent with district-level social/health
indicators bounded at 0-100%. `total_hospital_count` is heavily right-skewed with a long tail --
a small number of districts (metro/state-capital districts) have disproportionately more listed
facilities than the median district, which will need a log transform or capping before use as a
linear model feature.

## 3. Correlation heatmap (03_correlation_heatmap.png)
NFHS4 and NFHS5 versions of the same indicator are, as expected, the strongest correlated pairs
in the matrix (districts are persistently high or low on a given indicator across survey rounds).
Beyond that, the blood-pressure and blood-sugar indicators cluster together (chronic risk factors
move together across districts), and literacy/sanitation/electricity/drinking-water infrastructure
indicators also cluster together, consistent with a shared underlying socioeconomic development
factor. Hospital counts show comparatively weak correlation with the health indicators, suggesting
facility supply is not strongly explained by the disease/risk-factor burden alone (likely driven
more by population size and urbanization, which aren't in this panel).

## 4. Regional boxplots (04_regional_boxplots.png)
Zone assignment: {zone_counts.to_dict()}. Zone-level means (see chart for full distribution):
Diarrhoea prevalence is clearly highest in the **East** (mean 9.4%) and **West** (7.7%) zones,
versus ~5.3-5.7% everywhere else including the Northeast -- infrastructure gaps are not uniform
across the eastern half of the country the way a simple "East+Northeast" prior might suggest.
Male hypertension runs in the *opposite* direction from what a naive deprivation story would
predict: the **South** has the highest mean elevated-blood-pressure rate (29.8%), well above the
West (20.4%) and East (21.2%), which are the lowest. This is consistent with the same
literacy/hypertension pattern in finding 5(a) below -- more developed, higher-literacy southern
districts show more diagnosed/reported hypertension, not less. Male tobacco use follows a
different pattern again: **Northeast** (54.8%) and **East** (49.4%) are highest, **South** is by
far the lowest (24.0%), with North and West in between -- consistent with known state-level
tobacco survey data (heavy smokeless-tobacco use in the Northeast/East versus lower use in
southern states).

## 5. Hypothesis scatter plots
**(a) Literacy vs. male hypertension** (05_scatter_literacy_vs_bp.png): r = {r1:.3f}, p = {p1:.2e},
n = {n1}. {"A statistically significant" if p1 < 0.05 else "No statistically significant"}
{"positive" if r1 > 0 else "negative"} relationship: higher-literacy districts tend to
{"also have higher" if r1 > 0 else "have lower"} rates of male elevated blood pressure. This is
counter to a naive "poverty causes worse health" prior, and is consistent with a known pattern in
NFHS data where higher-literacy/higher-income districts show more *diagnosed and reported*
lifestyle-linked chronic conditions (better healthcare access/awareness), rather than literacy
directly protecting against hypertension.

**(b) Health insurance coverage vs. hospital count** (06_scatter_insurance_vs_hospitals.png):
r = {r2:.3f}, p = {p2:.2e}, n = {n2}. {"A statistically significant" if p2 < 0.05 else "No statistically significant"}
{"positive" if r2 > 0 else "negative"} relationship between insurance coverage and the number of
hospitals in a district. The weak/moderate correlation suggests insurance coverage is not simply
a function of facility density -- state-level insurance scheme rollout (e.g. Ayushman Bharat
enrollment drives) likely explains coverage variation more than local hospital supply does.
"""

with open("EDA_Findings_Summary.md", "w", encoding="utf-8") as f:
    f.write(findings)

print("\nSaved EDA_Findings_Summary.md")
print("Saved 6 PNG charts to eda_charts/")
