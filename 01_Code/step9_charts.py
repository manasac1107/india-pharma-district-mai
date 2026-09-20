import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

sns.set_theme(style="whitegrid")
os.makedirs("results_charts", exist_ok=True)

final = pd.read_csv("final_district_mai_scores.csv")
norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv")
chronic_w = pd.read_csv("chronic_mai_weights.csv", index_col=0)["weight"]
acute_w = pd.read_csv("acute_mai_weights.csv", index_col=0)["weight"]
sens = pd.read_csv("sensitivity_analysis.csv")

CLUSTER_COLORS = {
    "Elite Metro Powerhouses": "#c0392b",
    "Established High-Value": "#e67e22",
    "Mainstream Mid-Tier": "#7f8c8d",
    "Small Underserved, Chronic-Leaning": "#2980b9",
}

# ============================================================
# 1. Top-20 ranking bar chart
# ============================================================
top20 = final.sort_values("Overall_MAI_entropy", ascending=False).head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 9))
colors = [CLUSTER_COLORS[c] for c in top20["Cluster_Label"]]
ax.barh(top20["District"] + ", " + top20["State"].str.slice(0, 12), top20["Overall_MAI_entropy"], color=colors)
ax.set_xlabel("Overall MAI (entropy-weighted)")
ax.set_title("Top 20 districts by Overall MAI")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLUSTER_COLORS.values()]
ax.legend(handles, CLUSTER_COLORS.keys(), loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("results_charts/01_top20_ranking.png", dpi=120)
plt.close()

# ============================================================
# 2. Chronic vs Acute quadrant scatter, colored by cluster
# ============================================================
fig, ax = plt.subplots(figsize=(9, 8))
for label, color in CLUSTER_COLORS.items():
    sub = final[final["Cluster_Label"] == label]
    ax.scatter(sub["Chronic_MAI"], sub["Acute_MAI"], s=18, alpha=0.55, color=color, label=label)
ax.axvline(final["Chronic_MAI"].median(), color="gray", linestyle="--", linewidth=1)
ax.axhline(final["Acute_MAI"].median(), color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Chronic MAI")
ax.set_ylabel("Acute MAI")
ax.set_title("Chronic vs. Acute MAI quadrant (dashed lines = medians)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results_charts/02_chronic_acute_quadrant.png", dpi=120)
plt.close()

# ============================================================
# 3. Cluster scatter (PCA 2D on the scoring variable set)
# ============================================================
SCORING_VARS_36 = entropy_w["variable"].tolist()
SCORING_VARS = SCORING_VARS_36 + ["Population_2011_norm"]
X = norm[SCORING_VARS].fillna(norm[SCORING_VARS].mean())
# weight inputs the same way the actual weighted K-Means clustering did (0.85*entropy
# weight for the 36 base vars, 0.15 for population), so the PCA view is consistent with
# what actually produced the cluster assignments
entropy_w_indexed = entropy_w.set_index("variable")["entropy_weight"]
effective_weight = pd.concat([0.85 * entropy_w_indexed, pd.Series({"Population_2011_norm": 0.15})])
X_weighted = X[SCORING_VARS] * effective_weight[SCORING_VARS]
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_weighted)
fig, ax = plt.subplots(figsize=(9, 8))
for label, color in CLUSTER_COLORS.items():
    mask = final["Cluster_Label"] == label
    ax.scatter(pcs[mask.values, 0], pcs[mask.values, 1], s=18, alpha=0.55, color=color, label=label)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("K-Means clusters (4) in PCA 2D space of the entropy-weighted 37-variable scoring set")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results_charts/03_cluster_pca_scatter.png", dpi=120)
plt.close()

# ============================================================
# 4. Pillar-contribution stacked bar for 4 exemplar districts (one per cluster)
# ============================================================
pillar_map = pd.read_csv("pillar_mapping.csv").set_index("variable")["pillar"]
pillar_map["Population_2011_norm"] = "Market_Size"
# effective weight actually used in Overall_MAI_entropy: 0.85*entropy_weight for the 36
# base vars, flat 0.15 for the population floor
effective_weight_full = pd.concat([0.85 * entropy_w.set_index("variable")["entropy_weight"],
                                    pd.Series({"Population_2011_norm": 0.15})])
exemplars = final.loc[final.groupby("Cluster_Label")["Overall_MAI_entropy"].idxmax()]
exemplars = exemplars.set_index("Cluster_Label").loc[list(CLUSTER_COLORS.keys())].reset_index()

contrib_rows = []
for _, row in exemplars.iterrows():
    idx = final[(final["State"] == row["State"]) & (final["District"] == row["District"])].index[0]
    x_row = X.iloc[idx]
    contrib = x_row * effective_weight_full[SCORING_VARS]
    pillar_contrib = contrib.groupby(pillar_map[SCORING_VARS]).sum()
    contrib_rows.append({"label": f"{row['District']} ({row['Cluster_Label']})", **pillar_contrib.to_dict()})

contrib_df = pd.DataFrame(contrib_rows).set_index("label")
fig, ax = plt.subplots(figsize=(10, 6))
contrib_df.plot(kind="barh", stacked=True, ax=ax, colormap="tab10")
ax.set_xlabel("Contribution to Overall MAI (entropy-weighted)")
ax.set_title("Pillar contribution breakdown, one exemplar district per cluster")
ax.legend(title="Pillar", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("results_charts/04_pillar_contribution_exemplars.png", dpi=120)
plt.close()

# ============================================================
# 5. Sensitivity / weight-comparison chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sens_plot = sens.copy()
sens_plot["label"] = sens_plot["variable"].str.slice(0, 30) + "\n" + sens_plot["perturbation"]
axes[0].barh(sens_plot["label"], sens_plot["spearman_rho_vs_base"], color="#27ae60")
axes[0].set_xlim(0.9, 1.01)
axes[0].set_xlabel("Spearman rho vs. base case")
axes[0].set_title("Sensitivity: +/-10% perturbation of top-3 weights")

top10_entropy_w = entropy_w.sort_values("entropy_weight", ascending=False).head(10)
axes[1].barh(top10_entropy_w["variable"].str.slice(0, 40), top10_entropy_w["entropy_weight"], color="#8e44ad")
axes[1].set_xlabel("Entropy weight")
axes[1].set_title("Top 10 entropy weights (perturbation targets from top 3)")
plt.tight_layout()
plt.savefig("results_charts/05_sensitivity_weights.png", dpi=120)
plt.close()

# ============================================================
# 6. Rising-stars (Rank_Delta) chart
# ============================================================
rising = final.sort_values("Rank_Delta", ascending=False).head(15).iloc[::-1]
fading = final.sort_values("Rank_Delta", ascending=True).head(15)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].barh(rising["District"] + ", " + rising["State"].str.slice(0, 10), rising["Rank_Delta"], color="#27ae60")
axes[0].set_title("Top 15 rising stars (Rank_Delta, Current -> Future)")
axes[0].set_xlabel("Rank_Delta (positive = climbing)")
axes[1].barh(fading["District"] + ", " + fading["State"].str.slice(0, 10), fading["Rank_Delta"], color="#c0392b")
axes[1].set_title("Top 15 fading (Rank_Delta)")
axes[1].set_xlabel("Rank_Delta (negative = falling)")
plt.tight_layout()
plt.savefig("results_charts/06_rising_stars.png", dpi=120)
plt.close()

print("Saved 6 charts to results_charts/")
