import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]  # 36-var
overall = pd.read_csv("overall_mai_entropy_vs_judgment.csv")

BASE_WEIGHT = 0.85
POP_WEIGHT = 0.15
SCORING_VARS_36 = entropy_w.index.tolist()
SCORING_VARS = SCORING_VARS_36 + ["Population_2011_norm"]

X = norm[SCORING_VARS].copy()
X_imputed = X.fillna(X.mean())

# effective weight = same weighting Overall_MAI_entropy actually uses: 0.85*entropy_weight
# for the 36 base vars, 0.15 flat for population -- so clustering distance reflects the
# same relative importance as the index, per user's chosen fix.
effective_weight = pd.concat([BASE_WEIGHT * entropy_w, pd.Series({"Population_2011_norm": POP_WEIGHT})])
X_weighted = X_imputed[SCORING_VARS] * effective_weight[SCORING_VARS]

km = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_weighted)

four = pd.read_csv("four_indices_final.csv").drop(
    columns=["Overall_MAI_entropy", "Overall_MAI_judgment", "Overall_MAI_equal_weight",
             "Rank_entropy", "Rank_judgment", "Rank_Diff_abs", "Population_2011", "Population_2011_norm"],
    errors="ignore",
)
four = four.merge(overall, on=["State", "District"], how="left")
four["Cluster_raw"] = cluster_labels

cluster_stats = four.groupby("Cluster_raw").agg(
    n_districts=("District", "count"),
    mean_Overall_MAI=("Overall_MAI_entropy", "mean"),
    mean_Chronic_MAI=("Chronic_MAI", "mean"),
    mean_Acute_MAI=("Acute_MAI", "mean"),
    mean_Momentum=("Momentum", "mean"),
    mean_Population=("Population_2011", "mean"),
).round(4).sort_values("mean_Overall_MAI", ascending=False)

print("=== Cluster stats (weighted K-Means, entropy-weight-scaled inputs) ===")
print(cluster_stats.to_string())

four.to_csv("clustered_districts_raw_v3.csv", index=False)

print("\nWhere do the new Top 20 districts land now?")
top20 = four.sort_values("Overall_MAI_entropy", ascending=False).head(20)
pd.set_option("display.width", 160)
print(top20[["State", "District", "Overall_MAI_entropy", "Cluster_raw"]].to_string(index=False))
