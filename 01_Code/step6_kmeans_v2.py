import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv")  # 36-var base
overall = pd.read_csv("overall_mai_entropy_vs_judgment.csv")

# clustering inputs: the 36-var base PLUS Population_2011_norm (37 total), since
# population is now a first-class part of how districts are being distinguished
SCORING_VARS = entropy_w["variable"].tolist() + ["Population_2011_norm"]
X = norm[SCORING_VARS].copy()
X_imputed = X.fillna(X.mean())

km = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_imputed)

# rebuild the four_indices frame from four_indices_final (has Chronic/Acute/Momentum) but
# refresh Overall_MAI_entropy/judgment with the new population-floor version
four = pd.read_csv("four_indices_final.csv").drop(
    columns=["Overall_MAI_entropy", "Overall_MAI_judgment", "Overall_MAI_equal_weight",
             "Rank_entropy", "Rank_judgment", "Rank_Diff_abs"]
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

print("=== Cluster stats (raw KMeans labels 0-3, WITH population in clustering inputs) ===")
print(cluster_stats.to_string())

four.to_csv("clustered_districts_raw_v2.csv", index=False)

print("\nWhere do the new Top 20 districts land?")
top20 = four.sort_values("Overall_MAI_entropy", ascending=False).head(20)
print(top20[["State", "District", "Overall_MAI_entropy", "Cluster_raw"]].to_string(index=False))
