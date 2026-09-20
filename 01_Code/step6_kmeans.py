import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv")
four = pd.read_csv("four_indices_with_momentum.csv")

SCORING_VARS = entropy_w["variable"].tolist()
X = norm[SCORING_VARS].copy()
X_imputed = X.fillna(X.mean())

km = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_imputed)

four["Cluster_raw"] = cluster_labels

cluster_stats = four.groupby("Cluster_raw").agg(
    n_districts=("District", "count"),
    mean_Overall_MAI=("Overall_MAI_entropy", "mean"),
    mean_Chronic_MAI=("Chronic_MAI", "mean"),
    mean_Acute_MAI=("Acute_MAI", "mean"),
    mean_Momentum=("Momentum", "mean"),
).round(4).sort_values("mean_Overall_MAI", ascending=False)

print("=== Cluster stats (raw KMeans labels 0-3) ===")
print(cluster_stats.to_string())

four.to_csv("clustered_districts_raw.csv", index=False)
print("\nSaved clustered_districts_raw.csv (raw cluster IDs, labels pending your review)")
