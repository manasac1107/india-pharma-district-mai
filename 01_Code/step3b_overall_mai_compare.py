import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

norm = pd.read_csv("normalized_district_panel_full.csv")
entropy_w = pd.read_csv("entropy_weights.csv").set_index("variable")["entropy_weight"]
judgment_w = pd.read_csv("judgment_weights.csv").set_index("variable")["judgment_weight"]

SCORING_VARS = entropy_w.index.tolist()
X = norm[SCORING_VARS].copy()
X_imputed = X.fillna(X.mean())

out = norm[["State", "District"]].copy()
out["Overall_MAI_entropy"] = X_imputed[SCORING_VARS].values @ entropy_w[SCORING_VARS].values
out["Overall_MAI_judgment"] = X_imputed[SCORING_VARS].values @ judgment_w[SCORING_VARS].values

out["Rank_entropy"] = out["Overall_MAI_entropy"].rank(ascending=False, method="min").astype(int)
out["Rank_judgment"] = out["Overall_MAI_judgment"].rank(ascending=False, method="min").astype(int)

rho, pval = spearmanr(out["Overall_MAI_entropy"], out["Overall_MAI_judgment"])
print(f"Spearman rank correlation (entropy vs judgment Overall MAI): rho = {rho:.4f}, p = {pval:.2e}, n = {len(out)}")

out["Rank_Diff_abs"] = (out["Rank_entropy"] - out["Rank_judgment"]).abs()
biggest_movers = out.sort_values("Rank_Diff_abs", ascending=False).head(15)
print("\nTop 15 districts with the largest rank disagreement between entropy and judgment weighting:")
pd.set_option("display.width", 160)
print(biggest_movers[["State", "District", "Rank_entropy", "Rank_judgment", "Rank_Diff_abs"]].to_string(index=False))

out.to_csv("overall_mai_entropy_vs_judgment.csv", index=False)

# --- scatter chart ---
import os
os.makedirs("results_charts", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(out["Rank_entropy"], out["Rank_judgment"], alpha=0.4, s=15, color="#2980b9")
lims = [1, len(out)]
ax.plot(lims, lims, color="red", linestyle="--", linewidth=1, label="perfect agreement")
for _, row in biggest_movers.head(8).iterrows():
    ax.annotate(row["District"], (row["Rank_entropy"], row["Rank_judgment"]), fontsize=7, alpha=0.8)
ax.set_xlabel("Rank (entropy-weighted Overall MAI)")
ax.set_ylabel("Rank (judgment-weighted Overall MAI)")
ax.set_title(f"Entropy-rank vs. Judgment-rank, n={len(out)}\nSpearman rho = {rho:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig("results_charts/entropy_vs_judgment_rank.png", dpi=120)
plt.close()
print("\nSaved results_charts/entropy_vs_judgment_rank.png")
