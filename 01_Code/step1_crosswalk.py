import pandas as pd
from rapidfuzz import fuzz, process

india = pd.read_csv("India.csv", low_memory=False)
india["State"] = india["State"].astype(str).str.strip()
india["District"] = india["District"].astype(str).str.strip()

hosp = pd.read_csv("hospital_directory.csv", low_memory=False)
hosp["State"] = hosp["State"].astype(str).str.strip()
hosp["District"] = hosp["District"].astype(str).str.strip()

nfhs_pairs = india[["State", "District"]].drop_duplicates().reset_index(drop=True)
nfhs_pairs = nfhs_pairs[~nfhs_pairs["District"].str.contains("Data Not Available", case=False, na=False)]

hosp_pairs = hosp[["State", "District"]].drop_duplicates().reset_index(drop=True)

nfhs_states = sorted(nfhs_pairs["State"].unique())
hosp_states = sorted(hosp_pairs["State"].unique())

# Build state-level fuzzy map first, to constrain district matching to the right state.
# If the best state match is weak (<60), there is no real counterpart state in the
# hospital directory (e.g. NCT of Delhi, Ladakh) -- leave unmatched rather than force
# a nonsense match to a dissimilar state's districts.
STATE_MIN_SCORE = 60
state_match = {}
for s in nfhs_states:
    match, score, _ = process.extractOne(s, hosp_states, scorer=fuzz.token_sort_ratio)
    if score < STATE_MIN_SCORE:
        match = None
    state_match[s] = (match, score)

rows = []
for _, r in nfhs_pairs.iterrows():
    n_state, n_dist = r["State"], r["District"]
    best_state, state_score = state_match[n_state]

    if best_state is None:
        rows.append({
            "nfhs_state": n_state,
            "nfhs_district": n_dist,
            "hospital_dir_state": None,
            "hospital_dir_district": None,
            "match_confidence": 0.0,
        })
        continue

    candidates = hosp_pairs.loc[hosp_pairs["State"] == best_state, "District"].unique().tolist()

    if candidates:
        match, dist_score, _ = process.extractOne(n_dist, candidates, scorer=fuzz.token_sort_ratio)
    else:
        match, dist_score = None, 0

    overall_conf = round(min(state_score, dist_score), 1) if candidates else 0.0

    rows.append({
        "nfhs_state": n_state,
        "nfhs_district": n_dist,
        "hospital_dir_state": best_state,
        "hospital_dir_district": match,
        "match_confidence": overall_conf,
    })

crosswalk = pd.DataFrame(rows).sort_values(["match_confidence", "nfhs_state", "nfhs_district"])
crosswalk.to_csv("district_crosswalk.csv", index=False)

print("Total NFHS (State,District) pairs:", len(nfhs_pairs))
print("Total hospital_directory (State,District) pairs:", len(hosp_pairs))
print("Crosswalk rows saved:", len(crosswalk))
print()
below_90 = crosswalk[crosswalk["match_confidence"] < 90]
print("Rows with match_confidence < 90:", len(below_90))

no_state_match = below_90[below_90["hospital_dir_state"].isna()]
print("  ...of which have NO corresponding state at all in hospital_directory:", len(no_state_match))
print(no_state_match[["nfhs_state", "nfhs_district"]].drop_duplicates().to_string(index=False))
print()

low_but_matched = below_90[below_90["hospital_dir_state"].notna()]
print("  ...of which have a state match but weak district match:", len(low_but_matched))
print(low_but_matched.to_string(index=False))
