from __future__ import annotations

import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_CANDIDATES = ("foods.pkl", "foods1.pkl")
SIMILARITY_FILE_CANDIDATES = ("similarity.pkl", "similarity1.pkl")


def _load_pickle(candidates: tuple[str, ...]):
    for file_name in candidates:
        file_path = BASE_DIR / file_name
        if file_path.exists():
            with file_path.open("rb") as file:
                return pickle.load(file)

    searched = ", ".join(str(BASE_DIR / name) for name in candidates)
    raise FileNotFoundError(f"Could not find any model artifact. Searched: {searched}")


def load_artifacts():
    data = _load_pickle(DATA_FILE_CANDIDATES)
    cosine_sim = _load_pickle(SIMILARITY_FILE_CANDIDATES)
    return data, cosine_sim


def recommend_foods(food_name: str, data, cosine_sim, top_n: int = 20) -> list[str]:
    query = food_name.strip()
    if not query:
        return []

    matches = data[
        data["Name"].str.lower().str.contains(query.lower(), na=False)
        | data["Describe"].str.lower().str.contains(query.lower(), na=False)
    ]

    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    seen = set()

    for item_index, _ in scores[1:]:
        food_name = data["Name"].iloc[item_index]
        normalized_name = food_name.strip().lower()

        if normalized_name in seen:
            continue

        seen.add(normalized_name)
        recommendations.append(food_name)

        if len(recommendations) == top_n:
            break

    return recommendations
