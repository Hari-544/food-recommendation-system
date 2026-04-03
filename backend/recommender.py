from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_CANDIDATES = ("foods.csv", "foods.pkl", "foods1.pkl")
SIMILARITY_FILE_CANDIDATES = ("similarity.npy", "similarity.pkl", "similarity1.pkl")
CUISINE_RESTAURANTS = {
    "Indian": ["Dindigul Thalappakatti", "Anjappar", "A2B", "Sangeetha", "Saravana Bhavan"],
    "Chinese": ["Mainland China", "Wok Express", "Noodle Theory", "Beijing Bites", "Mamagoto"],
    "Japanese": ["Kuuraku", "Pa Pa Ya", "Harima", "Sushi Junction", "Daily Sushi"],
    "Mediterranean": ["Zaatar House", "Arabian Grill", "Olive Bistro", "Levant Kitchen", "Mezze Hub"],
    "American": ["Smoke House Deli", "Chili's", "Hard Rock Cafe", "Stack & Grill", "Route 66 Kitchen"],
    "Thai": ["Thai Basil", "Soi 7", "Bangkok Bowl", "Blue Ginger", "Thai Pavilion"],
    "Italian": ["Little Italy", "Toscano", "Pasta Street", "Olive Table", "La Piazza"],
    "Mexican": ["Taco Bell", "Chili's", "Salsa Street", "Tex Mex Kitchen", "Casa Mexicana"],
    "Healthy Food": ["Fresh Bowl", "Green Crave", "Fit Feast", "Salad Stories", "Nutri Kitchen"],
    "Dessert": ["Sweet Truth", "CakeZone", "The Dessert Lab", "Cream & Crust", "Sugar Studio"],
    "Snack": ["Snack House", "Tea Time Bites", "Crunch Corner", "Quick Treats", "Street Munch"],
}
FALLBACK_RESTAURANTS = ["Food Street Kitchen", "Taste Garden", "Spice Route", "The Hungry Table", "Urban Tiffin"]
POPULAR_DISH_TERMS = {
    "biryani": 52,
    "curry": 38,
    "masala": 34,
    "gravy": 28,
    "fry": 26,
    "fried rice": 22,
    "korma": 24,
    "tikka": 24,
}
NON_INDIAN_DISH_PENALTIES = {
    "pasta bowl": 28,
    "noodle bowl": 24,
    "stir fry": 20,
    "soup": 16,
}
POPULAR_QUERY_BOOSTS = {
    "chicken": {
        "chicken biryani": 50,
        "chicken fried rice": 34,
        "chicken curry": 28,
        "chicken masala": 24,
        "chicken gravy": 22,
    },
    "paneer": {
        "paneer butter masala": 52,
        "paneer biryani": 36,
        "paneer curry": 28,
        "paneer masala": 24,
    },
    "mutton": {
        "mutton biryani": 52,
        "mutton curry": 30,
        "mutton gravy": 26,
    },
    "fish": {
        "fish curry": 32,
        "fish biryani": 28,
        "fish fried rice": 24,
    },
}


def _load_pickle(candidates: tuple[str, ...]):
    for file_name in candidates:
        file_path = BASE_DIR / file_name
        if file_path.exists():
            with file_path.open("rb") as file:
                return pickle.load(file)

    searched = ", ".join(str(BASE_DIR / name) for name in candidates)
    raise FileNotFoundError(f"Could not find any model artifact. Searched: {searched}")


def _load_data():
    csv_path = BASE_DIR / "foods.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return _load_pickle(DATA_FILE_CANDIDATES[1:])


def _load_similarity():
    npy_path = BASE_DIR / "similarity.npy"
    if npy_path.exists():
        return np.load(npy_path)

    return _load_pickle(SIMILARITY_FILE_CANDIDATES[1:])


def load_artifacts():
    data = _load_data()
    cosine_sim = _load_similarity()

    required_columns = {"Name", "Describe", "C_Type", "Veg_Non"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if not isinstance(cosine_sim, np.ndarray):
        cosine_sim = np.array(cosine_sim)

    if cosine_sim.ndim != 2 or cosine_sim.shape[0] != cosine_sim.shape[1]:
        raise ValueError(
            "Similarity artifact must be a square 2D matrix. "
            f"Received shape: {getattr(cosine_sim, 'shape', 'unknown')}"
        )

    if cosine_sim.shape[0] != len(data):
        raise ValueError(
            "Artifact mismatch: foods data row count and similarity matrix size differ. "
            f"Rows: {len(data)}, similarity size: {cosine_sim.shape[0]}"
        )

    for column in ("Name", "Describe", "C_Type", "Veg_Non"):
        if column in data.columns:
            data[column] = data[column].fillna("").astype(str).str.strip()

    return data, cosine_sim


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _select_restaurants(row) -> list[str]:
    cuisine = row.get("C_Type", "")
    restaurant_pool = CUISINE_RESTAURANTS.get(cuisine, FALLBACK_RESTAURANTS)
    digest = hashlib.md5(f"{row.get('Name', '')}|{cuisine}".encode("utf-8")).hexdigest()
    start_index = int(digest[:8], 16) % len(restaurant_pool)

    return [
        restaurant_pool[start_index],
        restaurant_pool[(start_index + 1) % len(restaurant_pool)],
    ]


def _score_candidate(query: str, row, cosine_score: float) -> float:
    name = _normalize(row["Name"])
    description = _normalize(row.get("Describe", ""))
    query_terms = query.split()
    score = cosine_score * 20

    if name == query:
        score += 150
    if name.startswith(query):
        score += 120
    if query in name:
        score += 90
    if all(term in name for term in query_terms):
        score += 40
    if query in description:
        score += 20
    if query_terms and name.split() and name.split()[0] == query_terms[0]:
        score += 25

    for dish_term, boost in POPULAR_DISH_TERMS.items():
        if dish_term in name:
            score += boost

    for dish_term, penalty in NON_INDIAN_DISH_PENALTIES.items():
        if dish_term in name:
            score -= penalty

    query_boosts = POPULAR_QUERY_BOOSTS.get(query, {})
    for popular_name, boost in query_boosts.items():
        if popular_name == name:
            score += boost
        elif popular_name in name:
            score += boost * 0.6

    return score


def _food_tag(name: str) -> str:
    normalized_name = _normalize(name)

    if "biryani" in normalized_name:
        return "Indian favorite"
    if "curry" in normalized_name or "masala" in normalized_name or "gravy" in normalized_name:
        return "Indian-style main course"
    if "fry" in normalized_name or "fried rice" in normalized_name:
        return "Popular rice or fry item"

    return "Popular nearby dish"


def _is_relevant_match(query: str, row) -> bool:
    name = _normalize(row["Name"])
    description = _normalize(row.get("Describe", ""))
    query_terms = query.split()

    if query in name or query in description:
        return True

    return bool(query_terms) and all(term in name for term in query_terms)


def recommend_foods(food_name: str, data, cosine_sim, top_n: int = 20) -> list[dict]:
    query = food_name.strip()
    if not query:
        return []

    normalized_query = _normalize(query)
    matches = data[
        data["Name"].str.lower().str.contains(normalized_query, na=False)
        | data["Describe"].str.lower().str.contains(normalized_query, na=False)
    ]

    if matches.empty:
        return []

    idx = matches.index[0]
    scored_items = []

    seen_names = set()
    seen = set()

    for item_index, cosine_score in enumerate(cosine_sim[idx]):
        row = data.iloc[item_index]
        candidate_name = row["Name"]
        normalized_name = _normalize(candidate_name)

        if item_index != idx and normalized_name == normalized_query:
            continue

        if normalized_name in seen:
            continue

        seen.add(normalized_name)

        if not _is_relevant_match(normalized_query, row):
            continue

        scored_items.append(
            (
                _score_candidate(normalized_query, row, float(cosine_score)),
                item_index,
            )
        )

    scored_items.sort(key=lambda item: item[0], reverse=True)

    recommendations = []
    for score, item_index in scored_items:
        row = data.iloc[item_index]
        candidate_name = row["Name"]
        normalized_name = _normalize(candidate_name)

        if normalized_name in seen_names:
            continue

        seen_names.add(normalized_name)
        recommendations.append(
            {
                "name": candidate_name,
                "cuisine": "",
                "food_tag": _food_tag(candidate_name),
                "type": row.get("Veg_Non", ""),
                "restaurants": _select_restaurants(row),
                "restaurant_source": "fallback",
                "score": round(score, 3),
            }
        )

        if len(recommendations) == top_n:
            break

    return recommendations
