import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "food_dataset_6000.csv"
BACKEND_DIR = BASE_DIR / "backend"


def clean_text(text):
    text = text.lower()
    return re.sub("[^a-zA-Z ]", "", text)


def main():
    data = pd.read_csv(DATASET_PATH)
    data = data[["Name", "Describe"]].dropna().copy()

    data["Name"] = data["Name"].astype(str).str.strip()
    data["Describe"] = data["Describe"].astype(str).apply(clean_text)
    data["combined_features"] = data["Name"] + " " + data["Describe"]

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )

    tfidf_matrix = tfidf.fit_transform(data["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    BACKEND_DIR.mkdir(parents=True, exist_ok=True)

    with (BACKEND_DIR / "similarity.pkl").open("wb") as file:
        pickle.dump(cosine_sim, file)

    with (BACKEND_DIR / "foods.pkl").open("wb") as file:
        pickle.dump(data, file)

    print(f"Model training completed. Saved artifacts to {BACKEND_DIR}")


if __name__ == "__main__":
    main()
