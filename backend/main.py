from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

# allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = pickle.load(open(BASE_DIR / "foods.pkl","rb"))
cosine_sim = pickle.load(open(BASE_DIR / "similarity.pkl","rb"))

print(data['Name'].head(30))

def recommend(food_name):

    matches = data[
        data['Name'].str.lower().str.contains(food_name.lower(), na=False) |
        data['Describe'].str.lower().str.contains(food_name.lower(), na=False)
    ]

    if matches.empty:
        return ["Food not found in dataset"]

    idx = matches.index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:21]

    food_indices = [i[0] for i in scores]

    return data['Name'].iloc[food_indices].tolist()


@app.get("/recommend/{food}")
def get_recommendations(food: str):
    return {"recommendations": recommend(food)}
