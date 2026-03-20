from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.recommender import load_artifacts, recommend_foods

app = FastAPI()

# allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

data, cosine_sim = load_artifacts()

def recommend(food_name):
    recommendations = recommend_foods(food_name, data, cosine_sim, top_n=20)
    if not recommendations:
        return ["Food not found in dataset"]
    return recommendations


@app.get("/recommend/{food}")
def get_recommendations(food: str):
    return {"recommendations": recommend(food)}
