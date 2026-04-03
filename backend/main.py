from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'backend.places' can be imported when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Annotated
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.places import assign_nearby_restaurants
from backend.recommender import load_artifacts, recommend_foods


class RecommendationItem(BaseModel):
    name: str
    cuisine: str
    food_tag: str
    type: str
    restaurants: list[str]
    restaurant_source: Literal["fallback", "nearby", "unavailable"]
    score: float


class RecommendationResponse(BaseModel):
    query: str
    message: str
    location_used: bool
    nearby_results_available: bool
    recommendations: list[RecommendationItem]


app = FastAPI(
    title="Food Recommendation API",
    description="Recommends food dishes and nearby restaurants for a user query.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

data, cosine_sim = load_artifacts()


def _build_recommendations(
    food_name: str,
    latitude: float | None = None,
    longitude: float | None = None,
    limit: int = 10,
) -> tuple[list[dict], bool]:
    import logging
    logger = logging.getLogger(__name__)

    candidate_pool = max(limit * 4, 20)
    base_recommendations = recommend_foods(food_name, data, cosine_sim, top_n=candidate_pool)
    nearby_results_available = False

    if base_recommendations and latitude is not None and longitude is not None:
        location_recommendations = [dict(item) for item in base_recommendations]
        try:
            location_recommendations, nearby_results_available = assign_nearby_restaurants(
                location_recommendations,
                latitude,
                longitude,
            )
        except Exception as exc:
            logger.error("Nearby restaurant lookup failed: %s", exc, exc_info=True)
            nearby_results_available = False
            # Strip out hardcoded fake restaurants when location was requested but lookup failed
            for item in location_recommendations:
                item["restaurants"] = []
                item["restaurant_source"] = "unavailable"

        # Return items that got real nearby restaurants first
        nearby_only = [
            item for item in location_recommendations if item["restaurant_source"] == "nearby"
        ]

        if nearby_only:
            return nearby_only[:limit], True

        # Location was provided but no nearby restaurants found —
        # still return the food recommendations but WITHOUT fake hardcoded restaurant names
        for item in location_recommendations:
            if item["restaurant_source"] == "fallback":
                item["restaurants"] = []
                item["restaurant_source"] = "unavailable"
        return location_recommendations[:limit], False

    return base_recommendations[:limit], False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "foods_loaded": len(data),
        "similarity_shape": list(cosine_sim.shape),
    }


@app.get("/examples")
def examples():
    return {
        "examples": [
            "Chicken",
            "Fish",
            "Paneer",
            "Mutton",
            "Biryani",
            "Fried rice",
        ]
    }


@app.get("/recommend/{food}", response_model=RecommendationResponse)
def get_recommendations(
    food: str,
    lat: Annotated[float | None, Query(ge=-90.0, le=90.0)] = None,
    lon: Annotated[float | None, Query(ge=-180.0, le=180.0)] = None,
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
):
    recommendations, nearby_results_available = _build_recommendations(food, lat, lon, limit)
    if recommendations and lat is not None and lon is not None and not nearby_results_available:
        message = "No strong nearby matches found, showing best overall dish recommendations."
    elif recommendations:
        message = ""
    elif lat is not None and lon is not None:
        message = "No nearby restaurants found that strongly match this food item."
    else:
        message = "Food not found in dataset"

    return RecommendationResponse(
        query=food,
        message=message,
        location_used=lat is not None and lon is not None,
        nearby_results_available=nearby_results_available,
        recommendations=[RecommendationItem(**r) for r in recommendations],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
