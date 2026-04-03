# Food Recommendation System

A content-based food recommendation project that ranks dish suggestions from a food query such as `Chicken`, `Fish`, or `Biryani`, then attaches restaurant suggestions with a preference for nearby places when location access is available.

## What This Project Does

- Recommends likely dish names for broad user inputs
- Pushes common dishes higher for inputs like `Chicken` or `Paneer`
- Shows restaurant suggestions for each recommended dish
- Uses browser geolocation and a nearby restaurant lookup when location is available
- Falls back gracefully when location access or nearby restaurant data is unavailable

## Project Structure

- `backend/main.py`: FastAPI API with recommendation endpoints
- `backend/recommender.py`: core food ranking logic
- `backend/places.py`: nearby restaurant lookup and matching
- `model/train_model.py`: training script that builds portable artifacts
- `backend/foods.csv`: trained food metadata artifact
- `backend/similarity.npy`: trained similarity matrix
- `docs/index.html`: main web frontend
- `frontend/index.html`: redirect entry for IDE/frontend convenience
- `app.py`: Gradio demo
- `test_model.py`: terminal demo
- `dataset/food_dataset_6000.csv`: main training dataset

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train The Model

```powershell
python model\train_model.py
```

This creates:

- `backend/foods.csv`
- `backend/similarity.npy`

## Run The API

```powershell
uvicorn backend.main:app --reload
```

Useful API routes:

- `GET /health`
- `GET /examples`
- `GET /recommend/{food}`

Example:

`http://127.0.0.1:8000/recommend/chicken`

With location:

`http://127.0.0.1:8000/recommend/chicken?lat=13.0827&lon=80.2707`

## Run The Web Frontend

Open:

- `docs/index.html`

or

- `frontend/index.html`

Notes:

- The frontend tries `http://127.0.0.1:8000` automatically on localhost
- You must keep the FastAPI server running
- Allow browser location access if you want nearby restaurant suggestions

## Run The Gradio Demo

```powershell
python app.py
```

## Run The Terminal Demo

```powershell
python test_model.py
```

## Current Ranking Logic

The food recommender uses:

- TF-IDF text similarity on dish name plus ingredients
- direct name-match boosts
- relevance filtering
- popularity-style boosts for common dish patterns such as biryani, curry, fried rice, and masala

The restaurant matcher uses:

- nearby restaurants around the user location
- distance-aware scoring
- simple relevance checks against dish and cuisine text
- fallback suggestions when nearby lookup is unavailable

## Important Limitations

- The food dataset does not include real popularity counts, ratings, or order history
- Nearby restaurants depend on external map data quality
- Restaurant popularity is estimated heuristically, not from live review data

## Best Next Upgrade

If you want production-quality nearby restaurants, replace the current nearby restaurant lookup with a real Places API such as Google Places. That would let you rank by:

- distance
- rating
- review count
- opening status
- real cuisine tags
