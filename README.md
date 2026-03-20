# Food Recommendation System

A content-based food recommendation project built with TF-IDF and cosine similarity.

## Project Structure

- `backend/main.py`: FastAPI app for recommendations
- `backend/recommender.py`: shared model-loading and recommendation logic
- `model/train_model.py`: trains the model and saves artifacts to `backend/`
- `app.py`: Gradio interface
- `test_model.py`: simple terminal-based recommendation test
- `evaluate_model.py`: evaluation script
- `dataset/food_dataset_6000.csv`: training dataset
- `docs/index.html`: static frontend

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python model/train_model.py
```

This creates:

- `backend/foods.pkl`
- `backend/similarity.pkl`

## Run the API

```bash
uvicorn backend.main:app --reload
```

## Run the Gradio App

```bash
python app.py
```

## Test from the Terminal

```bash
python test_model.py
```

## Evaluate the Model

```bash
python evaluate_model.py
```
