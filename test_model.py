import pickle
import pandas as pd

# Load trained model
data = pickle.load(open("foods.pkl", "rb"))
cosine_sim = pickle.load(open("similarity.pkl", "rb"))

def recommend(food_name, top_n=5):

    # search in Name or Describe (case insensitive)
    matches = data[
        data['Name'].str.lower().str.contains(food_name.lower(), na=False) |
        data['Describe'].str.lower().str.contains(food_name.lower(), na=False)
    ]

    if matches.empty:
        print("\n❌ Food not found in dataset\n")
        return

    # take first matched food
    idx = matches.index[0]
    matched_food = data['Name'].iloc[idx]

    # get similarity scores
    scores = list(enumerate(cosine_sim[idx]))

    # sort highest similarity first
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # ignore first result (same food)
    scores = scores[1:top_n+1]

    print("\n🍽 Closest Match Found:", matched_food)
    print("\n🔎 Top Recommendations:\n")

    for i, score in scores:
        food = data['Name'].iloc[i]
        print(f"{food}  |  Similarity Score: {score:.3f}")


# --------------------------

# user input
food_input = input("Enter a food name: ")

recommend(food_input)