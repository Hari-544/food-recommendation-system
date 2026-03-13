import gradio as gr
import pickle
import pandas as pd

data = pickle.load(open("backend/foods.pkl","rb"))
cosine_sim = pickle.load(open("backend/similarity.pkl","rb"))

def recommend(food):

    matches = data[
        data['Name'].str.lower().str.contains(food.lower(), na=False) |
        data['Describe'].str.lower().str.contains(food.lower(), na=False)
    ]

    if matches.empty:
        return "Food not found in dataset"

    idx = matches.index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:6]

    foods = [data['Name'].iloc[i[0]] for i in scores]

    return "\n".join(foods)


demo = gr.Interface(
    fn=recommend,
    inputs="text",
    outputs="text",
    title="Food Recommendation System",
    description="Enter a food name and get similar food recommendations."
)

demo.launch()