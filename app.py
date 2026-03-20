import gradio as gr

from backend.recommender import load_artifacts, recommend_foods


data, cosine_sim = load_artifacts()


def recommend(food: str) -> str:
    recommendations = recommend_foods(food, data, cosine_sim, top_n=5)
    if not recommendations:
        return "Food not found in dataset"

    return "\n".join(recommendations)


demo = gr.Interface(
    fn=recommend,
    inputs="text",
    outputs="text",
    title="Food Recommendation System",
    description="Enter a food name and get similar food recommendations.",
)


if __name__ == "__main__":
    demo.launch()
