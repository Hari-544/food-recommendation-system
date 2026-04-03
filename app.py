import gradio as gr

from backend.recommender import load_artifacts, recommend_foods


data, cosine_sim = load_artifacts()


def recommend(food: str) -> str:
    recommendations = recommend_foods(food, data, cosine_sim, top_n=5)
    if not recommendations:
        return "Food not found in dataset."

    lines = []
    for index, item in enumerate(recommendations, start=1):
        restaurants = ", ".join(item["restaurants"]) if item["restaurants"] else "Not available"
        lines.append(
            f"{index}. {item['name']}\n"
            f"   Tag: {item['food_tag']} | Type: {item['type']}\n"
            f"   Restaurant suggestions: {restaurants}"
        )

    return "\n\n".join(lines)


demo = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(label="Food or ingredient", placeholder="Try Chicken, Fish, Paneer, or Biryani"),
    outputs=gr.Textbox(label="Recommendations", lines=18),
    title="Food Recommendation System",
    description="Enter a food name or ingredient to get ranked dish recommendations.",
)


if __name__ == "__main__":
    demo.launch()
