from backend.recommender import load_artifacts, recommend_foods


data, cosine_sim = load_artifacts()


def recommend(food_name: str, top_n: int = 5):
    recommendations = recommend_foods(food_name, data, cosine_sim, top_n=top_n)
    if not recommendations:
        print("\nFood not found in dataset.\n")
        return

    print("\nTop Recommendations:\n")
    for food in recommendations:
        print(food)


if __name__ == "__main__":
    food_input = input("Enter a food name: ")
    recommend(food_input)
