import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from backend.recommender import load_artifacts


data, cosine_sim = load_artifacts()
threshold = 0.5


def precision_at_k(sim_matrix, k=5):
    precisions = []

    for i in range(len(sim_matrix)):
        scores = list(enumerate(sim_matrix[i]))
        scores.sort(key=lambda item: item[1], reverse=True)
        top_k = scores[1 : k + 1]

        relevant = sum(1 for _, score in top_k if score >= threshold)
        precisions.append(relevant / k)

    return np.mean(precisions)


def recall_at_k(sim_matrix, k=5):
    recalls = []

    for i in range(len(sim_matrix)):
        scores = list(enumerate(sim_matrix[i]))
        scores.sort(key=lambda item: item[1], reverse=True)
        top_k = scores[1 : k + 1]

        relevant = sum(1 for _, score in top_k if score >= threshold)
        recalls.append(relevant / k)

    return np.mean(recalls)


def main():
    y_true = []
    y_pred = []

    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue

            name1 = data["Name"].iloc[i].lower()
            name2 = data["Name"].iloc[j].lower()

            true_label = int(name1.split()[0] == name2.split()[0])
            similarity_score = cosine_sim[i][j]
            predicted_label = int(similarity_score >= threshold)

            y_true.append(true_label)
            y_pred.append(predicted_label)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("\nConfusion Matrix:\n")
    print(cm)

    print("\nAccuracy:", accuracy)

    sim_df = pd.DataFrame(
        cosine_sim,
        index=data["Name"],
        columns=data["Name"],
    )

    print("\nSimilarity Matrix Sample:\n")
    print(sim_df.head())

    print("\nPrecision@5:", precision_at_k(cosine_sim, 5))
    print("Recall@5:", recall_at_k(cosine_sim, 5))


if __name__ == "__main__":
    main()
