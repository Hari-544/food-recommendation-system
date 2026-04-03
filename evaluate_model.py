import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from backend.recommender import load_artifacts


data, cosine_sim = load_artifacts()
threshold = 0.5
MAX_EVAL_ITEMS = 1200


def precision_at_k(sim_matrix, k=5):
    precisions = []

    for i in range(len(sim_matrix)):
        scores = list(enumerate(sim_matrix[i]))
        scores.sort(key=lambda item: item[1], reverse=True)
        top_k = scores[1 : k + 1]

        relevant = sum(1 for _, score in top_k if score >= threshold)
        precisions.append(relevant / max(1, len(top_k)))

    return np.mean(precisions)


def recall_at_k(sim_matrix, k=5):
    recalls = []

    for i in range(len(sim_matrix)):
        scores = list(enumerate(sim_matrix[i]))
        scores.sort(key=lambda item: item[1], reverse=True)
        top_k = scores[1 : k + 1]

        relevant_in_top_k = sum(1 for _, score in top_k if score >= threshold)
        relevant_total = sum(1 for _, score in scores[1:] if score >= threshold)
        recalls.append(relevant_in_top_k / max(1, relevant_total))

    return np.mean(recalls)


def main():
    n_items = min(len(data), cosine_sim.shape[0], cosine_sim.shape[1])
    eval_items = min(n_items, MAX_EVAL_ITEMS)

    names = (
        data["Name"]
        .astype(str)
        .str.lower()
        .str.split()
        .str[0]
        .fillna("")
        .iloc[:eval_items]
        .to_numpy()
    )
    sim_eval = cosine_sim[:eval_items, :eval_items]

    true_matrix = names[:, None] == names[None, :]
    np.fill_diagonal(true_matrix, False)

    pred_matrix = sim_eval >= threshold
    np.fill_diagonal(pred_matrix, False)

    # Use only unique pairs to avoid duplicate (i, j) and (j, i) comparisons.
    upper_i, upper_j = np.triu_indices(eval_items, k=1)
    y_true = true_matrix[upper_i, upper_j].astype(int)
    y_pred = pred_matrix[upper_i, upper_j].astype(int)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("\nConfusion Matrix:\n")
    print(cm)

    print("\nAccuracy:", accuracy)
    print(f"Evaluated on {eval_items} items ({len(y_true)} unique pairs).")

    sim_df = pd.DataFrame(
        cosine_sim,
        index=data["Name"],  # type: ignore
        columns=data["Name"],  # type: ignore
    )

    print("\nSimilarity Matrix Sample:\n")
    print(sim_df.head())

    print("\nPrecision@5:", precision_at_k(cosine_sim, 5))
    print("Recall@5:", recall_at_k(cosine_sim, 5))


if __name__ == "__main__":
    main()
