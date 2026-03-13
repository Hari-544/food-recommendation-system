import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# load model
data = pickle.load(open("backend/foods.pkl","rb"))
cosine_sim = pickle.load(open("backend/similarity.pkl","rb"))

threshold = 0.5

y_true = []
y_pred = []

# generate true and predicted labels
for i in range(len(data)):
    for j in range(len(data)):

        if i == j:
            continue

        name1 = data['Name'].iloc[i].lower()
        name2 = data['Name'].iloc[j].lower()

        # simple rule for ground truth
        true_label = int(name1.split()[0] == name2.split()[0])

        similarity_score = cosine_sim[i][j]

        predicted_label = int(similarity_score >= threshold)

        y_true.append(true_label)
        y_pred.append(predicted_label)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

# accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

print("\nAccuracy:", accuracy)


# similarity matrix
sim_df = pd.DataFrame(
    cosine_sim,
    index=data['Name'],
    columns=data['Name']
)

print("\nSimilarity Matrix Sample:\n")
print(sim_df.head())


# ---------- Precision@K and Recall@K ----------

def precision_at_k(sim_matrix, k=5):

    precisions = []

    for i in range(len(sim_matrix)):

        scores = list(enumerate(sim_matrix[i]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        top_k = scores[1:k+1]

        relevant = 0

        for idx, score in top_k:

            if score >= threshold:
                relevant += 1

        precisions.append(relevant / k)

    return np.mean(precisions)


def recall_at_k(sim_matrix, k=5):

    recalls = []

    for i in range(len(sim_matrix)):

        scores = list(enumerate(sim_matrix[i]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        top_k = scores[1:k+1]

        relevant = 0

        for idx, score in top_k:

            if score >= threshold:
                relevant += 1

        recalls.append(relevant / k)

    return np.mean(recalls)


print("\nPrecision@5:", precision_at_k(cosine_sim,5))
print("Recall@5:", recall_at_k(cosine_sim,5))