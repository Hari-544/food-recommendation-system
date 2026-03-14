import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("c:/Users/hari4/food-recommendation/dataset/food_dataset_6000.csv")

print(data.head())

# Keep required columns
data = data[['Name','Describe']]

# Remove missing values
data = data.dropna()

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z ]","",text)
    return text

# Clean description
data['Describe'] = data['Describe'].apply(clean_text)

# 🔥 Combine name + description
data['combined_features'] = data['Name'] + " " + data['Describe']

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_features=8000
)

tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save model
pickle.dump(cosine_sim, open("similarity1.pkl","wb"))
pickle.dump(data, open("foods1.pkl","wb"))

print("Model training completed!")