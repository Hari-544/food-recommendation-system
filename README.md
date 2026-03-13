# 🍔 Food Recommendation System

A **Machine Learning based Food Recommendation System** that suggests similar dishes using **Content-Based Filtering with TF-IDF and Cosine Similarity**.

Users can enter a food name and the system recommends similar foods based on ingredient descriptions.

---

# 📌 Project Overview

Recommendation systems are widely used in platforms like **Netflix, Amazon, and Spotify**.  
This project demonstrates how a **content-based recommendation system** can be built for food dishes using **text similarity techniques**.

The system analyzes food descriptions and calculates similarity between dishes to generate recommendations.

---

# 🚀 Features

- Content-based food recommendation
- Cosine similarity based recommendations
- FastAPI backend API
- Interactive web interface
- Model evaluation (Accuracy & Confusion Matrix)
- Precision@K and Recall@K metrics
- Clean project structure for GitHub

---

# 🧠 Machine Learning Approach

The model works using the following steps:

### 1️⃣ Data Preprocessing
- Remove missing values
- Clean text data
- Convert text to lowercase
- Remove special characters

### 2️⃣ Feature Extraction
Food descriptions are converted into numerical vectors using **TF-IDF Vectorization**.

### 3️⃣ Similarity Calculation
Similarity between food items is calculated using **Cosine Similarity**.

### 4️⃣ Recommendation Generation
The system recommends **Top-N similar foods** based on similarity scores.

---

# 📊 Evaluation Metrics

The recommendation model is evaluated using:

- Cosine Similarity Matrix
- Confusion Matrix
- Accuracy Score
- Precision@K
- Recall@K

These metrics help evaluate how well the system identifies similar foods.

---

# 🏗 Project Architecture
User Input (Food Name)
↓
Frontend (HTML / CSS / JS)
↓
FastAPI Backend
↓
TF-IDF Vectorization
↓
Cosine Similarity Matrix
↓
Top-N Food Recommendations


---

# 📁 Project Structure

food-recommendation-system
│
├── backend
│ ├── main.py
│ ├── foods.pkl
│ └── similarity.pkl
│
├── dataset
│ └── food.csv
│
├── docs
│ └── index.html
│
├── model
│ └── train_model.py
│
├── test_model.py
├── evaluate_model.py
├── requirements.txt
├── .gitignore
└── README.md


---

# ⚙️ Installation

### Clone the repository
git clone https://github.com/Hari-544/food-recommendation-system.git


### Navigate to the project folder
cd food-recommendation-system


### Create virtual environment
python -m venv .venv

### Activate virtual environment

Windows:
.venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt


---

# ▶️ Running the Project

Start the FastAPI backend server:
uvicorn backend.main:app --reload


Open the frontend page:
docs/index.html

Enter a food name and click **Recommend** to see similar dishes.

---

# 🧪 Testing the Model

To test the recommendation model from the terminal:
python test_model.py

You can enter a food name and the system will display recommended foods.

---

# 📈 Evaluating the Model

Run the evaluation script:
python evaluate_model.py


This script calculates:

- Confusion Matrix
- Accuracy
- Precision@K
- Recall@K

---

# 🖥 Example Output

Input:
chicken soup

Output:
Recommended Foods:

Chicken Curry

Chicken Paella

Lamb Rogan Josh

Fish Andlouse

Rice in Lamb Stock


---

# 🛠 Technologies Used

- Python
- Scikit-learn
- FastAPI
- Pandas
- NumPy
- HTML
- CSS
- JavaScript
- Git & GitHub

---

# 📚 Future Improvements

Possible improvements for this project:

- Add food images
- Ingredient-based recommendation
- Deep learning based recommendation models
- Deploy the application online

---

# 👨‍💻 Author

**Hari Krishna**

GitHub:  
https://github.com/Hari-544

---

# ⭐ If you like this project

Consider giving it a **star ⭐ on GitHub**!
