# Project 2: Data Classification Using AI


---

##  Project Overview
This project builds a basic classification model using the **K-Nearest Neighbors (KNN)** algorithm on the Iris dataset. The goal is to predict the species of a flower based on its measurements.

---

##  Goal
Train a supervised learning model that can classify iris flowers into 3 categories:
- Setosa
- Versicolor
- Virginica

---

##  Dataset
- **Name:** Iris Dataset
- **Source:** Built-in (`sklearn.datasets.load_iris`)
- **Samples:** 150
- **Classes:** 3
- **Features:** 4
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

---

##  How It Works
1. Load the Iris dataset
2. Split data → 80% Training / 20% Testing
3. Train a KNN model with K=5
4. Predict on test data
5. Calculate accuracy

---

##  Libraries Used
| Library | Purpose |
|--------|---------|
| scikit-learn | ML model, dataset, metrics |

---

##  How to Run

**Step 1: Install dependencies**
```
pip install scikit-learn
```

**Step 2: Run the script**
```
python project2_knn.py
```

---

##  Output
```
Data Classification Using AI
-----------------------------------
Model Accuracy: 100.0 %

Sample Flower Prediction:
Predicted Class: setosa
```

---

## Project Structure
```

├── Screenshot   
├── project2.py             
├── README.md               
└── requirements.txt   
```

---

##  Intern Info
- **Program:** Industrial Training — DecodeLabs
- **Track:** Artificial Intelligence
- **Project:** 2 of series