# Project 2: Data Classification Using AI

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()

X = iris.data      # Features
y = iris.target    # Target labels

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Classification Model
model = KNeighborsClassifier(n_neighbors=5)

# Train the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Data Classification Using AI")
print("-" * 35)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Example Prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

flower_names = iris.target_names

print("\nSample Flower Prediction:")
print("Predicted Class:", flower_names[prediction[0]])