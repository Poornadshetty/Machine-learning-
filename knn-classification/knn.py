import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Function to evaluate k-NN
def evaluate_knn(k_values, weighted=False):
    accuracies = []
    f1_scores = []

    for k in k_values:
        if weighted:
            model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        else:
            model = KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    return accuracies, f1_scores

# K values
k_values = [1, 3, 5]

# Evaluate
acc_normal, f1_normal = evaluate_knn(k_values, weighted=False)
acc_weighted, f1_weighted = evaluate_knn(k_values, weighted=True)

# Print results
print("Regular k-NN:")
for i, k in enumerate(k_values):
    print(f"K={k} -> Accuracy={acc_normal[i]:.3f}, F1={f1_normal[i]:.3f}")

print("\nWeighted k-NN:")
for i, k in enumerate(k_values):
    print(f"K={k} -> Accuracy={acc_weighted[i]:.3f}, F1={f1_weighted[i]:.3f}")

# Plot
plt.figure(figsize=(10, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(k_values, acc_normal, marker='o', label='Regular')
plt.plot(k_values, acc_weighted, marker='o', label='Weighted')
plt.title('Accuracy vs K')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()

# F1 Score plot
plt.subplot(1, 2, 2)
plt.plot(k_values, f1_normal, marker='o', label='Regular')
plt.plot(k_values, f1_weighted, marker='o', label='Weighted')
plt.title('F1 Score vs K')
plt.xlabel('K')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
