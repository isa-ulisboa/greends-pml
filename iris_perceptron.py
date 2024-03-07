import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Load the Iris dataset
iris = load_iris()
X = iris.data[:100, :2]  # Only taking the first two features for simplicity
y = iris.target[:100]

# Split data into training, development, and testing sets
X_train, X_devtest, y_train, y_devtest = train_test_split(X, y, test_size=0.4, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_dev_std = scaler.transform(X_dev)
X_test_std = scaler.transform(X_test)

# Initialize perceptron with default parameters
perceptron = Perceptron()

# Hyperparameters to tune
learning_rates = [0.001, 0.01, 0.1, 1]
n_iterations = [50, 100, 150]

best_accuracy = 0
best_hyperparams = {}

# Grid search over hyperparameters using development set
for lr in learning_rates:
    for n_iter in n_iterations:
        perceptron.learning_rate = lr
        perceptron.n_iterations = n_iter

        perceptron.fit(X_train_std, y_train)
        y_dev_pred = perceptron.predict(X_dev_std)
        accuracy = accuracy_score(y_dev, y_dev_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparams = {'learning_rate': lr, 'n_iterations': n_iter}

# Train with the best hyperparameters on combined training and development set
perceptron.learning_rate = best_hyperparams['learning_rate']
perceptron.n_iterations = best_hyperparams['n_iterations']
X_train_dev_std = np.vstack((X_train_std, X_dev_std))
y_train_dev = np.concatenate((y_train, y_dev))
perceptron.fit(X_train_dev_std, y_train_dev)

# Predictions on the test set
y_pred = perceptron.predict(X_test_std)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Best hyperparameters:", best_hyperparams)
print("Accuracy on test set:", accuracy)
