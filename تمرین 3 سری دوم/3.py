import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'x': np.array([-4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 20]).reshape(-1, 1),
    'y': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
}

# Models
linear_model = LinearRegression()
logistic_model = LogisticRegression()

# Fit models
linear_model.fit(data['x'], data['y'])
logistic_model.fit(data['x'], data['y'])

# Predictions with threshold for linear regression
linear_predictions = linear_model.predict(data['x'])
linear_classes = (linear_predictions >= 0.5).astype(int)

# Predictions for logistic regression
logistic_classes = logistic_model.predict(data['x'])

# Accuracy
linear_accuracy = accuracy_score(data['y'], linear_classes)
logistic_accuracy = accuracy_score(data['y'], logistic_classes)

# Plot decision boundaries
x_range = np.linspace(-5, 22, 500).reshape(-1, 1)
linear_boundary = linear_model.predict(x_range)
logistic_boundary = logistic_model.predict_proba(x_range)[:, 1]

plt.figure(figsize=(12, 6))

# Linear regression plot
plt.subplot(1, 2, 1)
plt.scatter(data['x'], data['y'], color='blue', label='Data Points')
plt.plot(x_range, linear_boundary, color='red', label='Linear Regression')
plt.axhline(0.5, color='green', linestyle='--', label='Threshold (0.5)')
plt.title('Linear Regression Decision Boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Logistic regression plot
plt.subplot(1, 2, 2)
plt.scatter(data['x'], data['y'], color='blue', label='Data Points')
plt.plot(x_range, logistic_boundary, color='red', label='Logistic Regression')
plt.axhline(0.5, color='green', linestyle='--', label='Threshold (0.5)')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

plt.tight_layout()
plt.show()

# Print accuracy
print(f"Linear Regression Accuracy: {linear_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%")
