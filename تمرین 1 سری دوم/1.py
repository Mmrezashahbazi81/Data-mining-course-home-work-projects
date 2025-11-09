import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_points = np.array([
    [-3, -2],
    [-2, -1],
    [0, 1],
    [2, 0],
    [3, 2],
    [4, 3]
])


X = data_points[:, 0].reshape(-1, 1) 
y = data_points[:, 1]  

model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


alpha = model.intercept_  
beta = model.coef_[0]  

mse = mean_squared_error(y, y_pred)

print(alpha, beta, mse)

plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="red", label="Data Points")
plt.plot(X, y_pred, color="blue", label=f"Regression Line: y = {alpha:.2f} + {beta:.2f}x")
plt.title("Linear Regression Analysis", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle="--")
plt.axvline(0, color='black', linewidth=0.8, linestyle="--")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.show()