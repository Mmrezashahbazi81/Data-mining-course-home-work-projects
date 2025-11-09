from sklearn.preprocessing import PolynomialFeatures
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


poly_data_points = np.array([
    [-4, 4],
    [-3, 0],
    [-1, -2],
    [1, 3],
    [3, 1]
])


X_poly = poly_data_points[:, 0].reshape(-1, 1)  
y_poly = poly_data_points[:, 1]  


poly_features = PolynomialFeatures(degree=3)
X_poly_transformed = poly_features.fit_transform(X_poly)


poly_model = LinearRegression()
poly_model.fit(X_poly_transformed, y_poly)


y_poly_pred = poly_model.predict(X_poly_transformed)


coefficients = poly_model.coef_ 
intercept = poly_model.intercept_ 

print(coefficients, intercept)


plt.figure(figsize=(8, 6))

plt.scatter(X_poly, y_poly, color="red", label="Data Points")

X_plot = np.linspace(min(X_poly), max(X_poly), 100).reshape(-1, 1)
X_plot_transformed = poly_features.transform(X_plot)
y_plot_pred = poly_model.predict(X_plot_transformed)

plt.plot(X_plot, y_plot_pred, color="blue", label="Polynomial Regression Curve")

plt.title("Polynomial Regression (Degree 3)", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle="--")
plt.axvline(0, color='black', linewidth=0.8, linestyle="--")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.show()
