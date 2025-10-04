import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (Years of Experience vs Salary)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1) # Independent variable
y = np.array([30000, 35000, 40000, 45000, 50000, 60000]) # Dependent variable

# Create Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Make prediction for new value (e.g., 7 years of experience)
predicted_salary = model.predict([[7]])
print("Predicted salary for 7 years of experience:", predicted_salary[0])

# Plotting the dataset and regression line
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.scatter(7, predicted_salary, color="green", marker="x", s=100, label="Prediction (7 yrs)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Example")
plt.legend()
plt.show()