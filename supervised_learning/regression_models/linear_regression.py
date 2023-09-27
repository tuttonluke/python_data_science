# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# %%
# load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y = True)

# use only one feature (for 1D linear regressoin)
diabetes_X = diabetes_X[:, np.newaxis, 2]

# split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
# %%
# create linear regression object
regression = linear_model.LinearRegression()

# Train the model using training sets
regression.fit(diabetes_X_train, diabetes_y_train)

# make predictions using the testing set
diabetes_y_pred = regression.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regression.coef_)

# mean squared error
print(f'Mean Squared Error: {round(mean_squared_error(diabetes_y_test, diabetes_y_pred), 2)}')

# The coefficient of determination: 1 is perfect prediction
print(f'Coefficient of determination: {round(r2_score(diabetes_y_test, diabetes_y_pred), 2)}')

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()