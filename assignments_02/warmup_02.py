# Part 1: Warmup Exercises 

# --- The scikit-learn API ---

# Q1: Create a LinearRegression model, fit it to this data, then predict the high for a day
import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

x = temp_min     # the lows
y = temp_max     # the highs
print("shape of x: ", x.shape)

model = LinearRegression()      # create
try:
    model.fit(x,y)      # fit
    print(f"slope = {model.coef_[0]:.2f}")
    print(f"intercept = {model.intercept_:.2f}")
except ValueError as e:
    print("ValueError: ", str(e).splitlines()[0])

# Print predicted values for a day with a 6 °C low and an 18 °C low
print(f"\nPrediction for a day with a 6 °C low: {model.predict([[6]])[0]:.1f}")      # predict
print(f"\nPrediction for a day with a 18 °C low: {model.predict([[18]])[0]:.1f}")      # predict

# Q2: Start with 1D array, reshape it and print new shape
x = np.array([10, 20, 30, 40, 50])
print("\noriginal shape of x: ", x.shape)

# Reshape
X = x.reshape(-1, 1)    # -1 = "work out the rows yourself", 1 = "one column"

# Print new shape
print("\nshape of X: ", X.shape)

# Explain why scikit-learn needs X to be 2D
# X must always be a 2D array with shape (num_samples, num_features), even if you only have a single feature. A 1D array will cause an error. 

# --- Linear Regression ---
# Load Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

#Q1: Create a scatterplot of temp_min on x-axis and temp_max on y-axis
plt.figure(figsize=(7,5))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm", alpha=0.35, s=12)
plt.title("Daily High vs. Low")
plt.xlabel("Low temperature (°C)")
plt.ylabel("High temperature (°C)")
plt.grid(alpha=0.3)
plt.savefig('assignments_02/outputs/high_vs_low.png', dpi=300, bbox_inches='tight')
plt.show()

# What do you see? Are there two visible groups? What does that suggested about is_rainy variable?
# I see a line the goes up diagonally from the bottom left to top right. There are two visible groups, rainy and not rainy days. 

# Q2: Split the data into training and tests using temp_min
# Reshape temp_min to a 2D
X = temp_min.reshape(-1, 1)    # -1 = "work out the rows yourself", 1 = "one column"
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the shapes of all four arrays
print("\nshape of X_train: ", X_train.shape)
print("\nshape of X_test: ", X_test.shape)
print("\nshape of y_train: ", y_train.shape)
print("\nshape of y_test: ", y_test.shape)

# Q3: Fit a Linear Regression model to training data from Q2
model = LinearRegression()    # Create
model.fit(X_train, y_train)    # Fit

# Print slope
print(f"slope = {model.coef_[0]:.2f}")

# Print intercept
print(f"intercept = {model.intercept_:.2f}")

# Predict on test set 
y_pred = model.predict(X_test)    # Predict

# Print RMSE
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Print R² 
print(f"Single Feature: R² = {model.score(X_test, y_test)}")

# What does the slope mean for the daily high (in plain English)?
# The daily high rises by about one degree for each degree the low rises.

# Q4: Now add is_rainy as a second feature and fit a new model
X_full = np.column_stack([temp_min, is_rainy])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression().fit(X_train_full, y_train_full)     # Create & Fit
y_full_pred = model_full.predict(X_test_full)      # Predict

# Print R²
print(f"\nMultiple Features: R² = {r2_score(y_test_full, y_full_pred):.3f}")

# Does adding the rain flag help?
# Yes, the rain flag improves R² to 0.945. 

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Interpret the is_rainy coefficient 
# The model predicts that a rainy day is about 2.9 °C lower than a non-rainy day with the same temperature minimum.

# Q5: Predicted vs. Actual plot
plt.figure(figsize=(10,6))
plt.scatter(y_full_pred, y_test_full, alpha=0.6, edgecolors="k", linewidth=0.5)
plt.plot([y_test_full.min(), y_test_full.max()],
         [y_test_full.min(), y_test_full.max()],
         "r--", label="Perfect fit")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs. Actual")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('assignments_02/outputs/predicted_vs_actual_warmup.png', dpi=300, bbox_inches='tight')
plt.show()

# What does it mean when a point falls above the diagonal? What about below?
# If the point falls above the diagonal, the model underestimated the actual value. If the point falls below the diagonal, the model overestimated the actual value. 
