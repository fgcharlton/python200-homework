## Part 1: Warmup Exercises 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---

# Q1: Fit a StandardScaler on X_train only, then transform both X_train and X_test
# Fit StandardScaler on X_train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn mean and std from training data only

# Transform both X_train and X_test
X_test_scaled = scaler.transform(X_test)  # apply the same scaling to the test data 

# Print the mean of each column of the scaled training data
train_means = X_train_scaled.mean(axis=0)
print("Mean of each column on scaled training data: ", train_means)

# Explain why you fit the scaler on X_train only
# The scaler learns two numbers from the data, the mean and the standard deviation of each feature.
# If you fit it on all of the data, those numbers are influenced by the test set. 
# When you evaluate on the test set, the model has indirectly seen it, and the results can look better than they truly are (i.e., data leakage).

# Q2: The season column in the dataset is categorical. One-hot encode it and print the resulting array's shape and category names.
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["season"]])

# Print array's shape
print("Encoded shape: ", encoded.shape)

# Print category names
print("Features names: ", encoder.get_feature_names_out())

# How many columns did the single season become, and why is that number what it is?
# It become 4 columns. One-hot encoding replaces a single column with one new column per category with each row having a 1 in the column for its category and a 0 elsewhere. 

# Q3: Build a ColumnTransformer that applies a StandardScaler to the 4 numeric columns and a OneHotEncoder to season
numeric_features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
categorical_features = ["season"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Wrap it with a LogisticRegression in a Pipeline
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# What does the pipeline do automatically that you would otherwise have to do by hand?
# A Pipeline is a sequence of steps where the output of each step becomes the input to the next.
# Otherwise, you would need to scale the numeric features, one-hot encode the categorical feature, and then fit a model.
# Doing it by hand means keeping several arrays and several fitted objects in sync, applying each transformation to the training set then to the test set.
# That would make it easy to forget a step, apply one out of order or accidentally fit a scaler on the test data.

# --- KNN ---

# Q1: Build a Pipeline of StandardScaler and KNeighborClassifier(n_neighbors=5)
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

# Fit it on the training data
knn_scaled.fit(X_train, y_train)

# Print the test accuracy and full classification report
print("Scaled accuracy: ", accuracy_score(y_test, knn_scaled.predict(X_test)))
y_pred = knn_scaled.predict(X_test)
print(classification_report(y_test, y_pred))

# Q2: Loop over k values [1, 3, 5, 7, 9, 11, 15, 21]. For each, build the scaled pipeline and compute the mean 5-fold cross-validation
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

# Which k to chose and why
# The best cross-validation score is around k = 11. The smaller values do slightly worse and the larger values start to fall off as the model underfits.

# Q3: Fit KNN once without scaling (on raw X_train) and once with scaling, both using n_neighbors=5
# Without Scaling 
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

# Print without scaling accuracy
print("Unscaled accuracy: ", accuracy_score(y_test, knn_unscaled.predict(X_test)))

# With Scaling
# Created above in Q1

# Print with scaling accuracy 
print("Scaled accuracy: ", accuracy_score(y_test, knn_scaled.predict(X_test)))

# Did scaling help, and why would scaling matter for a distance based model?
# Scaling raised accuracy from about 0.925 to about 0.958.
# Scaling allows for smaller-range features, such as precipitation, to count fairly in the distance for KNN.

# --- Logistic Regression ---

# Q1: Build a Pipeline of StandardScaler and LogisticRegression(max_iter=1000)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

# Fit it
model.fit(X_train, y_train)

# Print the test accuracy, precision, recall, and F1 (for the "good" class)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))

# Q2: Print each feature name alongside its coefficient from the fitted logistic regression (use pipeline.name_steps["..."].coef_[0])
logreg = model.named_steps["logreg"]
for name, coef in zip(numeric_features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# Which features push most strongly toward "skip", and does that match your intuition about running weather?
# Precipitation and wind speed push most strongly toward "skip", having large negative coefficients. 
# That does match my intuition for deciding whether it is a "good" day for running. It makes sense to skip running for days that have either a lot of rain or wind.

# Q3: Use predict_proba to print the probability of "good" for the first five test days
probabilities = model.predict_proba(X_test)[:, 1]   # probability of "good"
print(probabilities[:5].round(2))

# What does a probablity near 0.5 tell you about the model's confidence for that day?
# A probability of 0.5 means that there is basically a 50/50 chance that it will be a good day for running. Essentially the model has the same confidence as a coin flip.

# --- Evaluation ---

# Q1: For the logistic regression pipeline, build a confusion matrix on the test set and display it with ConfusionMatrixDisplay
labels=["skip", "good"]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(colorbar=False)
plt.title("Good Running Day Confusion Matrix")
plt.savefig('outputs/logreg_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# How many false positives and how many false negatives did the model make?
# There are 7 false positives where the model labeled it as a skip day when it was really a good day.
# There were 9 false negatives where the model labeled it as a good day when it was really a skip day. 

# Which kind of error would matter more for a running app?
# False negatives would matter more because you would be telling people that it is a good day for running when it is not. 
# Runners would run in high winds and rain. With a false positive, they would just simply miss a good running day.

# --- joblib ---

# Q1: Take your fitted logistic regression pipeline and save it to a models/warmup_model.pkl with joblib.dump
joblib.dump(model, "models/warmup_model.pkl")
print("Model saved.")

# Load it back with joblib.load() and confirm it produces identical predictions:
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == model.predict(X_test)).all()
print("Predictions match.")

# What would go wrong if you had saved only the LogisticRegression step without the scaler, then called .predict() on raw data?
# A trained logistic regression model knows it coefficients, but it does not know the scale of the data it was trained on.
# If you only save the classifier and call .predict() on raw, unscaled data, the predictions will be wrong. However, they will be silently wrong, with no error, because the model just computes whatever numbers it receives.


