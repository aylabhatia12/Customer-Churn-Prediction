import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

url = "https://raw.githubusercontent.com/SaeidRostami/Customer_Churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

data = pd.read_csv(url)

print()
print("----- Dataset Information -----")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])


# ---------------------------------------------------------
# 2. Clean the data
# ---------------------------------------------------------

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
data = data.dropna()

# Remove customer ID because it is not useful for prediction
data = data.drop(columns=["customerID"])

# Convert target to 0 and 1
data["Churn"] = data["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ---------------------------------------------------------
# 3. Separate features and target
# ---------------------------------------------------------

X = data.drop(columns=["Churn"])
y = data["Churn"]


# ---------------------------------------------------------
# 4. Identify column types
# ---------------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns


# ---------------------------------------------------------
# 5. Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ---------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 7. Logistic Regression model
# ---------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]


# ---------------------------------------------------------
# 8. Random Forest model
# ---------------------------------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test
)

random_forest_probabilities = (
    random_forest_model.predict_proba(X_test)[:, 1]
)


# ---------------------------------------------------------
# 9. Evaluation function
# ---------------------------------------------------------

def evaluate_model(
    model_name,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print()
    print("-----", model_name, "-----")

    print(
        "Accuracy:",
        round(accuracy, 3)
    )

    print(
        "Precision:",
        round(precision, 3)
    )

    print(
        "Recall:",
        round(recall, 3)
    )

    print(
        "F1 Score:",
        round(f1, 3)
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 3)
    )


# ---------------------------------------------------------
# 10. Evaluate both models
# ---------------------------------------------------------

evaluate_model(
    "Logistic Regression",
    logistic_predictions,
    logistic_probabilities
)

evaluate_model(
    "Random Forest",
    random_forest_predictions,
    random_forest_probabilities
)


# ---------------------------------------------------------
# 11. Confusion matrix
# ---------------------------------------------------------

confusion = confusion_matrix(
    y_test,
    random_forest_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Churn", "Churn"]
)

display.plot()

plt.title("Random Forest Confusion Matrix")
plt.show()


# ---------------------------------------------------------
# 12. ROC curves
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    logistic_probabilities,
    name="Logistic Regression"
)

RocCurveDisplay.from_predictions(
    y_test,
    random_forest_probabilities,
    name="Random Forest"
)

plt.title("ROC Curve Comparison")

plt.grid(True)

plt.show()


# ---------------------------------------------------------
# 13. Churn distribution
# ---------------------------------------------------------

churn_counts = data["Churn"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    ["No Churn", "Churn"],
    [
        churn_counts.get(0, 0),
        churn_counts.get(1, 0)
    ]
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.grid(axis="y")

plt.show()