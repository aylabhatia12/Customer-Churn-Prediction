# Customer Churn Prediction

## Overview

This project uses machine learning to predict whether a customer is likely to churn based on customer account, service, and billing information.

The project compares two classification models:

* Logistic Regression
* Random Forest

The models are evaluated using accuracy, precision, recall, F1 score, and ROC-AUC. The project also analyzes Random Forest feature importance to identify which variables contribute most to the model's predictions.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Features

* Loads and cleans customer churn data
* Converts categorical variables using one-hot encoding
* Standardizes numerical features
* Splits data into training and testing sets
* Trains Logistic Regression and Random Forest models
* Compares model performance
* Evaluates accuracy, precision, recall, F1 score, and ROC-AUC
* Generates a confusion matrix
* Generates ROC curve comparisons
* Visualizes the distribution of customer churn
* Analyzes Random Forest feature importance

## Example Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.804 |     0.648 |  0.572 |    0.608 |   0.836 |
| Random Forest       |    0.790 |     0.634 |  0.492 |    0.554 |   0.814 |

These results are based on one train/test split with a fixed random seed. Model performance can vary with different datasets, preprocessing choices, and train/test splits.

## Top Random Forest Features

The Random Forest model calculates feature importance to provide insight into which variables contribute most to its predictions.

In the current run, the top features included:

| Feature                          | Importance |
| -------------------------------- | ---------: |
| TotalCharges                     |      0.158 |
| Tenure                           |      0.142 |
| MonthlyCharges                   |      0.136 |
| Contract: Month-to-month         |      0.054 |
| OnlineSecurity: No               |      0.031 |
| Payment Method: Electronic check |      0.030 |
| Internet Service: Fiber optic    |      0.027 |
| Tech Support: No                 |      0.027 |
| Senior Citizen                   |      0.020 |
| Online Backup: No                |      0.019 |

Feature importance values describe the model's use of these variables and should not be interpreted as proof that a feature independently causes customer churn.

## Visualizations

The program generates:

1. Customer churn distribution
2. Random Forest confusion matrix
3. ROC curve comparison between the two models
4. Top 10 Random Forest feature importance

## How to Run

Create and activate a Python virtual environment, install the required packages, and run:

```bash
python src/main.py
```

The program downloads the dataset automatically and trains both models.

## Project Structure

```text
Customer-Churn-Prediction/
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    └── main.py
```

## Machine Learning Workflow

The project follows these steps:

1. Load the customer churn dataset
2. Clean missing and invalid values
3. Remove the customer ID
4. Convert the churn target into binary values
5. Separate features and target
6. Encode categorical variables
7. Standardize numerical variables
8. Split the data into training and testing sets
9. Train Logistic Regression
10. Train Random Forest
11. Evaluate both models
12. Analyze Random Forest feature importance
13. Generate visualizations

## Disclaimer

This project is intended for educational and portfolio purposes. Model predictions depend on the dataset and modeling choices and should not be treated as guaranteed predictions about individual customers.
