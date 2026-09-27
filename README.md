# Customer Churn Prediction

## Overview

This project uses machine learning to predict whether a customer is likely to churn based on customer account, service, and billing information.

The project compares two classification models:

* Logistic Regression
* Random Forest

The models are evaluated using accuracy, precision, recall, F1 score, and ROC-AUC.

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
* Generates a confusion matrix
* Generates ROC curve comparisons
* Visualizes the distribution of customer churn

## Example Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.804 |     0.648 |  0.572 |    0.608 |   0.836 |
| Random Forest       |    0.790 |     0.634 |  0.492 |    0.554 |   0.814 |

These results are based on one train/test split with a fixed random seed. Model performance can vary with different datasets, preprocessing choices, and train/test splits.

## Visualizations

The program generates:

1. Customer churn distribution
2. Random Forest confusion matrix
3. ROC curve comparison between the two models

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

## Disclaimer

This project is intended for educational and portfolio purposes. Model predictions depend on the dataset and modeling choices and should not be treated as guaranteed predictions about individual customers.
