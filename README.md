

# E-commerce Customer Behavior and Churn Forecasting

This project focuses on predicting customer churn for an e-commerce platform by analyzing behavioral, demographic, and transactional data. Using machine learning models, it provides insights into factors that influence churn and supports targeted customer retention strategies.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Objective](#objective)
4. [Installation and Setup](#installation-and-setup)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Data Preprocessing](#data-preprocessing)
7. [Modeling](#modeling)
8. [Evaluation and Results](#evaluation-and-results)
9. [Conclusion and Future Work](#conclusion-and-future-work)
10. [References](#references)

---

## Project Overview
In e-commerce, understanding customer churn is crucial for maintaining revenue and customer satisfaction. This project predicts customer churn using machine learning models trained on customer demographic and behavioral data.

## Dataset
The dataset used is from the [Kaggle Playground Series Season 4 Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1). It includes comprehensive customer data with columns such as:
- **CustomerId**: Unique identifier for each customer
- **Demographics**: `Age`, `Gender`, `Geography`
- **Financial Metrics**: `CreditScore`, `Balance`, `EstimatedSalary`
- **Behavioral Indicators**: `NumOfProducts`, `HasCrCard`, `IsActiveMember`
- **Target Variable**: `Exited` - 1 if the customer has churned, 0 otherwise

The dataset consists of two CSV files:
1. **ecommerce_customer_data_large.csv**
2. **ecommerce_customer_data_custom_ratios.csv**

### Objective
The goal is to predict which customers are likely to churn, using this information to develop effective customer retention strategies.

## Installation and Setup
To replicate the analysis, install the following libraries:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm catboost optuna plotly ipywidgets
```

## Exploratory Data Analysis (EDA)
**Key Findings**:
- **Distribution of `Exited`**: There is an imbalance, with more customers who have not churned than those who have.
- **Age and Churn**: Older customers show a higher tendency to churn.
- **Geography and Churn**: Churn rates vary across geographical locations.
- **Account Balance**: No significant trend found, but it's used for further modeling.

EDA was conducted using visualization libraries like `matplotlib` and `plotly`.

## Data Preprocessing
### Steps Involved:
1. **Handling Missing Values**: Checked for and imputed any missing values.
2. **Encoding Categorical Features**: 
   - **Geography** and **Gender** were encoded using `LabelEncoder`.
3. **Feature Scaling**: Used `RobustScaler` to scale numerical columns to manage outliers.
4. **Handling Class Imbalance**: `RandomUnderSampler` was applied to balance the target classes.

## Modeling
This project implements multiple models to predict churn, leveraging both traditional algorithms and ensemble techniques. Hyperparameter tuning was performed using `Optuna` for each model:

1. **Logistic Regression**: A baseline model to understand linear separability.
2. **Decision Tree**: Simple yet interpretable.
3. **Random Forest**: An ensemble model that improves accuracy through multiple decision trees.
4. **XGBoost**: Known for its effectiveness in handling tabular data.
5. **LightGBM**: A gradient-boosting model optimized for speed and accuracy.
6. **CatBoost**: Particularly effective with categorical features.

## Evaluation and Results
Each model was evaluated using the following metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Ability to correctly identify positive churn cases.
- **Recall**: Ability to capture all positive churn cases.
- **F1 Score**: Balance between precision and recall.

### Results Summary:
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 74.62%   | 74.63%    | 74.62% | 74.61%   |
| Decision Tree       | 72.57%   | 72.57%    | 72.57% | 72.57%   |
| Random Forest       | 79.46%   | 79.48%    | 79.47% | 79.46%   |
| XGBoost             | 80.13%   | 80.14%    | 80.13% | 80.13%   |
| LightGBM            | 80.61%   | 80.62%    | 80.61% | 80.61%   |
| CatBoost            | 80.62%   | 80.63%    | 80.62% | 80.62%   |

**Best Model**: Based on accuracy and F1 score, the model chosen for final deployment was `CatBoost` (with the hyper-parameters: 'depth': 5, 'learning_rate': 0.1052667277943049, 'l2_leaf_reg': 3.775242984771314, 'iterations': 172).

### ROC Curve Analysis
The ROC-AUC score provided additional insights into the model's performance, with the best-performing model showing a balanced trade-off between sensitivity and specificity.

## Conclusion and Future Work
### Key Insights:
- Age and geographic location are strong indicators of customer churn.
- Ensemble models, particularly `XGBoost` and `CatBoost`, provided high accuracy and interpretability.

### Future Enhancements:
1. **Feature Engineering**: Explore additional customer behavioral metrics (e.g., time between purchases).
2. **Real-time Prediction**: Integrate the model into an API for real-time churn prediction.
3. **Explainability**: Use `SHAP` values for further interpretability of the model's predictions.

## References
- [Kaggle Playground Series Season 4 Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1)
- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---
