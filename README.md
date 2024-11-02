# E-commerce Customer Behavior and Churn Forecasting

This project focuses on predicting customer churn for an e-commerce platform by analyzing behavioral, demographic, and transactional data. Using advanced machine learning models with techniques like hyperparameter tuning and SHAP analysis, it provides insights into factors that influence churn and supports targeted customer retention strategies.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Objective](#objective)
4. [Installation and Setup](#installation-and-setup)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Feature Engineering](#feature-engineering)
7. [Data Preprocessing](#data-preprocessing)
8. [Modeling](#modeling)
9. [Evaluation and Results](#evaluation-and-results)
10. [Conclusion and Future Work](#conclusion-and-future-work)
11. [How to Use This Project](#how-to-use-this-project)
12. [Repository Structure](#repository-structure)
13. [References](#references)
14. [License](#license)
15. [Contact](#contact)

---

## Project Overview

In e-commerce, understanding customer churn is crucial for maintaining revenue and customer satisfaction. This project predicts customer churn using machine learning models trained on customer demographic and behavioral data.

## Dataset

The dataset used is from the [Kaggle Playground Series Season 4 Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1). It includes comprehensive customer data with columns such as:

- **CustomerId**: Unique identifier for each customer
- **Demographics**: `Age`, `Gender`, `Geography`
- **Financial Metrics**: `CreditScore`, `Balance`, `EstimatedSalary`
- **Behavioral Indicators**: `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `Tenure`
- **Target Variable**: `Exited` - 1 if the customer has churned, 0 otherwise

### Objective

The goal is to predict which customers are likely to churn, using this information to develop effective customer retention strategies.

## Installation and Setup

To replicate the analysis, you need the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna plotly ipywidgets shap
```

## Exploratory Data Analysis (EDA)

The exploratory data analysis phase involved a thorough examination of customer characteristics and churn patterns. Key steps and findings include:

1. **Target Variable Analysis (`Exited`)**:
   - **Churn Distribution**: Visualized the distribution of churn, showing a higher proportion of retained customers compared to churned ones, indicating a class imbalance.
   - **Insights**: Highlighted the need for handling techniques during modeling, such as class weighting or resampling.

2. **Univariate Analysis**:
   - **Demographic Variables**:
     - **Age**: Older customers show a higher propensity to churn.
     - **Geography**: Churn rates vary across regions, with certain locations like Germany showing higher churn tendencies.
   - **Behavioral Variables**:
     - **NumOfProducts**: Higher numbers of products correlate with increased churn rates.
     - **IsActiveMember**: Inactive members have a higher likelihood of churn.
   - **Financial Metrics**:
     - **CreditScore and Balance**: No direct relationship with churn observed, but further explored in feature engineering.

3. **Bivariate Analysis**:
   - Analyzed relationships between pairs of variables to gain deeper insights into feature interactions, such as the correlation between `Age` and `Balance` in relation to churn behavior.

4. **Correlation Analysis**:
   - Created a heatmap to visualize correlations among numerical variables.
   - **Findings**: Significant correlation observed between `Balance` and the engineered `Balance_to_Age_Ratio`, leading to the decision to drop `Balance` to avoid multicollinearity.

Visualizations generated during EDA are available in the `__results___files` folder.

## Feature Engineering

### Steps Involved:

1. **Creating New Features**:

   - **Balance_to_Age_Ratio**:
     - **Definition**: The ratio of a customer's account balance to their age.
     - **Purpose**: Captures wealth accumulation relative to age, helping to identify high-value customers.

   - **Products_per_Year**:
     - **Definition**: The average number of products held per year of tenure.
     - **Purpose**: Gauges product engagement over time, identifying customers who are rapidly acquiring products, which may correlate with churn due to potential product overload or dissatisfaction.

2. **Insights from New Features**:

   - **Balance_to_Age_Ratio**: Customers with a high ratio might be less likely to churn, as they have more invested in the platform relative to their age.
   - **Products_per_Year**: A high rate may indicate aggressive upselling or customer enthusiasm, but could also signal risk of churn if customers feel overwhelmed.

## Data Preprocessing

### Steps Involved:

1. **Dropping Irrelevant Columns**:

   - Removed columns that do not contribute to predictive modeling, such as unique identifiers and personal information.
   - **Columns Dropped**: `id`, `CustomerId`, `Surname`.
   - **Reason**: These columns do not provide predictive value and could introduce noise into the model.

2. **Encoding Categorical Features**:

   - **Geography**:
     - Applied **One-Hot Encoding** to convert categorical geographical data into binary variables.
     - **Outcome**: Created separate columns for each geographic region, allowing the model to learn region-specific churn patterns.
   - **Gender**:
     - Used **Binary Encoding** to map gender to numerical values (`Male`: 0, `Female`: 1).
     - **Outcome**: Simplified gender representation for the model while preserving the potential impact of gender on churn.

3. **Feature Scaling**:

   - Applied **RobustScaler** to numerical features to handle outliers effectively.
   - **Reason**: Ensures that features with larger scales do not dominate the model training process and improves the convergence of gradient-based algorithms.

4. **Handling Class Imbalance**:

   - Noted an imbalance in the target variable `Exited` (`0`: 130,113 instances, `1`: 34,921 instances).
   - Applied **RandomUnderSampler** to balance the classes by undersampling the majority class.
   - **Outcome**: Created a balanced dataset with equal instances of churned and non-churned customers, improving the model's ability to learn from both classes.

5. **Correlation Analysis and Feature Selection**:

   - Performed a correlation analysis to identify multicollinearity among features.
   - **Findings**:
     - High correlation between `Balance` and `Balance_to_Age_Ratio`.
     - Decided to drop `Balance` to reduce redundancy and potential overfitting.
   - **Outcome**: Streamlined the feature set to enhance model performance.

## Modeling

Multiple models were implemented to predict churn, leveraging both traditional algorithms and ensemble techniques. Advanced hyperparameter tuning was performed using `Optuna` for the top-performing models.

### Handling Class Imbalance in Modeling

To address class imbalance within the modeling process, class weights were adjusted for each algorithm:

- **Logistic Regression**: Set `class_weight='balanced'`.
- **Decision Tree**: Set `class_weight='balanced'`.
- **Random Forest**: Set `class_weight='balanced'`.
- **XGBoost**: Adjusted `scale_pos_weight` to the ratio of negative to positive classes.
- **LightGBM**: Set `class_weight='balanced'`.
- **CatBoost**: Used `class_weights` parameter with calculated weights based on class distribution.

### Model Training and Evaluation

1. **Data Splitting**:

   - The balanced dataset was split into training and validation sets (80% train, 20% validation).
   - Ensured that both sets are representative of the overall data distribution.

2. **Baseline Models**:

   - **Logistic Regression**: Provided a baseline for linear relationships.
   - **Decision Tree**: Offered interpretability through simple decision rules.
   - **Random Forest**: Improved accuracy using ensemble learning by aggregating multiple decision trees.
   - **XGBoost**: Utilized gradient boosting for handling complex patterns in tabular data.
   - **LightGBM**: Offered speed and efficiency advantages, especially with large datasets.
   - **CatBoost**: Particularly effective with categorical features and handling of default values.

3. **Hyperparameter Tuning with Optuna**:

   - Performed hyperparameter optimization to enhance model performance.
   - **Random Forest**:
     - Tuned parameters included the number of trees, maximum depth, minimum samples for splits and leaves.
     - Achieved optimal settings that balanced bias and variance.
   - **CatBoost**:
     - Tuned parameters such as depth, learning rate, regularization, and iterations.
     - Found the best combination that maximized the ROC-AUC score.

4. **Model Selection**:

   - **Best Model**: The `CatBoostClassifier` with tuned hyperparameters was selected based on its superior performance across evaluation metrics.

### SHAP Analysis

To interpret the model's predictions and understand feature importance, SHAP (SHapley Additive exPlanations) values were utilized.

- **Dependence Plot for `Balance_to_Age_Ratio`**:

  - Showed that customers with a higher ratio tend to have a lower likelihood of churn.
  - Highlighted interactions with other features, such as `NumOfProducts`, indicating that multiple factors contribute to churn risk.
  - **Insight**: Customers with multiple products and a high balance relative to their age require targeted retention strategies.

  <p align="center">
    <img src="SHAP%20Study/shap_dependence_plot.png" alt="SHAP Dependence Plot" width="300"/>
  </p>

- **SHAP Summary Plot**:

  - Visualized the impact of all features on the model output.
  - Identified `NumOfProducts`, `Age`, `IsActiveMember`, and `Geography_Germany` as the most influential features.

  <p align="center">
    <img src="SHAP%20Study/shap_summary_plot.png" alt="SHAP Summary Plot" width="300"/>
  </p>

- **SHAP Bar Plot (Feature Importance)**:

  - Ranked features based on their average absolute SHAP values.
  - Provided actionable insights for prioritizing customer segments in retention strategies.

  <p align="center">
    <img src="SHAP%20Study/shap_bar_plot.png" alt="SHAP Bar Plot" width="300"/>
  </p>

The SHAP plots are available in the `SHAP Study` folder.

## Evaluation and Results

Each model was evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **AUC (Area Under ROC Curve)**: Measures the ability to distinguish between classes.
- **Precision**: The accuracy of positive predictions.
- **Recall**: The ability to find all positive instances.
- **F1 Score**: The harmonic mean of precision and recall.

### Results Summary:

| Model               | Accuracy | AUC    |
|---------------------|----------|--------|
| Logistic Regression | 74.62%   | 80.15% |
| Decision Tree       | 72.57%   | 78.34% |
| Random Forest       | 79.46%   | 85.27% |
| XGBoost             | 80.13%   | 86.10% |
| LightGBM            | 80.61%   | 86.45% |
| **CatBoost**        | **80.62%** | **86.50%** |

**Best Model**: **CatBoostClassifier** with hyperparameters tuned via Optuna.

#### Classification Report for CatBoost:

- **Precision**: 80.63%
- **Recall**: 80.62%
- **F1 Score**: 80.62%

### ROC Curve Analysis

- **CatBoost** achieved the highest ROC-AUC score of **86.50%**, indicating a strong ability to distinguish between churned and non-churned customers.
- The ROC curve showed a balanced trade-off between sensitivity and specificity for the CatBoost model.

## Conclusion and Future Work

### Key Insights:

- **Age and Geographic Location** are strong indicators of customer churn.
- **Financial Ratios** like `Balance_to_Age_Ratio` provide valuable insights into customer behavior.
- **Ensemble Models**, particularly `CatBoost`, offered high accuracy and interpretability.
- **Feature Engineering** significantly improved model performance by introducing meaningful variables.
- **Exploratory Data Analysis** was crucial in guiding feature engineering and model selection.

### Future Enhancements:

1. **Advanced Feature Engineering**:

   - Incorporate temporal features like `Time Since Last Purchase`.
   - Analyze customer interaction data for deeper insights.

2. **Real-time Prediction**:

   - Deploy the model as an API for integration into the e-commerce platform.
   - Enable dynamic updates and real-time churn predictions.

3. **Model Deployment and Monitoring**:

   - Implement continuous monitoring to track model performance over time.
   - Update the model as new data becomes available to maintain accuracy.

## How to Use This Project

### Steps to Use the Project:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/sandeeppandey1108/E-commerce-Customer-Behavior-and-Churn-Forecasting.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd E-commerce-Customer-Behavior-and-Churn-Forecasting
   ```

3. **Install Required Libraries**:

   - Ensure you have the necessary Python libraries installed:

     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna plotly ipywidgets shap
     ```

4. **Access the Dataset**:

   - The dataset is included in the `Data_Set` folder.

5. **Run the Notebook**:

   - Open the Jupyter notebook:

     ```bash
     jupyter notebook
     ```

   - Navigate to `Code.ipynb` and run each cell sequentially.

6. **Explore Visualizations and SHAP Plots**:

   - EDA visualizations are available in the `__results___files` folder.
   - SHAP analysis plots are available in the `SHAP Study` folder.
   - Open and review these files to understand the analysis outcomes.

7. **Power BI Dashboard (Optional)**:

   - The `Power Bi` folder contains a Power BI dashboard (`ISBI_Project.pbix`).
   - Open this file with Power BI Desktop to explore interactive visualizations.

8. **Customize and Experiment**:

   - Modify hyperparameters or try different models in the notebook.
   - Explore additional features or different resampling techniques.

9. **Make Predictions**:

   - Use the trained CatBoost model to make predictions on new data.
   - The notebook includes code to apply the model to the test dataset.

10. **Deployment (Optional)**:

    - To deploy the model, consider using frameworks like Flask or FastAPI.
    - The trained model can be saved and loaded using joblib or pickle.

## Repository Structure

- **Data_Set**: Contains the dataset files (`train.csv`, `test.csv`, `submission.csv`).
- **Power Bi**: Includes the Power BI dashboard file (`ISBI_Project.pbix`) and dataset files.
- **SHAP Study**: Contains SHAP plots in PNG format.
- **__results___files**: Contains all graphs and visualizations produced during EDA and modeling.
- **catboost_info**: Contains CatBoost-related information and files.
- **Code.ipynb**: Jupyter notebook containing the code for data analysis and modeling.
- **LICENSE.md**: Project license (Apache License 2.0).
- **README.md**: Project documentation.

## References

- [Kaggle Playground Series Season 4 Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1)
- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

## License

This project is licensed under the terms of the **Apache License 2.0**. Please refer to the [LICENSE.md](LICENSE.md) file for details.

## Contact

For any questions or collaboration opportunities, please reach out:

- **Emails**:
  - [sandeeppandey00880@gmail.com](mailto:sandeeppandey00880@gmail.com)
  - [alvi2241998@gmail.com](mailto:alvi2241998@gmail.com)
- **LinkedIn**:
  - [Sandeep Pandey](https://www.linkedin.com/in/sandy-pandey/)
  - [Alvi Rownok](https://www.linkedin.com/in/alvi-rownok/)
