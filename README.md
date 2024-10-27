

# **E-commerce Customer Behavior and Churn Forecasting**

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Approach & Methods](#approach--methods)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## **Project Overview**
This project is designed to provide valuable insights into customer behaviors within an E-commerce setting by identifying factors affecting customer churn and buying patterns. Through predictive modeling, we aim to deliver a solution that assists businesses in enhancing customer retention, boosting revenue, and minimizing cart abandonment.

## **Dataset Description**
The dataset used in this project comes from the [Kaggle Playground Series - Season 4, Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1), containing the following primary columns:
- **Customer ID**: Unique identifier for each customer
- **Purchase Date**: Date of customer purchases
- **Product Category**: Category of purchased products
- **Total Purchase Amount**: Total spending per transaction
- **Payment Method, Customer Age, Returns, Churn**: Additional metrics including customer demographics and retention

## **Installation**
To run this project, please ensure you have Python 3.8+ and install the necessary packages using:
```bash
pip install -r requirements.txt
```

#### **Dependencies**
- `pandas` for data manipulation
- `numpy` for numerical operations
- `scikit-learn` for machine learning models
- `plotly` and `ipywidgets` for interactive visualizations

## **Usage**
Clone the repository, set up the environment, and run the Jupyter Notebook for step-by-step analysis and modeling:

```bash
git clone https://github.com/sandeeppandey1108/E-commerce-Customer-Behavior-and-Churn-Forecasting.git
cd E-commerce-Customer-Behavior-and-Churn-Forecasting
jupyter notebook Code.ipynb
```

## **Project Structure**
- **Code.ipynb**: Main Jupyter Notebook with analysis, preprocessing, and model-building code
- **data/**: Contains CSV files for customer data and other resources
- **README.md**: This file, providing an overview and instructions
- **requirements.txt**: List of Python packages to install
- **images/**: (Optional) Any project images for documentation purposes

## **Approach & Methods**
### **Exploratory Data Analysis**
Analyzed customer purchase patterns, churn distribution, and correlations across various factors to uncover underlying trends:
- **Churn Rates**: Proportion of customers likely to churn
- **Purchase Frequency**: Patterns in purchase frequency and value
- **Category Preference**: Insights into product categories contributing to customer retention

### **Machine Learning Models**
We tested and compared several models to predict churn, with the final model being selected based on accuracy and interpretability. Models tested:
1. **Logistic Regression**
2. **Decision Trees**
3. **Random Forest**
4. **Support Vector Machines (SVM)**
5. **Gradient Boosting (XGBoost)**

## **Results**
Using metrics such as accuracy, recall, and F1 score, we evaluated model performance and selected the best model for predicting churn. The insights from this model have applications in customer segmentation, targeting, and personalized retention strategies.

## **Contributing**
Contributions are welcome! Feel free to fork the project, make improvements, and submit a pull request.

