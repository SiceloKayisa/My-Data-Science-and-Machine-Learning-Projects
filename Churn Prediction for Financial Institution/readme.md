## 📊 Churn Prediction for Financial Institution

### Table of Contents

- [Introduction](#introduction)

- [Dataset](#Dataset)

- [Key Steps](#Key-Steps)

    - [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-(EDA))

    - [Data Cleaning & Feature Engineering](#Data-Cleaning-&-Feature-Engineering)

    - [Model Building & Evaluation](#Model-Building-&-Evaluation)

- [Models Used](#Models-Used)

- [Results](#Results)

- [Libraries Used](#Libraries-Used)

### 🚀 Introduction
This project focuses on building a churn prediction model for a financial institution. Customer churn is a critical issue for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. By identifying customers at high risk of churning, institutions can implement targeted retention strategies.

This repository contains a comprehensive Jupyter Notebook that covers the entire machine learning pipeline, from data acquisition and exploratory data analysis to data preprocessing, feature engineering, model building, and evaluation, strictly using the sklearn library.

### 💾 Dataset
The dataset used in this project is the Bank Marketing Data Set from the UCI Machine Learning Repository. This dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (the y variable), which serves as our churn indicator.

### 🔑 Key Steps

#### Exploratory Data Analysis (EDA)
Initial Data Inspection: Understanding the dataset's dimensions, data types, and checking for missing values.
    
- **Target Variable Analysis:** Visualizing the distribution of churn (y variable) to identify class imbalance.
    
- **Univariate Analysis:** Examining the distributions of individual features (numerical and categorical).
    
- **Bivariate Analysis:** Investigating relationships between features and the target variable to uncover insights into potential churn drivers.

#### Data Cleaning & Feature Engineering ###
  - **Handling Irrelevant Features:** Dropping columns that do not contribute to the prediction.

  - **Categorical Encoding:** Converting categorical features into a numerical format suitable for machine learning models using One-Hot Encoding and Label Encoding.

  - **Feature Scaling:** Applying StandardScaler to numerical features to normalize their range, preventing features with larger values from dominating the model.

  - **Addressing Class Imbalance:** (Optional, but recommended for churn datasets) Utilizing techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the target classes, which can improve model performance for the minority class. Note: SMOTE requires the imblearn library, which is outside sklearn but often used in conjunction.

#### Model Building & Evaluation ###
- **Data Splitting:** Dividing the dataset into training and testing sets.

- **Pipeline Creation:** Constructing sklearn pipelines to streamline preprocessing and model training, ensuring consistency and preventing data leakage.

- **Model Training:** Training various classification algorithms on the preprocessed training data.

- **Performance Evaluation:** Assessing each model's effectiveness using metrics appropriate for imbalanced datasets:

    - Accuracy

    - Precision

    - Recall

    - F1-Score

    - ROC-AUC Score (Receiver Operating Characteristic - Area Under the Curve)

- **Comparison:** Comparing the performance of different models to identify the best-performing one.

### 🤖 Models Used ###
The following sklearn classification algorithms were implemented and evaluated:

  - Logistic Regression

  - Decision Tree Classifier

  - Random Forest Classifier

  - Support Vector Machine (SVC)

  - Gradient Boosting Classifier

  - K-Neighbors Classifier

### 📊 Results
The project provides a comparative analysis of the selected models, highlighting their performance across various metrics. The F1-Score and ROC-AUC are particularly crucial for this imbalanced classification problem, offering a balanced view of precision and recall. The Churn_Prediction_Notebook.ipynb will detail the specific metrics for each model, allowing for an informed decision on the best model for deployment.
