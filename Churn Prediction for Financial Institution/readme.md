## 📊 Churn Prediction for Financial Institution

### Table of Contents

- [Introduction](#-Introduction)

- [Dataset](#-Dataset)

- [Key Steps](#-Key-Steps)

    - [Exploratory Data Analysis (EDA)](#-Exploratory-Data-Analysis-(EDA))

    - [Data Cleaning & Feature Engineering](#-Data-Cleaning-&-Feature-Engineering)

    - [Model Building & Evaluation](#-Model-Building-&-Evaluation)

- [Models Used](#-Models-Used)
  
- [Model Perfomance Analysis](#-Model-Perfomance-Analysis)

- [Libraries Used](#-Libraries-Used)

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

### 📊 Model Perfomance Analysis

In churn prediction, correctly identifying customers who are likely to churn (class 1, the positive class) is often more critical than overall accuracy. Therefore, recall and F1-score for class 1 are particularly important metrics to focus on.

#### 1. Logistic Regression
Accuracy: 0.90

- Class 0 (Non-Churn): Very high precision (0.91) and recall (0.99), leading to an excellent F1-score (0.95). It's great at identifying non-churners.

Class 1 (Churn):

Precision: 0.68 (68% of customers predicted to churn actually churned).

Recall: 0.22 (Only 22% of actual churners were correctly identified).

F1-score: 0.33

Analysis: While overall accuracy is high, Logistic Regression performs poorly in identifying actual churners (low recall for class 1). This model would miss a significant number of customers who are about to churn.

#### 2. Decision Tree
Accuracy: 0.84

Class 0 (Non-Churn): Good precision (0.92) and recall (0.90), with an F1-score of 0.91.

Class 1 (Churn):

Precision: 0.31

Recall: 0.34

F1-score: 0.32

Analysis: Decision Tree has a lower overall accuracy compared to Logistic Regression. It shows slightly better recall for class 1 than Logistic Regression (0.34 vs 0.22), meaning it catches more actual churners, but its precision for class 1 is much lower, indicating more false positives.

#### 3. Random Forest
**Accuracy**: 0.89

Class 0 (Non-Churn): Strong precision (0.91) and recall (0.97), resulting in an F1-score of 0.94.

Class 1 (Churn):

Precision: 0.57

Recall: 0.29

F1-score: 0.38

**Analysis:** Random Forest provides a good balance, with high overall accuracy and better precision for class 1 (0.57) compared to Decision Tree, but its recall for churners (0.29) is still relatively low, though better than Logistic Regression. It improves the F1-score for class 1 to 0.38, making it the best so far for the churn class.

#### 4. SVC (Support Vector Classifier)
**Accuracy:** 0.90

Class 0 (Non-Churn): Excellent precision (0.91) and recall (0.99), with an F1-score of 0.95.

Class 1 (Churn):

  - Precision: 0.70

  - Recall: 0.24

  - F1-score: 0.36

**Analysis:** SVC achieves similar overall accuracy to Logistic Regression and the highest precision for class 1 (0.70), meaning when it predicts churn, it's quite often correct. However, its recall for class 1 (0.24) is still quite low, similar to Logistic Regression, meaning it misses most actual churners.

#### 5. Gradient Boosting
**Accuracy:** 0.90

Class 0 (Non-Churn): High precision (0.91) and recall (0.99), leading to an F1-score of 0.95.

Class 1 (Churn):

Precision: 0.68

Recall: 0.23

F1-score: 0.35

**Analysis:** Gradient Boosting performs very similarly to Logistic Regression in this scenario, with high overall accuracy but low recall for the churn class (0.23).


From this summary, Random Forest has the highest F1-score for class 1, indicating the best balance between precision and recall for churners among the models tested. SVC has the highest precision, but a relatively low recall. Decision Tree has the highest recall, but its precision is quite poor, meaning many of its churn predictions would be false alarms.
