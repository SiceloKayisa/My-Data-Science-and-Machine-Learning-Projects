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

- [Recommendations](#-recommendations)

- [Hyperparameter Tuning and Improving The Models Performance](#-Hyperparameter-Tuning-and-Improving-The-Models-Performance)

- [Libraries Used](#-Libraries-Used)

### 🚀 Introduction
This project focuses on building a churn prediction model for a financial institution. Customer churn is a critical issue for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. By identifying customers at high risk of churning, institutions can implement targeted retention strategies.

This project contains a comprehensive Jupyter Notebook that covers the entire machine learning pipeline, from data acquisition and exploratory data analysis to data preprocessing, feature engineering, model building, and evaluation, strictly using the sklearn library.

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


### 📊 Model Perfomance Analysis

In churn prediction, correctly identifying customers who are likely to churn (class 1, the positive class) is often more critical than overall accuracy. Therefore, recall and F1-score for class 1 are particularly important metrics to focus on.

#### 1. Logistic Regression
**Accuracy:** 0.90

- Class 0 (Non-Churn): Very high precision (0.91) and recall (0.99), leading to an excellent F1-score (0.95). It's great at identifying non-churners.

- Class 1 (Churn):

  - Precision: 0.68 (68% of customers predicted to churn actually churned).

  - Recall: 0.22 (Only 22% of actual churners were correctly identified).

  - F1-score: 0.33

Analysis: While overall accuracy is high, Logistic Regression performs poorly in identifying actual churners (low recall for class 1). This model would miss a significant number of customers who are about to churn.

#### 2. Decision Tree
**Accuracy:** 0.84

- Class 0 (Non-Churn): Good precision (0.92) and recall (0.90), with an F1-score of 0.91.

- Class 1 (Churn):

  - Precision: 0.31

  - Recall: 0.34

  - F1-score: 0.32

**Analysis:** Decision Tree has a lower overall accuracy compared to Logistic Regression. It shows slightly better recall for class 1 than Logistic Regression (0.34 vs 0.22), meaning it catches more actual churners, but its precision for class 1 is much lower, indicating more false positives.

#### 3. Random Forest
**Accuracy**: 0.89

- Class 0 (Non-Churn): Strong precision (0.91) and recall (0.97), resulting in an F1-score of 0.94.

- Class 1 (Churn):

  - Precision: 0.57

  - Recall: 0.29

  - F1-score: 0.38

**Analysis:** Random Forest provides a good balance, with high overall accuracy and better precision for class 1 (0.57) compared to Decision Tree, but its recall for churners (0.29) is still relatively low, though better than Logistic Regression. It improves the F1-score for class 1 to 0.38, making it the best so far for the churn class.

#### 4. SVC (Support Vector Classifier)
**Accuracy:** 0.90

- Class 0 (Non-Churn): Excellent precision (0.91) and recall (0.99), with an F1-score of 0.95.

- Class 1 (Churn):

  - Precision: 0.70

  - Recall: 0.24

  - F1-score: 0.36

**Analysis:** SVC achieves similar overall accuracy to Logistic Regression and the highest precision for class 1 (0.70), meaning when it predicts churn, it's quite often correct. However, its recall for class 1 (0.24) is still quite low, similar to Logistic Regression, meaning it misses most actual churners.

#### 5. Gradient Boosting
**Accuracy:** 0.90

- Class 0 (Non-Churn): High precision (0.91) and recall (0.99), leading to an F1-score of 0.95.

- Class 1 (Churn):

  - Precision: 0.68

  - Recall: 0.23

  - F1-score: 0.35

**Analysis:** Gradient Boosting performs very similarly to Logistic Regression in this scenario, with high overall accuracy but low recall for the churn class (0.23).


From this summary, Random Forest has the highest F1-score for class 1, indicating the best balance between precision and recall for churners among the models tested. SVC has the highest precision, but a relatively low recall. Decision Tree has the highest recall, but its precision is quite poor, meaning many of its churn predictions would be false alarms.


### 🎯 Recommendations 
**1. Prioritize Random Forest for initial deployment:** Given its superior F1-score for the churn class (0.38), Random Forest appears to be the most balanced performer for identifying churners effectively in your current setup.

**2. Focus on improving Recall for Class 1:** All models still have relatively low recall for churners (the highest is 0.34 from Decision Tree). This means a large percentage of actual churners are being missed. For a financial institution, missing churners can be costly.

- **Consider techniques for imbalanced data:** If you haven't already, strongly consider using SMOTE (Synthetic Minority Over-sampling Technique) or other over/under-sampling methods from imblearn on your training data. This can significantly boost recall for the minority class.
- **Adjust class weights:** Many sklearn classifiers allow you to adjust class_weight parameters to give more importance to the minority class during training.

- **Optimize for Recall during hyperparameter tuning:** When performing hyperparameter tuning (e.g., with GridSearchCV or RandomizedSearchCV), set scoring='recall' or scoring='f1' (if you want to balance precision and recall) for the churn class.

**3. Explore Threshold Tuning:** You can adjust the decision threshold for your models. By default, most models classify based on a 0.5 probability. Lowering this threshold might increase recall (catching more churners) at the cost of precision (more false positives), which might be acceptable depending on the business cost of missing a churner versus a false positive.

**4. Feature Importance:** For best performing model I will inspect feature importances (model.feature_importances_) to understand which features are most influential in predicting churn. This can provide actionable business insights.

### 🎯 Hyperparameter Tuning and Improving The Models Performance

Using the above recommendations I will continue to train Random Forest and Decision Trees as they have better performance metrics for our churn problem at hand. I will adjust these models individually to improve their perfomance by exploring the following ways listed below

- Consider the techniques for imbalanced dataset and in this case SMOTE
- Perform hyperparameter tuning using GridSearchCV
- Analyse feature importance and use all the features that are informative for developing the prediction model
- Lastly I will take the better performing model in preparation for deployment.

### Libraries used

- **Pandas:** For loading, cleaning, transforming, and manipulating data. It's the go-to tool for working with tabular data.

- **NumPy:** Used for high-performance numerical operations, especially with multi-dimensional arrays and mathematical functions. It's often the backbone for other libraries like Pandas
- **Seaborn:** For creating beautiful and informative statistical visualizations. It builds on Matplotlib and is perfect for exploring relationships between variables.

- **Scikit-learn:** A comprehensive library for building and evaluating machine learning models. It includes a wide range of algorithms for classification, regression, clustering, and more.

- **Warnings:** To manage and suppress warning messages in your code. This helps keep your output clean, especially when dealing with deprecation warnings from other libraries.

- **Imbalanced-learn (imblearn):** A specialized library for handling imbalanced datasets. It provides resampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution, which is crucial for building accurate predictive models.

  
