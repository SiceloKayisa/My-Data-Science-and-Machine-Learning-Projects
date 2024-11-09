# Sales Predictive Modeling using Linear Regression

## 1. Problem Statement

__Sales__ (in thousands of units) for a particular product as a __function__ of __advertising budgets__ (in thousands of dollars) for _TV, radio, and newspaper media_. Suppose that in our role as __Data Scientist__ we are asked to suggest.

- We want to find a function that given input budgets for TV, radio and newspaper __predicts the output sales__.

- Which media __contribute__ to sales?

- Visualize the __relationship__ between the _features_ and the _response_ using scatter plots.


## 2. Data Loading and Description

The advertising dataset captures sales revenue generated with respect to advertisement spends across multiple channels like radio, tv and newspaper.
- TV        - Spend on TV Advertisements
- Radio     - Spend on radio Advertisements
- Newspaper - Spend on newspaper Advertisements
- Sales     - Sales revenue generated

## 3. Exploratory Data Analysis

- Getting to know our data and it's quality
- Searching for any relationships
- Creating Visualisations to look for some patterns 

## 4. Model Implementation:

- Libraries: Python with scikit learn, pandas, seaborn, numpy and matplotib.pyplot
- Model Training:
  - Linear Regression
  - Ridge Regression (with hyperparameter tuning)
 
- Model Evaluation:
    - MSE and RMSE
    - MAE
    - R-squared
- Performance comparison between linear and ridge regression

 ## 5. Results:

 ![image](https://github.com/user-attachments/assets/e37ef56e-5e20-4b91-b24e-912c1999bb06)

##### Observation

- The diagonal of the above matirx shows the auto-correlation of the variables. It is always 1. You can observe that the correlation between TV and Sales is highest i.e. 0.78 and then between sales and radio i.e. 0.576.
- Correlations can vary from -1 to +1. Closer to +1 means strong positive correlation and close -1 means strong negative correlation. Closer to 0 means not very strongly correlated. variables with strong correlations are mostly probably candidates for model building.

__The below plot shows the visualization of the relationship strength of the above features to sales__

![image](https://github.com/user-attachments/assets/d336d4a5-e19f-4474-a2ca-b032ac6f4829)

- Strong relationship between TV ads and sales
- Weak relationship between Radio ads and sales
- Very weak to no relationship between Newspaper ads and sales

#### Model Results Analysis:

##### Linear Regression

  __y = 2.9 + 0.0468 `*` TV + 0.1785 `*` radio + 0.00258 `*` newspaper__

  - A "unit" increase in TV ad spending is **associated with** a _"0.0468_ unit" increase in Sales.
  - Or more clearly: An additional $1,000 spent on TV ads is **associated with** an increase in sales of ~0.0468 * 1000 = 47 widgets.
 
#### metrics scores:

- __Mean Absolute Error__ <br>
  - MAE for training set is 1.237
  - MAE for test set is 1.349
      

- __Mean Square Error__ <br>
  - MSE for training set is 2.766
  - MSE for test set is 3.21
       

- __Root Mean Squared Error__ <br>
  - RMSE for training set is 1.663
  - RMSE for test set is 1.792
  
- __R Squared:__ <br>
  - The R squared value : 0.723
    
  ##### Our value is very close to 1, recall that when building such model is that our r squared value must be closer to 1.
      
  ##### Ridge Regression
  - Suprisingly so we also accomplished the same __Rsquared value__ of 0.723

    __For more kindly go through my notebook__

    



  
      



