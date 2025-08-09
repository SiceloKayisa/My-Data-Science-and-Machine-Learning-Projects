## Movie Rental Duration Prediction 🎬
This project focuses on predicting the duration of movie rentals using various machine learning regression models. By analyzing historical rental data, we aim to build a robust model that can accurately estimate how long a movie will be rented.

### Project Goal 🎯
The primary objective of this project is to compare the performance of different regression techniques in predicting movie rental duration and identify the most effective model for this task.

### Dataset 📊
The project utilizes a rental_info.csv dataset, which contains information about movie rentals, including rental dates, return dates, and special features of the movies.

**Columns of interest in the dataset:**

`rental_date:` The date when the movie was rented.

`return_date:` The date when the movie was returned.

`special_features:` A string indicating special features (e.g., "Deleted Scenes", "Behind the Scenes").

### Features & Technologies ✨
- **Data Preparation:** Calculating rental duration in days and creating dummy variables for special_features.

- **Feature Engineering:** Deriving rental_length_days, deleted_scenes, and behind_the_scenes.

- **Machine Learning Models:**

  - **Lasso Regression:** Used for feature selection to identify important predictors.

  - **Ordinary Least Squares (OLS) Regression:** Applied to the features selected by Lasso.

  - **Random Forest Regressor:** A powerful ensemble model for robust predictions.

- **Hyperparameter Tuning:** Employing RandomizedSearchCV to optimize the Random Forest model's performance.

- **Model Evaluation:** Using Mean Squared Error (MSE) as the primary metric to assess model accuracy.

- **Libraries:**

  - pandas for data manipulation.

  - numpy for numerical operations.

  - scikit-learn for machine learning models and utilities (train_test_split, mean_squared_error, Lasso, LinearRegression, RandomForestRegressor, RandomizedSearchCV).

### Results 📈
The project compares the Mean Squared Error (MSE) of the combined Lasso + OLS model against the Random Forest Regressor. The notebook will output which model performed better based on this metric, along with its corresponding MSE.

(Based on the provided code, the Random Forest Regressor typically yields a lower MSE, making it the preferred model for this task.)

### Future Enhancements 💡
- Explore more advanced feature engineering, such as incorporating movie genres, audience ratings, or director information.

- Experiment with other regression models (e.g., Gradient Boosting, XGBoost, Support Vector Regressor).

- Perform more extensive hyperparameter tuning using GridSearchCV if computational resources allow.

- Implement cross-validation more broadly across all models for a more robust evaluation.

- Visualize the residuals and actual vs. predicted values to gain deeper insights into model performance.
