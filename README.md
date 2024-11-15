# Car Dheko - Used Car Price Prediction

## Overview

This project aims to predict the price of used cars based on various features like car model, year of manufacture, engine type, mileage, and more. Using machine learning algorithms, this model provides an estimate of the price of a used car, helping potential buyers and sellers make informed decisions.

The project involves data cleaning, exploratory data analysis (EDA), feature engineering, model development, and deployment of the predictive model using a Streamlit application.

## Key Features

- **Data Cleaning & Preprocessing**: Handles missing values, removes outliers, and transforms data to improve model performance.
- **Exploratory Data Analysis (EDA)**: Analyzes car attributes like make, model, year, mileage, and other features to understand the data better.
- **Model Development**: Implements machine learning algorithms (e.g., Random Forest, Linear Regression) to predict car prices.
- **Model Evaluation**: Assesses model performance using metrics like RMSE, MAE, and R².
- **Streamlit App Deployment**: Deploys the model as a web application using Streamlit for easy user interaction.

## Technologies Used

- **Programming Language**: Python
- **Libraries & Tools**:
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Streamlit
- **Machine Learning Algorithms**:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- **Data Visualization**: Matplotlib, Seaborn

## Data

The dataset used in this project contains various car features, such as:

- `make`, `model`, `year of manufacture`
- `mileage`, `engine displacement`, `fuel type`
- `seats`, `transmission`, `price`

## How It Works

- **Data Cleaning**: The dataset is cleaned to handle missing values, outliers, and irrelevant features. Redundant columns are removed, and necessary transformations are applied to ensure consistency.
- **Feature Engineering**: New features are created from the existing data (e.g., converting textual data into numerical values such as "mileage" and "fuel type").
- **Model Training**: Multiple machine learning models are trained on the cleaned dataset to predict the car price based on the features.
- **Model Evaluation**: The models are evaluated using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) to ensure the best-performing model is used.
- **Streamlit App**: A web application is developed using Streamlit to allow users to input car features and predict the price.

## Model Performance

- **Random Forest**: Achieved an RMSE of X and R² of Y.
- **Linear Regression**: Achieved an RMSE of X and R² of Y.
- **Decision tree Regression**: Achieved an RMSE of X and R² of Y.
- **Gradient Boosting**: Achieved an RMSE of X and R² of Y.

## Future Work

- Improve model performance by exploring additional algorithms (e.g., XGBoost, LightGBM).
- Add more features to enhance prediction accuracy (e.g., car history, location).
- Integrate the model with a web platform to allow users to compare multiple car prices.

