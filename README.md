## Cardiovascular Disease Prediction
This project is a machine learning application for predicting cardiovascular disease (CVD) based on demographic and health indicators, such as age, blood pressure, cholesterol levels, and lifestyle habits. The goal is to leverage machine learning techniques to assist in early detection and management of cardiovascular disease risk.

## Table of Contents
Project Overview
Features
Installation
Usage
Dataset
Model Training
Results


## Project Overview
This project analyzes health-related features to predict cardiovascular disease using machine learning models. The dataset used for training and testing includes various health and demographic indicators. The project involves data preprocessing, feature engineering, and training different models, with the goal of identifying individuals at risk of cardiovascular disease.

## Features
Data Cleaning: Handles missing values and transforms date features for analysis.
Feature Engineering: Processes health indicators into usable numerical formats.
Model Training: Trains models such as Logistic Regression, Random Forest, and XGBoost.
Prediction and Evaluation: Evaluates model performance using accuracy, F1-score, and AUC metrics.
Web Interface: Interactive Flask-based UI for users to input health data and get a CVD risk prediction.

Installation

Install Dependencies

Make sure you have Python 3.13.0 installed. Then, install the required packages:

pip install -r requirements.txt
Set Up Flask Application

Usage

python main.py

Making Predictions
Input the health and demographic data via the web interface to receive predictions on cardiovascular disease risk.
Dataset
The dataset used in this project includes indicators such as:

Demographic: Age, gender, etc.
Health: Blood pressure, cholesterol levels, etc.
Lifestyle: Smoking habits, physical activity, etc.
Data preprocessing involves handling missing values and transforming datetime columns into numeric formats for analysis.

Model Training
Model training includes steps such as:

Preprocessing: Cleaning the data, handling missing values, and transforming date features.
Feature Scaling: Scaling numerical features to improve model performance.
Model Selection: Testing various models (e.g., Logistic Regression, Random Forest, XGBoost).
Evaluation: Analyzing model performance metrics to select the best model for deployment.
Results
The best model achieved an accuracy of X%, F1-score of Y%, and an AUC of Z% on the test set, making it suitable for predicting cardiovascular disease risk.








