# Bank Customer Churn Prediction using PyTorch

## Overview
This repository contains a deep learning project aimed at predicting bank customer churn. The goal is to identify customers who are likely to leave the bank, allowing the business to take proactive retention measures. The model is built using a Fully Connected Neural Network (FCNN) in PyTorch.

## Dataset
The project uses the `Churn_Modelling.csv` dataset, which contains 10,000 records of bank customers with features such as:
* **Demographics:** Geography, Gender, Age.
* **Financial Info:** Credit Score, Balance, Estimated Salary.
* **Bank Relationship:** Tenure, Number of Products, Has Credit Card, Is Active Member.
* **Target Variable:** `Exited` (1 = Churned, 0 = Stayed).

## Project Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed data distribution and identified outliers using Seaborn boxplots.
2. **Data Preprocessing:** 
   * Dropped irrelevant columns (RowNumber, CustomerId, Surname).
   * Applied One-Hot Encoding for categorical features.
   * Standardized continuous numerical features using `StandardScaler`.
3. **Handling Class Imbalance:** The dataset was highly imbalanced (~80% stayed, ~20% churned). This was handled mathematically by calculating the `pos_weight` and integrating it directly into the `BCEWithLogitsLoss` function.
4. **Model Architecture:** Built an FCNN using PyTorch with two hidden layers (16 and 8 neurons) using ReLU activations.
5. **Training:** Trained for 200 epochs using the Adam Optimizer.

## Model Evaluation
The model was evaluated with a focus on **Recall**, as identifying actual churners is the primary business objective.
* **Accuracy:** ~75%
* **Recall:** ~77% (Successfully capturing the majority of churning customers)
* **Visualizations:** Included a Confusion Matrix and a Receiver Operating Characteristic (ROC) curve with AUC score calculation.

## Technologies Used
* Python
* PyTorch
* Scikit-Learn
* Pandas
* Matplotlib & Seaborn

## Author
**Rana Magdy**
