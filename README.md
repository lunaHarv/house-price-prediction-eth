# 🏡 Ethiopian Real Estate Price Prediction Dashboard

An end-to-end Machine Learning web application designed to estimate residential property prices in Ethiopia using multivariate linear regression. Built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

Valuing real estate in dynamic markets like Ethiopia requires evaluating a blend of structural, geographical, and accessibility metrics. This project analyzes a dataset of 955 residential property records to train a machine learning model capable of predicting house prices (`Price_ETB`). 

The final pipeline achieves an **$R^2$ score of 0.82**, accurately reflecting how site area, building materials, location, and structural characteristics dictate property market values.

---

## 📊 Model Performance & Features

* **Model Type:** Multivariate Linear Regression Pipeline
* **$R^2$ Score:** `0.82` (Explains ~82% of price variance)
* **Mean Absolute Error (MAE):** `~422,815 ETB`
* **Preprocessing:** Standard numeric passthrough & One-Hot Encoding for categorical features

### Input Features Used (11 Total)

| Feature Category | Features Included |
| :--- | :--- |
| **Physical Attributes** | `Built_Area_sqm`, `Site_Area_sqm`, `Number_of_Rooms`, `Property_Years` |
| **Building Specs** | `Construction_Materials` (Concrete, Mud&Wood), `Housing_Typology` (Detached, Semi-detached, Condominium) |
| **Location & Grade** | `Land_Value_Grading` (High, Medium, Low), `Type_of_Nearest_Road` (Asphalt, Gravel) |
| **Accessibility** | `Proximity_to_CBD_km`, `Proximity_to_Bus_Station_km`, `Proximity_to_Schools_km` |

---

## 📁 Repository Structure

```text
house-price-prediction-ethiopia/
├── .gitignore
├── README.md
├── requirements.txt
├── full_house_price_model.pkl
├── app.py
├── houses_improved_data.csv
└── notebooks/
    └── model_training.ipynb
