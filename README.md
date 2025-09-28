# 🏠 House Price Prediction

A machine learning project to predict house prices using **Linear Regression** on the [King County Housing Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction).

---

## 📌 Project Overview
1. **Data Loading**  
   - Dataset used: `kc_house_data.csv`  

2. **Data Cleaning**  
   - Removed duplicates and missing values.  
   - Handled outliers using the IQR method.  

3. **Feature Selection**  
   - `sqft_living` (Living area in square feet)  
   - `bedrooms` (Number of bedrooms)  
   - `bathrooms` (Number of bathrooms)  

4. **Model Training**  
   - Built a **Linear Regression** model using scikit-learn.  

5. **Model Evaluation**  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - R² Score  

6. **Visualization**  
   - Scatter plot showing **Actual vs Predicted Prices**.  

---

## ⚙️ Requirements
Install the dependencies before running the project:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
🚀 How to Run

Follow these steps to run the project on your local machine:

Clone the repository

git clone https://github.com/USERNAME/house_price_prediction.git


Navigate into the project folder

cd house_price_prediction


Make sure the dataset kc_house_data.csv is inside the project folder.

Run the script

python house_price_prediction.py
