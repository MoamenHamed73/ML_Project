# house_price_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load Dataset
df = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\kc_house_data.csv")

# 2. Data Cleaning
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    for col in ["sqft_living", "bedrooms", "bathrooms", "price"]:
        df = remove_outliers(df, col)
    df["bedrooms"] = df["bedrooms"].astype(int)
    return df

df = clean_data(df)

# 3. Feature Selection

X = df[["sqft_living", "bedrooms", "bathrooms"]]
y = df["price"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Square Score: {r2:.2f}")

# 7. Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.tight_layout()
plt.show()