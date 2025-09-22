
#Customer Churn Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("C:\\Users\\DELL\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 3. Explore Dataset
print(df.head())
print(df.info())
print(df["Churn"].value_counts())  # Target variable (Yes/No)

# 4. Data Preprocessing
# حول Yes/No لـ 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# شيل الأعمدة اللي ملهاش لازمة (ID مثلاً)
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# حول categorical → numerical (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# 5. Split Data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# 10. Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()