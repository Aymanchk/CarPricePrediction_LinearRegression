import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("CarPrice_Assignment.csv")

df['brand'] = df['CarName'].apply(lambda x: x.split()[0].lower())

le = LabelEncoder()
for col in df.select_dtypes(include='str').columns:
    df[col] = le.fit_transform (df[col])

X = df[['enginesize', 'horsepower', 'curbweight', 'carwidth',
        'carlength', 'cylindernumber', 'boreratio', 'brand',
        'highwaympg', 'stroke', 'carheight', 'drivewheel', 'aspiration']]

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Predicted Prices: {y_pred[:3]:}")
print(f"Actual Prices: {y_test.values[:3]}")
r2 = r2_score(y_test, y_pred)
print(f"Accuracy: {r2:.2f} ({r2*100:.1f}%)")

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved!")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='purple')
plt.plot([y_test.min(), y_test.max()],
         [y_pred.min(), y_pred.max()],
         color='black', linewidth=2)
plt.xlabel('Actual Price →')
plt.ylabel('Predicted Price')
plt.title('Linear Regression', fontsize=14, fontweight='bold', color='navy')
plt.tight_layout()
plt.show()

