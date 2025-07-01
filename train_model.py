
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'age': [5, 10, 3, 8, 2],
    'stairs': [1, 2, 1, 2, 3],
    'price': [50, 70, 85, 100, 130]  # in lakhs
}

df = pd.DataFrame(data)

# Features and Target
X = df[['area', 'age', 'stairs']]
y = df['price']

# Train Linear Regression model
model = LinearRegression()
# price=w1⋅area+w2⋅age+w3⋅stairs+b
model.fit(X, y)
# Save the trained model
joblib.dump(model, 'lr_model.pkl')
print("Model trained and saved as lr_model.pkl")
