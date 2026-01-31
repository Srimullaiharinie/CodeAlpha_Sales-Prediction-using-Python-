import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\srimullai\Downloads\archive (6)\Advertising.csv")
print(data.head())

#To Check missing values
print(data.isnull().sum())

# To Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# To select Feature
X = data.drop("Sales", axis=1)
y = data["Sales"]

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Make preidctions
y_pred = model.predict(X_test)

#Model evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#Advertising Impact Analysis
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Impact": model.coef_
})

print(coefficients.sort_values(by="Impact", ascending=False))

#Visualization
plt.figure()
sns.barplot(x="Impact", y="Feature", data=coefficients)
plt.title("Impact of Advertising on Sales")
plt.show()
