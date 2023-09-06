# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Load the Data
data_path = "./vehicles.csv"
data = pd.read_excel(data_path)

#Data Exploration
print(data.head())
print(data.info())
print(data.describe())

#Data Preprocessing
# Handle missing values
data = data.dropna()

#Feature Selection
#assume 'Year', 'Mileage', 'Condition', 'Maintenance', 'Repair' are relevant features
selected_features = ['Year', 'Mileage', 'Condition', 'Maintenance', 'Repair']

#Feature Engineering
data['Mileage_Age_Ratio'] = data['Mileage'] / (2023 - data['Year'])

#Data Splitting
X = data[selected_features]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Training can use other models as well if they perform better
#can be used to fine tune the model
model = LinearRegression()

model.fit(X_train, y_train)

#Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

#Model Interpretation
coefficients = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_})
print(coefficients)

#Visualization and Reporting (Optional)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()
