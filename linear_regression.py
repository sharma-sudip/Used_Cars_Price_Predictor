# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the Data
data_path = "path_to_your_excel_file.xlsx"
data = pd.read_excel(data_path)

# Step 2: Data Exploration
print(data.head())
print(data.info())
print(data.describe())

# Step 3: Data Preprocessing
# Handle missing values
data = data.dropna()

# Step 4: Feature Selection
# Let's assume 'Year', 'Mileage', 'Condition', 'Maintenance', 'Repair' are relevant features
selected_features = ['Year', 'Mileage', 'Condition', 'Maintenance', 'Repair']

# Step 5: Feature Engineering (Optional)
data['Mileage_Age_Ratio'] = data['Mileage'] / (2023 - data['Year'])

# Step 6: Data Splitting
X = data[selected_features]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Step 9: Model Interpretation (Coefficients)
coefficients = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_})
print(coefficients)

# Step 10: Visualization and Reporting (Optional)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()
