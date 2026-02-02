import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#loading dataset
df = pd.read_csv('Walmart.csv')

#first 5 rows
print("First 5 rows:")
print(df.head())

#last 5 rows
print("\nLast 5 rows:")
print(df.tail())

#information
print("\nDataset Info:")
print(df.info())

# Shape
print("\nDataset Shape (rows, columns):")
print(df.shape)

#date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

#sorting dates
df = df.sort_values('Date').reset_index(drop=True)

print("Date conversion and sorting done")

#shape after conversion
print(df.head())


#checking missing value
print("\n Missing values in each column")
print(df.isnull().sum())

print("\nfinal dataset info")
print(df.info())

#sales trend over time
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Weekly_Sales'])
plt.title("Weekly sales trend over time")
plt.xlabel("Date")
plt.ylabel("Weekly sales")
plt.show()

#Holiday vs Non-Holiday sales
plt.figure(figsize=(10, 5))
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title("Holiday vs Non-Holiday sales")
plt.xlabel("Holiday(1=Yes,0=No")
plt.ylabel("Weekly sales")
plt.show()

#monthly sales pattern
df['Month'] = df['Date'].dt.month
plt.figure(figsize=(10, 5))
sns.boxplot(x='Month', y='Weekly_Sales', data=df)
plt.title("Monthly sales distribution")
plt.xlabel("Month")
plt.ylabel("Weekly sales")
plt.show()

#Time-based features

df['Week'] = df['Date'].dt.isocalendar().week
df['Year'] = df['Date'].dt.year
print(df[['Date', 'Week', 'Year']].head(10))

#lag features
df['lag_1'] = df['Weekly_Sales'].shift(1)
df['lag_2'] = df['Weekly_Sales'].shift(2)
print(df[['Weekly_Sales', 'lag_1', 'lag_2']].head(10))

#rolling mean(4 week avg)

df['Rolling_Mean_4'] = df['Weekly_Sales'].rolling(window=4).mean()
print(df[['Weekly_Sales', 'Rolling_Mean_4']].head(10))

#removes rows with missing values

df.dropna(inplace=True)
print("Dataset shape after feature engineering:", df.shape)
print(df.columns)

#train-test split (80% train,20% test)

split_date = df['Date'].quantile(0.80)
train = df[df['Date'] < split_date].copy()
test = df[df['Date'] >= split_date].copy()

print("Train shape:", train.shape)
print("Test shape:", test.shape)

#features

features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Week', 'Month', 'Year', 'lag_1',
            'lag_2', 'Rolling_Mean_4']
X_train = train[features]
X_test = test[features]

#target
y_train = train['Weekly_Sales']
y_test = test['Weekly_Sales']

#linear regression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

#evaluate model performance

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Linear Regression MAE:", round(mae, 2))
print("Linear Regression RMSE:", round(rmse, 2))

#random forest
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)

#train
rf.fit(X_train, y_train)

#predict
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Random Forest MAE:", round(mae_rf, 2))
print("Random Forest RMSE:", round(rmse_rf, 2))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape_rf = mape(y_test, y_pred)

print("Random forest MAPE:", round(mape_rf, 2), "%")
print("Forecast accuracy:", round(100 - mape_rf, 2), "%")

#stockout and overstock

test['Forecast'] = y_pred_rf

#safety stock
test['Safety_Stock'] = 0.20 * test['Forecast']

#reorder point
test['Reorder_Point'] = test['Forecast'] + test['Safety_Stock']

#stockout condition
test['Stockout'] = test['Weekly_Sales'] > test['Reorder_Point']

#overstock condition
test['Overstock'] = test['Reorder_Point'] > (1.65 * test['Weekly_Sales'])

print("Stockout rate:", round(test['Stockout'].mean() * 100, 2), "%")
print("Overstock rate:", round(test['Overstock'].mean() * 100, 2), "%")
