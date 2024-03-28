import lightgbm as lgb  # Importing LightGBM library
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import pandas as pd  # Importing pandas for data manipulation
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)  # Importing evaluation metrics
from sklearn.model_selection import (
    train_test_split,
)  # Importing train_test_split function

# Loading the data
data = pd.read_csv("5y.csv", encoding="gbk")

# List of columns to specify as categorical features
categorical_features = ["航班号", "机型"]

# Preprocessing categorical features and extracting additional features
data["航班日期"] = pd.to_datetime(data["航班日期"]).astype(
    int
)  # Converting date to numerical representation
data["航班号"] = data["航班号"].astype("category").cat.codes  # Encoding categorical feature
data["机型"] = data["机型"].astype("category").cat.codes  # Encoding categorical feature

# Extracting relevant information from datetime features
data["起飞时间"] = pd.to_datetime(data["起飞时间"])
data["到达时间"] = pd.to_datetime(data["到达时间"])
data["起飞小时"] = data["起飞时间"].dt.hour
data["到达小时"] = data["到达时间"].dt.hour

# Dropping unnecessary columns
data.drop(["航班日期", "起飞时间", "到达时间"], axis=1, inplace=True)

# Separating target variable and features
features = data.drop("航线金额", axis=1)
target = data["航线金额"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=False
)

# Converting to LightGBM Dataset
train_data = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_features
)
test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features)

# Model parameters setting
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 50,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

# Model training
model = lgb.train(params, train_data)

# Predicting on test data
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Calculating evaluation metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying results
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.xlabel("Sample")
plt.ylabel("航线金额")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()


if __name__ == "__main__":
    print("end")
