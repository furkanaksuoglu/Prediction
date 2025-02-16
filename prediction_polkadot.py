import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Define the output directory
output_path = "C:/Users/User/OneDrive - metu.edu.tr/Desktop/polkadot/"
os.makedirs(output_path, exist_ok=True)

# Load the CSV file
file_path = os.path.join(output_path, "DOTUSDT_historical_data.csv")
data = pd.read_csv(file_path)

# Feature Engineering
data["ma7"] = data["close"].rolling(window=7).mean()
data["ma21"] = data["close"].rolling(window=21).mean()
data["ema7"] = data["close"].ewm(span=7, adjust=False).mean()
data["ema21"] = data["close"].ewm(span=21, adjust=False).mean()
data["volatility"] = data["close"].rolling(window=7).std()
data["momentum"] = data["close"] - data["close"].shift(4)
data["high_low_diff"] = data["high"] - data["low"]
data["close_open_diff"] = data["close"] - data["open"]

# RSI Calculation
delta = data["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data["rsi"] = 100 - (100 / (1 + rs))

# Normalize Moving Averages (Keeping raw price values)
data["ema7_norm"] = data["ema7"] / data["close"]
data["ema21_norm"] = data["ema21"] / data["close"]

# ✅ Add Lagged Close Prices as Features (Using past 5 closing prices)
for lag in range(1, 6):
    data[f"close_lag{lag}"] = data["close"].shift(lag)

# Drop NaN values (created by moving averages, RSI, and lagged features)
data = data.dropna()

# Define features and target (NO SCALING)
X = data[[
    "volume", "ema7_norm", "ema21_norm", "volatility", "momentum", 
    "high_low_diff", "close_open_diff", "rsi",
    "close_lag1", "close_lag2", "close_lag3", "close_lag4", "close_lag5"
]]
y = data["close"]

# Expanding window validation (to preserve time-series ordering)
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# ✅ Train a More Generalized Random Forest Model
rf = RandomForestRegressor(
    n_estimators=300,      # Balanced tree count
    max_depth=6,           # Prevent overfitting
    min_samples_split=10,  # Prevent deep splits (ensures each node has at least 10 samples before splitting)
    min_samples_leaf=5,    # Forces at least 5 samples per leaf
    max_features=0.5,      # Uses only 50% of features per tree (introduces randomness)
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_val)

# Evaluation
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
evs = explained_variance_score(y_val, y_pred)

# Save evaluation results
metrics_file = os.path.join(output_path, "model_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"Explained Variance Score: {evs:.4f}\n")

print(f"Model evaluation saved to: {metrics_file}")

# Plot actual vs predicted closing prices
plt.figure(figsize=(10, 6))
plt.plot(y_val.values, label="Actual Close Price", color="blue", linestyle="dashed")
plt.plot(y_pred, label="Predicted Close Price", color="red", linestyle="solid")
plt.title("Actual vs Predicted Closing Prices")
plt.xlabel("Index")
plt.ylabel("Closing Price")
plt.legend()
plt.savefig(os.path.join(output_path, "actual_vs_predicted_prices.png"))
plt.close()

# Feature Importance
feature_importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
importance_df.to_csv(os.path.join(output_path, "feature_importance.csv"), index=False)

# Sort and plot feature importance
importance_df = importance_df.sort_values(by="Importance", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="purple")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Prediction")
plt.savefig(os.path.join(output_path, "feature_importance_plot.png"))
plt.close()

print(f"All outputs saved to: {output_path}")
