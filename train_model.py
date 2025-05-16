import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load your dataset
df = pd.read_csv("inventory_data_without_supplier_performance.csv")

# Handle missing values if any
df = df.dropna(subset=["Season", "Consumption_Rate", "RollingAvg_Consumption_7", "Quantity"])

# Encode categorical features
label_encoder = LabelEncoder()
df["Season"] = label_encoder.fit_transform(df["Season"])

# Define features and target
features = ["Season", "Consumption_Rate", "RollingAvg_Consumption_7"]
target = "Quantity"

X = df[features]
y = df[target]

# Split into training and test (optional but good for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Validation RMSE: {rmse:.2f}")


# Save the model and encoder
joblib.dump(model, "xgb_model.pkl")
joblib.dump(label_encoder, "season_encoder.pkl")

print("✅ Model and encoder saved successfully.")
