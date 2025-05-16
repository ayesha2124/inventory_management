import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("inventory_data_without_supplier_performance.csv")

# Drop rows with missing values in key columns
df = df.dropna(subset=['Quantity', 'RollingAvg_Consumption_7'])

# Create stockout label
df['Stockout_Risk'] = (df['Quantity'] < df['RollingAvg_Consumption_7']).astype(int)

# Encode season
le = LabelEncoder()
df['Season'] = le.fit_transform(df['Season'])

# Select features
features = [
    'Consumption_Rate', 'Lag_Quantity', 'Lag_Consumption',
    'RollingAvg_Consumption_7', 'RollingAvg_Consumption_30',
    'IsWeekend', 'Season'
]

X = df[features]
y = df['Stockout_Risk']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoder
joblib.dump(model, 'stockout_model.pkl')
joblib.dump(le, 'season_encoder_stockout.pkl')
