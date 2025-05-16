import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("inventory_data_without_supplier_performance.csv")

# Drop rows with missing values in required columns
df = df.dropna(subset=['LeadTime', 'DefectRate', 'OnTimeRate'])

# Convert percentages to fractions if needed
if df['OnTimeRate'].max() > 1.0:
    df['OnTimeRate'] = df['OnTimeRate'] / 100
if df['DefectRate'].max() > 1.0:
    df['DefectRate'] = df['DefectRate'] / 100

# Define classification logic
def classify_performance(row):
    if row['OnTimeRate'] >= 0.90 and row['DefectRate'] <= 0.02 and row['LeadTime'] <= 5:
        return "Good"
    elif row['OnTimeRate'] >= 0.85 and row['DefectRate'] <= 0.05 and row['LeadTime'] <= 7:
        return "Average"
    else:
        return "Poor"

df['SupplierPerformance'] = df.apply(classify_performance, axis=1)

# Check distribution
print("\nClass distribution:")
print(df['SupplierPerformance'].value_counts())

# Encode target
label_encoder = LabelEncoder()
df['SupplierPerformance'] = label_encoder.fit_transform(df['SupplierPerformance'])  # e.g. Good=2, Average=0, Poor=1

# Features and target
X = df[['OnTimeRate', 'DefectRate', 'LeadTime']]
y = df['SupplierPerformance']

# Make sure there are at least 2 classes
if len(set(y)) < 2:
    raise ValueError("Target variable must have at least two classes to train the model.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='mlogloss')  # mlogloss for multi-class
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and encoder
joblib.dump(model, "supplier_model.pkl")
joblib.dump(label_encoder, "supplier_encoder.pkl")
print("\n✅ Model and encoder saved successfully.")
