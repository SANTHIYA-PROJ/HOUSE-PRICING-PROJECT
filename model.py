import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# === Step 1: Load Dataset ===
df = pd.read_csv('Housing.csv')  # Make sure the dataset file is named correctly and in the same folder

# === Step 2: Encode Categorical Columns ===
label_encoders = {}
categorical_cols = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning',
    'prefarea', 'furnishingstatus'
]

# Create label encoders for categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encoding categorical features
    label_encoders[col] = le  # Storing encoders for later use

# Save encoders to 'encoders.pkl'
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✅ Saved encoders.pkl")

# === Step 3: Define Features (X) and Target (y) ===
X = df.drop('price', axis=1)  # Features (excluding the target 'price')
y = df['price']  # Target (price)

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 split

# === Step 5: Train the XGBoost Model ===
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Train the model
print("✅ Model training complete")

# === Step 6: Save the Trained Model to 'model.pkl' ===
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Saved model.pkl")
