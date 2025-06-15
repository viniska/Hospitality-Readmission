import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("data/hospital_data.csv")

# Drop duplicates and nulls
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Label encoding for categorical columns
le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])  # Male=1, Female=0
data["admission_type"] = le.fit_transform(data["admission_type"])  # Emergency, Elective -> 0, 1...
data["readmitted"] = le.fit_transform(data["readmitted"])  # No=0, Yes=1

# Define features and target
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/readmission_model.pkl")
