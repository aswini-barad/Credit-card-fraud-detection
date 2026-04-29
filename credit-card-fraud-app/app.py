import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
data = pd.read_csv("creditcard.csv")

features = ["Amount","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"]
X = data[features]
y = data["Class"]

# Split data FIRST (important)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply SMOTE only on training data
smote = SMOTE()
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Scale data
scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_sm, y_train_sm)

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Random Forest + SMOTE model saved successfully!")
