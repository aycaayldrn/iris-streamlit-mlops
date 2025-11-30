# train_model.py
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model accuracy: {accuracy}%")

# 5. Save model
os.makedirs("app", exist_ok=True)
model_path = os.path.join("app", "model.joblib")
dump(model, model_path)
print(f"Model saved to {model_path}")
