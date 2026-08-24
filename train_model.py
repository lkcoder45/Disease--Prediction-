import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "symptom_disease.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=["disease"])
le = LabelEncoder()
y = le.fit_transform(df["disease"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_name, best_model, best_acc = None, None, -1

print("\nMODEL COMPARISON")
print("=" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"{name:22s} Test Accuracy: {acc:.4f} | CV: {cv:.4f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

pred = best_model.predict(X_test)
print("\nBEST MODEL:", best_name)
print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

joblib.dump(best_model, os.path.join(MODEL_DIR, "disease_model.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "symptoms.pkl"))

print(f"\nSaved model with test accuracy: {best_acc:.2%}")
print("You can now run: streamlit run app.py")
