import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.autolog()

print("Memuat data...")
df = pd.read_csv('preprocessing/heart_disease_preprocessing/heart_clean.csv')

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Basic_Autolog_Model"):
    print("Training model (Basic)...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Selesai! Akurasi: {acc}")

print("✅ modelling.py selesai dijalankan.")