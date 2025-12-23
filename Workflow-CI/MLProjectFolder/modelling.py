import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.autolog()

print("Memuat data...")
df = pd.read_csv('heart_preprocessing.csv')

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Mulai Training...")
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Akurasi: {accuracy_score(y_test, preds)}")

    with open("last_run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("✅ Model selesai dilatih.")