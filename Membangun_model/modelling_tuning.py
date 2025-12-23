import mlflow
import dagshub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dagshub.init(repo_owner='AlvinMI', repo_name='Eksperimen_SML_AlvinMahesaIrawadi', mlflow=True)

mlflow.set_experiment("Eksperimen_Jantung_Alvin")

print("Sedang memuat data...")
df = pd.read_csv('preprocessing/heart_disease_preprocessing/heart_clean.csv')

X = df.drop(columns=['HeartDisease']) 
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = [50, 100]  

for n_trees in param_grid:
    with mlflow.start_run(run_name=f"Run_Trees_{n_trees}"):
        print(f"Training model dengan {n_trees} trees...")
        
        model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        print(f"Akurasi: {acc}")

        mlflow.log_param("n_estimators", n_trees)
        
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model_random_forest")

        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Trees={n_trees})")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png") 
        mlflow.log_artifact("confusion_matrix.png") 
        plt.close()

        report = classification_report(y_test, predictions)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

print("✅ Selesai! Cek dashboard DagsHub lu sekarang.")