import requests
import json

url = "http://localhost:8090/invocations"

payload = {
    "dataframe_split": {
        "columns": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "data": [[63.0, 1.0, 3.0, 145.0, 233.0, 1.0, 0.0, 150.0, 0.0, 2.3, 0.0, 0.0, 1.0]]
    }
}

headers = {"Content-Type": "application/json"}

try:
    print(f"🚀 Mencoba format MLflow Legacy di: {url}")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("✅ GOL!!! STATUS 200 OK!")
        print(f"Hasil Prediksi: {response.json()}")
    else:
        print(f"❌ Masih Gagal ({response.status_code})")
        print(f"Detail: {response.text}")
        
except Exception as e:
    print(f"❌ Error Koneksi: {e}")