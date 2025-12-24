# Heart Disease Model Monitoring & Logging

Project ini mengimplementasikan sistem monitoring untuk model machine learning menggunakan stack MLServer, Prometheus, dan Grafana.

## Fitur Utama:
- **Model Serving**: Menggunakan MLServer dengan runtime MLflow.
- **Monitoring**: Metrik dikumpulkan oleh Prometheus dari endpoint internal MLServer (port 8082).
- **Visualization**: Dashboard Grafana untuk memantau trafik request secara real-time.
- **Alerting**: Notifikasi otomatis jika trafik melebihi ambang batas (Threshold > 5 requests).

## Cara Menjalankan:
1. Pastikan Docker sudah berjalan.
2. Jalankan perintah: `docker compose up --build`.
3. Lakukan pengujian dengan: `python inference.py`.
