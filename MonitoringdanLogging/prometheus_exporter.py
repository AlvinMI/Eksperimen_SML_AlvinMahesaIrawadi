# ==============================================================================
# PROMETHEUS EXPORTER LOGIC (MLSERVER BUILT-IN)
# ==============================================================================
# File ini mendokumentasikan bahwa fitur Prometheus Exporter telah dihandle 
# secara native oleh MLServer pada port 8082.
# 
# Detail Konfigurasi:
# 1. Endpoint: http://localhost:8082/metrics
# 2. Scrape Job: Telah dikonfigurasi di prometheus.yml untuk menarik data dari 
#    container 'model_serving' secara berkala (scrape_interval: 5s).
# 3. Metrik Utama: rest_server_request_duration_seconds_count
# ==============================================================================