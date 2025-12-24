FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/ml/model

RUN pip install --no-cache-dir \
    mlserver \
    mlserver-mlflow \
    scikit-learn \
    joblib \
    pandas

COPY . .

EXPOSE 8080

CMD ["mlserver", "start", "."]