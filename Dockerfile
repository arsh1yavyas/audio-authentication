FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY hearsay /app/hearsay
COPY models/hearsay.joblib /app/models/hearsay.joblib

ENTRYPOINT ["python", "-m", "hearsay"]
CMD ["predict", "--model", "/app/models/hearsay.joblib", "--input", "/data/test", "--output", "/predictions/hearsay_predictions.tsv"]
