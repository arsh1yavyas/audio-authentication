FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY hearsay /app/hearsay
COPY models/hearsay-split.joblib /app/models/hearsay-split.joblib
COPY models/hearsay-optimized.joblib /app/models/hearsay-optimized.joblib

ENTRYPOINT ["python", "-m", "hearsay"]
CMD ["predict", "--model", "/app/models/hearsay-optimized.joblib", "--input", "/data/test", "--output", "/predictions/hearsay_predictions.tsv"]
