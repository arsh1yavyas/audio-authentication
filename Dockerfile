FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY hearsay /app/hearsay
# Keep the runtime image focused on selectable scoring models, not research artifacts.
COPY models/hearsay-optimized.joblib /app/models/hearsay-optimized.joblib
COPY models/hearsay-split.joblib /app/models/hearsay-split.joblib
COPY models/arshiya_julia_lfcc_lowpass.joblib /app/models/arshiya_julia_lfcc_lowpass.joblib
COPY models/arshiya_lfcc_lowpass_candidate.joblib /app/models/arshiya_lfcc_lowpass_candidate.joblib
COPY models/arshiya_librispeech_julia_lfcc.joblib /app/models/arshiya_librispeech_julia_lfcc.joblib
COPY models/asvspoof5_lfcc_candidate.joblib /app/models/asvspoof5_lfcc_candidate.joblib

ENTRYPOINT ["python", "-m", "hearsay"]
CMD ["predict-lfcc", "--model", "/app/models/asvspoof5_lfcc_candidate.joblib", "--input", "/data/test", "--template", "/data/HGT_Hearsay_score_template.csv", "--output", "/predictions/hearsay_predictions.tsv"]
