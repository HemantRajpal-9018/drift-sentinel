FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY drift_sentinel/ drift_sentinel/

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["uvicorn", "drift_sentinel.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
