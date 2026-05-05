FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eda ./eda
COPY http-api/http_api ./http_api

EXPOSE 8080

CMD ["uvicorn", "http_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
