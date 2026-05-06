FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ git libgomp1 libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY airflow/requirements.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt
