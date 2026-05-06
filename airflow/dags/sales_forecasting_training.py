from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG  # type: ignore[import-not-found]
from airflow.operators.bash import BashOperator  # type: ignore[import-not-found]
from airflow.models.param import Param  # type: ignore[import-not-found]


PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
DATA_DIR = os.getenv("DATA_OUTPUT_DIR", f"{PROJECT_ROOT}/reports/eda/data")
FEATURES_DIR = os.getenv("FEATURES_OUTPUT_DIR", f"{PROJECT_ROOT}/reports/eda/features")
MODELS_DIR = os.getenv("MODEL_OUTPUT_DIR", f"{PROJECT_ROOT}/reports/eda/models")


default_args = {
    "owner": "forecast",
    "retries": 1,
}


with DAG(
    dag_id="sales_forecasting_training",
    description="Prepara datos, construye features, entrena modelos y promueve el ganador.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "selection_metric": Param("mae", enum=["mae", "rmse", "mape"], description="Métrica para elegir ganador."),
        "val_ratio": Param(0.2, type="number", minimum=0.05, maximum=0.5),
        "random_state": Param(42, type="integer"),
        "hyperparameters": Param(
            {
                "ridge_alpha": 1.0,
                "random_forest_n_estimators": 300,
                "random_forest_max_depth": None,
                "random_forest_min_samples_split": 2,
                "random_forest_min_samples_leaf": 1,
                "xgboost_n_estimators": 500,
                "xgboost_max_depth": 6,
                "xgboost_learning_rate": 0.05,
                "xgboost_subsample": 0.9,
                "xgboost_colsample_bytree": 0.9,
                "catboost_iterations": 500,
                "catboost_learning_rate": 0.05,
                "catboost_depth": 6,
                "lstm_epochs": 50,
                "lstm_batch_size": 32,
                "lstm_hidden_size": 64,
                "lstm_num_layers": 2,
                "lstm_dropout": 0.2,
                "lstm_learning_rate": 0.001,
                "neuralprophet_epochs": 40,
                "neuralprophet_learning_rate": 1.0,
                "tft_max_encoder_length": 28,
                "tft_max_epochs": 10,
                "tft_batch_size": 32,
                "tft_learning_rate": 0.03,
                "tft_hidden_size": 16,
                "tft_attention_head_size": 2,
                "tft_dropout": 0.1,
                "tft_hidden_continuous_size": 8,
            },
            type=["object", "null"],
            description="Hiperparámetros por modelo. Editar desde Trigger DAG w/ config.",
        ),
    },
    tags=["forecast", "training"],
) as dag:
    build_datasets = BashOperator(
        task_id="build_datasets",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"python -m eda.cli build-datasets "
            f"--output-dir {DATA_DIR} "
            f"--fmt ${{DATASET_FORMAT:-jsonl}}"
        ),
    )

    build_combined_dataset = BashOperator(
        task_id="build_combined_dataset",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"python -m eda.cli analyze "
            f"--input-dir {DATA_DIR} "
            f"--plots-dir ${{PLOTS_OUTPUT_DIR:-{PROJECT_ROOT}/reports/eda/plots}} "
            "--no-with-plots"
        ),
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"python -m eda.cli build-features "
            f"--input-path ${{COMBINED_PATH:-{DATA_DIR}/combined.jsonl}} "
            f"--output-dir {FEATURES_DIR} "
            f"--lags ${{LAGS:-1,7,30}} "
            f"--target-col ${{TARGET_COL:-orders}} "
            f"--window-size ${{WINDOW_SIZE:-28}}"
        ),
    )

    train_and_promote = BashOperator(
        task_id="train_and_promote",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"python -m eda.cli training-dag "
            f"--input-path ${{WINDOWS_PATH:-{FEATURES_DIR}/windows.npz}} "
            f"--output-dir {MODELS_DIR} "
            f"--series-path ${{FEATURES_PATH:-{FEATURES_DIR}/features.jsonl}} "
            f"--target-col ${{TARGET_COL:-orders}} "
            '--val-ratio "{{ params.val_ratio }}" '
            '--random-state "{{ params.random_state }}" '
            '--selection-metric "{{ params.selection_metric }}" '
            '--hyperparameters-json "$HYPERPARAMETERS_JSON"'
        ),
        env={
            "HYPERPARAMETERS_JSON": "{{ params.hyperparameters | tojson }}",
        },
    )

    build_datasets >> build_combined_dataset >> build_features >> train_and_promote
