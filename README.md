## Predicción de ventas por producto (orders.line_items)

Proyecto en Python para entrenar modelos SARIMAX y RNN (PyTorch) y predecir cantidad de ventas diarias por producto leyendo desde PostgreSQL (`orders` con `line_items` JSONB). Sin pandas: todo con NumPy/SQL.

### Requisitos
- Python 3.10+
- Docker (PostgreSQL corriendo en `localhost:5432`)
- **Windows**: Microsoft Visual C++ Redistributable (requerido para PyTorch)

### Docker Compose
El proyecto incluye un stack con MinIO, Postgres para MLflow, MLflow Tracking, un job de entrenamiento y una API FastAPI.

```bash
docker compose up --build
```

Servicios:
- FastAPI: http://localhost:8080
- MLflow: http://localhost:5000
- MinIO API: http://localhost:9000
- MinIO Console: http://localhost:9001 (`minio` / `minio123`)
- Airflow: http://localhost:8081 (`admin` / `admin`)
- Postgres de MLflow: `localhost:5433` en el host, `postgres:5432` dentro de Docker

El puerto host de Postgres es `5433` para no pisar una conexión local existente en `5432`.

Endpoints principales de la API:
```bash
curl http://localhost:8080/health
curl http://localhost:8080/metrics
curl -X POST http://localhost:8080/train
curl http://localhost:8080/best-model
curl "http://localhost:8080/predict?days=7"
curl "http://localhost:8080/predict/baseline?days=7"
```

El servicio `trainer` ejecuta `python -m eda.cli training-dag` una vez. El DAG entrena los modelos, compara las métricas, elige el ganador según `SELECTION_METRIC` (`mae` por defecto) y promueve el artefacto a `reports/eda/models/best_model.*` junto con `reports/eda/models/best_model.json`. Si `MLFLOW_TRACKING_URI` está definido, registra métricas y artefactos en MLflow.

Airflow también está disponible como orquestador visual. El DAG `sales_forecasting_training` vive en `airflow/dags/sales_forecasting_training.py` y ejecuta:
`build_datasets` → `build_combined_dataset` → `build_features` → `train_and_promote`.
Para levantarlo:
```bash
docker compose up --build airflow-init airflow-webserver airflow-scheduler
```
Desde la UI de Airflow se puede usar **Trigger DAG w/ config** para cambiar `selection_metric`, `val_ratio`, `random_state` e hiperparámetros de modelos como Random Forest, XGBoost, CatBoost, LSTM, NeuralProphet y TFT. Esos valores se guardan también en MLflow como parámetros `hp_*`.

### Instalación
1. Crear entorno virtual (opcional):
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash/PowerShell)
```
2. **Solo Windows**: Instalar Microsoft Visual C++ Redistributable:
   - PyTorch requiere las bibliotecas de Visual C++ en Windows
   - Descargar e instalar desde: https://aka.ms/vs/16/release/vc_redist.x64.exe
   - O la versión más reciente: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
   - **Reiniciar la terminal después de instalar**

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```
Nota: Si no encuentra las version que terminan con +cu124 instalarlo con el comando
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

4. Variables de entorno:
```bash
cp .env.example .env
# Editar .env
```

### Variables de entorno (DB y JSONB)
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `DB_DRIVER` (default: `psycopg2`)  # driver SQLAlchemy para PostgreSQL
- `DOCKER_DB_HOST` (default: `host.docker.internal`)  # host de la DB visto desde Docker
- `ORDERS_TABLE` (default: `orders`)
- `EVENTS_TABLE` (default: `events`)
- `ORDERS_DATE_COL` (default: `created`)  # columna fecha de la orden (timestamptz)
- `EVENTS_START_DATE_COL` (default: `startDate`)  # fecha de comienzo del evento
- `ORDERS_LINE_ITEMS_COL` (default: `line_items`)  # JSONB array de ítems
- `ITEM_PRODUCT_ID_KEY` (default: `product_id`)  # clave en cada ítem JSON
- `ITEM_QUANTITY_KEY` (default: `quantity`)  # clave en cada ítem JSON
- `MODEL_DIR` (default: `models`)

La tabla esperada se parece a:
```sql
CREATE TABLE public.orders (
  id varchar(255) PRIMARY KEY,
  created timestamptz NOT NULL,
  "totalPrice" float8 DEFAULT '0'::double precision NOT NULL,
  channel varchar(255) NOT NULL,
  "sourceName" varchar(255) NOT NULL,
  tags jsonb DEFAULT '[]'::jsonb NOT NULL,
  "customerId" varchar(255) NOT NULL,
  line_items jsonb NOT NULL DEFAULT '[]'::jsonb,
  -- ... otras columnas
);
CREATE INDEX orders_created ON public.orders USING btree (created);
```

Los `line_items` son un array JSONB; cada elemento debe incluir `product_id` y `quantity` (configurable por env).

### Uso CLI
- Probar conexión a la BD:
```bash
python -m eda.cli test-db
```
- Construir datasets persistidos (JSON Lines por defecto) para análisis offline:
```bash
python -m eda.cli build-datasets --output-dir reports/eda/data --fmt jsonl
# formatos: jsonl | csv | both
```
- Analizar y graficar usando datasets guardados:
```bash
python -m eda.cli analyze --input-dir reports/eda/data

```
- Construir features y ventanas deslizantes para modelos:
```bash
python -m eda.cli build-features \
  --input-path reports/eda/data/combined.jsonl \
  --output-dir reports/eda/features \
  --lags 1,7,30 \
  --target-col orders \
  --window-size 28
```
- Comparar múltiples modelos y elegir el mejor:
```bash
python -m eda.cli compare-models \
  --input-path reports/eda/features/windows.npz \
  --output-dir reports/eda/models \
  --series-path reports/eda/features/features.jsonl \
  --target-col orders \
  --val-ratio 0.2
```
- Entrenar modelo para un producto (o todos):
```bash
# Global agregado (sumas de todos los productos)
python -m sales_forecasting.cli train --model sarimax --product-id global --target both
python -m sales_forecasting.cli train --model rnn --product-id global --target both

# Por producto
python -m sales_forecasting.cli train --model sarimax --product-id 12345 --target quantity
python -m sales_forecasting.cli train --model rnn --product-id 12345 --target totalPrice
```
- Predecir próximos N días:
```bash
# Global (ambos targets)
python -m sales_forecasting.cli predict --model rnn --product-id global --target both --horizon 14

# Por producto (cantidad)
python -m sales_forecasting.cli predict --model sarimax --product-id 12345 --target quantity --horizon 14
```
- Correr API FastAPI (Uvicorn):
```bash
python -m sales_forecasting.cli run_api --port 8000
# GET  /health
# POST /train?model=sarimax&product_id=12345
# GET  /predict?model=sarimax&product_id=12345&horizon=14
```

### Evaluación rápida
```python
from sales_forecasting.evaluation import evaluate
print(evaluate("12345", model="sarimax", test_horizon=14))
```

### Notas
- La serie se agrega por día y se rellenan días sin ventas con 0.
- Features adicionales: lags de `adSpend` y `orders`, `revenue_growth` (pct_change de `totalRevenue`) y señales de inicio de eventos (`event_start`, `event_start_next_1`) usando solo `events.startDate`.
- La ventana deslizante genera `reports/eda/features/windows.npz` con `X`, `y` y `feature_columns`.
- `compare-models` entrena Linear/Ridge/RandomForest, XGBoost, CatBoost y NeuralProphet (si está disponible).
- Archivos de modelo usan IDs saneados (sin `/`, espacios → `_`).
- SARIMAX por defecto usa estacionalidad semanal implícita vía `seasonal_order=(1,0,1,7)`.
- La RNN es una LSTM simple con ventana de 28 días (configurable en código).
- En entrenamiento se excluye automáticamente el día actual si aparece (para evitar datos parciales del día).
- Para pronóstico de ingresos usar `--target totalPrice`; para cantidades `--target quantity`.

### Features del modelo RNN
El modelo RNN utiliza **12 features exógenas** por cada día:
- **Features de órdenes** (6): cantidad de órdenes, clientes únicos, precio promedio, canales, fuentes, promedio de tags
- **Features temporales** (6): día de la semana, día del mes, mes, trimestre, es fin de semana, semana del año

Estas features ayudan al modelo a capturar patrones estacionales y comportamientos relacionados con el calendario.


python -m eda.cli grid-search temporal_fusion_transformer --param-grid-json configs/grids/temporal_fusion_transformer_grid.json
python -m eda.cli grid-search random_forest --param-grid-json configs/grids/random_forest_grid.json
python -m eda.cli grid-search catboost --param-grid-json configs/grids/catboost_grid.json
python -m eda.cli grid-search xgboost --param-grid-json configs/grids/xgboost_grid.json
python -m eda.cli grid-search exponential_smoothing --param-grid-json configs/grids/exponential_smoothing_grid.json
python -m eda.cli grid-search linear_regression --param-grid-json configs/grids/linear_regression_grid.json
python -m eda.cli grid-search lstm --param-grid-json configs/grids/lstm_grid.json
python -m eda.cli grid-search neuralprophet --param-grid-json configs/grids/neuralprophet_grid.json
python -m eda.cli grid-search sarima --param-grid-json configs/grids/sarima_grid.json
python -m eda.cli grid-search ridge --param-grid-json configs/grids/ridge_grid.json


## Graficos 

python -m eda.cli plot-mape-distribution \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --model-name random_forest \
  --metric-col mape \
  --output-path reports/eda/plots/mape_distribution_random_forest.png \
  --no-show

python -m eda.cli plot-mape-distribution \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --model-name baseline_tm7_sw8_blend \
  --metric-col mape \
  --output-path reports/eda/plots/mape_distribution_baseline_tm7_sw8_blend.png \
  --no-show

python -m eda.cli plot-mape-distribution \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --model-name temporal_fusion_transformer \
  --metric-col mape \
  --output-path reports/eda/plots/mape_distribution_temporal_fusion_transformer.png \
  --no-show
  
python -m eda.cli plot-mape-distribution \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --model-name catboost \
  --metric-col mape \
  --output-path reports/eda/plots/mape_distribution_catboost.png \
  --no-show

# True vs prediction (validación); requiere haber corrido compare-models o training-dag antes
python -m eda.cli plot-predictions random_forest \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --output-path reports/eda/plots/true_vs_pred_random_forest.png \
  --target-col orders \
  --no-show

python -m eda.cli plot-predictions temporal_fusion_transformer --no-show

python -m eda.cli plot-predictions temporal_fusion_transformer \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --output-path reports/eda/plots/true_vs_pred_temporal_fusion_transformer.png \
  --target-col orders \
  --no-show

python -m eda.cli plot-predictions sarima \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --output-path reports/eda/plots/true_vs_pred_sarima.png \
  --target-col orders \
  --no-show

python -m eda.cli plot-predictions ridge \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --output-path reports/eda/plots/true_vs_pred_ridge.png \
  --target-col orders \
  --no-show

python -m eda.cli plot-predictions exponential_smoothing \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --output-path reports/eda/plots/true_vs_pred_exponential_smoothing.png \
  --target-col orders \
  --no-show

python -m eda.cli plot-predictions linear_regression \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --output-path reports/eda/plots/true_vs_pred_linear_regression.png \
  --target-col orders \
  --no-show

python -m eda.cli plot-predictions baseline_tm7_sw8_blend \
  --input-path reports/eda/models/mape_distribution.jsonl \
  --output-path reports/eda/plots/true_vs_pred_baseline_tm7_sw8_blend.png \
  --target-col orders \
  --no-show