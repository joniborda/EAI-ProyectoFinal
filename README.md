## Predicción de ventas por producto (orders.line_items)

Proyecto en Python para entrenar modelos SARIMAX y RNN (PyTorch) y predecir cantidad de ventas diarias por producto leyendo desde PostgreSQL (`orders` con `line_items` JSONB). Sin pandas: todo con NumPy/SQL.

### Requisitos
- Python 3.10+
- Docker (PostgreSQL corriendo en `localhost:5432`)
- **Windows**: Microsoft Visual C++ Redistributable (requerido para PyTorch)

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
- `ORDERS_TABLE` (default: `orders`)
- `ORDERS_DATE_COL` (default: `created`)  # columna fecha de la orden (timestamptz)
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
python -m sales_forecasting.cli test-db
```
- Preparación de datos (EDA) antes de entrenar:
```bash
# Global (ambos targets)
python -m sales_forecasting.cli prepare-data --product-id global --target both

# Un producto específico (cantidad)
python -m sales_forecasting.cli prepare-data --product-id 12345 --target quantity

# Todos los productos (ambos targets)
python -m sales_forecasting.cli prepare-data --product-id all --target both
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
