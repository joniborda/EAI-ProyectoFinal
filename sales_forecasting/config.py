import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "postgres")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "postgres")

    # Orders schema
    orders_table: str = os.getenv("ORDERS_TABLE", "orders")
    orders_date_col: str = os.getenv("ORDERS_DATE_COL", "created")
    orders_line_items_col: str = os.getenv("ORDERS_LINE_ITEMS_COL", "line_items")
    item_product_id_key: str = os.getenv("ITEM_PRODUCT_ID_KEY", "product_id")
    item_quantity_key: str = os.getenv("ITEM_QUANTITY_KEY", "quantity")

    model_dir: Path = Path(os.getenv("MODEL_DIR", "models"))

    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def get_settings() -> Settings:
    settings = Settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    return settings


def safe_id_for_path(value: str) -> str:
    # Reemplaza caracteres problemáticos para nombres de archivo
    safe = value.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(":", "-").replace(" ", "_")
    if not safe:
        safe = "unknown"
    return safe[:128]
