# ==========================
# storage/init_db_schema.py
# ==========================
# Initializes PostgreSQL schema for signal storage.

from sqlalchemy import create_engine, text
import os

# ------------------------------------------
# PostgreSQL Engine Setup
# ------------------------------------------
def get_pg_engine():
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "password")
    host = os.getenv("PG_HOST", "localhost")
    db = os.getenv("PG_DB", "quantdata")
    port = os.getenv("PG_PORT", "5432")
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

# ------------------------------------------
# Create Schema Table
# ------------------------------------------
def initialize_schema(table="signal_snapshots"):
    engine = get_pg_engine()
    schema_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            timestamp TIMESTAMPTZ PRIMARY KEY,
            {', '.join([f'log_{r}m DOUBLE PRECISION, alg_{r}m DOUBLE PRECISION' for r in [1,5,10,60,180,300,600,1440,2880,10080,43200]])[:-1]}
        );
    """)
    with engine.begin() as conn:
        conn.execute(schema_sql)
    print(f"[DB] Initialized schema for table '{table}'")

# ------------------------------------------
# Example Manual Test
# ------------------------------------------
if __name__ == "__main__":
    initialize_schema()

