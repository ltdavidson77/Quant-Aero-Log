# ==========================
# storage/pg_reader.py
# ==========================
# Retrieves snapshots and historical signal data.

import pandas as pd
from sqlalchemy import create_engine
import os

# ---------------------------------------------
# PostgreSQL Engine Setup
# ---------------------------------------------
def get_pg_engine():
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "password")
    host = os.getenv("PG_HOST", "localhost")
    db = os.getenv("PG_DB", "quantdata")
    port = os.getenv("PG_PORT", "5432")
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

# ---------------------------------------------
# Read Historical Data
# ---------------------------------------------
def load_snapshots(table="signal_snapshots", days=30):
    engine = get_pg_engine()
    query = f"""
        SELECT * FROM {table}
        WHERE timestamp >= NOW() - INTERVAL '{days} days'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, con=engine)
    df.set_index("timestamp", inplace=True)
    return df

# ---------------------------------------------
# Example Manual Test
# ---------------------------------------------
if __name__ == "__main__":
    print("[DB] Loading snapshots from PostgreSQL...")
    df = load_snapshots(days=7)
    print(df.head())
