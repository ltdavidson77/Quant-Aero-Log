# ==========================
# storage/pg_writer.py
# ==========================
# Writes signal data to PostgreSQL every 5-minute interval.

import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import os

# -------------------------------
# Setup PostgreSQL connection
# -------------------------------
def get_pg_engine():
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "password")
    host = os.getenv("PG_HOST", "localhost")
    db = os.getenv("PG_DB", "quantdata")
    port = os.getenv("PG_PORT", "5432")
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

# -------------------------------
# Write signals to PostgreSQL
# -------------------------------
def write_snapshot_to_db(df, table="signal_snapshots"):
    engine = get_pg_engine()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.index)
    df.to_sql(table, engine, if_exists="append", index=False)
    print(f"[DB] Snapshot written to '{table}' at {df['timestamp'].iloc[-1]}")

# -------------------------------
# Manual Test Example
# -------------------------------
if __name__ == "__main__":
    from generate_data import get_price_series
    from multi_timeframe import compute_multi_timeframe_signals

    df = get_price_series()
    signals = compute_multi_timeframe_signals(df)
    write_snapshot_to_db(signals)
