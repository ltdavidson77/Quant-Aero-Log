# ==========================
# storage/snapshot_rotator.py
# ==========================
# Maintains rolling snapshot size in database.

from sqlalchemy import create_engine, text
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
# Rotate Snapshots Beyond Max Days
# ---------------------------------------------
def rotate_snapshots(table="signal_snapshots", keep_days=180):
    engine = get_pg_engine()
    query = text(f"""
        DELETE FROM {table}
        WHERE timestamp < NOW() - INTERVAL '{keep_days} days'
    """)
    with engine.begin() as conn:
        conn.execute(query)
    print(f"[DB] Rotated snapshots: kept last {keep_days} days")

# ---------------------------------------------
# Example Manual Test
# ---------------------------------------------
if __name__ == "__main__":
    rotate_snapshots(keep_days=60)
