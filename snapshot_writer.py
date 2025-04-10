# ==========================
# snapshot_writer.py
# ==========================
# Writes periodic signal snapshots to PostgreSQL database.

import pandas as pd
from db_connector import get_pg_connection, close_pg_connection
from config_env import DEBUG_MODE

# ----------------------------------------------
# Write Signal Snapshot to PostgreSQL
# ----------------------------------------------
def write_snapshot_to_db(signal_df, table_name='signal_snapshots'):
    conn = None
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()

        signal_df = signal_df.copy()
        signal_df['timestamp'] = signal_df.index.astype(str)
        cols = signal_df.columns.tolist()

        insert_query = f"""
        INSERT INTO {table_name} ({', '.join(cols)})
        VALUES %s
        """

        values = [tuple(row) for row in signal_df.to_numpy()]
        execute_values(cursor, insert_query, values)
        conn.commit()

        if DEBUG_MODE:
            print(f"[DB] Snapshot written to table '{table_name}' with {len(values)} rows.")

    except Exception as e:
        print("[DB] Snapshot writing error:", e)
    finally:
        if conn:
            close_pg_connection(conn)
