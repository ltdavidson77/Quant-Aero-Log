# ==========================
# db_connector.py
# ==========================
# Establishes PostgreSQL connection for snapshot and signal storage.

import psycopg2
from psycopg2.extras import execute_values
import os
from config_env import PG_DB, PG_USER, PG_PASS, PG_HOST, PG_PORT, DEBUG_MODE

# ----------------------------------------------
# Establish DB Connection
# ----------------------------------------------
def get_pg_connection():
    try:
        conn = psycopg2.connect(
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS,
            host=PG_HOST,
            port=PG_PORT
        )
        if DEBUG_MODE:
            print("[DB] PostgreSQL connection established.")
        return conn
    except Exception as e:
        print("[DB] Connection Error:", e)
        raise

# ----------------------------------------------
# Close DB Connection
# ----------------------------------------------
def close_pg_connection(conn):
    try:
        conn.close()
        if DEBUG_MODE:
            print("[DB] PostgreSQL connection closed.")
    except Exception as e:
        print("[DB] Error closing connection:", e)
