# ==========================
# storage/backup_manager.py
# ==========================
# Handles full backup and restore operations.

import subprocess
import os

# ---------------------------------------------
# PostgreSQL Backup (pg_dump)
# ---------------------------------------------
def backup_database(output_file="backup/quantdata_backup.sql"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    user = os.getenv("PG_USER", "postgres")
    db = os.getenv("PG_DB", "quantdata")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    cmd = f"pg_dump -U {user} -h {host} -p {port} -d {db} -F p -f {output_file}"
    print(f"[BACKUP] Executing: {cmd}")
    subprocess.run(cmd, shell=True)

# ---------------------------------------------
# PostgreSQL Restore (psql)
# ---------------------------------------------
def restore_database(input_file="backup/quantdata_backup.sql"):
    user = os.getenv("PG_USER", "postgres")
    db = os.getenv("PG_DB", "quantdata")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    cmd = f"psql -U {user} -h {host} -p {port} -d {db} -f {input_file}"
    print(f"[RESTORE] Executing: {cmd}")
    subprocess.run(cmd, shell=True)

# ---------------------------------------------
# Example Manual Test
# ---------------------------------------------
if __name__ == "__main__":
    backup_database()
    # restore_database()  # Uncomment if needed
