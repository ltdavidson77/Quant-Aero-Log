# File: java_bridge.py
# Folder: root directory
# ----------------------------------------------
# Python bridge to call and capture Java engine output

import subprocess

def run_model_engine():
    try:
        result = subprocess.run(
            ["java", "-cp", "build", "ModelEngine"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("[PYTHON-JAVA BRIDGE] Java Output:\n")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("[PYTHON-JAVA BRIDGE] Java Error:\n")
        print(e.stderr)

if __name__ == "__main__":
    run_model_engine()
