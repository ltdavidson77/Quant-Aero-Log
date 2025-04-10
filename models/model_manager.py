# ==========================
# model_manager.py
# ==========================
# Manages model loading, saving, and versioning.

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import logging
import json
import pickle
import hashlib
import os
import time
import gc
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime
import sys

# Add physics directory to path
sys.path.append(str(Path(__file__).parent.parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

logger = logging.getLogger(__name__)

# ... existing code ... 