# ==========================
# __init__.py
# ==========================
# Root package initialization.

import os
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Quant-Aero-Log Team"
__email__ = "support@quant-aero-log.com"
__description__ = "Advanced Quantum Computing Framework for Financial Analysis"

# Import core modules
from physics import (
    QuantumState,
    QuantumStateManager,
    QuantumStateType,
    QuantumStateMetadata,
    QuantumDecoherence,
    QuantumEntropy,
    QuantumResonance,
    QuantumChaos,
    QuantumFractal,
    QuantumNeural,
    QuantumEvolution,
    QuantumOptimization,
    QuantumLearning
)

from models import (
    ModelManager,
    AdvancedEnsemble,
    FinancialMetrics,
    NextGenInferenceMachine
)

from utils import (
    load_config,
    setup_logging
)

from monitoring import PerformanceMonitor

# Export commonly used components
__all__ = [
    # Core components
    'QuantumState',
    'QuantumStateManager',
    'QuantumStateType',
    'QuantumStateMetadata',
    
    # Quantum algorithms
    'QuantumDecoherence',
    'QuantumEntropy',
    'QuantumResonance',
    'QuantumChaos',
    'QuantumFractal',
    'QuantumNeural',
    'QuantumEvolution',
    'QuantumOptimization',
    'QuantumLearning',
    
    # Model components
    'ModelManager',
    'AdvancedEnsemble',
    'FinancialMetrics',
    'NextGenInferenceMachine',
    
    # Utility components
    'load_config',
    'setup_logging',
    'PerformanceMonitor'
] 