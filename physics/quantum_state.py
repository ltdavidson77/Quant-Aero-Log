# ==========================
# quantum_state.py
# ==========================
# Defines the quantum state class and related functionality.

import numpy as np
import torch
import cupy as cp
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class QuantumState:
    """Represents a quantum state with various properties."""
    # Core quantum properties
    amplitude: np.ndarray  # Complex amplitude vector
    phase: np.ndarray     # Phase angles
    energy: float         # Energy level
    coherence: float      # Coherence measure
    
    # Additional quantum properties
    entanglement: Optional[np.ndarray] = None  # Entanglement matrix
    superposition: Optional[np.ndarray] = None # Superposition coefficients
    interference: Optional[np.ndarray] = None  # Interference pattern
    decoherence: Optional[float] = None        # Decoherence rate
    
    # Advanced quantum properties
    entropy: Optional[float] = None           # Quantum entropy
    resonance: Optional[float] = None         # Resonance frequency
    chaos: Optional[float] = None            # Chaos measure
    fractal: Optional[np.ndarray] = None     # Fractal dimension
    
    # Neural quantum properties
    neural_weights: Optional[torch.Tensor] = None  # Neural network weights
    neural_bias: Optional[torch.Tensor] = None     # Neural network bias
    
    # Evolution properties
    evolution_rate: float = 0.1              # Rate of quantum evolution
    optimization_step: int = 0               # Current optimization step
    learning_rate: float = 0.01              # Learning rate for quantum learning
    
    # Metadata
    metadata: Dict[str, Any] = None          # Additional metadata
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
            
    def to_tensor(self) -> torch.Tensor:
        """Convert quantum state to PyTorch tensor."""
        try:
            # Convert core properties to tensor
            amplitude_tensor = torch.from_numpy(self.amplitude)
            phase_tensor = torch.from_numpy(self.phase)
            
            # Combine into single tensor
            state_tensor = torch.cat([
                amplitude_tensor.real,
                amplitude_tensor.imag,
                phase_tensor
            ])
            
            return state_tensor
        except Exception as e:
            logger.error(f"Error converting state to tensor: {str(e)}")
            return None
            
    def to_cupy(self) -> cp.ndarray:
        """Convert quantum state to CuPy array."""
        try:
            # Convert core properties to CuPy array
            amplitude_cupy = cp.asarray(self.amplitude)
            phase_cupy = cp.asarray(self.phase)
            
            # Combine into single array
            state_cupy = cp.concatenate([
                cp.real(amplitude_cupy),
                cp.imag(amplitude_cupy),
                phase_cupy
            ])
            
            return state_cupy
        except Exception as e:
            logger.error(f"Error converting state to CuPy array: {str(e)}")
            return None
            
    def normalize(self) -> None:
        """Normalize the quantum state."""
        try:
            # Normalize amplitude
            norm = np.linalg.norm(self.amplitude)
            if norm > 0:
                self.amplitude /= norm
                
            # Normalize phase to [0, 2π]
            self.phase = np.mod(self.phase, 2 * np.pi)
            
            # Normalize other properties if they exist
            if self.entanglement is not None:
                self.entanglement = self.entanglement / np.linalg.norm(self.entanglement)
            if self.superposition is not None:
                self.superposition = self.superposition / np.linalg.norm(self.superposition)
            if self.interference is not None:
                self.interference = self.interference / np.linalg.norm(self.interference)
        except Exception as e:
            logger.error(f"Error normalizing state: {str(e)}")
            
    def evolve(self, time_step: float) -> None:
        """Evolve the quantum state over time."""
        try:
            # Update amplitude and phase
            self.amplitude *= np.exp(-1j * self.energy * time_step)
            self.phase += self.energy * time_step
            
            # Update other properties
            if self.entanglement is not None:
                self.entanglement *= np.exp(-self.decoherence * time_step)
            if self.superposition is not None:
                self.superposition *= np.exp(-self.decoherence * time_step)
            if self.interference is not None:
                self.interference *= np.exp(-self.decoherence * time_step)
                
            # Normalize after evolution
            self.normalize()
        except Exception as e:
            logger.error(f"Error evolving state: {str(e)}")
            
    def measure(self) -> Dict[str, Any]:
        """Measure the quantum state and return results."""
        try:
            results = {
                'amplitude': np.abs(self.amplitude),
                'phase': self.phase,
                'energy': self.energy,
                'coherence': self.coherence
            }
            
            # Add other measurements if they exist
            if self.entanglement is not None:
                results['entanglement'] = np.abs(self.entanglement)
            if self.superposition is not None:
                results['superposition'] = np.abs(self.superposition)
            if self.interference is not None:
                results['interference'] = np.abs(self.interference)
            if self.entropy is not None:
                results['entropy'] = self.entropy
            if self.resonance is not None:
                results['resonance'] = self.resonance
            if self.chaos is not None:
                results['chaos'] = self.chaos
                
            return results
        except Exception as e:
            logger.error(f"Error measuring state: {str(e)}")
            return {}
            
    def copy(self) -> 'QuantumState':
        """Create a deep copy of the quantum state."""
        try:
            return QuantumState(
                amplitude=self.amplitude.copy(),
                phase=self.phase.copy(),
                energy=self.energy,
                coherence=self.coherence,
                entanglement=self.entanglement.copy() if self.entanglement is not None else None,
                superposition=self.superposition.copy() if self.superposition is not None else None,
                interference=self.interference.copy() if self.interference is not None else None,
                decoherence=self.decoherence,
                entropy=self.entropy,
                resonance=self.resonance,
                chaos=self.chaos,
                fractal=self.fractal.copy() if self.fractal is not None else None,
                neural_weights=self.neural_weights.clone() if self.neural_weights is not None else None,
                neural_bias=self.neural_bias.clone() if self.neural_bias is not None else None,
                evolution_rate=self.evolution_rate,
                optimization_step=self.optimization_step,
                learning_rate=self.learning_rate,
                metadata=self.metadata.copy()
            )
        except Exception as e:
            logger.error(f"Error copying state: {str(e)}")
            return None 