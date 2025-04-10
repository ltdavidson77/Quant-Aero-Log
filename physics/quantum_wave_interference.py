# ==========================
# physics/quantum_wave_interference.py
# ==========================
# Simulated wave interference model using recursive logarithmic-trigonometric functions.
# This is NOT quantum hardware-dependent. It mimics wave mechanics computationally.

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("physics.wave")

@dataclass
class WaveConfig:
    """Configuration for wave interference simulation."""
    num_sources: int = 3
    base_amplitude: float = 1.0
    wavelength: float = 2 * np.pi
    depth: int = 3
    epsilon: float = 1e-9
    domain: Tuple[float, float] = (-10, 10)
    resolution: int = 1000

class WaveInterferenceSimulator:
    """
    Advanced wave interference simulator with recursive patterns and analysis.
    """
    
    def __init__(self, config: Optional[WaveConfig] = None):
        """
        Initialize the wave interference simulator.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = WaveConfig()
        
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.num_sources < 1:
            raise ValueError("Number of sources must be positive")
        if self.config.base_amplitude <= 0:
            raise ValueError("Base amplitude must be positive")
        if self.config.wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        if self.config.depth < 0:
            raise ValueError("Depth must be non-negative")
        if self.config.resolution < 2:
            raise ValueError("Resolution must be at least 2")
    
    def phase_shift(self,
                   x: np.ndarray,
                   k: float,
                   phi: float) -> np.ndarray:
        """
        Compute phase-shifted wave.
        
        Args:
            x: Position array
            k: Wave number
            phi: Phase angle
            
        Returns:
            Phase-shifted wave array
        """
        return (self.config.base_amplitude * 
                np.sin((2 * np.pi * k * x / self.config.wavelength) + phi))
    
    def recursive_interference(self,
                             x_values: np.ndarray,
                             k_values: np.ndarray,
                             phi_values: np.ndarray,
                             current_depth: Optional[int] = None) -> np.ndarray:
        """
        Recursive nested summation of interference patterns.
        
        Args:
            x_values: Position array
            k_values: Wave number array
            phi_values: Phase angle array
            current_depth: Current recursion depth
            
        Returns:
            Combined interference pattern
        """
        if current_depth is None:
            current_depth = self.config.depth
        if current_depth == 0:
            return np.zeros_like(x_values)
        
        # Calculate combined wave
        combined_wave = np.zeros_like(x_values)
        for i in range(self.config.num_sources):
            wave = self.phase_shift(x_values, k_values[i], phi_values[i])
            combined_wave += wave
        
        # Normalize and recurse
        interference_level = (np.log1p(np.abs(combined_wave)) / 
                            (current_depth + self.config.epsilon))
        
        return (interference_level + 
                self.recursive_interference(x_values, k_values, phi_values, 
                                          current_depth - 1))
    
    def generate_wavefield(self,
                          domain: Optional[Tuple[float, float]] = None,
                          resolution: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a 1D wave interference pattern.
        
        Args:
            domain: Optional domain range
            resolution: Optional resolution
            
        Returns:
            Tuple of (x_values, wave_pattern)
        """
        if domain is None:
            domain = self.config.domain
        if resolution is None:
            resolution = self.config.resolution
        
        x_values = np.linspace(domain[0], domain[1], resolution)
        k_values = np.random.uniform(0.8, 1.2, self.config.num_sources)
        phi_values = np.random.uniform(0, 2 * np.pi, self.config.num_sources)
        
        wave_pattern = self.recursive_interference(x_values, k_values, phi_values)
        
        logger.info("wavefield_generated",
                   domain=domain,
                   resolution=resolution,
                   pattern_range=(wave_pattern.min(), wave_pattern.max()))
        
        return x_values, wave_pattern
    
    def analyze_pattern(self,
                       wave_pattern: np.ndarray) -> dict:
        """
        Analyze wave interference pattern.
        
        Args:
            wave_pattern: Generated wave pattern
            
        Returns:
            Dictionary of analysis metrics
        """
        return {
            'max_amplitude': np.max(np.abs(wave_pattern)),
            'min_amplitude': np.min(np.abs(wave_pattern)),
            'mean_amplitude': np.mean(np.abs(wave_pattern)),
            'std_amplitude': np.std(np.abs(wave_pattern)),
            'zero_crossings': len(np.where(np.diff(np.signbit(wave_pattern)))[0])
        }

# Example usage
if __name__ == "__main__":
    # Create simulator with custom configuration
    config = WaveConfig(
        num_sources=3,
        base_amplitude=1.0,
        wavelength=2 * np.pi,
        depth=3
    )
    simulator = WaveInterferenceSimulator(config)
    
    # Generate and analyze wavefield
    x_values, wave_pattern = simulator.generate_wavefield()
    analysis = simulator.analyze_pattern(wave_pattern)
    
    print("Wave Pattern Analysis:")
    for metric, value in analysis.items():
        print(f"{metric}: {value:.4f}") 