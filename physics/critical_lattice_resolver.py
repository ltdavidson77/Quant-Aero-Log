# ==========================
# physics/critical_lattice_resolver.py
# ==========================
# Recursive lattice simulation based on RH principles (non-quantum).
# Useful for forward/reverse skip tracing and nested logarithmic modeling.

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("physics.lattice")

@dataclass
class LatticeConfig:
    """Configuration for critical lattice simulation."""
    depth: int = 5
    real_start: float = 0.1
    real_end: float = 0.9
    imaginary_range: float = 30
    epsilon: float = 1e-9
    layers: int = 3

class CriticalLatticeResolver:
    """
    Advanced critical lattice resolver with recursive patterns and analysis.
    """
    
    def __init__(self, config: Optional[LatticeConfig] = None):
        """
        Initialize the critical lattice resolver.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = LatticeConfig()
        
        self.config = config
        self._validate_config()
        self.real_line = np.linspace(config.real_start, config.real_end, config.depth)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.depth < 1:
            raise ValueError("Depth must be positive")
        if self.config.real_start >= self.config.real_end:
            raise ValueError("Real start must be less than real end")
        if self.config.imaginary_range <= 0:
            raise ValueError("Imaginary range must be positive")
        if self.config.layers < 1:
            raise ValueError("Number of layers must be positive")
    
    def rh_lattice(self) -> np.ndarray:
        """
        Simulate recursive critical strip approximations.
        
        Returns:
            Array of lattice points with recursive weights
        """
        lattice = []
        for i in range(1, self.config.layers + 1):
            # Generate imaginary components
            imag = np.linspace(-self.config.imaginary_range,
                             self.config.imaginary_range,
                             self.config.depth * i)
            
            # Tile real components to match length
            real = np.tile(self.real_line, len(imag) // self.config.depth + 1)[:len(imag)]
            
            # Create complex lattice layer
            lattice_layer = real + 1j * imag
            
            # Apply recursive weighting
            recursive_weight = np.log1p(np.abs(lattice_layer.imag)) / (i + self.config.epsilon)
            weighted = lattice_layer * (1 - recursive_weight)
            
            lattice.append(weighted)
            
            logger.debug("lattice_layer_generated",
                        layer=i,
                        points=len(weighted),
                        weight_range=(recursive_weight.min(), recursive_weight.max()))
        
        return np.array(lattice)
    
    def compute_density_map(self, lattice: np.ndarray) -> np.ndarray:
        """
        Compute a synthetic density metric across the critical lattice.
        
        Args:
            lattice: Array of lattice points
            
        Returns:
            Density map array
        """
        # Calculate density using complex magnitude
        density = (np.abs(np.real(lattice)) * 
                  np.abs(np.imag(lattice)) / 
                  (1 + np.abs(lattice)**2))
        
        # Normalize density
        density = density / (np.max(density) + self.config.epsilon)
        
        logger.info("density_map_computed",
                   density_range=(density.min(), density.max()))
        
        return density
    
    def analyze_lattice(self, lattice: np.ndarray) -> dict:
        """
        Analyze lattice properties and patterns.
        
        Args:
            lattice: Array of lattice points
            
        Returns:
            Dictionary of analysis metrics
        """
        # Calculate basic statistics
        real_parts = np.real(lattice)
        imag_parts = np.imag(lattice)
        
        return {
            'real_mean': np.mean(real_parts),
            'real_std': np.std(real_parts),
            'imag_mean': np.mean(imag_parts),
            'imag_std': np.std(imag_parts),
            'density_mean': np.mean(self.compute_density_map(lattice)),
            'density_std': np.std(self.compute_density_map(lattice)),
            'zero_crossings': len(np.where(np.diff(np.signbit(real_parts)))[0])
        }

# Example usage
if __name__ == "__main__":
    # Create resolver with custom configuration
    config = LatticeConfig(
        depth=5,
        real_start=0.1,
        real_end=0.9,
        imaginary_range=30,
        layers=3
    )
    resolver = CriticalLatticeResolver(config)
    
    # Generate and analyze lattice
    lattice = resolver.rh_lattice()
    density_map = resolver.compute_density_map(lattice)
    analysis = resolver.analyze_lattice(lattice)
    
    print("Lattice Analysis:")
    for metric, value in analysis.items():
        print(f"{metric}: {value:.4f}") 