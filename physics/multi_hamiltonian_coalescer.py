# ==========================
# physics/multi_hamiltonian_coalescer.py
# ==========================
# Unified orchestration layer to merge twisted + dissection + optional turbulence Hamiltonians
# Provides a harmonized energy metric for recursive physics analytics.

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from utils.logger import get_logger

from .twisted_hamiltonian import TwistedHamiltonian, scalar_field
from .dissection_hamiltonian import DissectionHamiltonian

logger = get_logger("physics.coalescer")

@dataclass
class CoalescerConfig:
    """Configuration for Hamiltonian coalescing."""
    twist_lambda: float = 1.5
    dissection_pieces: int = 3
    recursive_depth: int = 3
    epsilon: float = 1e-9
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6

class MultiHamiltonianCoalescer:
    """
    Advanced Hamiltonian coalescer for unified energy analysis.
    """
    
    def __init__(self, config: Optional[CoalescerConfig] = None):
        """
        Initialize the multi-Hamiltonian coalescer.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = CoalescerConfig()
        
        self.config = config
        self._validate_config()
        
        # Initialize component Hamiltonians
        self.twisted = TwistedHamiltonian(
            num_points=3,
            twist_lambda=config.twist_lambda
        )
        self.dissection = DissectionHamiltonian(
            num_pieces=config.dissection_pieces
        )
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.twist_lambda <= 0:
            raise ValueError("Twist lambda must be positive")
        if self.config.dissection_pieces < 1:
            raise ValueError("Number of dissection pieces must be positive")
        if self.config.recursive_depth < 0:
            raise ValueError("Recursive depth must be non-negative")
        if self.config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.config.max_iterations < 1:
            raise ValueError("Maximum iterations must be positive")
    
    def evaluate_twisted(self) -> float:
        """
        Evaluate the twisted Hamiltonian component.
        
        Returns:
            Twisted Hamiltonian energy
        """
        x_tf = tf.Variable([[1.0, 0.0], [0.5, 0.866], [0.0, 1.0]], 
                          dtype=tf.float64)
        theta_tf = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)
        
        H_twist = self.twisted.EvaluateHamiltonian(
            scalar_field,
            x_tf,
            theta_tf,
            depth=self.config.recursive_depth
        )
        
        logger.debug("twisted_hamiltonian_evaluated",
                    energy=float(H_twist.numpy()))
        
        return float(H_twist.numpy())
    
    def evaluate_dissection(self) -> float:
        """
        Evaluate the dissection Hamiltonian component.
        
        Returns:
            Dissection Hamiltonian energy
        """
        x_list = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.5, np.sqrt(3)/2])
        ]
        theta = [0.0, np.pi/6, -np.pi/6]
        dtheta_dt = [0.01, 0.02, 0.015]
        kappa = [0.0, 0.0, 0.0]
        d2kappa_dt2 = [0.001, 0.001, 0.001]
        
        H_dissect = self.dissection.EvaluateHamiltonian(
            x_list, theta, dtheta_dt, kappa, d2kappa_dt2
        )
        
        logger.debug("dissection_hamiltonian_evaluated",
                    energy=float(H_dissect))
        
        return float(H_dissect)
    
    def coalesce(self) -> Dict[str, float]:
        """
        Coalesce multiple Hamiltonian components into unified energy metric.
        
        Returns:
            Dictionary of energy components and total
        """
        # Evaluate individual components
        H1 = self.evaluate_twisted()
        H2 = self.evaluate_dissection()
        
        # Calculate weights
        weight_twist = np.log1p(np.abs(H1)) / (self.config.recursive_depth + 
                                              self.config.epsilon)
        weight_dissect = np.log1p(np.abs(H2)) / (self.config.recursive_depth + 
                                                self.config.epsilon)
        
        # Calculate total energy
        total = ((H1 * weight_twist + H2 * weight_dissect) / 
                (weight_twist + weight_dissect + self.config.epsilon))
        
        result = {
            "twisted_energy": H1,
            "dissection_energy": H2,
            "coalesced_energy": total,
            "twist_weight": weight_twist,
            "dissect_weight": weight_dissect
        }
        
        logger.info("hamiltonians_coalesced",
                   twisted_energy=H1,
                   dissection_energy=H2,
                   total_energy=total)
        
        return result
    
    def optimize(self) -> Tuple[Dict[str, float], int]:
        """
        Optimize the coalesced Hamiltonian system.
        
        Returns:
            Tuple of (optimized energies, iterations)
        """
        best_energies = None
        best_iteration = 0
        min_energy = float('inf')
        
        for i in range(self.config.max_iterations):
            # Evaluate current state
            energies = self.coalesce()
            current_energy = energies["coalesced_energy"]
            
            # Update best state
            if current_energy < min_energy:
                min_energy = current_energy
                best_energies = energies
                best_iteration = i
            
            # Check convergence
            if i > 0 and abs(current_energy - min_energy) < self.config.tolerance:
                break
        
        logger.info("optimization_completed",
                   iterations=best_iteration,
                   final_energy=min_energy)
        
        return best_energies, best_iteration

# Example usage
if __name__ == "__main__":
    # Create coalescer with custom configuration
    config = CoalescerConfig(
        twist_lambda=1.5,
        dissection_pieces=3,
        recursive_depth=3
    )
    coalescer = MultiHamiltonianCoalescer(config)
    
    # Coalesce Hamiltonians
    energies = coalescer.coalesce()
    print("Energy Components:")
    for component, value in energies.items():
        print(f"{component}: {value:.6f}")
    
    # Optimize system
    optimized_energies, iterations = coalescer.optimize()
    print("\nOptimized Energies:")
    for component, value in optimized_energies.items():
        print(f"{component}: {value:.6f}")
    print(f"Iterations: {iterations}") 