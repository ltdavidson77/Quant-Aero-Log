# ==========================
# physics/dissection_hamiltonian.py
# ==========================
# Implementation of dissection Hamiltonian for geometric analysis.

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("physics.dissection")

@dataclass
class DissectionConfig:
    """Configuration for dissection Hamiltonian."""
    num_pieces: int = 3
    epsilon: float = 1e-9
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6

class DissectionHamiltonian:
    """
    Advanced dissection Hamiltonian for geometric analysis.
    """
    
    def __init__(self, config: Optional[DissectionConfig] = None):
        """
        Initialize the dissection Hamiltonian.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = DissectionConfig()
        
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.num_pieces < 1:
            raise ValueError("Number of pieces must be positive")
        if self.config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.config.max_iterations < 1:
            raise ValueError("Maximum iterations must be positive")
    
    def _compute_curvature(self,
                         theta: float,
                         dtheta_dt: float,
                         kappa: float,
                         d2kappa_dt2: float) -> float:
        """
        Compute geometric curvature.
        
        Args:
            theta: Angle
            dtheta_dt: Angular velocity
            kappa: Curvature
            d2kappa_dt2: Curvature acceleration
            
        Returns:
            Computed curvature
        """
        return (kappa + dtheta_dt * np.sin(theta) + 
                d2kappa_dt2 * np.cos(theta))
    
    def _compute_energy(self,
                      x: np.ndarray,
                      theta: float,
                      dtheta_dt: float,
                      kappa: float,
                      d2kappa_dt2: float) -> float:
        """
        Compute piece energy.
        
        Args:
            x: Position array
            theta: Angle
            dtheta_dt: Angular velocity
            kappa: Curvature
            d2kappa_dt2: Curvature acceleration
            
        Returns:
            Piece energy
        """
        # Position energy
        pos_energy = np.sum(x**2)
        
        # Angular energy
        ang_energy = theta**2 + dtheta_dt**2
        
        # Curvature energy
        curv = self._compute_curvature(theta, dtheta_dt, kappa, d2kappa_dt2)
        curv_energy = curv**2 + d2kappa_dt2**2
        
        return pos_energy + ang_energy + curv_energy
    
    def EvaluateHamiltonian(self,
                          x_list: List[np.ndarray],
                          theta: List[float],
                          dtheta_dt: List[float],
                          kappa: List[float],
                          d2kappa_dt2: List[float]) -> float:
        """
        Evaluate the dissection Hamiltonian.
        
        Args:
            x_list: List of position arrays
            theta: List of angles
            dtheta_dt: List of angular velocities
            kappa: List of curvatures
            d2kappa_dt2: List of curvature accelerations
            
        Returns:
            Total Hamiltonian energy
        """
        if len(x_list) != self.config.num_pieces:
            raise ValueError("Number of pieces must match configuration")
        
        total_energy = 0.0
        
        for i in range(self.config.num_pieces):
            piece_energy = self._compute_energy(
                x_list[i],
                theta[i],
                dtheta_dt[i],
                kappa[i],
                d2kappa_dt2[i]
            )
            total_energy += piece_energy
        
        logger.debug("dissection_hamiltonian_evaluated",
                    energy=total_energy)
        
        return total_energy
    
    def optimize(self,
                initial_x: List[np.ndarray],
                initial_theta: List[float],
                initial_dtheta_dt: List[float],
                initial_kappa: List[float],
                initial_d2kappa_dt2: List[float]) -> Tuple[List[np.ndarray],
                                                         List[float],
                                                         List[float],
                                                         List[float],
                                                         List[float],
                                                         float]:
        """
        Optimize the dissection Hamiltonian system.
        
        Args:
            initial_x: Initial positions
            initial_theta: Initial angles
            initial_dtheta_dt: Initial angular velocities
            initial_kappa: Initial curvatures
            initial_d2kappa_dt2: Initial curvature accelerations
            
        Returns:
            Tuple of optimized parameters and final energy
        """
        # Initialize variables
        x = [np.array(xi) for xi in initial_x]
        theta = list(initial_theta)
        dtheta_dt = list(initial_dtheta_dt)
        kappa = list(initial_kappa)
        d2kappa_dt2 = list(initial_d2kappa_dt2)
        
        best_energy = float('inf')
        best_state = None
        
        for i in range(self.config.max_iterations):
            # Evaluate current state
            current_energy = self.EvaluateHamiltonian(
                x, theta, dtheta_dt, kappa, d2kappa_dt2
            )
            
            # Update best state
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = (x.copy(), theta.copy(), dtheta_dt.copy(),
                            kappa.copy(), d2kappa_dt2.copy())
            
            # Compute gradients
            dx = self._compute_gradients(x, theta, dtheta_dt, kappa, d2kappa_dt2)
            
            # Update parameters
            for j in range(self.config.num_pieces):
                x[j] -= self.config.learning_rate * dx[0][j]
                theta[j] -= self.config.learning_rate * dx[1][j]
                dtheta_dt[j] -= self.config.learning_rate * dx[2][j]
                kappa[j] -= self.config.learning_rate * dx[3][j]
                d2kappa_dt2[j] -= self.config.learning_rate * dx[4][j]
            
            # Check convergence
            if i > 0 and abs(current_energy - best_energy) < self.config.tolerance:
                break
        
        logger.info("optimization_completed",
                   iterations=i,
                   final_energy=best_energy)
        
        return (*best_state, best_energy)
    
    def _compute_gradients(self,
                         x: List[np.ndarray],
                         theta: List[float],
                         dtheta_dt: List[float],
                         kappa: List[float],
                         d2kappa_dt2: List[float]) -> Tuple[List[np.ndarray],
                                                          List[float],
                                                          List[float],
                                                          List[float],
                                                          List[float]]:
        """
        Compute numerical gradients for all parameters.
        
        Args:
            x: Position arrays
            theta: Angles
            dtheta_dt: Angular velocities
            kappa: Curvatures
            d2kappa_dt2: Curvature accelerations
            
        Returns:
            Tuple of parameter gradients
        """
        h = 1e-5
        dx = [np.zeros_like(xi) for xi in x]
        dtheta = [0.0] * self.config.num_pieces
        ddtheta_dt = [0.0] * self.config.num_pieces
        dkappa = [0.0] * self.config.num_pieces
        dd2kappa_dt2 = [0.0] * self.config.num_pieces
        
        for i in range(self.config.num_pieces):
            # Position gradients
            for j in range(len(x[i])):
                x_plus = [xk.copy() for xk in x]
                x_minus = [xk.copy() for xk in x]
                x_plus[i][j] += h
                x_minus[i][j] -= h
                
                E_plus = self.EvaluateHamiltonian(
                    x_plus, theta, dtheta_dt, kappa, d2kappa_dt2
                )
                E_minus = self.EvaluateHamiltonian(
                    x_minus, theta, dtheta_dt, kappa, d2kappa_dt2
                )
                dx[i][j] = (E_plus - E_minus) / (2 * h)
            
            # Angle gradients
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += h
            theta_minus[i] -= h
            
            E_plus = self.EvaluateHamiltonian(
                x, theta_plus, dtheta_dt, kappa, d2kappa_dt2
            )
            E_minus = self.EvaluateHamiltonian(
                x, theta_minus, dtheta_dt, kappa, d2kappa_dt2
            )
            dtheta[i] = (E_plus - E_minus) / (2 * h)
            
            # Angular velocity gradients
            dtheta_dt_plus = dtheta_dt.copy()
            dtheta_dt_minus = dtheta_dt.copy()
            dtheta_dt_plus[i] += h
            dtheta_dt_minus[i] -= h
            
            E_plus = self.EvaluateHamiltonian(
                x, theta, dtheta_dt_plus, kappa, d2kappa_dt2
            )
            E_minus = self.EvaluateHamiltonian(
                x, theta, dtheta_dt_minus, kappa, d2kappa_dt2
            )
            ddtheta_dt[i] = (E_plus - E_minus) / (2 * h)
            
            # Curvature gradients
            kappa_plus = kappa.copy()
            kappa_minus = kappa.copy()
            kappa_plus[i] += h
            kappa_minus[i] -= h
            
            E_plus = self.EvaluateHamiltonian(
                x, theta, dtheta_dt, kappa_plus, d2kappa_dt2
            )
            E_minus = self.EvaluateHamiltonian(
                x, theta, dtheta_dt, kappa_minus, d2kappa_dt2
            )
            dkappa[i] = (E_plus - E_minus) / (2 * h)
            
            # Curvature acceleration gradients
            d2kappa_dt2_plus = d2kappa_dt2.copy()
            d2kappa_dt2_minus = d2kappa_dt2.copy()
            d2kappa_dt2_plus[i] += h
            d2kappa_dt2_minus[i] -= h
            
            E_plus = self.EvaluateHamiltonian(
                x, theta, dtheta_dt, kappa, d2kappa_dt2_plus
            )
            E_minus = self.EvaluateHamiltonian(
                x, theta, dtheta_dt, kappa, d2kappa_dt2_minus
            )
            dd2kappa_dt2[i] = (E_plus - E_minus) / (2 * h)
        
        return dx, dtheta, ddtheta_dt, dkappa, dd2kappa_dt2

# Example usage
if __name__ == "__main__":
    # Create dissection Hamiltonian
    config = DissectionConfig(num_pieces=3)
    hamiltonian = DissectionHamiltonian(config)
    
    # Generate initial conditions
    initial_x = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, np.sqrt(3)/2])
    ]
    initial_theta = [0.0, np.pi/6, -np.pi/6]
    initial_dtheta_dt = [0.01, 0.02, 0.015]
    initial_kappa = [0.0, 0.0, 0.0]
    initial_d2kappa_dt2 = [0.001, 0.001, 0.001]
    
    # Evaluate initial state
    initial_energy = hamiltonian.EvaluateHamiltonian(
        initial_x,
        initial_theta,
        initial_dtheta_dt,
        initial_kappa,
        initial_d2kappa_dt2
    )
    print(f"Initial energy: {initial_energy:.6f}")
    
    # Optimize system
    x_opt, theta_opt, dtheta_dt_opt, kappa_opt, d2kappa_dt2_opt, final_energy = (
        hamiltonian.optimize(
            initial_x,
            initial_theta,
            initial_dtheta_dt,
            initial_kappa,
            initial_d2kappa_dt2
        )
    )
    print(f"Final energy: {final_energy:.6f}") 