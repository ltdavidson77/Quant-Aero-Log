# ==========================
# hamiltonian_numpy.py
# ==========================
# Implements the recursive dissection Hamiltonian using NumPy.

import numpy as np

class DissectionHamiltonian:
    def __init__(self, num_pieces=3):
        self.num_pieces = num_pieces
        self.epsilon = 1e-9

    def RigidTransform(self, x, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return R @ x

    def Logarithmic_Euclidean(self, x_i, x_j, x_i_prime, x_j_prime):
        norm_original = np.linalg.norm(x_i - x_j) + self.epsilon
        norm_transformed = np.linalg.norm(x_i_prime - x_j_prime) + self.epsilon
        return np.log(norm_original / norm_transformed)

    def Trig_Elasticity(self, theta, dtheta_dt):
        return dtheta_dt * (1 - np.cos(theta)) * np.log(1 + theta**2 + self.epsilon)

    def Curvature_Energy(self, kappa, d2kappa_dt2):
        return (1 / (1 + np.log(1 + kappa**2 + self.epsilon))) * d2kappa_dt2

    def RecursiveTier(self, x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list, depth=3):
        if depth == 0:
            return 0
        sum_terms = 0
        for i in range(self.num_pieces):
            for j in range(i + 1, self.num_pieces):
                x_i, x_j = x_list[i], x_list[j]
                theta_i, theta_j = theta_list[i], theta_list[j]
                x_i_prime = self.RigidTransform(x_i, theta_i)
                x_j_prime = self.RigidTransform(x_j, theta_j)
                log_euclidean = self.Logarithmic_Euclidean(x_i, x_j, x_i_prime, x_j_prime)
                trig_elastic = self.Trig_Elasticity(theta_list[i], dtheta_dt_list[i])
                curvature_energy = self.Curvature_Energy(kappa_list[i], d2kappa_dt2_list[i])
                sum_terms += log_euclidean + trig_elastic + curvature_energy
        return np.log(1 + sum_terms + self.RecursiveTier(
            x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list, depth - 1))

    def EvaluateHamiltonian(self, x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list):
        return self.RecursiveTier(x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list)
