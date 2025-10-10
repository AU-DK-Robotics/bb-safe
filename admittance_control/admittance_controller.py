# admittance_controller.py
import numpy as np


class ComputeAdmittance:
    def __init__(self, Md, Cd, Kd, dt):
        """
        Initialize the admittance controller.

        Parameters:
          Md: (6x6 numpy array) Virtual inertia matrix.
          Cd: (6x6 numpy array) Virtual damping matrix.
          Kd: (6x6 numpy array) Virtual stiffness matrix.
          dt: (float) Time step for integration.

        The state vector is assumed to be:
            state = [position_offset (6), velocity (6)]
        """
        self._dt = dt
        n = len(Md)
        # Build state-space matrices:
        #   x_dot = A*x + B*f, with x = [position_offset, velocity]
        Mat_A1 = np.zeros((n, n), dtype='float64')
        Mat_A2 = np.eye(n, dtype='float64')
        Mat_A3 = -np.matmul(Kd, np.linalg.inv(Md))
        Mat_A4 = -np.matmul(Cd, np.linalg.inv(Md))
        Mat_A12 = np.concatenate((Mat_A1, Mat_A2), axis=1)
        Mat_A34 = np.concatenate((Mat_A3, Mat_A4), axis=1)
        self.Mat_A = np.concatenate((Mat_A12, Mat_A34), axis=0)
        self.Mat_B = np.concatenate((np.zeros((n, n), dtype='float64'),
                                     np.linalg.inv(Md)), axis=0)

    def _update_matrices(self, NewM, NewC, NewK):
        n = len(NewM)
        Mat_A1 = np.zeros((n, n), dtype='float64')
        Mat_A2 = np.eye(n, dtype='float64')
        Mat_A3 = -np.matmul(NewK, np.linalg.inv(NewM))
        Mat_A4 = -np.matmul(NewC, np.linalg.inv(NewM))
        Mat_A12 = np.concatenate((Mat_A1, Mat_A2), axis=1)
        Mat_A34 = np.concatenate((Mat_A3, Mat_A4), axis=1)
        self.Mat_A = np.concatenate((Mat_A12, Mat_A34), axis=0)
        self.Mat_B = np.concatenate((np.zeros((n, n), dtype='float64'),
                                     np.linalg.inv(NewM)), axis=0)

    def state_fun(self, f, x):
        """
        Computes the derivative of the state.

        Parameters:
          f: (numpy array) external force/torque vector (6 elements).
          x: (numpy array) current state vector (12 elements).

        Returns:
          The derivative dx/dt.
        """
        return np.matmul(self.Mat_A, x) + np.matmul(self.Mat_B, f)

    def __call__(self, tau_ext, state, dt = 0):
        """
        Update the state vector using RK4 integration.

        Parameters:
          tau_ext: (numpy array) External force/torque (6 elements).
          state: (numpy array) Current state vector (12 elements) where
                 the first 6 elements are the position offset and the next 6 are velocity.

        Returns:
          Updated state vector (12 elements).
        """
        if not dt:
          dt = self._dt
        k1 = self.state_fun(tau_ext, state) * dt
        k2 = self.state_fun(tau_ext, state + 0.5 * k1) * dt
        k3 = self.state_fun(tau_ext, state + 0.5 * k2) * dt
        k4 = self.state_fun(tau_ext, state + k3) * dt
        state_new = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return state_new

    def update_matrices(self, M, C, K):
        """
        Update the dynamic matrices.
        """
        self._update_matrices(M, C, K)

    def derivative(self, tau_ext, state):
        """
        Compute the derivative dx/dt of the state using RK4 integration steps.
        (Useful for logging or debugging.)

        Returns:
          The time derivative of the state.
        """
        dt = self._dt
        k1 = self.state_fun(tau_ext, state) * dt
        k2 = self.state_fun(tau_ext, state + 0.5 * k1) * dt
        k3 = self.state_fun(tau_ext, state + 0.5 * k2) * dt
        k4 = self.state_fun(tau_ext, state + k3) * dt
        dydt = (k1 + 2 * k2 + 2 * k3 + k4) / (6.0 * dt)
        return dydt
