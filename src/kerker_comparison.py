""" Compare the implementation of Kerker qspace_preconditioner in real space to reciprocal space
for some trial density.
- Ultimately should give the same result
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

from realspace import sin_wave, expand_density_in_sin, build_density, l_matrix_sin_basis, overlap_matrix_sin_basis


class System:
    """1D system"""
    def __init__(self, system_length: float, n_points: int):
        """
        In this case, the grid samples the whole system
        """
        self.length = system_length
        self.n_points = n_points
        self.grid = np.linspace(0, system_length, n_points, endpoint=True)
        self.spacing = self.length / (self.n_points - 1)


def real_space_preconditioning(system: System, rho: np.ndarray, n_basis: int, q0: float) -> np.ndarray:
    """
    Apply preconditioning in real space
    :return:
    """
    rho_coefficients = expand_density_in_sin(n_basis, system.grid, system.length, rho)

    # Construct L and S this basis
    l_matrix = l_matrix_sin_basis(n_basis, system.length)
    s_matrix = overlap_matrix_sin_basis(n_basis, system.length)

    assert l_matrix.shape == (n_basis, n_basis)
    assert s_matrix.shape == (n_basis, n_basis)
    assert scipy.linalg.issymmetric(l_matrix), "l_matrix is not symmetric"

    # Solve Ax = y for x, where A = (L - q0^2 S) rho', y = L rho and x = rho'
    A = (l_matrix - (q0 * q0 * s_matrix))
    y = l_matrix @ rho_coefficients
    preconditioned_rho_coefficients, info = scipy.sparse.linalg.cg(A, y)
    assert info == 0, "CG failed"

    preconditioned_rho = build_density(preconditioned_rho_coefficients, system.grid, system.length)
    return preconditioned_rho


def reciprocal_space_preconditioning(system: System, rho: np.ndarray, q0: float) -> np.ndarray:
    """
    Apply preconditioning in reciprocal space
    Return preconditioned, real-space density

    q0^2 is scaled by (2pi)^2 to account for the convention of the FFTs used in numpy/sypy.
    That it, exp(-2 pi i G' . r), where G' is not the colloquial definition but G' = G / 2pi
    To be consistent with the real space definition, one must either use (G, q0), which would require
     G' -> G, application of preconditioner with q0, then G -> G', or stick with G' and scale q0,
    such that the filter wave lengths agree in both methods. The latter was simpler.

    :return:
    """
    frequencies = np.fft.fftfreq(system.n_points, d=system.spacing)
    rho_q = np.fft.fft(rho)
    G = frequencies**2 / (frequencies**2 + q0**2 / (4 * np.pi * np.pi))
    preconditioned_rho_q = G * rho_q
    preconditioned_rho_realspace = np.fft.ifft(preconditioned_rho_q)
    return preconditioned_rho_realspace


if __name__ == '__main__':
    system = System(system_length=10, n_points=500)

    # Preconditioning constant
    q0 = 10

    # Construct trial density with long and short wave lengths
    rho = sin_wave(1, system.length, system.grid)
    for i in range(5, 30):
        rho += sin_wave(i, system.length, system.grid)

    # 1. Real space treatment
    # Expand the density in a basis of sin functions (kind of trivial expansion)
    # Clearly need to go to the max(n) if the basis is going to be complete.
    n_basis = 30
    pre_rho_real_treatment = real_space_preconditioning(system, rho, n_basis, q0)

    # 2. Preconditioning applied in reciprocal space
    pre_rho_recip_treatment = reciprocal_space_preconditioning(system, rho, q0)

    # 3. Comparison
    plt.plot(system.grid, rho, label='Original')
    plt.plot(system.grid, pre_rho_real_treatment, color='red', linestyle='--', label='Real-space Treatment')
    plt.plot(system.grid, pre_rho_recip_treatment, color='green', linestyle='-.', label='Reciprocal-space Treatment')
    plt.xlabel('x')
    plt.ylabel(r'Preconditioned $\rho(x)$')
    plt.legend()
    plt.savefig('comparison.pdf', dpi=300)

    plt.show()
