"""Real space application of Kerker preconditioner, based on:

Modelling Simul. Mater. Sci. Eng. 16 (2008) 035004 (11pp) doi:10.1088/0965-0393/16/3/035004
Real-space Kerker method for self-consistent calculation using non-orthogonal basis functions
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def sin_wave(n: int, l: float, x: np.ndarray) -> np.ndarray:
    """
    Define a 1D sin wave in the range [0, l]
    :param n: Integer wave number
    :param l: System length
    :param x: Grid that defines the sin wave in real space
    :return:
    """
    n_factor = np.sqrt(2. / l)
    return n_factor * np.sin(2 * np.pi * n * x / l)


def expand_density_in_sin(n_basis: int, x: np.ndarray, l: float, rho: np.ndarray) -> np.ndarray:
    """
    Not sure if this is the optimal implementation
    a_n = <sin(2 pi n x / L) * rho(x)

    :return:
    """
    coeff = np.empty(shape=n_basis)
    for i in range(0, n_basis):
        n = i + 1
        integrand = sin_wave(n, l, x) * rho
        coeff[i] = integrate.simpson(integrand, x)

    # Normalise the coefficients
    # coeff *= 1. / np.linalg.norm(coeff)

    return coeff


def l_matrix_sin_basis(n_basis: int, system_length: float) -> np.ndarray:
    """
    Sin functions are orthogonal, so only define the diagonals.
    Solution of int_0^L sin(2 pi n / L x) d^2/dx^2 sin(2 pi m / L x) dx
    found analytically.

    :param n_basis:
    :param system_length:
    :return:
    """
    l_mat = np.zeros(shape=(n_basis, n_basis))
    for i in range(0, n_basis):
        n = i + 1
        l_mat[i, i] = - (2. / system_length) * (n * np.pi)**2
    return l_mat


def overlap_matrix_sin_basis(n_basis, system_length: float) -> np.ndarray:
    """

    Sin functions are orthogonal, so only define the diagonals.
    Solution is also analytic

    :return:
    """
    S = np.zeros(shape=(n_basis, n_basis))
    np.fill_diagonal(S, 0.5 * system_length)
    return S


if __name__ == '__main__':
    # System
    system_length = 10.
    x = np.linspace(0, system_length, 500, endpoint=True)
    # Check - definition could be off by 1
    dx = system_length / len(x)

    # Construct trial density
    # Trial charge with long and short wave lengths
    rho = sin_wave(1, system_length, x)
    for i in range(5, 30):
        rho += sin_wave(i, system_length, x)

    # Expand the density in a basis of sin functions (kind of trivial expansion)
    # Clearly need to go to the max(n) if the basis is going to be complete.
    n_basis = 30
    rho_coefficients = expand_density_in_sin(n_basis, x, system_length, rho)

    # Rebuild rho(x) and plot, to confirm the expansion has worked
    rebuild_rho = np.zeros_like(x)
    for i in range(0, n_basis):
        n = i + 1
        rebuild_rho += rho_coefficients[i] * sin_wave(n, system_length, x)

    # TODO Need to include normalisation constant in L and S expressions
    plt.plot(x, rho, label='Original')
    plt.plot(x, rebuild_rho, label='Expanded')
    plt.xlabel('x')
    plt.ylabel(r'$\rho(x)$')
    plt.legend()
    plt.show()

    # Construct L, S rho in this basis (rho will be coefficients)
    # l_matrix = l_matrix_sin_basis(n_basis, system_length)
    # s_matrix = overlap_matrix_sin_basis(n_basis, system_length)
    #
    # # Solve Ax = y for x, where A = (L - q0^2 S) rho', y = L rho and x = rho'
    # q0 = 2.9
    # TODO. Check eigenvalues of A == - (|q|^2 + q0^2)
    # A = (l_matrix - q0 * q0 * s_matrix)
    # Might need to transpose rho_coefficients
    # y = l_matrix @ rho_coefficients
    # Add CG call here

    # Check pre-conditioned rho' gives the same result as applying in Fourier space, then FTing back
    # Start by plotting for sanity
