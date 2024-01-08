"""Reciprocal space application of Kerker preconditioner, based on:
 https://doi.org/10.1016/0927-0256(96)00008-0
 Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set

See eq. 82

Additional reading:
Efficient iteration scheme for self-consistent pseudopotential calculations
"""
import numpy as np
import matplotlib.pyplot as plt


def sin_wave(n: int, l: float, x: np.ndarray) -> np.ndarray:
    """
    Define a 1D sin wave in the range [0, l]
    :param n: Integer wave number
    :param l: System length
    :param x: Grid that defines the sin wave in real space
    :return:
    """
    # TODO Normalise
    return np.sin(2 * np.pi * n * x / l)


def qspace_preconditioner(q: np.ndarray, q0: float) -> np.ndarray:
    """

    :param q:
    :param q0:
    :return:
    """
    return q**2 / (q**2 + q0**2)


if __name__ == '__main__':
    # System
    l = 10.
    x = np.linspace(0, l, 500, endpoint=True)
    dx = l / len(x)

    # Trial charge with long and short wave lengths
    rho = sin_wave(1, l, x)
    for i in range(5, 30):
        rho += sin_wave(i, l, x)

    # FT to frequency space, where d = sample spacing
    frequencies = np.fft.fftfreq(len(x), d=dx)
    rho_fft = np.fft.fft(rho)

    # Filter low q-numbers, where q is the wave number (would be wave vector, if a vector).
    # So this is equivalent to damping the longest wave lengths. In the case of my sin example,
    # these large wave lengths are what result in large peaks at the system boundaries
    # k = 2pi * n / L, such that q = n / L
    #
    # q0 = 5. / l will dampen all frequencies for n = 5 and below, plus some above it
    # Specifically, q = 5 / L will get multiplied by 0.5
    # q0 = 5. / l
    #
    # One can see from the frequency plot that 2.9 is the last low-frequency component with sizeable
    # magnitude. Therefore, setting q0  between 2.9 and 5.8 should remove most of the large peaks at
    # the edges of the system in real space (which are actually a consequence of the Gibbs phenomenon
    # in Fourier series).
    q0 = 2.9
    # For realistic systems, Kresse suggests that 1.5 Angstrom^-1 is always sufficient
    G = qspace_preconditioner(frequencies, q0)
    filtered_rho_fft = G * rho_fft

    # Back FT
    recovered_rho = np.fft.ifft(filtered_rho_fft)

    # Plot each
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(x, rho)
    plt.title(r'$\sum^n \sin\left(\frac{2\pi n x}{L}\right)$')
    plt.xlabel('x')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.stem(np.abs(frequencies), np.abs(rho_fft))
    plt.title('Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 3)
    plt.stem(np.abs(frequencies), G)
    plt.axvline(x=q0, color='red', linestyle='--', label='Wave number filter constant')
    plt.title('Preconditioner')
    plt.xlabel('Frequency')
    plt.ylabel('Preconditioner (arb)')

    plt.subplot(4, 1, 4)
    # Take the real part since there might be small imaginary components due to numerical errors
    plt.plot(x, rho, label='Original density')
    plt.plot(x, recovered_rho, label='Preconditioned density')
    plt.title('Inverse Fourier Transform')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig('pre_reciprocal.pdf', dpi=300)
    plt.show()
