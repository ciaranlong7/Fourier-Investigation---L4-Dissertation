import numpy as np
from matplotlib import pyplot as plt
import scipy.fft as fft

##Investigation into the relationship between harmonics in freq domain and attosecond pulses in the time domain:
def modulus(input):
    return np.sqrt(np.real(input)**2+ np.imag(input)**2)

# Define parameters
t_min, t_max = -20, 20   # Time range (arbitrary units)
num_t = 4096             # Number of time points
t = np.linspace(t_min, t_max, num_t)  # Time array
dt = t[1] - t[0]         # Time step

# Define a function to generate a harmonic spectrum (only odd harmonics)
def harmonic_spectrum(frequencies, base_freq=10, max_order=15):
    spectrum = np.zeros_like(frequencies, dtype=complex)
    for n in range(1, max_order + 1, 2):  # Only odd harmonics (1, 3, 5, ...)
        omega_n = n * base_freq
        if omega_n > 0:  # Ensure only positive frequencies
            idx = np.argmin(np.abs(frequencies - omega_n))
            spectrum[idx] = 1.0  # Set amplitude for each harmonic
    return spectrum

# Compute the frequency domain representation (harmonic spectrum)
frequencies = fft.fftfreq(num_t, d=dt) * 2 * np.pi  # Frequency array (ω)
harmonic_spectrum_values = harmonic_spectrum(modulus(frequencies), base_freq=10, max_order=15)

# Compute the attosecond pulse train by taking the Fourier transform
E_t_values = modulus(fft.fft(fft.fftshift(harmonic_spectrum_values)))

# Convert time array to wave periods (relative to the base frequency)
wave_periods = (t - t_min) / (2 * np.pi / 10)  # Base frequency period is 2π/10

# Plot results
plt.figure(figsize=(12, 5))

# Frequency-domain plot (|E(x,ω)|)
plt.subplot(1, 2, 1)
plt.plot(modulus(frequencies), modulus(harmonic_spectrum_values))
plt.xlabel("Frequency ω / arb. units")
plt.ylabel("|E(x,ω)| / arb. units")
plt.title("Harmonic Spectrum in Frequency Domain (Odd Harmonics)")
plt.xlim(0, 160)

# Time-domain plot (attosecond pulse train in wave periods)
plt.subplot(1, 2, 2)
plt.plot(wave_periods, E_t_values)
plt.xlabel("Time / wave periods")
plt.ylabel("|E(x,t)| / arb. units")
plt.title("Train of Attosecond Pulses in Time Domain")
plt.xlim(0, 10)  # Restrict to the first 10 wave periods

plt.tight_layout()
plt.show()