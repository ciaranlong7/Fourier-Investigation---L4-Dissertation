import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import PolyCollection
import scipy.fft as fft

def modulus(input):
    return np.sqrt(np.real(input)**2+ np.imag(input)**2)

##Investigation 1 - The relationship between harmonics in freq domain and attosecond pulses in the time domain:
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

# Frequency-domain plot (|E(z,ω)|)
plt.subplot(1, 2, 1)
plt.plot(np.abs(frequencies), np.abs(harmonic_spectrum_values))
plt.xlabel("Frequency ω / arb. units")
plt.ylabel("|E(z,ω)| / arb. units")
plt.title("Harmonic Spectrum in Frequency Domain (Odd Harmonics)")
plt.xlim(0, 160)

# Time-domain plot (attosecond pulse train in wave periods)
plt.subplot(1, 2, 2)
plt.plot(wave_periods, E_t_values)
plt.xlabel("Time / wave periods")
plt.ylabel("|E(z,t)| / arb. units")
plt.title("Train of Attosecond Pulses in Time Domain")
plt.xlim(0, 10)  # Restrict to the first 10 wave periods

plt.tight_layout()
plt.show()


##Investigation 2 - Solving the hedgehog-in-time equation to observe how pulses propagate with time
#Input pulse options
def gauss(z_points, E_0, a, b):
    #a cannot equal 0
    E_z_t0 = np.array([E_0*np.exp(-((z-b)**2)/(a**2)) for z in z_points])
    return E_z_t0

def rect(z_points, min, max, height):
    E_z_t0 = np.where(np.logical_and(z_points >= min, z_points <= max), height, 0)
    return E_z_t0

#Solving hedgehog-in-time equation:
#E_z_t0: An array of the input electric field
#t: The time at which the electric field is calculated
#dz: Spatial resolution
def electric_field_zt(E_z_t0, t, dz):

    # Compute the Fourier transform of the input electric field
    E_k = fft.fft(E_z_t0)

    # Number of spatial points
    N = len(E_z_t0)

    # Wave numbers corresponding to Fourier transform
    w_k = fft.fftfreq(N, dz)*2*np.pi

    E_k_t = E_k * np.exp(-1j*w_k*t)

    # Inverse Fourier transform to get E(z, t) - final step of hedgehog in time equation.
    E_z_t = fft.ifft(E_k_t)

    modulus_E_z_t = modulus(E_z_t)

    return modulus_E_z_t

dz = 0.01  #Spatial resolution (how far apart points are plotted in space)
z_min = -3
z_max = 20
z_points = np.arange(z_min, z_max, dz)  # min,max of z-axis for the plot

#Gauss function
E_0 = 1
b = 0
a = 1
E_z_t0 = gauss(z_points, E_0, a, b)

#Rect function
# rect_min = 0
# rect_max = 1
# height = 5
# E_z_t0 = rect(z_points, rect_min, rect_max, height)

#Electric field after time t
t = 5
E_z_t = electric_field_zt(E_z_t0, t, dz)


#Plot of E field at time t'=0 and t'=t:
plt.figure(figsize=(12, 5))
plt.plot(z_points, E_z_t0, label="E(z, t=0)")
plt.fill(z_points, E_z_t0, color="#1f77b4", alpha=0.3)
plt.plot(z_points, E_z_t, label=f"E(z, t={t})")
plt.fill(z_points, E_z_t, color="#ff7f0e", alpha=0.3)
plt.xlabel("Z / arb. units")
plt.ylabel("|E(z,t)| / arb. units")
plt.title("Electric Field Evolution - Hedgehog-In-Time Equation Solved")
plt.legend()
plt.show()


# A plot that can be updated in real time:
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25) #making space for slider

# ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgray")
# slider = Slider(ax_slider, "Time (t)", valmin=0, valmax=10, valinit=t, valstep=0.01)

# ax.plot(z_points, E_z_t0, label="E(z, t=0)")
# ax.fill_between(z_points, E_z_t0, color="#1f77b4", alpha=0.3)
# line, = ax.plot(z_points, E_z_t, color="#ff7f0e", label=f"E(z, t={t})")
# fill = ax.fill_between(z_points, E_z_t, color="#ff7f0e", alpha=0.3)
# ax.set_xlabel("Z")
# ax.set_ylabel("Electric Field")
# ax.set_title("Electric Field Evolution")
# ax.legend()

# def update(val):
#     t = slider.val
#     E_z_t = electric_field_zt(E_z_t0, t, dz)
#     line.set_ydata(E_z_t)
#     for collection in [c for c in ax.collections if isinstance(c, PolyCollection)]:
#         collection.remove()
#     ax.fill_between(z_points, E_z_t0, color="#1f77b4", alpha=0.3)
#     ax.fill_between(z_points, E_z_t, color="#ff7f0e", alpha=0.3)
#     line.set_label(f"E(z, t={t:.2f})")
#     ax.legend()
#     fig.canvas.draw_idle()

# slider.on_changed(update)

# plt.show()