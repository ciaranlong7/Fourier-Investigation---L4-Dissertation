import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import PolyCollection
import scipy.fft as fft

def modulus(input):
    return np.sqrt(np.real(input)**2+ np.imag(input)**2)

#Input pulse options
def gauss(x_points, E_0, a, b):
    #a cannot equal 0
    E_x_t0 = np.array([E_0*np.exp(-((x-b)**2)/(a**2)) for x in x_points])
    return E_x_t0

def rect(x_points, min, max, height):
    E_x_t0 = np.where(np.logical_and(x_points >= min, x_points <= max), height, 0)
    return E_x_t0

def triangle(x_points, centre, width, height):
    min_val = centre - width / 2
    max_val = centre + width / 2
    slope = height / (width / 2)
    E_x_t0 = np.zeros_like(x_points)
    for i, x in enumerate(x_points):
        if min_val <= x < centre:
            E_x_t0[i] = slope * (x - min_val)
        elif centre <= x <= max_val:
            E_x_t0[i] = height - slope * (x - centre)
    return E_x_t0

c = 1  # Speed of light in medium. Arbitrary units

#Solving hedgehog-in-time equation:
#E_x_t0: An array of the input electric field
#t: The time at which the electric field is calculated
#dx: Spatial resolution
#omega_function: Function defining the dispersion relation ω(k)
def electric_field_xt_dispersion(E_x_t0, t, dx, omega_function):
    # Compute the Fourier transform of the input electric field
    E_k = fft.fft(E_x_t0)

    # Number of spatial points
    N = len(E_x_t0)

    # Wave numbers corresponding to the Fourier transform
    k = fft.fftfreq(N, dx)*2*np.pi

    # Apply the dispersion phase shift
    dispersion_phase = np.exp(-1j*omega_function(k)*t)

    # Compute the Fourier transform after phase shift
    E_k_t = E_k*dispersion_phase

    # Inverse Fourier transform to get E(x, t) - final step of hedgehog in time equation.
    E_x_t = fft.ifft(E_k_t)

    modulus_E_x_t = modulus(E_x_t)

    return modulus_E_x_t

# Define the dispersion relation using Taylor expansion
# Ignore terms higher order than squared.
def dispersion_relation(k, k_c, v_g, beta):
    omega_c = c*k_c  # Central angular frequency
    return omega_c + v_g*(k - k_c) + 0.5*beta*(k - k_c)**2

dx = 0.01  #Spatial resolution (how far apart points are plotted in space)
x_min = -3
x_max = 20
x_points = np.arange(x_min, x_max, dx)  # min,max of x-axis for the plot

# Dispersion parameters
k_c = np.pi/4  # Central wave number - arbitrarily choose pi/4
v_g = c  # Group velocity (for simplicity, equals c here)
beta = 0.06  # Group velocity dispersion coefficient.
# beta = 0 corresponds to a dispersionless medium

omega_function = lambda k: dispersion_relation(k, k_c, v_g, beta)

#Gauss function
E_0 = 1
b = 0
a = 1
E_x_t0 = gauss(x_points, E_0, a, b)

#Rect function
# rect_min = 0
# rect_max = 1
# height = 5
# E_x_t0 = rect(x_points, rect_min, rect_max, height)

#Triangle function
centre = 0
width = 1
height = 1
E_x_t0 = triangle(x_points, centre, width, height)


#Plot of E field at time t'=0 and t'=t:
#Electric field after time t
plt.figure(figsize=(12, 5))
t = 4
E_x_t = electric_field_xt_dispersion(E_x_t0, t, dx, omega_function)
plt.plot(x_points, E_x_t0, color="#1f77b4",label="E(x,t=0)")
plt.fill(x_points, E_x_t0, color="#1f77b4", alpha=0.3)
plt.plot(x_points, E_x_t, color="#ff7f0e",label=f"E(x,t={t})")
plt.fill(x_points, E_x_t, color="#ff7f0e", alpha=0.3)
t = 8
E_x_t = electric_field_xt_dispersion(E_x_t0, t, dx, omega_function)
plt.plot(x_points, E_x_t, color="#2ca02c", label=f"E(x,t={t})")
plt.fill(x_points, E_x_t, color="#2ca02c", alpha=0.3)
plt.xlim(-3,12)
plt.xlabel("X / arb. units")
plt.ylabel("|E(x,t)| / arb. units")
plt.title("Light Pulse Evolution - Hedgehog-In-Time Equation Solved")
plt.legend()
plt.show()


# A plot that can be updated in real time:
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25) #making space for slider

ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgray")
slider = Slider(ax_slider, "Time (t)", valmin=0, valmax=15, valinit=t, valstep=0.01)

ax.plot(x_points, E_x_t0, label="E(x, t=0)")
ax.fill_between(x_points, E_x_t0, color="#1f77b4", alpha=0.3)
line, = ax.plot(x_points, E_x_t, color="#ff7f0e", label=f"E(x, t={t})")
fill = ax.fill_between(x_points, E_x_t, color="#ff7f0e", alpha=0.3)
ax.set_xlabel("X / arb. units")
ax.set_ylabel("|E(x,t)| / arb. units")
ax.set_title("Light Pulse Evolution - Hedgehog-In-Time Equation Solved")
ax.legend()

def update(val):
    t = slider.val
    E_x_t = electric_field_xt_dispersion(E_x_t0, t, dx, omega_function)
    line.set_ydata(E_x_t)
    for collection in [c for c in ax.collections if isinstance(c, PolyCollection)]:
        collection.remove()
    ax.fill_between(x_points, E_x_t0, color="#1f77b4", alpha=0.3)
    ax.fill_between(x_points, E_x_t, color="#ff7f0e", alpha=0.3)
    line.set_label(f"E(x, t={t:.2f})")
    ax.legend()
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()