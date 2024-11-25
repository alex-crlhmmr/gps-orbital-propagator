import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ecef_to_eci(ecef_positions, gst):
    """Convert ECEF positions to ECI using GST."""
    gst_rad = np.radians(gst)
    rotation_matrix = np.array([
        [np.cos(gst_rad), np.sin(gst_rad), 0],
        [-np.sin(gst_rad), np.cos(gst_rad), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, ecef_positions.T).T

def estimate_velocity(positions, times):
    """Estimate velocity using central differences."""
    velocities = np.zeros_like(positions)
    dt = np.diff(times)
    for i in range(1, len(positions) - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt[i - 1])
    velocities[0] = (positions[1] - positions[0]) / dt[0]
    velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
    return velocities

def calculate_orbital_elements(r, v, mu=398600.4418):
    """Calculate classical orbital elements."""
    h = np.cross(r, v)  # Angular momentum
    h_norm = np.linalg.norm(h)
    i = np.degrees(np.arccos(h[2] / h_norm))  # Inclination

    n = np.cross([0, 0, 1], h)  # Node vector
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raan = 0
    else:
        raan = np.degrees(np.arctan2(n[1], n[0]))

    e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
    e = np.linalg.norm(e_vec)

    if n_norm == 0 or e == 0:
        omega = 0
    else:
        omega = np.degrees(np.arctan2(np.dot(np.cross(n, e_vec), h) / h_norm, np.dot(n, e_vec)))

    nu = np.degrees(np.arctan2(np.dot(np.cross(e_vec, r), h) / h_norm, np.dot(e_vec, r)))

    a = 1 / (2 / np.linalg.norm(r) - np.dot(v, v) / mu)

    return {'a': a, 'e': e, 'i': i, 'raan': raan, 'omega': omega, 'nu': nu}

def orbital_elements_over_time(ecef_positions, times, gst_start, mu=398600.4418):
    """Calculate and plot orbital elements over time."""
    gst_step = (360 / 86400) * (times - times[0])
    elements = {'a': [], 'e': [], 'i': [], 'raan': [], 'omega': [], 'nu': []}

    for i, (pos, gst) in enumerate(zip(ecef_positions, gst_step)):
        eci_pos = ecef_to_eci(pos, gst_start + gst)
        if i > 0:
            dt = times[i] - times[i - 1]
            velocity = (eci_pos - prev_pos) / dt
            orbital_elements = calculate_orbital_elements(prev_pos, velocity, mu)
            for key, value in orbital_elements.items():
                elements[key].append(value)
        prev_pos = eci_pos

    return elements

def plot_orbital_elements(times, elements):
    """Plot orbital elements over time."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    keys = ['a', 'e', 'i', 'raan', 'omega', 'nu']
    titles = ['Semi-Major Axis (km)', 'Eccentricity', 'Inclination (deg)', 
              'RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)']

    for ax, key, title in zip(axes.flat, keys, titles):
        ax.plot(times[1:], elements[key])  # Skip the first time (no velocity)
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.grid()

    plt.tight_layout()
    plt.show()

# Example input data
#ecef_positions = np.array([
#    [6524.834, 6862.875, 6448.296],
#    [6525.0, 6863.0, 6448.5], 
#    [6525.2, 6863.1, 6448.6]
#])
#times = np.array([0, 60, 120])
#gst_start = 30.0

# Load GPS data
file_path = "gmat_gps.gmd"  # Path to your .gmd file
columns = ['Timestamp', 'MeasurementType', 'SatelliteID', 'AdditionalID', 'X', 'Y', 'Z']
df = pd.read_csv(file_path, sep='\s+', names=columns)
df["Timestamp"] = (df["Timestamp"] - 300) * 24 * 60 * 60 # fractional days --> seconds
timestamps = df["Timestamp"].values
X = df["X"].values
Y = df["Y"].values
Z = df["Z"].values

ecef_positions = np.column_stack((X, Y, Z))
times = np.array(timestamps) 
gst_start = 0


# Calculate orbital elements over time
elements = orbital_elements_over_time(ecef_positions, times, gst_start)

# Plot the results
plot_orbital_elements(times, elements)

#print(elements[:10,:])