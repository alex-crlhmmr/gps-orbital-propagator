import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ecef_to_eci(ecef_positions, gst):
    """Convert ECEF positions to ECI using GST."""
    eci_coords = np.zeros_like(ecef_positions)
    # Perform the rotation for each time step
    for i in range(gst.shape[0]):
        # Rotation matrix for the current GST angle
        theta = gst[i]
        rotation_matrix = np.array([
            [ np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [           0,            0, 1]
        ])
        
        # Apply the rotation to the ECEF position
        eci_coords[i] = rotation_matrix @ ecef_positions[i]
    
    return eci_coords

#not used
def estimate_velocity(positions, times):
    """Estimate velocity using central differences."""
    velocities = np.zeros_like(positions)
    dt = np.diff(times)
    for i in range(1, len(positions) - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (dt[i - 1]+dt[i])
    velocities[0] = (positions[1] - positions[0]) / dt[0]
    velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
    return velocities

#not used
def get_velocities(positions: np.array, mu):
    """Estimate velocity from positions, assuming elliptical orbit"""
    r = np.linalg.norm(positions)
    velocities = np.sqrt(mu/r)
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
    #if e>1: e=0

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
    eci_coords = ecef_to_eci(ecef_positions, gst_step)
    velocities = estimate_velocity(eci_coords, times)

    for i, (pos, gst) in enumerate(zip(ecef_positions, gst_step)):
        orbital_elements = calculate_orbital_elements(eci_coords[i], velocities[i], mu)
        for key, value in orbital_elements.items():
            elements[key].append(value)

    return elements

def plot_orbital_elements(times, elements):
    """Plot orbital elements over time."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    keys = ['a', 'e', 'i', 'raan', 'omega', 'nu']
    titles = ['Semi-Major Axis (km)', 'Eccentricity', 'Inclination (deg)', 
              'RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)']

    for ax, key, title in zip(axes.flat, keys, titles):
        ax.plot(times, elements[key])
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.grid()

    plt.tight_layout()
    plt.show()

def plot_ecef(time, X, Y, Z, alt):

    fig, axes = plt.subplots(4,1)
    axes[0].plot(time,X)
    axes[0].set_ylabel('X pos')
    axes[0].grid()
    axes[1].plot(time,Y)
    axes[1].set_ylabel('Y pos')
    axes[1].grid()
    axes[2].plot(time,Z)
    axes[2].set_ylabel('Z pos')
    axes[2].grid()
    axes[3].plot(time,alt)
    axes[3].set_ylabel('Altitude')
    axes[3].set_xlabel('Time')
    axes[3].grid()

    plt.show()


if __name__ == "__main__":

    # Load GPS data
    file_path = "gmat_gps.gmd"
    columns = ['Timestamp', 'MeasurementType', 'SatelliteID', 'AdditionalID', 'X', 'Y', 'Z']
    df = pd.read_csv(file_path, sep='\s+', names=columns)
    df = df.sort_values(by='Timestamp')
    df["Timestamp"] = (df["Timestamp"] - 300) * 24 * 60 * 60 # fractional days --> seconds
    timestamps = df["Timestamp"].values
    X = np.array(df["X"].values)
    Y = np.array(df["Y"].values)
    Z = np.array(df["Z"].values)
    ecef_positions = np.column_stack((X, Y, Z))
    alt_data = np.linalg.norm(ecef_positions, axis=1)
    times = np.array(timestamps) 

    #removing weird data points
    condition = (times<2.5720e9) & (times>2.5711e9) & (alt_data>6820)
    times = times[condition]
    X = X[condition]
    Y = Y[condition]
    Z = Z[condition]
    alt_data = alt_data[condition]
    gst_start = 0
    ecef_positions = np.column_stack((X, Y, Z)) #overriding for filtered ecef

    # Calculate orbital elements over time
    elements = orbital_elements_over_time(ecef_positions, times, gst_start)

    # Plot the results
    plot_orbital_elements(times, elements)

    # Plot XYZ coords, altitude
    plot_ecef(times, X, Y, Z, alt_data)

    #Print first elements:
    print(f'Semi-Major Axis: {elements['a'][0]}')
    print(f'Eccentricity: {elements['e'][0]}')
    print(f'Inclination: {elements['i'][0]}')
    print(f'RAAN: {elements['raan'][0]}')
    print(f'Argument of Perigee: {elements['omega'][0]}')
    print(f'True Anamoly: {elements['nu'][0]}')