from tle_utils import process_tle
import numpy as np
from typing import Tuple
import pandas as pd
import copy


def newton_raphson_eccentric_anomaly(mean_anomalies: np.ndarray, eccentricities: np.ndarray, tolerance=1e-6, max_iter=1000) -> np.ndarray:
    """
    Compute the eccentric anomaly for arrays of mean anomalies and eccentricities 
    using the Newton-Raphson method.

    Parameters:
    - mean_anomalies (array-like): Array of mean anomalies (in radians).
    - eccentricities (array-like): Array of eccentricities.
    - tolerance (float): Convergence tolerance.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - eccentric_anomalies (numpy array): Array of eccentric anomalies.
    """
    # Convert inputs to numpy arrays
    mean_anomalies = np.asarray(mean_anomalies)
    eccentricities = np.asarray(eccentricities)

    # Ensure input arrays are of the same length
    if mean_anomalies.shape != eccentricities.shape:
        raise ValueError("mean_anomaly and eccentricity must have the same shape.")
    
    # Initialize the iterations from the mean anomaly
    eccentric_anomalies = copy.deepcopy(mean_anomalies)

    for _ in range(max_iter):
        # Compute the function and its derivative
        f_E = eccentric_anomalies - eccentricities * np.sin(eccentric_anomalies) - mean_anomalies
        f_prime_E = 1 - eccentricities * np.cos(eccentric_anomalies)
        
        # Handle zero derivative (to avoid division by zero)
        zero_derivative = f_prime_E == 0
        if np.any(zero_derivative):
            raise ZeroDivisionError("Derivative is zero; Newton-Raphson method fails.")
        
        # Update eccentric anomaly
        delta_E = f_E / f_prime_E
        eccentric_anomalies -= delta_E
        
        # Check for convergence
        if np.all(np.abs(delta_E) < tolerance):
            break
    else:
        # If the loop completes without breaking, convergence failed
        raise RuntimeError("Newton-Raphson method did not converge.")

    return eccentric_anomalies



def get_true_anomaly(mean_anomalies, eccentricities):
    """
    Compute the true anomalies for arrays of mean anomalies and eccentricities.

    Parameters:
    - mean_anomalies (array-like): Array of mean anomalies (rad).
    - eccentricities (array-like): Array of eccentricities.

    Returns:
    - true_anomalies (numpy array): Array of true anomalies (rad).
      Shape matches the input mean_anomalies and eccentricities.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mean_anomalies = np.asarray(mean_anomalies)
    eccentricities = np.asarray(eccentricities)

    # Validate input shapes
    if mean_anomalies.shape != eccentricities.shape:
        raise ValueError("mean_anomalies and eccentricities must have the same shape.")

    # Compute eccentric anomalies using the Newton-Raphson method
    eccentric_anomalies = newton_raphson_eccentric_anomaly(
        mean_anomalies, eccentricities, tolerance=1e-12, max_iter=1000
    )

    # Compute true anomalies using eccentric anomalies
    true_anomalies = np.arctan2(
        np.sqrt(1 - eccentricities**2) * np.sin(eccentric_anomalies),
        np.cos(eccentric_anomalies) - eccentricities
    )

    return true_anomalies



def get_orbital_radius(semi_major_axis: np.ndarray, mean_anomalies: np.ndarray, eccentricities: np.ndarray)-> np.ndarray:
    """
    Compute the orbital radius for arrays of semi-major axes, mean anomalies, and eccentricities.

    Parameters:
    - semi_major_axis (array-like): Array of semi-major axes (km or other units).
    - mean_anomalies (array-like): Array of mean anomalies (rad).
    - eccentricities (array-like): Array of eccentricities.

    Returns:
    - orbital_radius (numpy array): Array of orbital radii (same units as semi_major_axis).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    semi_major_axis = np.asarray(semi_major_axis)
    mean_anomalies = np.asarray(mean_anomalies)
    eccentricities = np.asarray(eccentricities)

    # Validate input shapes
    if not (semi_major_axis.shape == mean_anomalies.shape == eccentricities.shape):
        raise ValueError("semi_major_axis, mean_anomalies, and eccentricities must have the same shape.")

    # Compute true anomalies
    true_anomalies = get_true_anomaly(mean_anomalies, eccentricities)

    # Compute orbital radii
    orbital_radius = semi_major_axis * (1 - eccentricities**2) / (1 + eccentricities * np.cos(true_anomalies))

    return orbital_radius


def get_plane_coordinates(semi_major_axis: np.ndarray,mean_anomalies: np.ndarray, eccentricities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D coordinates in orbital planes for arrays of orbital parameters.

    Parameters:
    - semi_major_axis (array-like): Array of semi-major axes (km or other units).
    - mean_anomalies (array-like): Array of mean anomalies (rad).
    - eccentricities (array-like): Array of eccentricities.

    Returns:
    - x (numpy array): Array of x-coordinates in the orbital planes.
    - y (numpy array): Array of y-coordinates in the orbital planes.
    """
    # Ensure inputs are numpy arrays
    semi_major_axis = np.asarray(semi_major_axis)
    mean_anomalies = np.asarray(mean_anomalies)
    eccentricities = np.asarray(eccentricities)

    # Validate input shapes
    if not (semi_major_axis.shape == mean_anomalies.shape == eccentricities.shape):
        raise ValueError("semi_major_axis, mean_anomalies, and eccentricities must have the same shape.")

    # Compute true anomalies
    true_anomaly = get_true_anomaly(mean_anomalies, eccentricities)

    # Compute orbital radii
    orbital_radius = get_orbital_radius(semi_major_axis, mean_anomalies, eccentricities)

    # Calculate plane coordinates
    x = orbital_radius * np.cos(true_anomaly)
    y = orbital_radius * np.sin(true_anomaly)

    return x, y



def get_ECEF_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    inclinations: np.ndarray,
    raans: np.ndarray,
    arg_periapsis: np.ndarray,
    epochs: np.ndarray
) -> np.ndarray:
    """
    Compute ECEF coordinates for the entire mission with evolving orbital parameters.

    Parameters:
    - x, y (array-like): Orbital plane coordinates (km).
    - inclinations (array-like): Inclinations (radians).
    - raans (array-like): Right Ascension of Ascending Nodes (radians).
    - arg_periapsis (array-like): Argument of periapsis (radians).
    - epochs (array-like): Array of times corresponding to TLE epochs (seconds).

    Returns:
    - ecef_positions (numpy array): ECEF coordinates (km) with shape (3, len(epochs)).
    """
    # Orbital radius (3D position in the orbital plane)
    pos_orbit = np.array([x, y, np.zeros_like(x)])  # (3, len(epochs))

    # Earth's rotation rate in rad/s
    earth_rotation_rate = 7.2921159e-5  # Earth's rotation rate (rad/s)

    ecef_positions = []

    for i in range(len(epochs)):
        # Time-varying orbital elements
        raan = raans[i]
        arg_peri = arg_periapsis[i]
        inclination = inclinations[i]

        # Rotation matrices for orbital to ECI
        R_periapsis = np.array([
            [np.cos(arg_peri), -np.sin(arg_peri), 0],
            [np.sin(arg_peri), np.cos(arg_peri), 0],
            [0, 0, 1]
        ])
        R_inclination = np.array([
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)]
        ])
        R_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])

        # Full rotation matrix: Orbital plane to ECI
        rotation_matrix = R_raan @ R_inclination @ R_periapsis
        pos_eci = rotation_matrix @ pos_orbit[:, i]

        # Earth's rotation angle at this epoch
        theta = earth_rotation_rate * epochs[i]
        R_earth_rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Convert ECI to ECEF
        pos_ecef = R_earth_rotation @ pos_eci
        ecef_positions.append(pos_ecef)

    return np.array(ecef_positions).T  # Shape (3, len(epochs))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_ECEF_coordinates(ecef_coords):
    """
    Plot satellite ECEF coordinates in 3D with an Earth reference sphere.

    Parameters:
    - ecef_coords (numpy array): ECEF coordinates of shape (3, len(time)).
    """
    # Earth's radius in kilometers
    earth_radius = 6378.137  # km

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot satellite ECEF coordinates
    ax.plot(
        ecef_coords[0], ecef_coords[1], ecef_coords[2],
        label="Satellite Trajectory", color='blue'
    )

    # Create a sphere to represent Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the Earth sphere
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='green', alpha=0.5, linewidth=0.5)

    # Labels and title
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Satellite Trajectory in ECEF Coordinates with Earth Reference Sphere")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # Load the TLE data
    orbital_data = process_tle('tle.txt')
    print(orbital_data)

    # Earth's standard gravitational parameter in km^3/s^2
    mu = 398600.4418

    # Extract the orbital parameters
    epochs = orbital_data['Epoch']
    mean_motions = orbital_data['Mean Motion (rev/day)']
    eccentricities = orbital_data['Eccentricity']
    mean_anomalies = orbital_data['Mean Anomaly (deg)']
    inclinations = orbital_data['Inclination (deg)']
    raans = orbital_data['RAAN (deg)']
    apogees = orbital_data['Argument of Perigee (deg)']

    # Convert the mean motion to radians per second
    mean_motions = np.deg2rad(mean_motions) * 2 * np.pi / 86400

    # Convert the mean anomaly to radians
    mean_anomalies = np.deg2rad(mean_anomalies)

    # Convert the orbital plane angles to radians
    inclinations = np.deg2rad(inclinations)
    raans = np.deg2rad(raans)
    apogees = np.deg2rad(apogees)
    
    # Compute the semi-major axis
    semi_major_axis = (mu / mean_motions**2)**(1/3)

    # Convert epochs to seconds since the first epoch
    reference_epoch = pd.to_datetime(epochs.iloc[0])
    epochs = (pd.to_datetime(epochs) - reference_epoch).dt.total_seconds().values

    # Compute the satellite's orbital plane coordinates
    x, y = get_plane_coordinates(semi_major_axis, mean_anomalies, eccentricities)

    # Compute the ECEF coordinates
    ecef_positions = get_ECEF_coordinates(x, y, inclinations, raans, apogees, epochs)

    # Plot the satellite's ECEF coordinates
    plot_ECEF_coordinates(ecef_positions)


    