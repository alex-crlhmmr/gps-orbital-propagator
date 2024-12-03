from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

def newton_raphson__eccentric_anomaly(mean_anomaly, eccentricity, tolerance, max_iter):
    
    # array operations
    mean_anomaly = np.asarray(mean_anomaly)
    eccentricity = [eccentricity for _ in mean_anomaly]

    # Initialize the iterations from the mean anomaly
    ecc_anomaly = copy.deepcopy(mean_anomaly)

    for _ in np.arange(0, max_iter):
        f_E = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - mean_anomaly
        f_prime_E = 1 - eccentricity * np.cos(ecc_anomaly)
        zero_derivative = f_prime_E == 0
        if np.any(zero_derivative):
            raise ZeroDivisionError("Derivative is zero; Newton-Raphson method fails.")
        delta_E = f_E / f_prime_E
        ecc_anomaly -= delta_E
        if np.all(np.abs(delta_E) < tolerance):
            break
    else:
        raise RuntimeError("Newton-Raphson method did not converge.")
    
    return ecc_anomaly


def get_true_anomaly(mean_motion, mean_ecentricity, initial_mean_anomaly, time):
 
    mean_anomaly = [initial_mean_anomaly+mean_motion*t for t in time]

    eccentric_anomaly = newton_raphson__eccentric_anomaly(mean_anomaly, mean_ecentricity, 1e-12, 1000)
    true_anomaly = np.arctan2(np.sqrt(1-mean_ecentricity**2)*np.sin(eccentric_anomaly),(np.cos(eccentric_anomaly)-mean_ecentricity))
    return true_anomaly


def get_orbital_radius(semi_major_axis, mean_motion, mean_ecentricity, initial_mean_anomaly, time):
    true_anomaly = get_true_anomaly(mean_motion, mean_ecentricity, initial_mean_anomaly, time)
    orbital_radius = semi_major_axis * (1 - mean_ecentricity**2) / (1 + mean_ecentricity * np.cos(true_anomaly))
    return orbital_radius

def get_plane_coordinates(semi_major_axis, mean_motion, mean_ecentricity, initial_mean_anomaly, time):
    true_anomaly = get_true_anomaly(mean_motion, mean_ecentricity, initial_mean_anomaly, time)
    orbital_radius = get_orbital_radius(semi_major_axis, mean_motion, mean_ecentricity, initial_mean_anomaly, time)
    x = orbital_radius * np.cos(true_anomaly)
    y = orbital_radius * np.sin(true_anomaly)
    return x, y


def get_ECEF_coordinates(x, y, semi_major_axis, inclination, raan, arg_periapsis, time):
    orbital_radius = np.sqrt(x**2 + y**2)

    pos_orbit = np.array([x, y, np.zeros_like(x)])  # 2D to 3D orbital coordinates

    R_periapsis = np.array([
        [np.cos(arg_periapsis), -np.sin(arg_periapsis), 0],
        [np.sin(arg_periapsis), np.cos(arg_periapsis), 0],
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

    # Full rotation from orbital plane to ECI
    rotation_matrix = R_raan @ R_inclination @ R_periapsis
    pos_eci = rotation_matrix @ pos_orbit

    # Earth's rotation rate in rad/s
    earth_rotation_rate = 7.2921159e-5

    # Convert from ECI to ECEF
    theta = earth_rotation_rate * time  # Earth's rotation angle
    ecef_positions = []
    for i, t in enumerate(time):
        R_earth_rotation = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]), 0],
            [np.sin(theta[i]), np.cos(theta[i]), 0],
            [0, 0, 1]
        ])
        ecef_positions.append(R_earth_rotation @ pos_eci[:, i])

    return np.array(ecef_positions).T  # Transpose for consistency



def plot_ECEF_coordinates(ecef_coords):
    # Earth's radius in meters
    earth_radius = 6378137/1000

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot satellite ECEF coordinates
    ax.plot(ecef_coords[0], ecef_coords[1], ecef_coords[2], label="Satellite Trajectory", color='blue')

    # Create a sphere to represent Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the Earth sphere
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='green', alpha=0.5, linewidth=0.5)

    # Labels and title
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("Satellite Trajectory in ECEF Coordinates with Earth Reference Sphere")
    plt.legend()
    plt.show()

# Plot in-plane coordinates
def plot_in_plane_coordinates(x, y):
    plt.plot(x, y)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Satellite Trajectory in Orbital Plane")
    plt.show()



# Plot ECEF coordinates with Earth reference sphere
def plot_ECEF_coordinates_with_GPS(ecef_coords, gps_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the satellite ECEF coordinates
    ax.plot(ecef_coords[0] / 1000, ecef_coords[1] / 1000, ecef_coords[2] / 1000, label="Satellite Trajectory", color='blue')

    # Create a sphere to represent Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x_sphere = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y_sphere = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z_sphere = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the Earth sphere
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='green', alpha=0.5, linewidth=0.5)

    # Plot the GPS data points on top in red
    ax.scatter(gps_data['X_normalized'], gps_data['Y_normalized'], gps_data['Z_normalized'], color='red', s=10, label="GPS Data Points")

    # Labels and title
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Satellite Trajectory with GPS Data Points")
    plt.legend()
    plt.show()



def plot_partial_ECEF_coordinates_with_filtered_GPS(ecef_coords, gps_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Earth's radius in kilometers
    earth_radius = 6378137 / 1000

    # Generate partial sphere (1/8 of the sphere) coordinates
    u = np.linspace(0, np.pi / 2, 50)    # Only the northern quarter
    v = np.linspace(0, np.pi / 2, 25)    # Only one-eighth section
    
    x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the partial Earth sphere
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='green', alpha=0.5, linewidth=0.5)

    # Filter ECEF satellite coordinates for points in the positive X, Y, Z octant
    filtered_ecef_coords = ecef_coords[:, (ecef_coords[0] > 0) & (ecef_coords[1] > 0) & (ecef_coords[2] > 0)]
    
    # Plot the filtered satellite ECEF coordinates
    ax.plot(filtered_ecef_coords[0] / 1000, filtered_ecef_coords[1] / 1000, filtered_ecef_coords[2] / 1000, label="Filtered Satellite Trajectory", color='blue')

    # Filter GPS data for points in the positive X, Y, Z octant
    filtered_gps_data = gps_data[(gps_data['X_normalized'] > 0) & (gps_data['Y_normalized'] > 0) & (gps_data['Z_normalized'] > 0)]
    
    # Plot the filtered GPS data points in red
    ax.scatter(filtered_gps_data['X_normalized'], filtered_gps_data['Y_normalized'], filtered_gps_data['Z_normalized'], color='red', s=10, label="Filtered GPS Data Points")

    # Labels and title
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Filtered Partial Earth with Satellite and GPS Data")
    plt.legend()
    plt.show()

# Compute the average error between the GPS data and satellite trajectory in the z-coordinate
def compute_average_error(ecef_coords, gps_data):
    # Averge z-coord of the computed ECEF coordinates
    avg_z = np.mean(ecef_coords[2])
    # Average z-coord of the GPS data
    avg_gps_z = np.mean(gps_data['Z_normalized'])
    # Compute the average error
    avg_error = np.abs(avg_z - avg_gps_z) / 1000  # Convert to kilometers
    return avg_error

#Compute average altitude error between GPS data and satellite trajectory
def compute_average_altitude_error(ecef_coords, gps_data):

    gps_x = np.array(gps_data['X_normalized'])
    gps_y = np.array(gps_data['Y_normalized'])
    gps_z = np.array(gps_data['Z_normalized'])

    gps_data_points = np.stack((gps_x, gps_y, gps_z))*1000

    avg_alt = np.mean(np.linalg.norm(ecef_coords, axis=0))
    avg_gps_alt = np.mean(np.linalg.norm(gps_data_points, axis=0))
    
    avg_error = np.abs(avg_alt - avg_gps_alt) /1000
    return avg_error

if __name__ == '__main__':

    mu = 3.986004418e14

    #Original orbital parameters from data sheet
    mean_motion = 15.22972908 * 2 * np.pi / 86400  # rad/sec
    mean_ecentricity = 0.0013769 
    initial_mean_anomaly = 0 
    semi_major_axis = (mean_motion**(-2) * mu)**(1/3) 
    time = np.arange(0, 86400, 60) # 1 day in seconds
    inclination = np.radians(97.4085)
    raan = np.radians(4.4766)
    arg_periapsis = np.radians(207.9913)

    #Calculated guesses of orbital parameters from GPS Data
    # semi_major_axis = 7192188.005891686
    # mean_ecentricity = 0.13426412998948875
    # inclination = np.radians(103.60603181398038)
    # raan = np.radians(-59.43694893107553)
    # arg_periapsis = np.radians(94.65823728204472)

    x, y = get_plane_coordinates(semi_major_axis, mean_motion, mean_ecentricity, initial_mean_anomaly, time)

    # Load GPS data
    file_path = "gmat_gps.gmd"  # Path to your .gmd file
    columns = ['Timestamp', 'MeasurementType', 'SatelliteID', 'AdditionalID', 'X', 'Y', 'Z']
    df = pd.read_csv(file_path, sep='\s+', names=columns)

    # Define normalization to Earth's surface function if needed (e.g., for better visualization)
    EARTH_RADIUS = 6378137 / 1000  # Convert to kilometers

    # Normalize GPS data points to Earth's surface radius if necessary
    df['X_normalized'], df['Y_normalized'], df['Z_normalized'] = df['X'], df['Y'], df['Z']

    # Generate the ECEF coordinates using the provided function
    ecef_coords = get_ECEF_coordinates(x, y, semi_major_axis, inclination, raan, arg_periapsis, time)

    # Plot the in-plane coordinates
    #plot_in_plane_coordinates(x, y)

    # Plot the ECEF coordinates
    plot_ECEF_coordinates(ecef_coords)

    # Plot the ECEF coordinates with GPS data points
    plot_ECEF_coordinates_with_GPS(ecef_coords, df)

    # Plot the partial ECEF coordinates with filtered GPS data points
    plot_partial_ECEF_coordinates_with_filtered_GPS(ecef_coords, df)

    # Compute the average error between the GPS data and satellite trajectory in the z-coordinate
    avg_error = compute_average_error(ecef_coords, df)
    print(f"Average error in the z-coordinate: {avg_error} km")

    #Compute average altitude error, propogator vs GPS
    avg_alt_error = compute_average_altitude_error(ecef_coords, df)
    print(f"Average error in altitude: {avg_alt_error} km")

    # plot z coordinate variation with time
    plt.plot(time, ecef_coords[2])
    plt.xlabel("Time (s)")
    plt.ylabel("Z Coordinate (m)")
    plt.title("Z Coordinate Variation with Time")
    plt.show()

    #plot altitude with time
    alt_data = np.linalg.norm(ecef_coords, axis=0)
    plt.plot(time, alt_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs Time')
    plt.show()