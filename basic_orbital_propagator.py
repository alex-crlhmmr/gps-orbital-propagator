import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    mu = 6378137
    mean_motion = 16.47925117*2*np.pi/86400 #revolutions per/day --> rad/sec
    mean_ecentricity = 0.0010327 
    initial_mean_anomaly = 0
    semi_major_axis = (mean_motion**(-2) * mu)**(1/3)
    time = np.arange(0, 86400, 60)
    x, y = get_plane_coordinates(semi_major_axis, mean_motion, mean_ecentricity, initial_mean_anomaly, time)
    # plot the orbit
    plt.plot(x, y)
    plt.show()

    # plot radius
    r = get_orbital_radius(semi_major_axis, mean_motion, mean_ecentricity, initial_mean_anomaly, time)
    plt.plot(time, r)
    plt.show()

    # plot true anomaly
    true_anomaly = get_true_anomaly(mean_motion, mean_ecentricity, initial_mean_anomaly, time)
    plt.plot(time, true_anomaly)
    plt.show()