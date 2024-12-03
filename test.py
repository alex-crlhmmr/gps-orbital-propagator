from datetime import datetime, timedelta
from tle_utils import process_tle
import pandas as pd
import numpy as np

# File path and column names
file_path = "gmat_gps.gmd"  # Replace with your actual file path
columns = ['Timestamp', 'MeasurementType', 'SatelliteID', 'AdditionalID', 'X', 'Y', 'Z']
df = pd.read_csv(file_path, sep=r'\s+', names=columns)

# Define GMAT MJD to DateTime conversion function
mjd_offset = 2430000.0
def gmat_mjd_to_datetime(gmat_mjd):
    jd = gmat_mjd + mjd_offset  # Convert GMAT MJD to Julian Date
    days_since_ref = jd - 2400000.5  # Days since 17 Nov 1858
    datetime_val = datetime(1858, 11, 17) + timedelta(days=days_since_ref)
    return datetime_val

# Convert and replace the 'Timestamp' column directly
df['Timestamp'] = df['Timestamp'].apply(gmat_mjd_to_datetime)

# Add a 'Day' column to group by day
df['Day'] = df['Timestamp'].dt.date  # Extract just the date part

# Define a helper function to assign chunk labels
def assign_hour_chunk(timestamp):
    start_of_day = datetime.combine(timestamp.date(), datetime.min.time())
    seconds_since_start = (timestamp - start_of_day).total_seconds()
    n = 1.5 # chunk size in hours
    chunk_label = int(seconds_since_start // (3600*n))  # Compute chunk label (3600*n = n hour)
    return chunk_label

# Add a 'Chunk' column to group by chunks of 3600 seconds within each day
df['Chunk'] = df['Timestamp'].apply(assign_hour_chunk)


# Save the processed data to a new file
df.to_csv('gmat_gps_utc.gmd', sep=' ', index=False, header=False)


# Define the function to find the chunk with the most points for each day
def find_day_max_chunk_with_points(df):
    results = {}
    for day, day_data in df.groupby('Day'):
        # Group the data by chunks for the current day
        chunk_counts = day_data.groupby('Chunk').size()
        # Identify the chunk with the most points
        max_chunk = chunk_counts.idxmax()  # Chunk label with the most points
        max_chunk_data = day_data[day_data['Chunk'] == max_chunk]
        # Collect all X, Y, Z values from the chunk with the most points
        xyz_list = max_chunk_data[['X', 'Y', 'Z']].values.tolist()
        # Store the result
        results[str(day)] = xyz_list
    return results

# Call the function and store the results
day_max_chunk_with_points = find_day_max_chunk_with_points(df)


def find_two_nearest_points_with_time_diff(day_max_chunk_with_points, df):
    day_two_nearest = {}
    for day, points in day_max_chunk_with_points.items():
        if len(points) < 2:
            # If fewer than 2 points, skip the day
            continue
        min_distance = float('inf')
        nearest_pair = []
        time_diff = None
        # Iterate through points to find the two closest
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            # Compute Euclidean distance
            distance = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_pair = [p1, p2]
                # Get timestamps for the points
                time1 = pd.to_datetime(
                    df[(df['X'] == p1[0]) & (df['Y'] == p1[1]) & (df['Z'] == p1[2])]['Timestamp'].values[0]
                )
                time2 = pd.to_datetime(
                    df[(df['X'] == p2[0]) & (df['Y'] == p2[1]) & (df['Z'] == p2[2])]['Timestamp'].values[0]
                )
                time_diff = abs((time2 - time1).total_seconds())
        day_two_nearest[day] = {"points": nearest_pair, "time_difference": time_diff}
    return day_two_nearest


# Example: Print the two nearest points for a specific day
# 2023-04-18
day = '2023-05-01'
day_two_nearest = find_two_nearest_points_with_time_diff(day_max_chunk_with_points, df)
print(day_two_nearest[day])



import numpy as np



def compute_keplerian_elements(day_two_nearest_with_time, mu=398600.4418):
    def keplerian_elements(r, v, mu):
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        e = np.cross(v, h) / mu - r / np.linalg.norm(r)
        e_norm = np.linalg.norm(e)
        a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v) ** 2 / mu)
        i = np.arccos(h[2] / h_norm)
        n = np.cross([0, 0, 1], h)
        n_norm = np.linalg.norm(n)
        Omega = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
        omega = np.arccos(np.dot(n, e) / (n_norm * e_norm))
        if e[2] < 0:
            omega = 2 * np.pi - omega
        nu = np.arccos(np.dot(e, r) / (e_norm * np.linalg.norm(r)))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
        return {
            "semi_major_axis": a,
            "eccentricity": e_norm,
            "inclination": np.degrees(i),
            "raan": np.degrees(Omega),
            "arg_periapsis": np.degrees(omega),
            "true_anomaly": np.degrees(nu)
        }

    keplerian_elements_dict = {}
    for day, info in day_two_nearest_with_time.items():
        if 'points' in info and 'time_difference' in info:
            p0 = np.asarray(info['points'][0])
            p1 = np.asarray(info['points'][1])
            time_diff = info['time_difference']
            # Position vector (r) and velocity vector (v)
            r = p1
            v = (p1 - p0) / time_diff
            # Compute Keplerian elements
            keplerian_elements_dict[day] = keplerian_elements(r, v, mu)

    return keplerian_elements_dict


# Example: Use the provided day_two_nearest_with_time dictionary for testing
keplerian_elements_dict = compute_keplerian_elements(day_two_nearest)

# Output the results
# print("Keplerian Elements for each day:")
# for day, elements in keplerian_elements_dict.items():
#     print(f"{day}: {elements}")


# print(keplerian_elements_dict)


# orbital_data = process_tle('tle.txt')
# print(orbital_data)


# Process the TLE data
orbital_data = process_tle('tle.txt')

# Compute Semi-Major Axis from Mean Motion
mu = 398600.4418  # Earth's gravitational parameter in km^3/s^2


def mean_motion_to_semi_major_axis(mean_motion):
    n = mean_motion * (2 * np.pi) / 86400  # Convert rev/day to rad/s
    a = (mu / n**2) ** (1/3)
    return a

# Apply the function to compute semi-major axis
orbital_data['Semi-Major Axis (km)'] = orbital_data['Mean Motion (rev/day)'].apply(mean_motion_to_semi_major_axis)

# Extract the date from 'Epoch'
orbital_data['Date'] = orbital_data['Epoch'].dt.date

# Remove entries with NaN values from 'keplerian_elements_dict'
keplerian_elements_dict = {day: elements for day, elements in keplerian_elements_dict.items()
                           if not any(np.isnan(list(elements.values())))}

# Create DataFrame from 'keplerian_elements_dict'
gps_elements = pd.DataFrame.from_dict(keplerian_elements_dict, orient='index')
gps_elements.reset_index(inplace=True)
gps_elements.rename(columns={'index': 'Date'}, inplace=True)
gps_elements['Date'] = pd.to_datetime(gps_elements['Date']).dt.date

# Add '_GPS' suffix to columns in 'gps_elements' except 'Date'
gps_elements.columns = [col + '_GPS' if col != 'Date' else col for col in gps_elements.columns]

# Add '_TLE' suffix to columns in 'orbital_data' except 'Date' and 'Epoch'
orbital_data.columns = [col + '_TLE' if col not in ['Date', 'Epoch'] else col for col in orbital_data.columns]

# Merge the DataFrames on 'Date'
merged_data = pd.merge(gps_elements, orbital_data, on='Date')

# Select the relevant columns
merged_data = merged_data[['Date',
                           'semi_major_axis_GPS', 'Semi-Major Axis (km)_TLE',
                           'eccentricity_GPS', 'Eccentricity_TLE',
                           'inclination_GPS', 'Inclination (deg)_TLE',
                           'raan_GPS', 'RAAN (deg)_TLE',
                           'arg_periapsis_GPS', 'Argument of Perigee (deg)_TLE']]

import matplotlib.pyplot as plt

# Convert 'Date' to datetime objects for plotting
dates = pd.to_datetime(merged_data['Date'])

# plt.figure(figsize=(14, 12))

# # Semi-Major Axis
# plt.subplot(3, 2, 1)
# plt.plot(dates, merged_data['semi_major_axis_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['Semi-Major Axis (km)_TLE'], marker='x', label='TLE Data')
# plt.title('Semi-Major Axis Comparison')
# plt.xlabel('Date')
# plt.ylabel('Semi-Major Axis (km)')
# plt.legend()
# plt.grid(True)

# # Eccentricity
# plt.subplot(3, 2, 2)
# plt.plot(dates, merged_data['eccentricity_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['Eccentricity_TLE'], marker='x', label='TLE Data')
# plt.title('Eccentricity Comparison')
# plt.xlabel('Date')
# plt.ylabel('Eccentricity')
# plt.legend()
# plt.grid(True)


# # Inclination
# plt.subplot(3, 2, 3)
# plt.plot(dates, merged_data['inclination_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['Inclination (deg)_TLE'], marker='x', label='TLE Data')
# plt.title('Inclination Comparison')
# plt.xlabel('Date')
# plt.ylabel('Inclination (degrees)')
# plt.legend()
# plt.grid(True)

# # RAAN
# plt.subplot(3, 2, 4)
# plt.plot(dates, merged_data['raan_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['RAAN (deg)_TLE'], marker='x', label='TLE Data')
# plt.title('RAAN Comparison')
# plt.xlabel('Date')
# plt.ylabel('RAAN (degrees)')
# plt.legend()
# plt.grid(True)

# # Argument of Periapsis
# plt.subplot(3, 2, 5)
# plt.plot(dates, merged_data['arg_periapsis_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['Argument of Perigee (deg)_TLE'], marker='x', label='TLE Data')
# plt.title('Argument of Periapsis Comparison')
# plt.xlabel('Date')
# plt.ylabel('Argument of Periapsis (degrees)')
# plt.legend()
# plt.grid(True)

# #plt.tight_layout()
# plt.show()




# import matplotlib.pyplot as plt

# # Assuming 'dates' and 'merged_data' are already defined

# # Set up the figure size
# plt.figure(figsize=(12, 9))  # Adjusted the figure size to suit three plots

# # Eccentricity
# plt.subplot(3, 1, 1)  # Changed the grid to 3 rows, 1 column
# plt.plot(dates, merged_data['eccentricity_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['Eccentricity_TLE'], marker='x', label='TLE Data')
# plt.title('Eccentricity Comparison')
# plt.xlabel('Date')
# plt.ylabel('Eccentricity')
# plt.legend()
# plt.grid(True)

# # Inclination
# plt.subplot(3, 1, 2)
# plt.plot(dates, merged_data['inclination_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['Inclination (deg)_TLE'], marker='x', label='TLE Data')
# plt.title('Inclination Comparison')
# plt.xlabel('Date')
# plt.ylabel('Inclination (degrees)')
# plt.legend()
# plt.grid(True)

# # Argument of Periapsis
# plt.subplot(3, 1, 3)
# plt.plot(dates, merged_data['arg_periapsis_GPS'], marker='o', label='GPS Data')
# plt.plot(dates, merged_data['Argument of Perigee (deg)_TLE'], marker='x', label='TLE Data')
# plt.title('Argument of Periapsis Comparison')
# plt.xlabel('Date')
# plt.ylabel('Argument of Periapsis (degrees)')
# plt.legend()
# plt.grid(True)

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Display the plots
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Create a figure for plotting
fig = plt.figure(figsize=(14, 12))

# Define a GridSpec with 2 rows and 2 columns
gs = GridSpec(2, 2, figure=fig)

# Eccentricity (Graph 1) - Top left
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(dates, merged_data['eccentricity_GPS'], marker='o', label='GPS Data')
ax1.plot(dates, merged_data['Eccentricity_TLE'], marker='x', label='TLE Data')
ax1.set_title('Eccentricity Comparison')
ax1.set_xlabel('Date')
ax1.set_ylabel('Eccentricity')
ax1.legend()
ax1.grid(True)

# Inclination (Graph 2) - Top right
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(dates, merged_data['inclination_GPS'], marker='o', label='GPS Data')
ax2.plot(dates, merged_data['Inclination (deg)_TLE'], marker='x', label='TLE Data')
ax2.set_title('Inclination Comparison')
ax2.set_xlabel('Date')
ax2.set_ylabel('Inclination (degrees)')
ax2.legend()
ax2.grid(True)

# RAAN (Graph 3) - Bottom left
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(dates, merged_data['raan_GPS'], marker='o', label='GPS Data')
ax3.plot(dates, merged_data['RAAN (deg)_TLE'], marker='x', label='TLE Data')
ax3.set_title('RAAN Comparison')
ax3.set_xlabel('Date')
ax3.set_ylabel('RAAN (degrees)')
ax3.legend()
ax3.grid(True)

# Argument of Periapsis (Graph 4) - Bottom right
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(dates, merged_data['arg_periapsis_GPS'], marker='o', label='GPS Data')
ax4.plot(dates, merged_data['Argument of Perigee (deg)_TLE'], marker='x', label='TLE Data')
ax4.set_title('Argument of Periapsis Comparison')
ax4.set_xlabel('Date')
ax4.set_ylabel('Argument of Periapsis (degrees)')
ax4.legend()
ax4.grid(True)

# Format x-axis dates for all axes
for ax in [ax1, ax2, ax3, ax4]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.savefig('comparison_plot.png')
plt.show()

