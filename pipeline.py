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


# Example: Print the points with the most data for a specific day
# 2023-04-18
# day = '2023-04-19'
# for points in day_max_chunk_with_points[day]:
#     print(points)



# # Final function to find the two nearest points based on consecutive proximity
# def find_two_nearest_points(day_max_chunk_with_points):
#     day_two_nearest = {}
#     for day, points in day_max_chunk_with_points.items():
#         if len(points) < 2:
#             # If fewer than 2 points, skip the day
#             continue
#         min_distance = float('inf')
#         nearest_pair = []
#         # Iterate through points to find the two closest
#         for i in range(len(points) - 1):
#             p1 = points[i]
#             p2 = points[i + 1]
#             # Compute Euclidean distance
#             distance = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_pair = [p1, p2]
#         day_two_nearest[day] = nearest_pair
#     return day_two_nearest

# # Test the function on the example dataset
# day_two_nearest = find_two_nearest_points(day_max_chunk_with_points)

# # Example: Print the two nearest points for a specific day
# # 2023-04-18
# day = '2023-04-19'
# # for points in day_two_nearest[day]:
# #     # print posittion
# #     print(points)
# #     # print time (search in orginal df)
# #     print(df[(df['X'] == points[0]) & (df['Y'] == points[1]) & (df['Z'] == points[2])]['Timestamp'].values[0])
# print(day_two_nearest[day])


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

def compute_keplerian_elements(day_two_nearest_with_time, mu=398600.4418):  # mu for Earth in km^3/s^2
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
print("Keplerian Elements for each day:")
for day, elements in keplerian_elements_dict.items():
    print(f"{day}: {elements}")



orbital_data = process_tle('tle.txt')
print(orbital_data)









import matplotlib.pyplot as plt
import datetime

# Assuming 'keplerian_elements_dict' is available from your previous computations

# Extract days and elements
days = []
semi_major_axes = []
eccentricities = []
inclinations = []
raans = []
arg_periapses = []
true_anomalies = []

for day_str, elements in keplerian_elements_dict.items():
    day = datetime.datetime.strptime(day_str, '%Y-%m-%d')
    days.append(day)
    semi_major_axes.append(elements['semi_major_axis'])
    eccentricities.append(elements['eccentricity'])
    inclinations.append(elements['inclination'])
    raans.append(elements['raan'])
    arg_periapses.append(elements['arg_periapsis'])
    true_anomalies.append(elements['true_anomaly'])

# Sort the data by days
sorted_indices = sorted(range(len(days)), key=lambda k: days[k])
days = [days[i] for i in sorted_indices]
semi_major_axes = [semi_major_axes[i] for i in sorted_indices]
eccentricities = [eccentricities[i] for i in sorted_indices]
inclinations = [inclinations[i] for i in sorted_indices]
raans = [raans[i] for i in sorted_indices]
arg_periapses = [arg_periapses[i] for i in sorted_indices]
true_anomalies = [true_anomalies[i] for i in sorted_indices]


# Plot each Keplerian element over time
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(days, semi_major_axes, marker='o')
plt.title('Semi-Major Axis')
plt.xlabel('Date')
plt.ylabel('Semi-Major Axis (km)')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(days, eccentricities, marker='o', color='orange')
plt.title('Eccentricity')
plt.xlabel('Date')
plt.ylabel('Eccentricity')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(days, inclinations, marker='o', color='green')
plt.title('Inclination')
plt.xlabel('Date')
plt.ylabel('Inclination (degrees)')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(days, raans, marker='o', color='red')
plt.title('Right Ascension of Ascending Node')
plt.xlabel('Date')
plt.ylabel('RAAN (degrees)')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(days, arg_periapses, marker='o', color='purple')
plt.title('Argument of Periapsis')
plt.xlabel('Date')
plt.ylabel('Argument of Periapsis (degrees)')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(days, true_anomalies, marker='o', color='brown')
plt.title('True Anomaly')
plt.xlabel('Date')
plt.ylabel('True Anomaly (degrees)')
plt.grid(True)

plt.tight_layout()
plt.show()




# # delete 2023-04-20 and 2023-04-21 ffrom keplerian_elements_dict

# # Plot the keplerian elements each day
# import matplotlib.pyplot as plt

# # Extract the Keplerian elements for plotting
# days = list(keplerian_elements_dict.keys())
# semi_major_axes = [elements["semi_major_axis"] for elements in keplerian_elements_dict.values()]
# eccentricities = [elements["eccentricity"] for elements in keplerian_elements_dict.values()]
# inclinations = [elements["inclination"] for elements in keplerian_elements_dict.values()]
# raans = [elements["raan"] for elements in keplerian_elements_dict.values()]
# arg_periapses = [elements["arg_periapsis"] for elements in keplerian_elements_dict.values()]
# true_anomalies = [elements["true_anomaly"] for elements in keplerian_elements_dict.values()]

# # Convert days to indices for plotting
# day_indices = range(len(days))

# # Plot the Keplerian elements over time
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 2, 1)
# plt.plot(day_indices, semi_major_axes, marker='o')
# plt.title("Semi-Major Axis vs. Time")
# plt.xlabel("Days")
# plt.ylabel("Semi-Major Axis (km)")

# plt.subplot(3, 2, 2)
# plt.plot(day_indices, eccentricities, marker='o')
# plt.title("Eccentricity vs. Time")
# plt.xlabel("Days")
# plt.ylabel("Eccentricity")

# plt.subplot(3, 2, 3)
# plt.plot(day_indices, inclinations, marker='o')
# plt.title("Inclination vs. Time")
# plt.xlabel("Days")
# plt.ylabel("Inclination (degrees)")

# plt.subplot(3, 2, 4)
# plt.plot(day_indices, raans, marker='o')
# plt.title("Right Ascension of Ascending Node vs. Time")
# plt.xlabel("Days")
# plt.ylabel("RAAN (degrees)")

# plt.subplot(3, 2, 5)
# plt.plot(day_indices, arg_periapses, marker='o')
# plt.title("Argument of Periapsis vs. Time")
# plt.xlabel("Days")
# plt.ylabel("Argument of Periapsis (degrees)")

# plt.subplot(3, 2, 6)
# plt.plot(day_indices, true_anomalies, marker='o')
# plt.title("True Anomaly vs. Time")
# plt.xlabel("Days")
# plt.ylabel("True Anomaly (degrees)")

# plt.tight_layout()
# plt.show()
