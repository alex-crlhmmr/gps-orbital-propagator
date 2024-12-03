from datetime import datetime, timedelta
import pandas as pd

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



# Final function to find the two nearest points based on consecutive proximity
def find_two_nearest_points(day_max_chunk_with_points):
    day_two_nearest = {}
    for day, points in day_max_chunk_with_points.items():
        if len(points) < 2:
            # If fewer than 2 points, skip the day
            continue
        min_distance = float('inf')
        nearest_pair = []
        # Iterate through points to find the two closest
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            # Compute Euclidean distance
            distance = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_pair = [p1, p2]
        day_two_nearest[day] = nearest_pair
    return day_two_nearest

# Test the function on the example dataset
day_two_nearest = find_two_nearest_points(day_max_chunk_with_points)

# Example: Print the two nearest points for a specific day
# 2023-04-18
day = '2023-04-19'
for points in day_two_nearest[day]:
    print(points)
