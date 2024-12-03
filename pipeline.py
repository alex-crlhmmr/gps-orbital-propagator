import pandas as pd
from datetime import datetime, timedelta

file_path = 'gmat_gps.gmd'

gmat_reference_date = datetime(1941, 1, 5, 12)  # 05 Jan 1941 12:00:00
mjd_offset = 2430000.0



# Reloading data
columns = ['GMAT MJD', 'MeasurementType', 'SatelliteID', 'AdditionalID', 'X', 'Y', 'Z']
df = pd.read_csv(file_path, sep='\s+', names=columns)

# Step 1: Convert GMAT MJD to Gregorian DateTime
def gmat_mjd_to_datetime(gmat_mjd):
    jd = gmat_mjd + mjd_offset  # Convert GMAT MJD to Julian Date
    days_since_ref = jd - 2400000.5  # Days since 17 Nov 1858
    datetime_val = datetime(1858, 11, 17) + timedelta(days=days_since_ref)
    return datetime_val

df['Timestamp'] = df['GMAT MJD'].apply(gmat_mjd_to_datetime)

# Step 2: Create a dictionary of dictionaries with hierarchical grouping
hierarchical_dict = {}

for _, row in df.iterrows():
    timestamp = row['Timestamp']
    day = timestamp.date()  # Extract the date
    time_in_seconds = (timestamp - datetime.combine(day, datetime.min.time())).total_seconds()
    time_bin = int(time_in_seconds // 3600)  # Divide into 1-hour bins

    # Initialize the dictionary structure if needed
    if day not in hierarchical_dict:
        hierarchical_dict[day] = {}

    if time_bin not in hierarchical_dict[day]:
        hierarchical_dict[day][time_bin] = []

    # Append the row data to the appropriate time bin
    hierarchical_dict[day][time_bin].append({
        'Timestamp': timestamp,
        'MeasurementType': row['MeasurementType'],
        'SatelliteID': row['SatelliteID'],
        'AdditionalID': row['AdditionalID'],
        'X': row['X'] * 1000,  # Convert km to meters
        'Y': row['Y'] * 1000,
        'Z': row['Z'] * 1000
    })

# Summarize the structure into a readable format
summary = [
    {
        "Day": day,
        "Time Bin (Hours)": time_bin,
        "Data Points": len(data)
    }
    for day, time_bins in hierarchical_dict.items()
    for time_bin, data in time_bins.items()
]

summary_df = pd.DataFrame(summary)


print(summary_df)