from datetime import datetime, timedelta

# Parse the new TLE file and extract orbital parameters
tle_path = 'tle.txt'

# Read the TLE file
with open(tle_path, 'r') as file:
    tle_lines = file.read().splitlines()

# Group lines into sets of 2 (Line 1 and Line 2 for each satellite)
tle_sets = [tle_lines[i:i+2] for i in range(0, len(tle_lines), 2)]

# Function to convert TLE epoch to datetime
def tle_epoch_to_datetime(epoch):
    year = int(epoch // 1000) + 2000  # The year is the first four digits of the epoch
    day_of_year = epoch % 1000  # The day of year is the remainder
    day_of_year = int(day_of_year)
    # exract decimale part of epoch
    frac_day = epoch - int(epoch)
    hour = int(frac_day * 24)
    minute = int((frac_day * 24 - hour) * 60)
    second = int(((frac_day * 24 - hour) * 60 - minute) * 60)

    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour, minutes=minute, seconds=second)

# Extract the orbital parameters
epochs = []
inclinations = []
eccentricities = []
raan = []
apogees = []
mean_anomalies = []
mean_motions = []

# Parse each TLE entry and extract values
for tle_set in tle_sets:
    if len(tle_set) < 2:
        continue
    line1 = tle_set[0].strip()
    line2 = tle_set[1].strip()

    # Extract the epoch from Line 1 (position 19-32)
    try:
        # eg of float(line1[18:32].strip()) 23108.67589746 -> 23 ggives 2023, day 108 and 67589746/3600 hours
        #line1_epoch = float(line1[18:21].strip())+2000 # The first 2 digits are the year
        #day_of_year = int(line1[21:24].strip())  # The day of year is the remainder
        #frac_day = float(line1[24:32].strip())/3600 # The fraction of the day in hours
        epoch = float(line1[18:33])
        
    except ValueError:
        continue  # Skip lines that do not have valid epoch data

    # Convert to datetime
    epoch_datetime = tle_epoch_to_datetime(epoch)
    epochs.append(epoch_datetime)

    # Extract orbital parameters from Line 2
    inclination = float(line2[8:16].strip())  # Inclination (degrees)
    eccentricity = float(line2[26:33].strip()) / 100000  # Eccentricity (no decimal point)
    raan_value = float(line2[18:25].strip())  # RAAN (Right Ascension of Ascending Node)
    apogee = float(line2[34:42].strip())  # Argument of Perigee
    mean_anomaly = float(line2[44:51].strip())  # Mean Anomaly
    mean_motion = float(line2[53:63].strip())  # Mean Motion (revolutions/day)

    # Append extracted values
    inclinations.append(inclination)
    eccentricities.append(eccentricity)
    raan.append(raan_value)
    apogees.append(apogee)
    mean_anomalies.append(mean_anomaly)
    mean_motions.append(mean_motion)

# Create a DataFrame to display the extracted data
import pandas as pd

orbital_data = pd.DataFrame({
    'Epoch': epochs,
    'Inclination (deg)': inclinations,
    'Eccentricity': eccentricities,
    'RAAN (deg)': raan,
    'Argument of Perigee (deg)': apogees,
    'Mean Anomaly (deg)': mean_anomalies,
    'Mean Motion (rev/day)': mean_motions
})

#import ace_tools as tools; tools.display_dataframe_to_user(name="Orbital Parameters Data", dataframe=orbital_data)

# print(tle_epoch_to_datetime(23108.67589746))


# print(orbital_data)  # Display the first few rows for verification

# plot all the data

# import matplotlib.pyplot as plt

# # Plot the orbital parameters
# plt.figure(figsize=(12, 8))
# plt.plot(orbital_data['Epoch'], orbital_data['Inclination (deg)'], label='Inclination (deg)')
# plt.plot(orbital_data['Epoch'], orbital_data['Eccentricity'], label='Eccentricity')
# plt.plot(orbital_data['Epoch'], orbital_data['RAAN (deg)'], label='RAAN (deg)')
# plt.plot(orbital_data['Epoch'], orbital_data['Argument of Perigee (deg)'], label='Argument of Perigee (deg)')
# plt.plot(orbital_data['Epoch'], orbital_data['Mean Anomaly (deg)'], label='Mean Anomaly (deg)')
# plt.plot(orbital_data['Epoch'], orbital_data['Mean Motion (rev/day)'], label='Mean Motion (rev/day)')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.title('Orbital Parameters Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()


