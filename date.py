from datetime import datetime, timedelta

# Given GPS time in milliseconds
gps_millis = 30051.28888888889

# GPS epoch: January 6, 1980
gps_epoch = datetime(1980, 1, 6)

# Convert milliseconds to seconds
gps_seconds = gps_millis / 1000

# Add GPS seconds to the epoch
utc_time = gps_epoch + timedelta(seconds=gps_seconds)

# Account for leap seconds (18 as of 2024)
utc_time_adjusted = utc_time - timedelta(seconds=18)

print(utc_time_adjusted)
