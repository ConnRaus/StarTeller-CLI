#!/usr/bin/env python3
"""
StarTeller-CLI - Optimal Deep Sky Object Viewing Time Calculator
A command-line tool to find the best times to observe deep sky objects throughout the year.
"""

import os
import pickle
import hashlib
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder
from tqdm import tqdm

try:
    from .catalog_manager import load_ngc_catalog
except ImportError:
    from catalog_manager import load_ngc_catalog

# Process counts for parallel operations
NUM_WORKERS = cpu_count() or 8
NIGHT_MIDPOINT_WORKERS = 2


def get_user_data_dir():
    if sys.platform == 'win32':
        # Windows: %LOCALAPPDATA%\StarTeller-CLI
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return Path(base) / 'StarTeller-CLI'
    elif sys.platform == 'darwin':
        # macOS: ~/Library/Application Support/StarTeller-CLI
        return Path.home() / 'Library' / 'Application Support' / 'StarTeller-CLI'
    else:
        # Linux: ~/.local/share/starteller-cli
        return Path.home() / '.local' / 'share' / 'starteller-cli'


def get_cache_dir():
    if sys.platform == 'win32':
        # Windows: %LOCALAPPDATA%\StarTeller-CLI\cache
        return get_user_data_dir() / 'cache'
    elif sys.platform == 'darwin':
        # macOS: ~/Library/Caches/StarTeller-CLI
        return Path.home() / 'Library' / 'Caches' / 'StarTeller-CLI'
    else:
        # Linux: ~/.cache/starteller-cli
        return Path.home() / '.cache' / 'starteller-cli'


def get_output_dir():
    return Path.cwd() / 'starteller_output'

# Global variables for worker processes
_worker_latitude = None
_worker_longitude = None
_worker_lst_array = None
_worker_night_dates = None
_worker_night_midpoint_ts = None 
_worker_night_dark_start_ts = None
_worker_night_dark_end_ts = None
_worker_local_tz = None
_worker_local_tz_str = None

def _calculate_lst(jd_array, longitude_deg):
    """
    Calculate Local Sidereal Time for an array of Julian dates
    Chapter 12 of https://auass.com/wp-content/uploads/2021/01/Astronomical-Algorithms.pdf

    Takes: numpy array of Julian dates (UT1) and longitude in degrees
    Returns: numpy array of Local Sidereal Times in radians
    """
    
    # Find 0h UT1 Julian date (needed for equation) and fraction of day
    jd_floor = np.floor(jd_array - 0.5) + 0.5  
    day_fraction = jd_array - jd_floor  
    
    # Julian centuries (Eq 12.1)
    T = (jd_floor - 2451545.0) / 36525.0 
    
    # GMST at midnight in degrees (Eq 12.3)
    gmst_midnight = 100.46061837 + 36000.770053608 * T + 0.000387933 * T**2 - (T**3) / 38710000.0
    
    # Add rotation for time since midnight (360.98564736629 deg per day = sidereal rate)
    gmst_deg = gmst_midnight + 360.98564736629 * day_fraction
    
    # Convert to LST by adding longitude
    lst_deg = gmst_deg + longitude_deg
    
    # Normalize to 0-360
    lst_deg = lst_deg % 360.0
    
    # Convert to radians
    return np.deg2rad(lst_deg)

def _precess_coordinates(ra_j2000_deg, dec_j2000_deg, jd_target):
    """
    Precess coordinates from J2000.0 epoch to target Julian date.

    Chapter 21 of https://auass.com/wp-content/uploads/2021/01/Astronomical-Algorithms.pdf
    
    Takes: Right Ascension and Declination in J2000 degrees, Target Julian Date
    Returns: New RA and Dec in degrees
    """
    # Julian centuries from J2000.0
    t = (jd_target - 2451545.0) / 36525.0
    
    # Accurate to within a few arcseconds for dates within ~100 years of J2000.0
    zeta = 2306.2181*t + 0.30188*t**2 - 0.017998*t**3
    z = 2306.2181*t + 1.09468*t**2 + 0.018203*t**3
    theta = 2004.3109*t-0.42665*t**2 - 0.041833*t**3
    
    # Convert to radians
    zeta_rad = np.deg2rad(zeta / 3600.0)
    z_rad = np.deg2rad(z / 3600.0)
    theta_rad = np.deg2rad(theta / 3600.0)
    
    ra_rad = np.deg2rad(ra_j2000_deg)
    dec_rad = np.deg2rad(dec_j2000_deg)
    
    # Precession formulas
    A = np.cos(dec_rad) * np.sin(ra_rad + zeta_rad)
    B = np.cos(theta_rad) * np.cos(dec_rad) * np.cos(ra_rad + zeta_rad) - np.sin(theta_rad) * np.sin(dec_rad)
    C = np.sin(theta_rad) * np.cos(dec_rad) * np.cos(ra_rad + zeta_rad) + np.cos(theta_rad) * np.sin(dec_rad)
    
    # New declination
    dec_new_rad = np.arcsin(np.clip(C, -1.0, 1.0))
    
    # New right ascension
    ra_new_rad = np.arctan2(A, B) + z_rad
    
    # Convert back to degrees
    ra_new_deg = np.rad2deg(ra_new_rad) % 360.0
    dec_new_deg = np.rad2deg(dec_new_rad)
    
    return ra_new_deg, dec_new_deg

def _calc_alt_az(ra_deg, dec_deg, lst_rad, lat_rad):
    """
    Calculate altitude and azimuth
    
    Takes: Right Ascension, Declination, Local Sidereal Time array, and observer latitude
    Returns: alt_deg, az_deg: numpy arrays of altitude and azimuth in degrees
    """
    # Convert to radians
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    # Hour angle = LST - RA
    ha_rad = lst_rad - ra_rad
    
    # Altitude calculation
    sin_alt = (np.sin(dec_rad) * np.sin(lat_rad) + 
               np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad))
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    
    # Azimuth calculation
    cos_alt = np.cos(alt_rad)
    # Avoid division by zero at zenith
    cos_alt = np.where(np.abs(cos_alt) < 1e-10, 1e-10, cos_alt)
    
    sin_az = -np.cos(dec_rad) * np.sin(ha_rad) / cos_alt
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * np.sin(alt_rad)) / (np.cos(lat_rad) * cos_alt)
    
    az_rad = np.arctan2(sin_az, cos_az)
    
    # Convert to degrees
    alt_deg = np.rad2deg(alt_rad)
    az_deg = np.rad2deg(az_rad) % 360.0  # Normalize to 0-360
    
    return alt_deg, az_deg

def _init_worker(latitude, longitude, t_array_data, night_dates_tuples, 
                 night_midpoint_ts, night_dark_start_ts, night_dark_end_ts, local_tz_str):
    """Initialize worker process with shared data (called once per worker)."""
    global _worker_latitude, _worker_longitude, _worker_lst_array
    global _worker_night_dates, _worker_night_midpoint_ts
    global _worker_night_dark_start_ts, _worker_night_dark_end_ts
    global _worker_local_tz, _worker_local_tz_str
    
    from datetime import date
    import numpy as np
    
    _worker_latitude = latitude
    _worker_longitude = longitude
    _worker_local_tz_str = local_tz_str
    _worker_local_tz = pytz.timezone(local_tz_str)
    
    # Convert Unix timestamps to Julian dates and pre-calculate LST
    # Pre-calculate Local Sidereal Time for all nights
    t_array_np = np.asarray(t_array_data)
    jd_array = t_array_np / 86400.0 + 2440587.5
    _worker_lst_array = _calculate_lst(jd_array, longitude)
    
    # Reconstruct dates from (year, month, day) tuples
    _worker_night_dates = [date(y, m, d) for y, m, d in night_dates_tuples]
    
    # Store timestamps directly - create datetime objects on-demand in worker function
    _worker_night_midpoint_ts = night_midpoint_ts
    _worker_night_dark_start_ts = night_dark_start_ts
    _worker_night_dark_end_ts = night_dark_end_ts

def _process_object_worker(args):
    """Worker function to process a single object (runs in subprocess)."""
    global _worker_latitude, _worker_longitude, _worker_lst_array
    global _worker_night_dates, _worker_night_midpoint_ts
    global _worker_night_dark_start_ts, _worker_night_dark_end_ts
    global _worker_local_tz, _worker_local_tz_str
    
    # Only receive minimal object data
    obj_id, ra, dec, name, obj_type, min_altitude, direction_filter = args
    
    import numpy as np
    
    try:
        lat_rad = np.deg2rad(_worker_latitude)
        alt_degrees, az_degrees = _calc_alt_az(ra, dec, _worker_lst_array, lat_rad)
        above_altitude = alt_degrees >= min_altitude
        
        if direction_filter:
            min_az, max_az = direction_filter
            if min_az <= max_az:
                meets_direction = (az_degrees >= min_az) & (az_degrees <= max_az)
            else:
                meets_direction = (az_degrees >= min_az) | (az_degrees <= max_az)
            valid_mask = above_altitude & meets_direction
        else:
            valid_mask = above_altitude
        
        total_good_nights = int(np.sum(valid_mask))
        
        if total_good_nights == 0:
            return (obj_id, name, obj_type, ra, dec,'N/A', 'N/A', 'Never visible', 'N/A', 
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 0, 'N/A', 'N/A')
        
        # Find best night
        masked_altitudes = np.where(valid_mask, alt_degrees, -999)
        best_idx = int(np.argmax(masked_altitudes))
        
        best_altitude = round(float(alt_degrees[best_idx]), 1)
        best_azimuth = round(float(az_degrees[best_idx]), 1)
        best_date = _worker_night_dates[best_idx]
        
        # Create datetime objects from timestamps (only for best night)
        best_midpoint = datetime.fromtimestamp(_worker_night_midpoint_ts[best_idx], tz=_worker_local_tz)
        best_dark_start = datetime.fromtimestamp(_worker_night_dark_start_ts[best_idx], tz=_worker_local_tz)
        best_dark_end = datetime.fromtimestamp(_worker_night_dark_end_ts[best_idx], tz=_worker_local_tz)
        
        # Calculate rise/set times by sampling altitude every 15 mins
        start_ts = _worker_night_dark_start_ts[best_idx]
        end_ts = _worker_night_dark_end_ts[best_idx]
        
        num_samples = 48
        sample_ts = np.linspace(start_ts, end_ts, num_samples)
        jd_samples = sample_ts / 86400.0 + 2440587.5
        lst_samples = _calculate_lst(jd_samples, _worker_longitude)
        
        sample_alt, sample_az = _calc_alt_az(ra, dec, lst_samples, lat_rad)
        
        # Apply direction filter if specified
        def meets_dir(az):
            if direction_filter is None:
                return True
            min_az, max_az = direction_filter
            if min_az <= max_az:
                return min_az <= az <= max_az
            return az >= min_az or az <= max_az
        
        # Find visibility mask
        visible = (sample_alt >= min_altitude)
        if direction_filter:
            dir_ok = np.array([meets_dir(az) for az in sample_az])
            visible = visible & dir_ok
        
        # Find rise and set times by looking for transitions
        rise_idx = None
        set_idx = None
        
        # Find first transition from not-visible to visible (rise)
        for i in range(len(visible) - 1):
            if not visible[i] and visible[i + 1]:
                rise_idx = i + 1
                break
        
        # Find last transition from visible to not-visible (set)
        for i in range(len(visible) - 1, 0, -1):
            if visible[i - 1] and not visible[i]:
                set_idx = i - 1
                break
        
        # Determine rise/set times and directions
        dark_duration_hours = (best_dark_end - best_dark_start).total_seconds() / 3600
        
        if visible[0] and visible[-1]:
            # Visible entire night
            rise_time = best_dark_start.strftime('%H:%M')
            set_time = best_dark_end.strftime('%H:%M')
            rise_dir = _azimuth_to_cardinal(sample_az[0])
            set_dir = _azimuth_to_cardinal(sample_az[-1])
            duration = round(dark_duration_hours, 1)
        elif visible[0]:
            # Visible at start, sets during night
            rise_time = best_dark_start.strftime('%H:%M')
            rise_dir = _azimuth_to_cardinal(sample_az[0])
            if set_idx is not None:
                set_datetime = datetime.fromtimestamp(sample_ts[set_idx], tz=_worker_local_tz)
                set_time = set_datetime.strftime('%H:%M')
                set_dir = _azimuth_to_cardinal(sample_az[set_idx])
                duration = round((set_datetime - best_dark_start).total_seconds() / 3600, 1)
            else:
                set_time = best_dark_end.strftime('%H:%M')
                set_dir = _azimuth_to_cardinal(sample_az[-1])
                duration = round(dark_duration_hours, 1)
        elif visible[-1]:
            # Rises during night, visible at end
            set_time = best_dark_end.strftime('%H:%M')
            set_dir = _azimuth_to_cardinal(sample_az[-1])
            if rise_idx is not None:
                rise_datetime = datetime.fromtimestamp(sample_ts[rise_idx], tz=_worker_local_tz)
                rise_time = rise_datetime.strftime('%H:%M')
                rise_dir = _azimuth_to_cardinal(sample_az[rise_idx])
                duration = round((best_dark_end - rise_datetime).total_seconds() / 3600, 1)
            else:
                rise_time = best_dark_start.strftime('%H:%M')
                rise_dir = _azimuth_to_cardinal(sample_az[0])
                duration = round(dark_duration_hours, 1)
        else:
            # Object rises AND sets during the night
            if rise_idx is not None and set_idx is not None:
                rise_datetime = datetime.fromtimestamp(sample_ts[rise_idx], tz=_worker_local_tz)
                set_datetime = datetime.fromtimestamp(sample_ts[set_idx], tz=_worker_local_tz)
                rise_time = rise_datetime.strftime('%H:%M')
                set_time = set_datetime.strftime('%H:%M')
                rise_dir = _azimuth_to_cardinal(sample_az[rise_idx])
                set_dir = _azimuth_to_cardinal(sample_az[set_idx])
                duration = round((set_datetime - rise_datetime).total_seconds() / 3600, 1)
            elif rise_idx is not None:
                rise_datetime = datetime.fromtimestamp(sample_ts[rise_idx], tz=_worker_local_tz)
                rise_time = rise_datetime.strftime('%H:%M')
                rise_dir = _azimuth_to_cardinal(sample_az[rise_idx])
                set_time = best_dark_end.strftime('%H:%M')
                set_dir = _azimuth_to_cardinal(sample_az[-1])
                duration = round((best_dark_end - rise_datetime).total_seconds() / 3600, 1)
            elif set_idx is not None:
                set_datetime = datetime.fromtimestamp(sample_ts[set_idx], tz=_worker_local_tz)
                rise_time = best_dark_start.strftime('%H:%M')
                rise_dir = _azimuth_to_cardinal(sample_az[0])
                set_time = set_datetime.strftime('%H:%M')
                set_dir = _azimuth_to_cardinal(sample_az[set_idx])
                duration = round((set_datetime - best_dark_start).total_seconds() / 3600, 1)
            else:
                # Fallback - shouldn't happen if object is visible at midpoint
                rise_time = best_midpoint.strftime('%H:%M')
                set_time = best_midpoint.strftime('%H:%M')
                rise_dir = _azimuth_to_cardinal(best_azimuth)
                set_dir = _azimuth_to_cardinal(best_azimuth)
                duration = round(np.sum(visible) / num_samples * dark_duration_hours, 1)
        
        return (obj_id, name, obj_type, ra, dec, best_date, best_midpoint.strftime('%H:%M'),
                best_altitude, best_azimuth, _azimuth_to_cardinal(best_azimuth),
                rise_time, rise_dir, set_time, set_dir, duration,
                total_good_nights, total_good_nights,
                best_dark_start.strftime('%H:%M'), best_dark_end.strftime('%H:%M'))
        
    except Exception as e:
        return (obj_id, name, obj_type, ra, dec, 'N/A', 'N/A', 'Error', 'N/A', 
                'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 0, 'N/A', 'N/A')

def _calc_sun_position(jd_array):
    """
    Calculate sun's RA and Dec for an array of julian dates.
    Equations from https://aa.usno.navy.mil/faq/sun_approx
    
    Takes: numpy array of julian dates
    Returns: numpy arrays of sun RA and Dec in degrees 
    """
    # Days since J2000.0
    n = jd_array - 2451545.0
    
    # Mean anomaly of the Sun (degrees)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)

    # Mean longitude of the Sun (degrees)
    q = (280.459 + 0.985647436 * n) % 360.0
    
    # Apprent ecliptic longitude of the Sun (degrees)
    L = q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    L_rad = np.deg2rad(L)
    
    # Obliquity of the ecliptic (degrees)
    e = np.deg2rad(23.439 - 0.00000036 * n)
    
    # Sun's Right Ascension
    ra_rad = np.arctan2(np.cos(e) * np.sin(L_rad), np.cos(L_rad))
    ra_deg = np.rad2deg(ra_rad) % 360.0
    
    # Sun's Declination
    dec_rad = np.arcsin(np.sin(e) * np.sin(L_rad))
    dec_deg = np.rad2deg(dec_rad)
    
    return ra_deg, dec_deg

def _calc_sun_altitude(jd_array, latitude, longitude):
    """
    Calculate sun altitude for an array of julian dates

    Takes: numpy array of julian dates, lat(deg), long(deg)
    Returns: numpy array of sun altitudes in degrees
    """
    # Get sun RA/Dec
    sun_ra, sun_dec = _calc_sun_position(jd_array)
    
    # Calculate Local Sidereal Time
    lst_rad = _calculate_lst(jd_array, longitude)
    
    # Calculate altitude
    lat_rad = np.deg2rad(latitude)
    alt_deg, _ = _calc_alt_az(sun_ra, sun_dec, lst_rad, lat_rad)
    
    return alt_deg

# Worker function for parallel year calculation
def _calculate_year_midpoints_worker(args):
    """
    Worker function to calculate night midpoints for a single year.
    
    Takes: Tuple of (year, latitude, longitude, local_tz_str, location_hash)
    Returns: tuple: (year, list of night midpoints) or (year, None) on error
    """
    year, latitude, longitude, local_tz_str, location_hash = args
    
    try:
        from datetime import date, datetime, timedelta
        import pytz
        import numpy as np
        
        local_tz = pytz.timezone(local_tz_str)
        
        # Calculate full year
        full_year_start = date(year, 1, 1)
        full_year_days = (date(year + 1, 1, 1) - full_year_start).days
        
        # For current year, don't calculate past today + 365 days for efficiency
        if year == datetime.now().year:
            max_date = date.today() + timedelta(days=365)
            if date(year + 1, 1, 1) > max_date:
                full_year_days = (max_date - full_year_start).days
        
        # 81 samples per day (15:00 to 11:00 next day, every 15 min)
        samples_per_day = 81
        total_samples = full_year_days * samples_per_day
        
        # Build timestamp array
        base_timestamps = np.zeros(total_samples, dtype=np.float64)
        day_indices = np.zeros(total_samples, dtype=np.int32)
        
        for day_offset in range(full_year_days):
            check_date = full_year_start + timedelta(days=day_offset)
            # 15:00 local time
            afternoon = local_tz.localize(datetime.combine(check_date, datetime.min.time().replace(hour=15)))
            base_ts = afternoon.timestamp()
            
            start_idx = day_offset * samples_per_day
            for i in range(samples_per_day):
                base_timestamps[start_idx + i] = base_ts + i * 900  # 900 seconds = 15 minutes
                day_indices[start_idx + i] = day_offset
        
        # Convert all timestamps to Julian dates
        jd_array = base_timestamps / 86400.0 + 2440587.5
        
        # Calculate sun altitude for all times at once
        sun_altitudes = _calc_sun_altitude(jd_array, latitude, longitude)
        is_dark = sun_altitudes < -18.0
        
        # Find dark periods for each day
        night_midpoints = []
        
        for day_offset in range(full_year_days):
            check_date = full_year_start + timedelta(days=day_offset)
            
            # Get indices for this day's samples
            start_idx = day_offset * samples_per_day
            end_idx = start_idx + samples_per_day
            
            day_altitudes = sun_altitudes[start_idx:end_idx]
            day_dark = is_dark[start_idx:end_idx]
            day_timestamps = base_timestamps[start_idx:end_idx]
            
            # Find the dark period from coarse samples
            dark_start_sample = None
            dark_end_sample = None
            
            # Find dark start time
            for i in range(len(day_dark) - 1):
                if not day_dark[i] and day_dark[i + 1]:
                    dark_start_sample = i
                    break
            
            # Find dark end time
            for i in range(len(day_dark) - 1, 0, -1):
                if day_dark[i - 1] and not day_dark[i]:
                    dark_end_sample = i
                    break
            
            # Handle edge cases
            if dark_start_sample is None:
                if day_dark[0]:
                    dark_start_sample = 0
                else:
                    continue
            
            if dark_end_sample is None:
                if day_dark[-1]:
                    dark_end_sample = len(day_dark) - 1
                else:
                    continue
            
            # Linear interpolate dark start time
            if dark_start_sample > 0:
                ts0 = day_timestamps[dark_start_sample - 1]
                ts1 = day_timestamps[dark_start_sample]
                alt0 = day_altitudes[dark_start_sample - 1]
                alt1 = day_altitudes[dark_start_sample]
                
                if alt0 != alt1:
                    fraction = (-18.0 - alt0) / (alt1 - alt0)
                    dark_start_ts = ts0 + (ts1 - ts0) * fraction
                else:
                    dark_start_ts = day_timestamps[dark_start_sample]
            else:
                dark_start_ts = day_timestamps[dark_start_sample]
            
            # Linear interpolate dark end time
            if dark_end_sample < len(day_dark) - 1:
                ts0 = day_timestamps[dark_end_sample]
                ts1 = day_timestamps[dark_end_sample + 1]
                alt0 = day_altitudes[dark_end_sample]
                alt1 = day_altitudes[dark_end_sample + 1]
                
                if alt0 != alt1:
                    fraction = (-18.0 - alt0) / (alt1 - alt0)
                    dark_end_ts = ts0 + (ts1 - ts0) * fraction
                else:
                    dark_end_ts = day_timestamps[dark_end_sample]
            else:
                dark_end_ts = day_timestamps[dark_end_sample]
            
            # Calculate midpoint and convert timestamps to datetime objects
            if dark_end_ts > dark_start_ts:
                midpoint_ts = (dark_start_ts + dark_end_ts) / 2
                
                # Convert timestamps to timezone-aware datetime objects
                dark_start = datetime.fromtimestamp(dark_start_ts, tz=local_tz)
                dark_end = datetime.fromtimestamp(dark_end_ts, tz=local_tz)
                midpoint = datetime.fromtimestamp(midpoint_ts, tz=local_tz)
                
                night_midpoints.append((check_date, midpoint, dark_start, dark_end))
        
        return (year, night_midpoints)
    except Exception as e:
        print(f"Error calculating midpoints for year {year}: {e}")
        return (year, None)

def _azimuth_to_cardinal(azimuth):
    if isinstance(azimuth, np.ndarray):
        # Vectorized version for numpy arrays
        result = np.empty(azimuth.shape, dtype='U2')
        result[:] = 'N'
        result[(azimuth >= 22.5) & (azimuth < 67.5)] = 'NE'
        result[(azimuth >= 67.5) & (azimuth < 112.5)] = 'E'
        result[(azimuth >= 112.5) & (azimuth < 157.5)] = 'SE'
        result[(azimuth >= 157.5) & (azimuth < 202.5)] = 'S'
        result[(azimuth >= 202.5) & (azimuth < 247.5)] = 'SW'
        result[(azimuth >= 247.5) & (azimuth < 292.5)] = 'W'
        result[(azimuth >= 292.5) & (azimuth < 337.5)] = 'NW'
        return result
    else:
        # Scalar version
        directions = [
            (0, 22.5, "N"), (22.5, 67.5, "NE"), (67.5, 112.5, "E"), (112.5, 157.5, "SE"),
            (157.5, 202.5, "S"), (202.5, 247.5, "SW"), (247.5, 292.5, "W"), (292.5, 337.5, "NW"),
            (337.5, 360, "N")
        ]
        for min_az, max_az, direction in directions:
            if min_az <= azimuth < max_az:
                return direction
        return "N"

class StarTellerCLI:
    # ============================================================================
    # CONSTRUCTOR AND SETUP
    # ============================================================================
    
    def __init__(self, latitude, longitude, elevation=0, catalog_filter="all"):
        """
        Initialize StarTellerCLI with observer location.      
        Args: lat(float), long(float), *elevation(m), *catalog filter(string)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        
        # Create location hash for caching
        self.location_hash = self._generate_location_hash()
        
        # Detect local timezone
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=latitude, lng=longitude)
        if tz_name:
            self.local_tz = pytz.timezone(tz_name)
            print(f"✓ Timezone: {tz_name}")
        else:
            self.local_tz = pytz.UTC
            print("✓ Timezone: UTC (could not auto-detect)")
        
        # Load catalog
        self.dso_catalog = self._setup_catalog(catalog_filter)
    
    def _generate_location_hash(self):
        """Generate a unique hash for this location for caching purposes."""
        # Round coordinates to 4 decimal places (~11m precision) for caching
        lat_rounded = round(self.latitude, 4)
        lon_rounded = round(self.longitude, 4)
        location_string = f"{lat_rounded},{lon_rounded}"
        return hashlib.md5(location_string.encode()).hexdigest()[:8]
    
    def _setup_catalog(self, catalog_filter):
        # Takes catalog filter, returns catalog dictionary
        filter_names = {
            "messier": "Messier Objects",
            "ic": "IC Objects", 
            "ngc": "NGC Objects",
            "all": "All Objects"
        }
        
        try:
            # Load NGC catalog with filter
            catalog_df = load_ngc_catalog(catalog_filter=catalog_filter)
            
            if catalog_df.empty:
                print("Failed to load NGC catalog - please ensure NGC.csv file is present")
                return {}
            
            # Convert to dictionary, allows instant access by obj_id
            # Also better to give dict for multiprocessing than pickled pandas db
            catalog_dict = {}
            for _, row in catalog_df.iterrows():
                obj_id = row['object_id']
                
                # Use common name if available, otherwise use name
                display_name = row.get('common_name', '') or row['name']
                
                catalog_dict[obj_id] = {
                    'ra': float(row['ra_deg']),
                    'dec': float(row['dec_deg']),
                    'name': display_name,
                    'type': row['type']
                }
            
            print(f"✓ Catalog: {len(catalog_dict)} {filter_names.get(catalog_filter, 'objects')}")
            return catalog_dict
            
        except Exception as e:
            print(f"Error loading catalog: {e}")
            print("Please ensure NGC.csv file is downloaded from OpenNGC")
            return {}
    
    # ============================================================================
    # CACHE MANAGEMENT
    # ============================================================================
    
    def _get_cache_filepath(self, year=None):
        if year is None:
            year = datetime.now().year
        cache_dir = get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"night_midpoints_{self.location_hash}_{year}.pkl"
    
    def _save_cache(self, night_midpoints, year):
        try:
            cache_file = self._get_cache_filepath(year)
            cache_data = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'timezone': str(self.local_tz),
                'year': year,
                'night_midpoints': night_midpoints,
                'created_date': datetime.now().isoformat()
            }
            
            with open(str(cache_file), 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            print(f"Warning: Could not save night midpoints cache: {e}")
            return False
    
    def _load_cache(self, year):
        try:
            cache_file = self._get_cache_filepath(year)
            if not cache_file.exists():
                return None
            
            with open(str(cache_file), 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify the cache is for the same location and timezone
            if (abs(cache_data['latitude'] - self.latitude) < 0.0001 and 
                abs(cache_data['longitude'] - self.longitude) < 0.0001 and
                cache_data['timezone'] == str(self.local_tz)):
                
                pass  # Cache loaded
                return cache_data['night_midpoints']
            else:
                # Cache mismatch, recalculate
                return None
                
        except Exception as e:
            print(f"Warning: Could not load night midpoints cache: {e}")
            return None
    
    # ============================================================================
    # NIGHT MIDPOINT CALCULATION
    # ============================================================================
    
    def get_night_midpoints(self, start_date=None, days=365):
        """
        Get night midpoints for period, use cache when available.
        
        Takes start date and *amount of days to calculate
        Returns list of (date, midpoint_datetime_local, dark_start_local, dark_end_local) tuples
        """
        from datetime import date
        
        if start_date is None:
            start_date = date.today()
        
        end_date = start_date + timedelta(days=days-1)
        
        # Check for cached data
        years_needed = set()
        current_date = start_date
        for day_offset in range(days):
            check_date = current_date + timedelta(days=day_offset)
            years_needed.add(check_date.year)
        
        # Try to load cached data for all needed years
        all_cached_midpoints = []
        missing_years = []
        
        for year in sorted(years_needed):
            cached_midpoints = self._load_cache(year)
            if cached_midpoints:
                # Filter to only include dates in our range
                for date_obj, midpoint, dark_start, dark_end in cached_midpoints:
                    if start_date <= date_obj <= end_date:
                        all_cached_midpoints.append((date_obj, midpoint, dark_start, dark_end))
            else:
                missing_years.append(year)
        
        # If all needed data in cache, return it
        if not missing_years and len(all_cached_midpoints) >= days * 0.95:  # Allow 5% missing for edge cases
            return sorted(all_cached_midpoints, key=lambda x: x[0])
        
        # Else calculate missing years
        if missing_years:
            import time
            t_calc_start = time.perf_counter()
            print(f"Calculating night darkness times for {len(missing_years)} year(s)...")
        
        all_calculated_midpoints = []
        
        if missing_years:
            local_tz_str = str(self.local_tz)
            year_args = [
                (year, self.latitude, self.longitude, local_tz_str, self.location_hash)
                for year in sorted(missing_years)
            ]
            
            # Calculate years in parallel
            if len(missing_years) > 1 and NIGHT_MIDPOINT_WORKERS > 1:
                with Pool(processes=min(NIGHT_MIDPOINT_WORKERS, len(missing_years))) as pool:
                    results = pool.map(_calculate_year_midpoints_worker, year_args)
            else:
                # Single year or single worker - calculate sequentially
                results = [_calculate_year_midpoints_worker(args) for args in year_args]
            
            # Process results and save to cache
            for year, year_midpoints in results:
                if year_midpoints:
                    self._save_cache(year_midpoints, year)
                    all_calculated_midpoints.extend(year_midpoints)
            
            print(f"✓ Night calculations completed in {time.perf_counter() - t_calc_start:.2f}s")
        
        # Combine cached and calculated data
        all_midpoints = all_cached_midpoints + all_calculated_midpoints
        
        # Filter to requested date range and sort
        result = []
        for date_obj, midpoint, dark_start, dark_end in all_midpoints:
            if start_date <= date_obj <= end_date:
                result.append((date_obj, midpoint, dark_start, dark_end))
        
        return sorted(result, key=lambda x: x[0])
    
    # ============================================================================
    # MAIN FUNCTIONALITY
    # ============================================================================
    
    def find_optimal_viewing_times(self, min_altitude=20, direction_filter=None):
        """
        Find optimal viewing times for all objects in the catalog
        
        Takes *minimum altitude in degrees, *direction filter (tuple)
        Returns pandas final result dataframe
        """
        print("Calculating optimal viewing times for deep sky objects...")
        print(f"Observer location: {self.latitude:.2f}°, {self.longitude:.2f}°")
        print(f"Local timezone: {self.local_tz}")
        print(f"Minimum altitude: {min_altitude}°")
        print("Dark time criteria: Sun below -18° (astronomical twilight)")
        
        if direction_filter:
            print(f"Direction filter: {direction_filter[0]}° to {direction_filter[1]}° azimuth")
        
        # Remove duplicates (keep messier names over ngc ids and other catalogs)
        unique_objects = {}
        for obj_id, obj_data in self.dso_catalog.items():
            coord_key = (round(obj_data['ra'], 4), round(obj_data['dec'], 4))
            
            if coord_key not in unique_objects:
                unique_objects[coord_key] = (obj_id, obj_data)
            else:
                existing_id, existing_data = unique_objects[coord_key]
                # Prefer Messier objects (M45) over Melotte (Mel022) and other catalogs
                if obj_id.startswith('M') and not existing_id.startswith('M'):
                    unique_objects[coord_key] = (obj_id, obj_data)
                # Also prefer Messier over Melotte even if Melotte was added first
                elif obj_id.startswith('M') and existing_id.startswith('Mel'):
                    unique_objects[coord_key] = (obj_id, obj_data)
        
        print(f"Processing {len(unique_objects)} unique objects")
        
        import time
        t_total_start = time.perf_counter()
        
        # Get night midpoints (cached)
        night_midpoints = self.get_night_midpoints()
        num_nights = len(night_midpoints)
        
        # Pre-convert data to serializable formats for worker initialization
        utc_tz = pytz.UTC
        
        # Single pass through night_midpoints to extract all data at once
        night_dates_tuples = []
        t_array_data = np.empty(num_nights, dtype=np.float64)
        night_midpoint_ts = np.empty(num_nights, dtype=np.float64)
        night_dark_start_ts = np.empty(num_nights, dtype=np.float64)
        night_dark_end_ts = np.empty(num_nights, dtype=np.float64)
        
        for i, (date_obj, midpoint, dark_start, dark_end) in enumerate(night_midpoints):
            night_dates_tuples.append((date_obj.year, date_obj.month, date_obj.day))
            t_array_data[i] = midpoint.astimezone(utc_tz).timestamp()
            night_midpoint_ts[i] = midpoint.timestamp()
            night_dark_start_ts[i] = dark_start.timestamp()
            night_dark_end_ts[i] = dark_end.timestamp()
        
        items = list(unique_objects.values())
        local_tz_str = str(self.local_tz)
        
        # Calculate mean epoch for precession (middle of calculation period)
        # Use the middle date of the night midpoints for the target epoch
        if night_midpoints:
            mid_idx = len(night_midpoints) // 2
            mid_date, mid_midpoint, _, _ = night_midpoints[mid_idx]
            # Convert to Julian date for precession
            mid_jd = mid_midpoint.timestamp() / 86400.0 + 2440587.5
            epoch_date_str = mid_midpoint.strftime('%Y-%m-%d')
        else:
            # Fallback to current date
            from datetime import date
            today = date.today()
            mid_jd = (datetime(today.year, 7, 1, 12, 0, 0, tzinfo=pytz.UTC).timestamp() / 86400.0 + 2440587.5)
            epoch_date_str = f"{today.year}-07-01"
        
        # Precess coordinates from J2000.0 to current epoch
        # NGC.csv provides coordinates in J2000.0 epoch, but we need current epoch for accurate calculations
        print(f"Precessing coordinates from J2000.0 to epoch {epoch_date_str} (accounting for Earth's precession)...")
        
        # Prepare work items with precessed coordinates
        work_items = []
        for obj_id, obj_data in items:
            ra_j2000 = obj_data['ra']
            dec_j2000 = obj_data['dec']
            # Precess from J2000.0 to current epoch
            ra_current, dec_current = _precess_coordinates(ra_j2000, dec_j2000, mid_jd)
            work_items.append(
                (obj_id, ra_current, dec_current, obj_data['name'], obj_data['type'],
                 min_altitude, direction_filter)
            )
        
        results = []
        columns = ['Object', 'Name', 'Type', 'Right_Ascension', 'Declination', 'Best_Date', 'Best_Time_Local',
                   'Max_Altitude_deg', 'Azimuth_deg', 'Direction',
                   'Rise_Time_Local', 'Rise_Direction', 'Set_Time_Local', 'Set_Direction',
                   'Observing_Duration_Hours', 'Dark_Nights_Per_Year', 'Good_Viewing_Periods',
                   'Dark_Start_Local', 'Dark_End_Local']
        
        with Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(self.latitude, self.longitude, t_array_data, night_dates_tuples,
                      night_midpoint_ts, night_dark_start_ts, night_dark_end_ts, local_tz_str)
        ) as pool:
            chunksize = max(100, len(work_items) // NUM_WORKERS)
            for result in tqdm(
                pool.imap_unordered(_process_object_worker, work_items, chunksize=chunksize),
                total=len(items),
                desc=f"Processing {len(items)} objects",
                unit="obj"
            ):
                results.append(result)
        
        print(f"✓ Processing completed in {time.perf_counter() - t_total_start:.2f}s")
        
        # Convert tuple results to DataFrame
        results_df = pd.DataFrame(results, columns=columns)
        results_df['Timezone'] = local_tz_str
        
        if results_df.empty:
            return results_df
        
        # Put never visible objects at bottom of list, 
        # sort everything else by closest best date
        def sort_key(x):
            return (isinstance(x, str) and (x == 'Never visible' or x == 'Error'))

        results_df['never_visible'] = results_df['Max_Altitude_deg'].apply(sort_key)

        results_df = results_df.sort_values(
            by=['never_visible', 'Best_Date', 'Object'],
            ascending=[True, True, True]
        ).drop('never_visible', axis=1)
        
        return results_df


def save_location(latitude, longitude, elevation):
    try:
        user_data_dir = get_user_data_dir()
        user_data_dir.mkdir(parents=True, exist_ok=True)
        
        location_file = user_data_dir / 'user_location.txt'
        with open(str(location_file), 'w') as f:
            f.write(f"{latitude},{longitude},{elevation}")
        print(f"✓ Location saved: {latitude:.2f}°, {longitude:.2f}°, {elevation}m")
    except Exception as e:
        print(f"Warning: Could not save location: {e}")

def load_location():
    try:
        location_file = get_user_data_dir() / 'user_location.txt'
        if not location_file.exists():
            return None
        with open(str(location_file), 'r') as f:
            data = f.read().strip().split(',')
            if len(data) == 3:
                latitude = float(data[0])
                longitude = float(data[1])
                elevation = float(data[2])
                return latitude, longitude, elevation
    except:
        pass
    return None

def get_user_location():
    saved_location = load_location()
    
    if saved_location:
        latitude, longitude, elevation = saved_location
        print(f"\nSaved location found: {latitude:.2f}°, {longitude:.2f}°, {elevation}m")
        use_saved = input("Use saved location? (y/n, default y): ").strip().lower()
        
        if use_saved in ['', 'y', 'yes']:
            return latitude, longitude, elevation
    
    # Get new location manually
    print("\nEnter your observing location:")
    latitude = float(input("Latitude (degrees, positive for North): "))
    longitude = float(input("Longitude (degrees, positive for East): "))
    elevation = float(input("Elevation (meters, optional, press Enter for 0): ") or 0)
    
    # Ask if they want to save it
    save_choice = input("Save this location for future use? (y/n, default y): ").strip().lower()
    if save_choice in ['', 'y', 'yes']:
        save_location(latitude, longitude, elevation)
    
    return latitude, longitude, elevation

def main():
    """Main function to run StarTeller-CLI."""
    print("=" * 60)
    print("                   StarTeller-CLI")
    print("        Deep Sky Object Optimal Viewing Calculator")
    print("=" * 60)
    
    # === Collect all User Input ===
    
    latitude, longitude, elevation = get_user_location()
    
    print("\nChoose catalog:")
    print("1. Messier Objects (~110 famous deep sky objects)")
    print("2. IC Objects (~5,000 Index Catalog objects)")
    print("3. NGC Objects (~8,000 New General Catalog objects)")
    print("4. All Objects (~13,000 NGC + IC objects)")
    
    catalog_choice = input("Enter choice (1-4, default 4): ").strip() or "4"

    print("\nViewing preferences:")
    min_alt = float(input("Minimum altitude (degrees, default 20): ") or 20)
    
    direction_input = input("Direction filter? (e.g., '90,180' for East-South, or Enter for no filter): ")
    direction_filter = None
    if direction_input.strip():
        try:
            min_az, max_az = map(float, direction_input.split(','))
            direction_filter = (min_az, max_az)
        except:
            print("Invalid direction format, proceeding without direction filter.")
    
    print("\n" + "=" * 60)
    print("PROCESSING...")
    print("=" * 60)
    
    # === User input collected, start processing ===

    # Create StarTellerCLI instance with appropriate catalog
    catalog_params = {
        "1": "messier",
        "2": "ic",
        "3": "ngc",
        "4": "all"
    }
    catalog_type = catalog_params.get(catalog_choice)
    st = StarTellerCLI(latitude, longitude, elevation, catalog_filter=catalog_type)
    
    if st is None:
        print("Failed to create StarTellerCLI instance. Exiting.")
        return
    
    # Calculate optimal viewing times
    results = st.find_optimal_viewing_times(min_altitude=min_alt, direction_filter=direction_filter)
    
    # Save to output directory (created in current working directory)
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"optimal_viewing_times_{datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M')}.csv"
    results.to_csv(str(filename), index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"✓ Results saved to: {filename}")
    print(f"✓ Found optimal viewing times for {len(results)} objects")
    
    # Print summary
    visible_count = len(results[results['Max_Altitude_deg'] != 'Never visible'])
    print(f"✓ {visible_count} objects visible above {min_alt}°")
    
    print("\nOpen the CSV file to see complete viewing schedule!")
    print("=" * 60)


if __name__ == "__main__":
    main() 