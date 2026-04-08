#!/usr/bin/env python3
"""
Core observation planning: coordinates, sun/night model, catalog viewing.

No CLI I/O — use starteller_cli for prompts and messages.
"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder

try:
    from .catalog_manager import load_ngc_catalog
except ImportError:
    from catalog_manager import load_ngc_catalog


def local_sidereal_time_rad(jd_array, longitude_deg):
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
    gmst_midnight = 100.46061837 + (36000.770053608 * T) + (0.000387933 * T**2) - (T**3 / 38710000.0)

    # Add rotation for time since midnight (360.98564736629 deg per day = sidereal rate)
    gmst_deg = gmst_midnight + 360.98564736629 * day_fraction

    # Convert to LST by adding longitude and normalizing to 0-360
    lst_deg = (gmst_deg + longitude_deg) % 360.0

    # Convert to radians
    return np.deg2rad(lst_deg)


def precess_equatorial_j2000(ra_j2000_deg, dec_j2000_deg, jd_target):
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


def equatorial_to_horizontal_deg(ra_deg, dec_deg, lst_rad, lat_rad):
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
    az_deg = np.rad2deg(az_rad) % 360.0

    return alt_deg, az_deg


# -------------------------------------------------------------------------------------
# 3. Sun position & night midpoints (only for get_night_midpoints → find_optimal…)
# -------------------------------------------------------------------------------------


def sun_equatorial_deg(jd_array):
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


def sun_altitude_deg(jd_array, latitude, longitude):
    """
    Calculate sun altitude for an array of julian dates

    Takes: numpy array of julian dates, lat(deg), long(deg)
    Returns: numpy array of sun altitudes in degrees
    """
    # Get sun RA/Dec
    sun_ra, sun_dec = sun_equatorial_deg(jd_array)

    # Calculate Local Sidereal Time
    lst_rad = local_sidereal_time_rad(jd_array, longitude)

    # Calculate altitude
    lat_rad = np.deg2rad(latitude)
    alt_deg, _ = equatorial_to_horizontal_deg(sun_ra, sun_dec, lst_rad, lat_rad)

    return alt_deg


def compute_year_night_midpoints(args):
    """
    Compute astronomical-dark midpoints for every night in one calendar year.

    Args: (year, latitude, longitude, local_tz_str)
    Returns: (year, list of night midpoint tuples) or (year, None) on error.
    """
    year, latitude, longitude, local_tz_str = args

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
        sun_altitudes = sun_altitude_deg(jd_array, latitude, longitude)
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
    except Exception:
        return (year, None)


# -----------------------------------------------------------------------------
# 4. Shared night arrays + per-object viewing (find_optimal_viewing_times)
# -----------------------------------------------------------------------------


class ObservationContext:
    """Precomputed LST and night metadata shared while scanning the catalog"""

    __slots__ = (
        'latitude', 'longitude', 'lst_array', 'night_dates',
        'night_midpoint_ts', 'night_dark_start_ts', 'night_dark_end_ts',
        'local_tz', 'local_tz_str',
    )

    def __init__(self, latitude, longitude, t_array_data, night_dates_tuples,
                 night_midpoint_ts, night_dark_start_ts, night_dark_end_ts, local_tz_str):
        from datetime import date

        self.latitude = latitude
        self.longitude = longitude
        self.local_tz_str = local_tz_str
        self.local_tz = pytz.timezone(local_tz_str)
        t_array_np = np.asarray(t_array_data)
        jd_array = t_array_np / 86400.0 + 2440587.5
        self.lst_array = local_sidereal_time_rad(jd_array, longitude)
        self.night_dates = [date(y, m, d) for y, m, d in night_dates_tuples]
        self.night_midpoint_ts = night_midpoint_ts
        self.night_dark_start_ts = night_dark_start_ts
        self.night_dark_end_ts = night_dark_end_ts


def compute_catalog_object_viewing(ctx, args):
    """Compute one catalog row: visibility over all nights and best-night rise/set detail."""
    obj_id, ra_j2000, dec_j2000, ra_now, dec_now, name, obj_type, messier_num, min_altitude, direction_filter = args

    try:
        lat_rad = np.deg2rad(ctx.latitude)
        alt_degrees, az_degrees = equatorial_to_horizontal_deg(ra_now, dec_now, ctx.lst_array, lat_rad)
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
            return (obj_id, name, obj_type, messier_num, ra_j2000, dec_j2000, 'N/A', 'N/A', 'Never visible', 'N/A',
                    'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 'N/A', 'N/A')

        # Find best night
        masked_altitudes = np.where(valid_mask, alt_degrees, -999)
        best_idx = int(np.argmax(masked_altitudes))

        best_altitude = round(float(alt_degrees[best_idx]), 1)
        best_azimuth = round(float(az_degrees[best_idx]), 1)
        best_date = ctx.night_dates[best_idx]

        # Create datetime objects from timestamps (only for best night)
        best_midpoint = datetime.fromtimestamp(ctx.night_midpoint_ts[best_idx], tz=ctx.local_tz)
        best_dark_start = datetime.fromtimestamp(ctx.night_dark_start_ts[best_idx], tz=ctx.local_tz)
        best_dark_end = datetime.fromtimestamp(ctx.night_dark_end_ts[best_idx], tz=ctx.local_tz)

        # Calculate rise/set times by sampling altitude every 15 mins
        start_ts = ctx.night_dark_start_ts[best_idx]
        end_ts = ctx.night_dark_end_ts[best_idx]

        num_samples = 48
        sample_ts = np.linspace(start_ts, end_ts, num_samples)
        jd_samples = sample_ts / 86400.0 + 2440587.5
        lst_samples = local_sidereal_time_rad(jd_samples, ctx.longitude)

        sample_alt, sample_az = equatorial_to_horizontal_deg(ra_now, dec_now, lst_samples, lat_rad)

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
            rise_az = round(float(sample_az[0]), 1)
            set_az = round(float(sample_az[-1]), 1)
            duration = round(dark_duration_hours, 1)
        elif visible[0]:
            # Visible at start, sets during night
            rise_time = best_dark_start.strftime('%H:%M')
            rise_az = round(float(sample_az[0]), 1)
            if set_idx is not None:
                set_datetime = datetime.fromtimestamp(sample_ts[set_idx], tz=ctx.local_tz)
                set_time = set_datetime.strftime('%H:%M')
                set_az = round(float(sample_az[set_idx]), 1)
                duration = round((set_datetime - best_dark_start).total_seconds() / 3600, 1)
            else:
                set_time = best_dark_end.strftime('%H:%M')
                set_az = round(float(sample_az[-1]), 1)
                duration = round(dark_duration_hours, 1)
        elif visible[-1]:
            # Rises during night, visible at end
            set_time = best_dark_end.strftime('%H:%M')
            set_az = round(float(sample_az[-1]), 1)
            if rise_idx is not None:
                rise_datetime = datetime.fromtimestamp(sample_ts[rise_idx], tz=ctx.local_tz)
                rise_time = rise_datetime.strftime('%H:%M')
                rise_az = round(float(sample_az[rise_idx]), 1)
                duration = round((best_dark_end - rise_datetime).total_seconds() / 3600, 1)
            else:
                rise_time = best_dark_start.strftime('%H:%M')
                rise_az = round(float(sample_az[0]), 1)
                duration = round(dark_duration_hours, 1)
        else:
            # Object rises AND sets during the night
            if rise_idx is not None and set_idx is not None:
                rise_datetime = datetime.fromtimestamp(sample_ts[rise_idx], tz=ctx.local_tz)
                set_datetime = datetime.fromtimestamp(sample_ts[set_idx], tz=ctx.local_tz)
                rise_time = rise_datetime.strftime('%H:%M')
                set_time = set_datetime.strftime('%H:%M')
                rise_az = round(float(sample_az[rise_idx]), 1)
                set_az = round(float(sample_az[set_idx]), 1)
                duration = round((set_datetime - rise_datetime).total_seconds() / 3600, 1)
            elif rise_idx is not None:
                rise_datetime = datetime.fromtimestamp(sample_ts[rise_idx], tz=ctx.local_tz)
                rise_time = rise_datetime.strftime('%H:%M')
                rise_az = round(float(sample_az[rise_idx]), 1)
                set_time = best_dark_end.strftime('%H:%M')
                set_az = round(float(sample_az[-1]), 1)
                duration = round((best_dark_end - rise_datetime).total_seconds() / 3600, 1)
            elif set_idx is not None:
                set_datetime = datetime.fromtimestamp(sample_ts[set_idx], tz=ctx.local_tz)
                rise_time = best_dark_start.strftime('%H:%M')
                rise_az = round(float(sample_az[0]), 1)
                set_time = set_datetime.strftime('%H:%M')
                set_az = round(float(sample_az[set_idx]), 1)
                duration = round((set_datetime - best_dark_start).total_seconds() / 3600, 1)
            else:
                # Fallback - shouldn't happen if object is visible at midpoint
                rise_time = best_midpoint.strftime('%H:%M')
                set_time = best_midpoint.strftime('%H:%M')
                rise_az = best_azimuth
                set_az = best_azimuth
                duration = round(np.sum(visible) / num_samples * dark_duration_hours, 1)

        return (obj_id, name, obj_type, messier_num, ra_j2000, dec_j2000, best_date, best_midpoint.strftime('%H:%M'),
                best_altitude, best_azimuth,
                rise_time, rise_az, set_time, set_az, duration,
                total_good_nights,
                best_dark_start.strftime('%H:%M'), best_dark_end.strftime('%H:%M'))

    except Exception as e:
        return (obj_id, name, obj_type, messier_num, ra_j2000, dec_j2000, 'N/A', 'N/A', 'Error', 'N/A',
                'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 'N/A', 'N/A')



class StarTellerCLI:
    """Observer site + catalog; compute viewing windows (silent — CLI handles messaging)."""

    def __init__(self, latitude, longitude, elevation=0):
        self.latitude = latitude
        self.longitude = longitude
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=latitude, lng=longitude)
        self.timezone_name = tz_name
        self.local_tz = pytz.timezone(tz_name) if tz_name else pytz.UTC
        self.catalog_df = self.setup_dso_catalog()

    def setup_dso_catalog(self):
        try:
            catalog_df = load_ngc_catalog()
            if catalog_df.empty:
                return pd.DataFrame()
            return catalog_df.copy()
        except Exception:
            return pd.DataFrame()

    def get_night_midpoints(self, start_date=None, days=365):
        if start_date is None:
            start_date = date.today()
        end_date = start_date + timedelta(days=days - 1)
        years_needed = set()
        current_date = start_date
        for day_offset in range(days):
            check_date = current_date + timedelta(days=day_offset)
            years_needed.add(check_date.year)
        local_tz_str = str(self.local_tz)
        year_args = [
            (year, self.latitude, self.longitude, local_tz_str)
            for year in sorted(years_needed)
        ]
        results = [compute_year_night_midpoints(args) for args in year_args]
        all_midpoints = []
        for year, year_midpoints in results:
            if year_midpoints:
                all_midpoints.extend(year_midpoints)
        result = []
        for date_obj, midpoint, dark_start, dark_end in all_midpoints:
            if start_date <= date_obj <= end_date:
                result.append((date_obj, midpoint, dark_start, dark_end))
        return sorted(result, key=lambda x: x[0])

    def find_optimal_viewing_times(
        self,
        min_altitude=20,
        direction_filter=None,
        messier_only=False,
        use_tqdm=True,
        night_midpoints=None,
    ):
        df_work = self.catalog_df
        if messier_only:
            m = df_work["Messier"].fillna("").astype(str).str.strip()
            df_work = df_work[m != ""].reset_index(drop=True)
        if night_midpoints is None:
            night_midpoints = self.get_night_midpoints()
        num_nights = len(night_midpoints)
        utc_tz = pytz.UTC
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
        local_tz_str = str(self.local_tz)
        if night_midpoints:
            mid_idx = len(night_midpoints) // 2
            _, mid_midpoint, _, _ = night_midpoints[mid_idx]
            mid_jd = mid_midpoint.timestamp() / 86400.0 + 2440587.5
        else:
            today = date.today()
            mid_jd = (datetime(today.year, 7, 1, 12, 0, 0, tzinfo=pytz.UTC).timestamp() / 86400.0 + 2440587.5)
        ra_j2000 = df_work["Right_Ascension"].to_numpy(dtype=np.float64, copy=False)
        dec_j2000 = df_work["Declination"].to_numpy(dtype=np.float64, copy=False)
        ra_now, dec_now = precess_equatorial_j2000(ra_j2000, dec_j2000, mid_jd)
        messier_col = df_work["Messier"].fillna("").astype(str).to_numpy()
        object_ids = df_work["Object"].to_numpy()
        display_names = df_work["Name"].to_numpy()
        types = df_work["Type"].to_numpy()
        work_items = [
            (
                object_ids[i],
                float(ra_j2000[i]),
                float(dec_j2000[i]),
                float(ra_now[i]),
                float(dec_now[i]),
                display_names[i],
                types[i],
                messier_col[i],
                min_altitude,
                direction_filter,
            )
            for i in range(len(df_work))
        ]
        columns = ['Object', 'Name', 'Type', 'Messier', 'Right_Ascension', 'Declination', 'Best_Date', 'Best_Time_Local',
                   'Max_Altitude_deg', 'Azimuth_deg',
                   'Rise_Time_Local', 'Rise_Direction_deg', 'Set_Time_Local', 'Set_Direction_deg',
                   'Observing_Duration_Hours', 'Visible_Nights_Per_Year',
                   'Dark_Start_Local', 'Dark_End_Local']
        ctx = ObservationContext(
            self.latitude, self.longitude, t_array_data, night_dates_tuples,
            night_midpoint_ts, night_dark_start_ts, night_dark_end_ts, local_tz_str,
        )
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(work_items, total=len(work_items), desc=f"Processing {len(df_work)} objects", unit="obj")
        else:
            iterator = work_items
        results = [compute_catalog_object_viewing(ctx, item) for item in iterator]
        results_df = pd.DataFrame(results, columns=columns)
        extra = pd.DataFrame(
            {
                "Major_Axis_arcmin": df_work["Major_Axis_arcmin"].to_numpy(copy=False),
                "Minor_Axis_arcmin": df_work["Minor_Axis_arcmin"].to_numpy(copy=False),
                "Position_Angle_deg": df_work["Position_Angle_deg"].to_numpy(copy=False),
                "Constellation": df_work["Constellation"].fillna("").to_numpy(),
                "V_Mag": df_work["V_Mag"].to_numpy(copy=False),
                "SurfBr": df_work["SurfBr"].to_numpy(copy=False),
            }
        )
        results_df = pd.concat([results_df, extra], axis=1)
        results_df["Timezone"] = local_tz_str
        final_columns = [
            'Object', 'Name', 'Type', 'Messier', 'Constellation', 'Right_Ascension', 'Declination',
            'Major_Axis_arcmin', 'Minor_Axis_arcmin', 'Position_Angle_deg',
            'V_Mag', 'SurfBr',
            'Best_Date', 'Best_Time_Local', 'Max_Altitude_deg', 'Azimuth_deg',
            'Rise_Time_Local', 'Rise_Direction_deg', 'Set_Time_Local', 'Set_Direction_deg',
            'Observing_Duration_Hours', 'Visible_Nights_Per_Year',
            'Dark_Start_Local', 'Dark_End_Local', 'Timezone'
        ]
        results_df = results_df[final_columns]
        if results_df.empty:
            return results_df
        def sort_key(x):
            return (isinstance(x, str) and (x == 'Never visible' or x == 'Error'))
        results_df['never_visible'] = results_df['Max_Altitude_deg'].apply(sort_key)
        results_df = results_df.sort_values(
            by=['never_visible', 'Best_Date', 'Object'],
            ascending=[True, True, True]
        ).drop('never_visible', axis=1)
        return results_df
