#!/usr/bin/env python3
"""
StarTeller Math Stuff
"""
from datetime import date, datetime, time, timedelta

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


def _lst_rad_from_jd_scalar(jd, longitude_deg):
    """Local sidereal time (rad) for scalar Julian date; same model as local_sidereal_time_rad."""
    jd_floor = np.floor(jd - 0.5) + 0.5
    day_fraction = jd - jd_floor
    T = (jd_floor - 2451545.0) / 36525.0
    gmst_midnight = 100.46061837 + (36000.770053608 * T) + (0.000387933 * T**2) - (T**3 / 38710000.0)
    gmst_deg = gmst_midnight + 360.98564736629 * day_fraction
    lst_deg = (gmst_deg + longitude_deg) % 360.0
    return np.deg2rad(lst_deg)


def _alt_deg_at_ts_scalar(ts_unix, ra_rad, dec_rad, lat_rad, lon_deg):
    """Topocentric altitude (deg) at one Unix time; same spherical model as the vector/matrix paths."""
    jd = float(ts_unix) / 86400.0 + 2440587.5
    lst_rad = float(_lst_rad_from_jd_scalar(jd, lon_deg))
    ha_rad = lst_rad - float(ra_rad)
    sin_alt = float(
        np.sin(dec_rad) * np.sin(lat_rad)
        + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad)
    )
    sin_alt = max(-1.0, min(1.0, sin_alt))
    return float(np.rad2deg(np.arcsin(sin_alt)))


def _alt_deg_at_ts_batch(ts_unix, ra_rad, dec_rad, lat_rad, lon_deg):
    """Topocentric altitude (deg); ``ts_unix`` and ``ra_rad`` / ``dec_rad`` same leading shape (vectorized LST)."""
    jd = np.asarray(ts_unix, dtype=np.float64) / 86400.0 + 2440587.5
    lst = local_sidereal_time_rad(jd, lon_deg)
    ra = np.asarray(ra_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    ha = lst - ra
    sin_alt = np.sin(dec) * np.sin(lat_rad) + np.cos(dec) * np.cos(lat_rad) * np.cos(ha)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_alt))


def _alt_deg_shared_unix_ts(ts_scalar, ra_rad, dec_rad, lat_rad, lon_deg):
    """Altitude (deg) for many objects at one Unix time (single LST solve, then vector HA)."""
    jd = float(ts_scalar) / 86400.0 + 2440587.5
    lst0 = _lst_rad_from_jd_scalar(jd, lon_deg)
    ha = lst0 - ra_rad
    sin_alt = np.sin(dec_rad) * np.sin(lat_rad) + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_alt))


def _alt_az_deg_vector_ts(t_vec, ra_vec, dec_vec, lat_rad, lon_deg):
    """Altitude and azimuth (deg); vectorized topocentric transform (same spherical model as ``equatorial_to_horizontal_deg``)."""
    jd = np.asarray(t_vec, dtype=np.float64) / 86400.0 + 2440587.5
    lst = local_sidereal_time_rad(jd, lon_deg)
    ha = lst - ra_vec
    sdec = np.sin(dec_vec)
    cdec = np.cos(dec_vec)
    slat = np.sin(lat_rad)
    clat = np.cos(lat_rad)
    sin_alt = sdec * slat + cdec * clat * np.cos(ha)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt_rad = np.arcsin(sin_alt)
    cos_alt = np.cos(alt_rad)
    cos_alt = np.where(np.abs(cos_alt) < 1e-10, np.copysign(1e-10, cos_alt), cos_alt)
    sin_az = -cdec * np.sin(ha) / cos_alt
    cos_az = (sdec - slat * np.sin(alt_rad)) / (clat * cos_alt)
    az_rad = np.arctan2(sin_az, cos_az)
    return np.rad2deg(alt_rad), np.rad2deg(az_rad) % 360.0


def _ternary_search_peak_times_batch(seg_ta, seg_tb, ra_vec, dec_vec, lat_rad, lon_deg, n_iter=50):
    """Vector ternary search for time of maximum altitude on each [seg_ta, seg_tb]; returns peak Unix times."""
    lo = np.asarray(seg_ta, dtype=np.float64).copy()
    hi = np.asarray(seg_tb, dtype=np.float64).copy()
    for _ in range(n_iter):
        span = hi - lo
        active = span >= 0.25
        if not np.any(active):
            break
        t1 = lo + span / 3.0
        t2 = hi - span / 3.0
        a1 = _alt_deg_at_ts_batch(t1, ra_vec, dec_vec, lat_rad, lon_deg)
        a2 = _alt_deg_at_ts_batch(t2, ra_vec, dec_vec, lat_rad, lon_deg)
        move_lo = active & (a1 < a2)
        lo = np.where(move_lo, t1, lo)
        hi = np.where(active & ~move_lo, t2, hi)
    return 0.5 * (lo + hi)


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
# 3. Sun position & astronomical-dark windows (smooth alt(ts); roots at −18°)
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


ASTRO_DARK_ALT_DEG = -18.0


def _sun_equatorial_deg_scalar(jd):
    """Sun RA/Dec (deg) for scalar Julian date; same low-precision model as sun_equatorial_deg."""
    n = jd - 2451545.0
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
    q = (280.459 + 0.985647436 * n) % 360.0
    L = q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    L_rad = np.deg2rad(L)
    e = np.deg2rad(23.439 - 0.00000036 * n)
    ra_rad = np.arctan2(np.cos(e) * np.sin(L_rad), np.cos(L_rad))
    dec_rad = np.arcsin(np.sin(e) * np.sin(L_rad))
    return float(np.rad2deg(ra_rad) % 360.0), float(np.rad2deg(dec_rad))


def _sun_altitude_deg_scalar(ts_unix, latitude, longitude):
    """Sun altitude (deg) at Unix time; scalar path for root-finding without array allocations."""
    jd = ts_unix / 86400.0 + 2440587.5
    sun_ra_deg, sun_dec_deg = _sun_equatorial_deg_scalar(jd)
    lst_rad = float(_lst_rad_from_jd_scalar(jd, longitude))
    lat_rad = np.deg2rad(latitude)
    ra_rad = np.deg2rad(sun_ra_deg)
    dec_rad = np.deg2rad(sun_dec_deg)
    ha_rad = lst_rad - float(ra_rad)
    sin_alt = float(
        np.sin(dec_rad) * np.sin(lat_rad)
        + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad)
    )
    sin_alt = max(-1.0, min(1.0, sin_alt))
    return float(np.rad2deg(np.arcsin(sin_alt)))


def _scalar_sun_alt_minus(ts_unix, latitude, longitude, target_alt=ASTRO_DARK_ALT_DEG):
    return _sun_altitude_deg_scalar(ts_unix, latitude, longitude) - target_alt


def _bisect_sun_alt_crossing(ts_lo, ts_hi, latitude, longitude, target_alt=ASTRO_DARK_ALT_DEG):
    """
    Unix timestamp in [ts_lo, ts_hi] where sun altitude == target_alt.
    Requires ts_lo < ts_hi and (alt(ts_lo)-target) * (alt(ts_hi)-target) <= 0.
    """
    flo = _scalar_sun_alt_minus(ts_lo, latitude, longitude, target_alt)
    fhi = _scalar_sun_alt_minus(ts_hi, latitude, longitude, target_alt)
    if flo * fhi > 0:
        return 0.5 * (ts_lo + ts_hi)
    lo, hi = float(ts_lo), float(ts_hi)
    fl, fh = flo, fhi
    for _ in range(32):
        m = 0.5 * (lo + hi)
        fm = _scalar_sun_alt_minus(m, latitude, longitude, target_alt)
        if fl * fm <= 0:
            hi, fh = m, fm
        else:
            lo, fl = m, fm
    return 0.5 * (lo + hi)


def _dark_window_local_noon_day(check_date, latitude, longitude, local_tz, n_scan=33):
    """
    Astronomical dark (Sun altitude < ASTRO_DARK_ALT_DEG) within one local
    **noon → next noon** span. Same sun model as sun_altitude_deg (smooth in time).

    Returns (dark_start_ts, dark_end_ts) unix floats, or None if no darkness.
    """
    t0 = local_tz.localize(datetime.combine(check_date, time(12, 0))).timestamp()
    next_day = check_date + timedelta(days=1)
    t1 = local_tz.localize(datetime.combine(next_day, time(12, 0))).timestamp()
    if t1 <= t0:
        return None

    ts = np.linspace(t0, t1, n_scan)
    jd = ts / 86400.0 + 2440587.5
    alt = sun_altitude_deg(jd, latitude, longitude)
    f = alt - ASTRO_DARK_ALT_DEG

    if np.all(f < 0):
        return float(t0), float(t1)
    if np.all(f > 0):
        return None

    roots = []
    for i in range(n_scan - 1):
        if f[i] == 0.0:
            roots.append(float(ts[i]))
        elif f[i] * f[i + 1] < 0.0:
            r = _bisect_sun_alt_crossing(float(ts[i]), float(ts[i + 1]), latitude, longitude, ASTRO_DARK_ALT_DEG)
            roots.append(r)

    roots = sorted(roots)
    knots = [float(t0)] + roots + [float(t1)]
    best_lo, best_hi = None, None
    best_span = -1.0
    for j in range(len(knots) - 1):
        ta, tb = knots[j], knots[j + 1]
        if tb - ta < 1e-6:
            continue
        mid = 0.5 * (ta + tb)
        if _scalar_sun_alt_minus(mid, latitude, longitude, ASTRO_DARK_ALT_DEG) < 0.0:
            span = tb - ta
            if span > best_span:
                best_span = span
                best_lo, best_hi = ta, tb

    if best_lo is None or best_hi is None or best_hi <= best_lo:
        return None
    return best_lo, best_hi


def compute_year_dark_windows(args):
    """
    Compute astronomical-dark [start, end] for each local calendar day.

    For each ``check_date``, uses **local noon → next noon**: Sun altitude is a smooth
    function of time (same low-precision sun + LST as everywhere else). Roots of
    (alt - (−18°)) are found with a short scan + bisection—no 15-minute ladder.

    Args: (year, latitude, longitude, local_tz_str)
    Returns: (year, list of (check_date, dark_start, dark_end)) or (year, None) on error.
    """
    year, latitude, longitude, local_tz_str = args

    try:
        local_tz = pytz.timezone(local_tz_str)

        full_year_start = date(year, 1, 1)
        full_year_days = (date(year + 1, 1, 1) - full_year_start).days

        if year == datetime.now().year:
            max_date = date.today() + timedelta(days=365)
            if date(year + 1, 1, 1) > max_date:
                full_year_days = (max_date - full_year_start).days

        nights_out = []

        for day_offset in range(full_year_days):
            check_date = full_year_start + timedelta(days=day_offset)
            span = _dark_window_local_noon_day(check_date, latitude, longitude, local_tz)
            if span is None:
                continue
            dark_start_ts, dark_end_ts = span
            dark_start = datetime.fromtimestamp(dark_start_ts, tz=local_tz)
            dark_end = datetime.fromtimestamp(dark_end_ts, tz=local_tz)
            nights_out.append((check_date, dark_start, dark_end))

        return (year, nights_out)
    except Exception:
        return (year, None)


# -----------------------------------------------------------------------------
# 4. Shared night arrays + per-object viewing (find_optimal_viewing_times)
# -----------------------------------------------------------------------------


class ObservationContext:
    # Precomputed night metadata shared while scanning the catalog
    __slots__ = ('latitude', 'longitude', 'night_dates', 'night_dark_start_ts', 'night_dark_end_ts', 'local_tz', 'local_tz_str')

    def __init__(self, latitude, longitude, night_dates_tuples, night_dark_start_ts, night_dark_end_ts, local_tz_str):
        from datetime import date

        self.latitude = latitude
        self.longitude = longitude
        self.local_tz_str = local_tz_str
        self.local_tz = pytz.timezone(local_tz_str)
        self.night_dates = [date(y, m, d) for y, m, d in night_dates_tuples]
        self.night_dark_start_ts = np.asarray(night_dark_start_ts, dtype=np.float64)
        self.night_dark_end_ts = np.asarray(night_dark_end_ts, dtype=np.float64)


def _bisect_altitude_equals(ts_lo, ts_hi, ra_rad, dec_rad, lat_rad, lon_deg, h_deg, max_iter=56):
    """Unix time where altitude equals h_deg on [ts_lo, ts_hi]; bisection on f(t)=alt(t)−h (bracket < 0.25 s)."""
    f_lo = _alt_deg_at_ts_scalar(ts_lo, ra_rad, dec_rad, lat_rad, lon_deg) - h_deg
    f_hi = _alt_deg_at_ts_scalar(ts_hi, ra_rad, dec_rad, lat_rad, lon_deg) - h_deg
    if f_lo * f_hi > 0:
        return None
    lo, hi = float(ts_lo), float(ts_hi)
    fl, fh = f_lo, f_hi
    for _ in range(max_iter):
        if abs(hi - lo) < 0.25:
            break
        m = 0.5 * (lo + hi)
        fm = _alt_deg_at_ts_scalar(m, ra_rad, dec_rad, lat_rad, lon_deg) - h_deg
        if fl * fm <= 0:
            hi, fh = m, fm
        else:
            lo, fl = m, fm
    return 0.5 * (lo + hi)


def _bisect_altitude_batch(ts_lo, ts_hi, ra_rad, dec_rad, lat_rad, lon_deg, h_deg, max_iter=22):
    """Vector bisection for many objects sharing the same time bracket (invalid brackets -> NaN)."""
    ra_rad = np.asarray(ra_rad, dtype=np.float64)
    dec_rad = np.asarray(dec_rad, dtype=np.float64)
    n = int(ra_rad.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    ts_lo = float(ts_lo)
    ts_hi = float(ts_hi)
    lo = np.full(n, ts_lo, dtype=np.float64)
    hi = np.full(n, ts_hi, dtype=np.float64)
    fl = _alt_deg_shared_unix_ts(ts_lo, ra_rad, dec_rad, lat_rad, lon_deg) - h_deg
    fh = _alt_deg_shared_unix_ts(ts_hi, ra_rad, dec_rad, lat_rad, lon_deg) - h_deg
    bad = fl * fh > 0
    for it in range(max_iter):
        m = 0.5 * (lo + hi)
        # First step only: all brackets share the same [ts_lo, ts_hi], so m is identical for every object.
        if it == 0:
            fm = _alt_deg_shared_unix_ts(float(m[0]), ra_rad, dec_rad, lat_rad, lon_deg) - h_deg
        else:
            fm = _alt_deg_at_ts_batch(m, ra_rad, dec_rad, lat_rad, lon_deg) - h_deg
        take_hi = (~bad) & (fl * fm <= 0)
        take_lo = (~bad) & (~take_hi)
        hi = np.where(take_hi, m, hi)
        fh = np.where(take_hi, fm, fh)
        lo = np.where(take_lo, m, lo)
        fl = np.where(take_lo, fm, fl)
        if np.bitwise_and.reduce(bad | (hi - lo < 0.25)):
            break
    out = 0.5 * (lo + hi)
    out[bad] = np.nan
    return out


def _visible_segments_above_alt_from_grid(ts_grid, alt_row, ts0, ts1, ra_rad, dec_rad, lat_rad, lon_deg, h_deg):
    """
    Intervals in [ts0, ts1] where altitude >= h_deg.
    The time grid only brackets sign changes; each crossing is refined with bisection on the same alt model.
    """
    n_scan = len(ts_grid)
    dh = alt_row - h_deg
    roots = []
    for i in range(n_scan - 1):
        if dh[i] == 0.0:
            roots.append(float(ts_grid[i]))
        elif dh[i] * dh[i + 1] < 0.0:
            r = _bisect_altitude_equals(
                float(ts_grid[i]), float(ts_grid[i + 1]), ra_rad, dec_rad, lat_rad, lon_deg, h_deg
            )
            if r is not None:
                roots.append(r)
    roots = sorted(set(roots))
    knots = [float(ts0)] + roots + [float(ts1)]
    segs = []
    for j in range(len(knots) - 1):
        ta, tb = knots[j], knots[j + 1]
        if tb - ta < 1e-6:
            continue
        mid = 0.5 * (ta + tb)
        if _alt_deg_at_ts_scalar(mid, ra_rad, dec_rad, lat_rad, lon_deg) >= h_deg:
            segs.append((ta, tb))
    return segs


def _night_visibility_refined(ts_grid, alt_row, ts0, ts1, ra_rad, dec_rad, lat_rad, lon_deg, h_deg):
    """
    Longest continuous span >= h_deg during [ts0, ts1], using grid + bisection for boundaries.
    Returns (duration_h, seg_ta, seg_tb, peak_alt_rank) for ranking; no datetime formatting.
    """
    segs = _visible_segments_above_alt_from_grid(
        ts_grid, alt_row, ts0, ts1, ra_rad, dec_rad, lat_rad, lon_deg, h_deg
    )
    if not segs:
        return None
    ta, tb = max(segs, key=lambda ab: ab[1] - ab[0])
    duration_h = (tb - ta) / 3600.0
    if duration_h <= 0.0:
        return None
    aa = _alt_deg_at_ts_scalar(ta, ra_rad, dec_rad, lat_rad, lon_deg)
    ab = _alt_deg_at_ts_scalar(tb, ra_rad, dec_rad, lat_rad, lon_deg)
    peak_alt_rank = max(aa, ab)
    return duration_h, ta, tb, peak_alt_rank


def _catalog_sorted_crossings_J_Rt(dh, ts_grid, refine_mask, ra_rad_vec, dec_rad_vec, lat_rad, lon_deg, h):
    """
    For all objects, collect times where alt crosses the horizon threshold on the coarse grid:
    exact hits on a sample, sign changes between samples (refined by bisection). Returns sorted
    (object_index, crossing_time) rows with duplicate pairs removed.
    """
    _, g = dh.shape
    j_parts = []
    t_parts = []
    for i in range(g):
        jz = np.flatnonzero((dh[:, i] == 0.0) & refine_mask)
        if jz.size:
            j_parts.append(jz)
            t_parts.append(np.full(jz.shape[0], ts_grid[i], dtype=np.float64))
    for i in range(g - 1):
        jc = np.flatnonzero(dh[:, i] * dh[:, i + 1] < 0.0)
        if jc.size:
            jc = jc[refine_mask[jc]]
            if jc.size:
                r = _bisect_altitude_batch(
                    ts_grid[i],
                    ts_grid[i + 1],
                    ra_rad_vec[jc],
                    dec_rad_vec[jc],
                    lat_rad,
                    lon_deg,
                    h,
                )
                ok = ~np.isnan(r)
                if np.any(ok):
                    j_parts.append(jc[ok])
                    t_parts.append(r[ok])
    if not j_parts:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.float64)
    J = np.concatenate(j_parts)
    Rt = np.concatenate(t_parts)
    order = np.lexsort((Rt, J))
    J, Rt = J[order], Rt[order]
    if J.size > 1:
        nodup = (J[1:] != J[:-1]) | (Rt[1:] != Rt[:-1])
        keep = np.empty(J.shape[0], dtype=bool)
        keep[0] = True
        keep[1:] = nodup
        J, Rt = J[keep], Rt[keep]
    return J, Rt


def _apply_night_catalog_update(ni, ts0, ts1, ts_grid, alt, dh, ra_rad_vec, dec_rad_vec, lat_rad, lon_deg, h, best_duration, best_peak_rank, best_night_idx, best_seg_ta, best_seg_tb, refine_mask):
    # One night: (1) crossing times → knot intervals, (2) vector midpoint test for alt≥h,
    # (3) pick best interval per object vs current best, (4) scalar fallback if the grid missed a case.
    n_obj, g = dh.shape
    ts0 = float(ts0)
    ts1 = float(ts1)
    h = float(h)

    J, Rt = _catalog_sorted_crossings_J_Rt(dh, ts_grid, refine_mask, ra_rad_vec, dec_rad_vec, lat_rad, lon_deg, h)

    if not np.any(refine_mask):
        return

    j_always = np.flatnonzero((np.min(dh, axis=1) >= 0.0) & refine_mask)
    unique_j, start = (np.unique(J, return_index=True) if J.size else (np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)))
    end = np.append(start[1:], len(J))
    in_root = np.zeros(n_obj, dtype=bool)
    if unique_j.size:
        in_root[unique_j] = True
    j_full = j_always[~in_root[j_always]]

    max_bt = int(np.count_nonzero(refine_mask)) * (g + 3) + 8
    b_j = np.empty(max_bt, dtype=np.intp)
    b_mid = np.empty(max_bt, dtype=np.float64)
    b_ta = np.empty(max_bt, dtype=np.float64)
    b_tb = np.empty(max_bt, dtype=np.float64)
    pos = 0

    u_keep = refine_mask[unique_j]
    uj = unique_j[u_keep]
    us = start[u_keep]
    ue = end[u_keep]
    for k in range(uj.shape[0]):
        j = int(uj[k])
        roots = Rt[int(us[k]) : int(ue[k])]
        nk = int(roots.size)
        knots = np.empty(nk + 2, dtype=np.float64)
        knots[0] = ts0
        if nk:
            knots[1:-1] = roots
        knots[-1] = ts1
        taa = knots[:-1]
        tbb = knots[1:]
        ok_seg = (tbb - taa) >= 1e-6
        n_add = int(np.count_nonzero(ok_seg))
        if n_add:
            b_j[pos : pos + n_add] = j
            b_mid[pos : pos + n_add] = 0.5 * (taa[ok_seg] + tbb[ok_seg])
            b_ta[pos : pos + n_add] = taa[ok_seg]
            b_tb[pos : pos + n_add] = tbb[ok_seg]
            pos += n_add

    for j in j_full:
        b_j[pos] = j
        b_mid[pos] = 0.5 * (ts0 + ts1)
        b_ta[pos] = ts0
        b_tb[pos] = ts1
        pos += 1

    processed = np.zeros(n_obj, dtype=bool)

    if pos > 0:
        bj = b_j[:pos]
        bta = b_ta[:pos]
        btb = b_tb[:pos]
        al_m = _alt_deg_at_ts_batch(b_mid[:pos], ra_rad_vec[bj], dec_rad_vec[bj], lat_rad, lon_deg)
        vis = al_m >= h
        dur = (btb - bta) / 3600.0
        aa = _alt_deg_at_ts_batch(bta, ra_rad_vec[bj], dec_rad_vec[bj], lat_rad, lon_deg)
        ab = _alt_deg_at_ts_batch(btb, ra_rad_vec[bj], dec_rad_vec[bj], lat_rad, lon_deg)
        pr = np.maximum(aa, ab)

        w = np.flatnonzero(vis)
        if w.size:
            bjv = bj[w]
            dvv = dur[w]
            prv = pr[w]
            ordv = np.lexsort((-prv, -dvv, bjv))
            bj_so = bjv[ordv]
            fst = np.concatenate([[True], bj_so[1:] != bj_so[:-1]])
            pick_w = w[ordv[fst]]
            jj = bj[pick_w]
            keep = refine_mask[jj]
            pick_w, jj = pick_w[keep], jj[keep]
            cand_d = dur[pick_w]
            cand_pr = pr[pick_w]
            cand_ta = bta[pick_w]
            cand_tb = btb[pick_w]
            old_d = best_duration[jj]
            old_pr = best_peak_rank[jj]
            better = (cand_d > old_d) | ((cand_d == old_d) & (cand_pr > old_pr))
            if np.any(better):
                jj_b = jj[better]
                best_duration[jj_b] = cand_d[better]
                best_peak_rank[jj_b] = cand_pr[better]
                best_night_idx[jj_b] = ni
                best_seg_ta[jj_b] = cand_ta[better]
                best_seg_tb[jj_b] = cand_tb[better]
            processed[jj] = True

    miss = np.flatnonzero(refine_mask & ~processed)
    for j in miss:
        nv = _night_visibility_refined(ts_grid, alt[j], ts0, ts1, float(ra_rad_vec[j]), float(dec_rad_vec[j]), lat_rad, lon_deg, h)
        if nv is None:
            continue
        duration_h, seg_ta, seg_tb, peak_alt_rank = nv
        cand_pair = (duration_h, peak_alt_rank)
        rank_pair = (best_duration[j], best_peak_rank[j])
        if cand_pair > rank_pair:
            best_duration[j] = duration_h
            best_peak_rank[j] = peak_alt_rank
            best_night_idx[j] = ni
            best_seg_ta[j] = seg_ta
            best_seg_tb[j] = seg_tb


def _compute_viewing_rows_batch(ctx, object_ids, names, types, messier_col, ra_j2000, dec_j2000, ra_now, dec_now, min_altitude, n_scan=25, progress_nights=False):
    """
    Per object: best astro-dark night by longest time above min altitude (tie-break: higher alt at segment ends).
    Each night: vectorized altitudes on a time grid for bracketing; crossings solved by bisection on alt(t)=h;
    peak time by ternary search on the same alt model.
    """
    lat_deg = float(ctx.latitude)
    lon_deg = float(ctx.longitude)
    lat_rad = np.deg2rad(lat_deg)
    ra_rad_vec = np.deg2rad(np.asarray(ra_now, dtype=np.float64))
    dec_rad_vec = np.deg2rad(np.asarray(dec_now, dtype=np.float64))
    sin_dec_v = np.sin(dec_rad_vec)
    cos_dec_v = np.cos(dec_rad_vec)
    sin_lat = float(np.sin(lat_rad))
    cos_lat = float(np.cos(lat_rad))
    n_obj = int(ra_rad_vec.shape[0])
    num_nights = len(ctx.night_dates)
    g = max(5, int(n_scan))
    h = float(min_altitude)

    best_duration = np.full(n_obj, -1.0, dtype=np.float64)
    best_peak_rank = np.full(n_obj, -999.0, dtype=np.float64)
    best_night_idx = np.full(n_obj, -1, dtype=np.int32)
    total_good = np.zeros(n_obj, dtype=np.int32)
    best_seg_ta = np.zeros(n_obj, dtype=np.float64)
    best_seg_tb = np.zeros(n_obj, dtype=np.float64)

    ts0_all = np.asarray(ctx.night_dark_start_ts, dtype=np.float64)
    ts1_all = np.asarray(ctx.night_dark_end_ts, dtype=np.float64)
    ts_grid_all = np.zeros((num_nights, g), dtype=np.float64)
    for ni in range(num_nights):
        t0, t1 = float(ts0_all[ni]), float(ts1_all[ni])
        if t1 > t0:
            ts_grid_all[ni] = np.linspace(t0, t1, g, dtype=np.float64)
    jd_all = ts_grid_all / 86400.0 + 2440587.5
    lst_all = local_sidereal_time_rad(jd_all, lon_deg)

    night_chunk = 48
    night_list = list(range(num_nights))
    if progress_nights:
        try:
            from tqdm import tqdm

            night_list = tqdm(night_list, desc="Nights", unit="night", leave=False)
        except ImportError:
            pass

    cur_c0 = -1
    alt_blk = None
    for ni in night_list:
        ts0 = float(ts0_all[ni])
        ts1 = float(ts1_all[ni])
        if ts1 <= ts0:
            continue
        c0 = (ni // night_chunk) * night_chunk
        if c0 != cur_c0:
            cur_c0 = c0
            c1 = min(c0 + night_chunk, num_nights)
            lst_b = lst_all[c0:c1]
            ha_b = lst_b[:, np.newaxis, :] - ra_rad_vec[np.newaxis, :, np.newaxis]
            sin_alt_b = (
                sin_dec_v[np.newaxis, :, np.newaxis] * sin_lat
                + cos_dec_v[np.newaxis, :, np.newaxis] * cos_lat * np.cos(ha_b)
            )
            alt_blk = np.rad2deg(np.arcsin(np.clip(sin_alt_b, -1.0, 1.0)))
        alt = alt_blk[ni - c0]
        ts_grid = ts_grid_all[ni]
        dh = alt - h
        if not np.any(dh >= 0.0):
            continue

        total_good += (np.max(dh, axis=1) >= h).astype(np.int32)
        night_len_h = (ts1 - ts0) / 3600.0
        refine_mask = (np.max(dh, axis=1) >= h) & (night_len_h >= best_duration)
        _apply_night_catalog_update(ni, ts0, ts1, ts_grid, alt, dh, ra_rad_vec, dec_rad_vec, lat_rad, lon_deg, h, best_duration, best_peak_rank, best_night_idx, best_seg_ta, best_seg_tb, refine_mask)

    good = best_night_idx >= 0
    gid = np.flatnonzero(good)
    p_ts_arr = np.zeros(n_obj, dtype=np.float64)
    peak_alt_arr = np.zeros(n_obj, dtype=np.float64)
    peak_az_arr = np.zeros(n_obj, dtype=np.float64)
    rise_az_arr = np.zeros(n_obj, dtype=np.float64)
    set_az_arr = np.zeros(n_obj, dtype=np.float64)
    if gid.size:
        ta = best_seg_ta[gid]
        tb = best_seg_tb[gid]
        ra_g = ra_rad_vec[gid]
        dec_g = dec_rad_vec[gid]
        p_ts = _ternary_search_peak_times_batch(ta, tb, ra_g, dec_g, lat_rad, lon_deg)
        n = int(gid.size)
        ra3 = np.concatenate([ra_g, ra_g, ra_g])
        dec3 = np.concatenate([dec_g, dec_g, dec_g])
        alt_all, az_all = _alt_az_deg_vector_ts(
            np.concatenate([ta, tb, p_ts]), ra3, dec3, lat_rad, lon_deg
        )
        rise_az_arr[gid] = az_all[:n]
        set_az_arr[gid] = az_all[n : 2 * n]
        peak_alt_arr[gid] = alt_all[2 * n :]
        peak_az_arr[gid] = az_all[2 * n :]
        p_ts_arr[gid] = p_ts

    rows = []
    for j in range(n_obj):
        obj_id = object_ids[j]
        name = names[j]
        obj_type = types[j]
        mnum = messier_col[j]
        ra_j = float(ra_j2000[j])
        dec_j = float(dec_j2000[j])
        bi = int(best_night_idx[j])
        if bi < 0:
            rows.append(
                (obj_id, name, obj_type, mnum, ra_j, dec_j, 'N/A', 'N/A', 'Never visible', 'N/A',
                 'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 'N/A', 'N/A')
            )
            continue
        seg_ta = float(best_seg_ta[j])
        seg_tb = float(best_seg_tb[j])
        p_ts = float(p_ts_arr[j])
        peak_alt = float(peak_alt_arr[j])
        peak_az = float(peak_az_arr[j])
        rise_az = float(rise_az_arr[j])
        set_az = float(set_az_arr[j])
        dark_start = datetime.fromtimestamp(float(ctx.night_dark_start_ts[bi]), tz=ctx.local_tz)
        dark_end = datetime.fromtimestamp(float(ctx.night_dark_end_ts[bi]), tz=ctx.local_tz)
        best_date = ctx.night_dates[bi]
        rise_hm = datetime.fromtimestamp(seg_ta, tz=ctx.local_tz).strftime('%H:%M')
        set_hm = datetime.fromtimestamp(seg_tb, tz=ctx.local_tz).strftime('%H:%M')
        best_time = datetime.fromtimestamp(p_ts, tz=ctx.local_tz).strftime('%H:%M')
        best_altitude = round(peak_alt, 1)
        best_azimuth = round(peak_az, 1)
        duration = round(float(best_duration[j]), 1)
        rows.append(
            (obj_id, name, obj_type, mnum, ra_j, dec_j, best_date, best_time,
             best_altitude, best_azimuth,
             rise_hm, round(rise_az, 1), set_hm, round(set_az, 1),
             duration, int(total_good[j]),
             dark_start.strftime('%H:%M'), dark_end.strftime('%H:%M'))
        )
    return rows


def compute_catalog_object_viewing(ctx, args):
    """Compute one catalog row (legacy path; batch pipeline is preferred)."""
    obj_id, ra_j2000, dec_j2000, ra_now, dec_now, name, obj_type, messier_num, min_altitude = args
    try:
        rows = _compute_viewing_rows_batch(ctx, np.array([obj_id], dtype=object), np.array([name], dtype=object), np.array([obj_type], dtype=object), np.array([messier_num], dtype=object), np.array([ra_j2000], dtype=np.float64), np.array([dec_j2000], dtype=np.float64), np.array([ra_now], dtype=np.float64), np.array([dec_now], dtype=np.float64), min_altitude)
        return rows[0]
    except Exception:
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

    def get_dark_windows(self, start_date=None, days=365):
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
        results = [compute_year_dark_windows(args) for args in year_args]
        all_windows = []
        for year, year_windows in results:
            if year_windows:
                all_windows.extend(year_windows)
        result = []
        for date_obj, dark_start, dark_end in all_windows:
            if start_date <= date_obj <= end_date:
                result.append((date_obj, dark_start, dark_end))
        return sorted(result, key=lambda x: x[0])

    def find_optimal_viewing_times(self, min_altitude=20, messier_only=False, use_tqdm=True, dark_windows=None, time_grid_points=25):
        """
        ``time_grid_points``: uniform samples per astro-dark span used only to *bracket* crossings of
        the minimum-altitude level; each crossing is refined with bisection on alt(t)=h, and the peak
        uses ternary search on the same continuous model. Use 25+ for minute-level agreement with
        refined outputs; lower values risk missing short visibility gaps between samples.
        """
        df_work = self.catalog_df
        if messier_only:
            m = df_work["Messier"].fillna("").astype(str).str.strip()
            df_work = df_work[m != ""].reset_index(drop=True)
        if dark_windows is None:
            dark_windows = self.get_dark_windows()
        num_nights = len(dark_windows)
        night_dates_tuples = []
        night_dark_start_ts = np.empty(num_nights, dtype=np.float64)
        night_dark_end_ts = np.empty(num_nights, dtype=np.float64)
        for i, (date_obj, dark_start, dark_end) in enumerate(dark_windows):
            night_dates_tuples.append((date_obj.year, date_obj.month, date_obj.day))
            night_dark_start_ts[i] = dark_start.timestamp()
            night_dark_end_ts[i] = dark_end.timestamp()
        local_tz_str = str(self.local_tz)
        if dark_windows:
            mid_idx = len(dark_windows) // 2
            _, ds, de = dark_windows[mid_idx]
            mid_jd = (ds.timestamp() + de.timestamp()) * 0.5 / 86400.0 + 2440587.5
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
        # Visible_Nights_Per_Year: count of nights with any time above min altitude during astro dark
        columns = ['Object', 'Name', 'Type', 'Messier', 'Right_Ascension', 'Declination', 'Best_Date', 'Best_Time_Local',
                   'Max_Altitude_deg', 'Azimuth_deg',
                   'Rise_Time_Local', 'Rise_Direction_deg', 'Set_Time_Local', 'Set_Direction_deg',
                   'Observing_Duration_Hours', 'Visible_Nights_Per_Year',
                   'Dark_Start_Local', 'Dark_End_Local']
        ctx = ObservationContext(self.latitude, self.longitude, night_dates_tuples, night_dark_start_ts, night_dark_end_ts, local_tz_str)
        results = _compute_viewing_rows_batch(ctx, object_ids, display_names, types, messier_col, ra_j2000, dec_j2000, ra_now, dec_now, min_altitude, n_scan=int(time_grid_points), progress_nights=use_tqdm)
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
