#!/usr/bin/env python3
"""
StarTeller Math Stuff
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

try:
    from .catalog_manager import load_ngc_catalog
except ImportError:
    from catalog_manager import load_ngc_catalog


# --- 1. Time & sky geometry (Julian date, LST, precession, horizon coordinates) ---


def unix_timestamp_to_julian_date(ts_unix):
    # Takes: scalar or array-like of Unix timestamps
    # Returns: float64 array of Julian date(s), same shape as np.asarray(ts_unix)
    return np.asarray(ts_unix, dtype=np.float64) / 86400.0 + 2440587.5


def local_sidereal_time_rad(jd_array, longitude_deg):
    """
    Local sidereal time for each UT Julian date (GMST + longitude).

    Takes: numpy array of Julian dates (UT1); observer longitude in degrees, east positive
    Returns: numpy array of local sidereal angles in radians (shape matches jd_array)
    """
    # Meeus, Astronomical Algorithms, Ch. 12; https://auass.com/wp-content/uploads/2021/01/Astronomical-Algorithms.pdf
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
    Precess J2000.0 equatorial coordinates to the epoch of a target Julian date.

    Takes: RA and Dec in degrees at J2000.0 (scalar or arrays); jd_target Julian date(s), same broadcast shape as coords
    Returns: (ra_deg, dec_deg) at that epoch in degrees
    """
    # Meeus, Astronomical Algorithms, Ch. 21; https://auass.com/wp-content/uploads/2021/01/Astronomical-Algorithms.pdf
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


def equatorial_to_horizontal(ra_rad, dec_rad, lst_rad, lat_rad, return_azimuth=True):
    """
    Horizon altitude (and optionally azimuth) from equatorial coords at a given LST.

    Takes: right ascension, declination, local sidereal time, observer latitude (all radians); return_azimuth bool
    Returns: if return_azimuth: (altitude_deg, azimuth_deg); else altitude_deg only (hour angle = LST − RA)
    """
    # https://astronomy.stackexchange.com/questions/13067/conversion-from-equatorial-coordinate-to-horizon-coordinates
    # https://en.wikipedia.org/wiki/Astronomical_coordinate_systems#Equatorial_.E2.86.90.E2.86.92_horizontal
    ha_rad = lst_rad - ra_rad
    sin_alt = (
        np.cos(ha_rad) * np.cos(dec_rad) * np.cos(lat_rad)
        + np.sin(dec_rad) * np.sin(lat_rad)
    )
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    alt_deg = np.rad2deg(alt_rad)

    if not return_azimuth:
        return alt_deg

    cos_alt = np.cos(alt_rad)
    cos_alt = np.where(np.abs(cos_alt) < 1e-10, np.copysign(1e-10, cos_alt), cos_alt)

    sin_az = -np.cos(dec_rad) * np.sin(ha_rad) / cos_alt
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * np.sin(alt_rad)) / (np.cos(lat_rad) * cos_alt)

    az_rad = np.arctan2(sin_az, cos_az)
    az_deg = np.rad2deg(az_rad) % 360.0

    return alt_deg, az_deg


# --- 2. Sun altitude & −18° (astronomical dark) crossing search ---


def sun_equatorial_deg(jd):
    """
    Low-precision geocentric Sun equatorial coordinates (approximate ephemeris).

    Takes: Julian date as scalar or numpy array (UT)
    Returns: (sun_ra_deg, sun_dec_deg) in degrees, same scalar-vs-array kind as jd
    """
    # https://aa.usno.navy.mil/faq/sun_approx
    is_scalar = np.ndim(jd) == 0
    if is_scalar:
        jd = float(jd)

    n = jd - 2451545.0
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
    q = (280.459 + 0.985647436 * n) % 360.0
    L = q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    L_rad = np.deg2rad(L)
    e = np.deg2rad(23.439 - 0.00000036 * n)
    ra_rad = np.arctan2(np.cos(e) * np.sin(L_rad), np.cos(L_rad))
    dec_rad = np.arcsin(np.sin(e) * np.sin(L_rad))

    ra_deg = np.rad2deg(ra_rad) % 360.0
    dec_deg = np.rad2deg(dec_rad)
    if is_scalar:
        return float(ra_deg), float(dec_deg)
    return ra_deg, dec_deg


def sun_altitude_deg(jd, latitude, longitude):
    """
    Sun altitude above the mathematical horizon for the observer (LST-based).

    Takes: Julian date (scalar or numpy array); observer latitude and longitude in degrees (east positive)
    Returns: Sun altitude in degrees, same scalar-vs-array style as jd
    """
    # https://aa.usno.navy.mil/faq/alt_az
    if np.ndim(jd) == 0:
        sun_ra_deg, sun_dec_deg = sun_equatorial_deg(float(jd))
        lst_rad = float(local_sidereal_time_rad(float(jd), longitude))
        lat_rad = float(np.deg2rad(latitude))
        ra_rad = float(np.deg2rad(sun_ra_deg))
        dec_rad = float(np.deg2rad(sun_dec_deg))
        ha_rad = lst_rad - ra_rad
        # sin a = cos(LHA) cos δ cos φ + sin δ sin φ (https://aa.usno.navy.mil/faq/alt_az); ha_rad = LHA
        sin_alt = float(
            np.cos(ha_rad) * np.cos(dec_rad) * np.cos(lat_rad)
            + np.sin(dec_rad) * np.sin(lat_rad)
        )
        sin_alt = max(-1.0, min(1.0, sin_alt))
        return float(np.rad2deg(np.arcsin(sin_alt)))

    sun_ra, sun_dec = sun_equatorial_deg(jd)
    lst_rad = local_sidereal_time_rad(jd, longitude)
    lat_rad = np.deg2rad(latitude)
    ra_rad = np.deg2rad(sun_ra)
    dec_rad = np.deg2rad(sun_dec)
    ha_rad = lst_rad - ra_rad
    sin_alt = np.cos(ha_rad) * np.cos(dec_rad) * np.cos(lat_rad) + np.sin(dec_rad) * np.sin(lat_rad)
    return np.rad2deg(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


ASTRO_DARK_ALT_DEG = -18.0


def sun_altitude_minus_target(ts_unix, latitude, longitude, target_alt=ASTRO_DARK_ALT_DEG):
    """
    Sun altitude minus a threshold (positive when the Sun is above that altitude).

    Takes: Unix timestamp(s); site latitude and longitude (deg); target_alt in degrees (default −18, astronomical dark)
    Returns: difference in degrees, same shape as ts_unix after numpy broadcasting rules
    """
    return sun_altitude_deg(unix_timestamp_to_julian_date(ts_unix), latitude, longitude) - target_alt


def bisect_sun_altitude_crossings(ts_lo, ts_hi, latitude, longitude, target_alt=ASTRO_DARK_ALT_DEG):
    """
    Refine each [ts_lo, ts_hi] bracket to a Unix time where Sun altitude equals target_alt (vectorized bisection).

    Takes: ts_lo and ts_hi as equal-length float64 arrays (Unix seconds); site lat/lon (deg); target_alt in degrees
    Returns: one Unix timestamp per bracket; if f(ts_lo)*f(ts_hi)>0 (no sign change), returns the interval midpoint
    """
    n_brackets = int(ts_lo.shape[0])
    if n_brackets == 0:
        return np.zeros(0, dtype=np.float64)
    lo = ts_lo.astype(np.float64, copy=True)
    hi = ts_hi.astype(np.float64, copy=True)
    f_lo = sun_altitude_minus_target(lo, latitude, longitude, target_alt)
    f_hi = sun_altitude_minus_target(hi, latitude, longitude, target_alt)
    bad_bracket = np.asarray(f_lo * f_hi > 0, dtype=np.bool_)
    out = np.empty(n_brackets, dtype=np.float64)
    out[bad_bracket] = 0.5 * (lo[bad_bracket] + hi[bad_bracket])
    active = np.logical_not(bad_bracket)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        f_mid = sun_altitude_minus_target(mid, latitude, longitude, target_alt)
        move_hi = active & (f_lo * f_mid <= 0)
        move_lo = active & (f_lo * f_mid > 0)
        hi = np.where(move_hi, mid, hi)
        f_hi = np.where(move_hi, f_mid, f_hi)
        lo = np.where(move_lo, mid, lo)
        f_lo = np.where(move_lo, f_mid, f_lo)
    out[active] = 0.5 * (lo[active] + hi[active])
    return out


# --- 3. Astronomical-dark nights: which local days → noon–noon grids → crossings → datetimes ---


def check_dates_for_local_years(years_sorted):
    """
    Flatten local calendar days for each year (current year may be truncated)

    Takes: sorted unique calendar years
    Returns: list of datetime.date values in order
    """
    out = []
    for year in years_sorted:
        full_year_start = date(year, 1, 1)
        num_days = (date(year + 1, 1, 1) - full_year_start).days
        if year == datetime.now().year:
            max_date = date.today() + timedelta(days=365)
            if date(year + 1, 1, 1) > max_date:
                num_days = (max_date - full_year_start).days
        if num_days > 0:
            out += [full_year_start + timedelta(days=day_offset) for day_offset in range(num_days)]
    return out


def local_noon_window_timestamps(check_dates, local_tz_str):
    """
    Unix timestamps for local civil noon on each date and on the following day

    Takes: list of local dates, IANA timezone name
    Returns: arrays t0_arr and t1_arr (seconds since epoch) for each row of check_dates
    """
    zi = ZoneInfo(local_tz_str)
    n_days = len(check_dates)
    t0_arr = np.empty(n_days, dtype=np.float64)
    t1_arr = np.empty(n_days, dtype=np.float64)
    for i, check_date in enumerate(check_dates):
        t0_arr[i] = datetime(check_date.year, check_date.month, check_date.day, 12, tzinfo=zi).timestamp()
        next_day = check_date + timedelta(days=1)
        t1_arr[i] = datetime(next_day.year, next_day.month, next_day.day, 12, tzinfo=zi).timestamp()
    return t0_arr, t1_arr


def build_noon_to_noon_grids(t0_arr, t1_arr, latitude, longitude, n_scan):
    """
    For each local calendar day, build a uniform time grid from that day's civil noon to the next day's civil noon,
    and the Sun's altitude in degrees minus the astronomical-dark limit (ASTRO_DARK_ALT_DEG, −18°).

    Takes: Unix arrays for window start/end per day, site latitude and longitude (deg), sample count
    Returns: ts_grid and alt_minus_dark, each shape (n_days, n_scan)
    """
    u = np.linspace(0.0, 1.0, n_scan, dtype=np.float64)
    ts_grid = t0_arr[:, None] + (t1_arr[:, None] - t0_arr[:, None]) * u
    alt_minus_dark = sun_altitude_deg(unix_timestamp_to_julian_date(ts_grid), latitude, longitude) - ASTRO_DARK_ALT_DEG
    return ts_grid, alt_minus_dark


def coarse_crossing_brackets(partial, alt_minus_dark, ts_grid, n_scan):
    """
    Build coarse Unix time brackets around each sign change of (Sun alt − (−18°)) on the noon–noon grid.

    Takes: partial — 1D bool mask, True for days that are neither all-dark nor all-light; alt_minus_dark and ts_grid shape (n_days, n_scan); n_scan column count
    Returns: (bracket_lo, bracket_hi) as two Python lists of Unix seconds (paired by index)
    """
    bracket_lo, bracket_hi = [], []
    for day_i in np.flatnonzero(partial):
        f_row = alt_minus_dark[day_i]
        ts_row = ts_grid[day_i]
        for i in range(n_scan - 1):
            if f_row[i] * f_row[i + 1] < 0.0:
                bracket_lo.append(float(ts_row[i]))
                bracket_hi.append(float(ts_row[i + 1]))
    return bracket_lo, bracket_hi


@dataclass
class DarkNightGrid:
    check_dates: list
    local_tz: object
    t0_arr: np.ndarray
    t1_arr: np.ndarray
    valid: np.ndarray
    alt_minus_dark: np.ndarray
    ts_grid: np.ndarray
    row_all_dark: np.ndarray
    row_all_light: np.ndarray
    refined_roots: np.ndarray
    n_scan: int
    latitude: float
    longitude: float


def assemble_dark_night_datetimes(grid: DarkNightGrid):
    """
    Turn per-day Sun grids and refined −18° crossings into astronomical-dark local intervals.

    Takes: DarkNightGrid — noon–noon windows, alt_minus_dark/ts_grid, all-dark/all-light/partial masks, refined_roots aligned with coarse sign changes, n_scan, site lat/lon
    Returns: list of (check_date, dark_start, dark_end); datetimes use grid.local_tz; for partial days, keeps the longest sub-interval whose midpoint is still below −18°
    """
    refined_idx = 0
    segment_mid_times = []
    nights_out = []
    partial_rows = []
    n_days = len(grid.check_dates)

    for day_i in range(n_days):
        if not grid.valid[day_i]:
            continue
        check_date = grid.check_dates[day_i]
        window_t0 = float(grid.t0_arr[day_i])
        window_t1 = float(grid.t1_arr[day_i])
        f_row = grid.alt_minus_dark[day_i]
        ts_row = grid.ts_grid[day_i]

        if grid.row_all_dark[day_i]:
            nights_out.append(
                (
                    check_date,
                    datetime.fromtimestamp(window_t0, tz=grid.local_tz),
                    datetime.fromtimestamp(window_t1, tz=grid.local_tz),
                )
            )
        elif grid.row_all_light[day_i]:
            continue
        else:
            roots = []
            for i in range(grid.n_scan - 1):
                if f_row[i] == 0.0:
                    roots.append(float(ts_row[i]))
                elif f_row[i] * f_row[i + 1] < 0.0:
                    roots.append(float(grid.refined_roots[refined_idx]))
                    refined_idx += 1
            knots = [window_t0] + sorted(roots) + [window_t1]
            partial_rows.append((check_date, knots))
            for j in range(len(knots) - 1):
                if knots[j + 1] - knots[j] >= 1e-6:
                    segment_mid_times.append(0.5 * (knots[j] + knots[j + 1]))

    mid_alt_minus = (
        sun_altitude_minus_target(np.asarray(segment_mid_times, dtype=np.float64), grid.latitude, grid.longitude)
        if segment_mid_times
        else np.zeros(0, dtype=np.float64)
    )
    mid_idx = 0
    for check_date, knots in partial_rows:
        best_lo, best_hi, best_span = None, None, -1.0
        for j in range(len(knots) - 1):
            seg_t0, seg_t1 = knots[j], knots[j + 1]
            if seg_t1 - seg_t0 < 1e-6:
                continue
            span = seg_t1 - seg_t0
            if float(mid_alt_minus[mid_idx]) < 0.0 and span > best_span:
                best_span, best_lo, best_hi = span, seg_t0, seg_t1
            mid_idx += 1
        if best_lo is None or best_hi is None or best_hi <= best_lo:
            continue
        nights_out.append(
            (check_date, datetime.fromtimestamp(best_lo, tz=grid.local_tz), datetime.fromtimestamp(best_hi, tz=grid.local_tz))
        )
    return nights_out


def compute_dark_windows_for_years(years_sorted, latitude, longitude, local_tz_str, n_scan=100):
    """
    Astronomical-dark local windows for one or more calendar years (vectorized Sun altitude on a day grid).

    Takes: sorted unique integer years; observer latitude and longitude (deg); IANA timezone name; n_scan samples
    Returns: list of (check_date, dark_start, dark_end) sorted by date (tz-aware datetimes) or empty list on error
    """
    try:
        local_tz = ZoneInfo(local_tz_str)
        check_dates = check_dates_for_local_years(years_sorted)
        if not check_dates:
            return []

        t0_arr, t1_arr = local_noon_window_timestamps(check_dates, local_tz_str)
        valid = t1_arr > t0_arr
        ts_grid, alt_minus_dark = build_noon_to_noon_grids(t0_arr, t1_arr, latitude, longitude, n_scan)
        row_all_dark = np.all(alt_minus_dark < 0.0, axis=1)
        row_all_light = np.all(alt_minus_dark > 0.0, axis=1)
        partial = valid & ~row_all_dark & ~row_all_light

        bracket_lo, bracket_hi = coarse_crossing_brackets(partial, alt_minus_dark, ts_grid, n_scan)
        refined_roots = bisect_sun_altitude_crossings(
            np.asarray(bracket_lo, dtype=np.float64),
            np.asarray(bracket_hi, dtype=np.float64),
            latitude,
            longitude,
            ASTRO_DARK_ALT_DEG,
        )

        nights_out = assemble_dark_night_datetimes(
            DarkNightGrid(
                check_dates=check_dates,
                local_tz=local_tz,
                t0_arr=t0_arr,
                t1_arr=t1_arr,
                valid=valid,
                alt_minus_dark=alt_minus_dark,
                ts_grid=ts_grid,
                row_all_dark=row_all_dark,
                row_all_light=row_all_light,
                refined_roots=refined_roots,
                n_scan=n_scan,
                latitude=latitude,
                longitude=longitude,
            )
        )
        return sorted(nights_out, key=lambda x: x[0])
    except Exception:
        return None


def compute_year_dark_windows(args):
    """
    Astronomical-dark windows for a single local calendar year

    Takes: tuple (year, latitude, longitude, local_tz_str)
    Returns: (year, list of nights) or (year, None) on error
    """
    year, latitude, longitude, local_tz_str = args
    nights_out = compute_dark_windows_for_years([year], latitude, longitude, local_tz_str)
    if nights_out is None:
        return (year, None)
    return (year, nights_out)


# --- 4. Optimal viewing: best night per catalog object during astronomical dark ---


class ObservationContext:
    """
    Hold site location and one row per local night (calendar date + dark window timestamps).

    Takes: latitude, longitude (deg); night_dates_tuples as (year, month, day) per night; parallel arrays
    night_dark_start_ts and night_dark_end_ts (Unix seconds, astro-dark bounds); IANA timezone name string
    Returns: None; sets night_dates as date objects and timestamps as float64 ndarrays plus local_tz ZoneInfo
    """

    __slots__ = ('latitude', 'longitude', 'night_dates', 'night_dark_start_ts', 'night_dark_end_ts', 'local_tz', 'local_tz_str')

    def __init__(self, latitude, longitude, night_dates_tuples, night_dark_start_ts, night_dark_end_ts, local_tz_str):
        self.latitude = latitude
        self.longitude = longitude
        self.local_tz_str = local_tz_str
        self.local_tz = ZoneInfo(local_tz_str)
        self.night_dates = [date(y, m, d) for y, m, d in night_dates_tuples]
        self.night_dark_start_ts = np.asarray(night_dark_start_ts, dtype=np.float64)
        self.night_dark_end_ts = np.asarray(night_dark_end_ts, dtype=np.float64)


TWO_PI = 2.0 * np.pi
SEG_HA_SHIFTS = np.array([-TWO_PI, 0.0, TWO_PI], dtype=np.float64)
TRANSIT_K = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
NO_VIEW_ROW_SUFFIX = (
    'N/A', 'N/A', 'Never visible', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 'N/A', 'N/A',
)


def wrap_hour_angle_to_pi_rad(ha_rad):
    # Takes: hour angle in radians (scalar or numpy array)
    # Returns: same shape, values in (-π, π]
    return (ha_rad + np.pi) % TWO_PI - np.pi


def hour_angle_half_width_rad(sin_h_min_alt, sin_dec_v, sin_lat, cos_dec_v, cos_lat):
    """
    Per-object hour-angle half-width H (rad) such that altitude >= threshold when |HA| <= H (great-circle model).

    Takes: sin(min_altitude); sin(dec), cos(dec) arrays per object; sin(lat), cos(lat) scalars for the site
    Returns: (always_above, never_above, H) as boolean/boolean/float64 arrays (length n_objects); H is 0 when never up, π when up all HA
    """
    # https://astronomy.stackexchange.com/questions/13067/conversion-from-equatorial-coordinate-to-horizon-coordinates
    # https://en.wikipedia.org/wiki/Astronomical_coordinate_systems#Equatorial_.E2.86.90.E2.86.92_horizontal
    denom = cos_dec_v * cos_lat
    denom_safe = np.where(np.abs(denom) < 1e-14, np.nan, denom)
    cos_ha_lim = (sin_h_min_alt - sin_dec_v * sin_lat) / denom_safe
    always_above = np.isfinite(cos_ha_lim) & (cos_ha_lim <= -1.0)
    never_above = ~always_above & (np.isnan(cos_ha_lim) | (cos_ha_lim > 1.0))
    n_obj = int(cos_ha_lim.shape[0])
    h_width = np.empty(n_obj, dtype=np.float64)
    h_width[always_above] = np.pi
    h_width[never_above] = 0.0
    mid = ~(always_above | never_above)
    h_width[mid] = np.arccos(np.clip(cos_ha_lim[mid], -1.0, 1.0))
    if np.any(np.isnan(cos_ha_lim)):
        sin_alt_const = sin_dec_v * sin_lat
        const_above = np.isnan(cos_ha_lim) & (sin_alt_const >= sin_h_min_alt)
        always_above = always_above | const_above
        never_above = never_above | (np.isnan(cos_ha_lim) & ~const_above)
        h_width[const_above] = np.pi
        h_width[np.isnan(cos_ha_lim) & ~const_above] = 0.0
    return always_above, never_above, h_width


@dataclass
class ViewingInput:
    ctx: ObservationContext
    object_ids: np.ndarray
    names: np.ndarray
    types: np.ndarray
    messier_col: np.ndarray
    ra_j2000: np.ndarray
    dec_j2000: np.ndarray
    ra_now: np.ndarray
    dec_now: np.ndarray
    min_altitude: float
    progress_nights: bool = False


def compute_viewing_rows(view: ViewingInput):
    """
    Find the best night and time to view each catalog object

    Takes: ViewingInput (night context, catalog arrays, min altitude, tqdm flag)
    Returns: list of tuples, one row per object for the results DataFrame
    """
    ctx = view.ctx
    object_ids, names, types, messier_col = view.object_ids, view.names, view.types, view.messier_col
    ra_j2000, dec_j2000 = view.ra_j2000, view.dec_j2000
    min_altitude, progress_nights = view.min_altitude, view.progress_nights

    # 1. Site and object directions
    lat_deg = float(ctx.latitude)
    lon_deg = float(ctx.longitude)
    lat_rad = np.deg2rad(lat_deg)
    ra_rad_vec = np.deg2rad(np.asarray(view.ra_now, dtype=np.float64))
    dec_rad_vec = np.deg2rad(np.asarray(view.dec_now, dtype=np.float64))
    sin_dec_v = np.sin(dec_rad_vec)
    cos_dec_v = np.cos(dec_rad_vec)
    sin_lat = float(np.sin(lat_rad))
    cos_lat = float(np.cos(lat_rad))
    n_obj = int(ra_rad_vec.shape[0])
    num_nights = len(ctx.night_dates)
    sin_h = float(np.sin(np.deg2rad(float(min_altitude))))

    best_duration = np.full(n_obj, -1.0, dtype=np.float64)
    best_peak_rank = np.full(n_obj, -999.0, dtype=np.float64)
    best_night_idx = np.full(n_obj, -1, dtype=np.int32)
    total_good = np.zeros(n_obj, dtype=np.int32)
    best_seg_ta = np.zeros(n_obj, dtype=np.float64)
    best_seg_tb = np.zeros(n_obj, dtype=np.float64)

    # 2. Sidereal advance per night (LST at dark start / end)
    ts0_all = ctx.night_dark_start_ts
    ts1_all = ctx.night_dark_end_ts
    jd0 = unix_timestamp_to_julian_date(ts0_all)
    jd1 = unix_timestamp_to_julian_date(ts1_all)
    lst_ends = local_sidereal_time_rad(np.concatenate((jd0, jd1)), lon_deg)
    lst0_all = lst_ends[:num_nights]
    lst1_all = lst_ends[num_nights:]
    dlst = lst1_all - lst0_all
    dlst = np.where(dlst <= 0.0, dlst + TWO_PI, dlst)
    dt_all = ts1_all - ts0_all
    omega_all = np.where(dt_all > 0.0, dlst / dt_all, 0.0)

    # 3. Per-object hour-angle cap for the altitude threshold (whole night / never / partial)
    always_above, never_above, H = hour_angle_half_width_rad(sin_h, sin_dec_v, sin_lat, cos_dec_v, cos_lat)

    night_list = list(range(num_nights))
    if progress_nights:
        try:
            from tqdm import tqdm

            night_list = tqdm(night_list, desc="Nights", unit="night", leave=False)
        except ImportError:
            pass

    # 4. Each night: visible HA segment, keep best duration (tie: higher peak altitude)
    for ni in night_list:
        ts0 = float(ts0_all[ni])
        ts1 = float(ts1_all[ni])
        if ts1 <= ts0:
            continue
        omega = float(omega_all[ni])
        if omega <= 0.0:
            continue
        lst0 = float(lst0_all[ni])

        # HA at night start, wrapped to [-π, π] for a stable local unwrapped segment.
        ha0 = wrap_hour_angle_to_pi_rad(lst0 - ra_rad_vec)
        dt = ts1 - ts0
        ha1 = ha0 + omega * dt  # unwrapped forward

        seg_len = np.zeros(n_obj, dtype=np.float64)
        seg_ta = np.full(n_obj, np.nan, dtype=np.float64)
        seg_tb = np.full(n_obj, np.nan, dtype=np.float64)
        seg_shift = np.zeros(n_obj, dtype=np.float64)

        if np.any(always_above):
            seg_len[always_above] = dt
            seg_ta[always_above] = ts0
            seg_tb[always_above] = ts1
            seg_shift[always_above] = 0.0

        cand = ~(always_above | never_above)
        if np.any(cand):
            hc0 = ha0[cand]
            hc1 = ha1[cand]
            h_c = H[cand]
            s = SEG_HA_SHIFTS[:, None]
            lo = -h_c + s
            hi = h_c + s
            a = np.maximum(hc0, lo)
            b = np.minimum(hc1, hi)
            L = np.maximum(0.0, b - a)
            si = np.argmax(L, axis=0)
            m = int(hc0.shape[0])
            ar = np.arange(m, dtype=np.intp)
            best_ta = a[si, ar]
            best_tb = b[si, ar]
            best_s = SEG_HA_SHIFTS[si]
            idx = np.flatnonzero(cand)
            good_seg = L[si, ar] > 0.0
            if np.any(good_seg):
                ii = idx[good_seg]
                aa = best_ta[good_seg]
                bb = best_tb[good_seg]
                h0 = ha0[ii]
                ta = ts0 + (aa - h0) / omega
                tb = ts0 + (bb - h0) / omega
                seg_len[ii] = np.maximum(0.0, tb - ta)
                seg_ta[ii] = ta
                seg_tb[ii] = tb
                seg_shift[ii] = best_s[good_seg]

        # Update total_good count (any visibility that night).
        total_good += (seg_len > 0.0).astype(np.int32)

        # Candidate night only matters if it can beat current best duration (duration tie-break handled below).
        night_len_h = dt / 3600.0
        refine_mask = (seg_len > 0.0) & (night_len_h >= best_duration)
        if not np.any(refine_mask):
            continue

        # Peak time: time when HA is closest to transit (HA == 2πk), within the chosen segment.
        # For our selected segment, the relevant transit target is HA == seg_shift (0, ±2π).
        jj = np.flatnonzero(refine_mask)
        # Transit time in unwrapped coords
        t_transit = ts0 + (seg_shift[jj] - ha0[jj]) / omega
        # Clamp to visibility segment
        t_peak = np.clip(t_transit, seg_ta[jj], seg_tb[jj])

        # Rank by duration (hours), then peak altitude at t_peak.
        dur_h = seg_len[jj] / 3600.0
        # Peak altitude for tie-break: use the same linear HA model as the segment math
        ha_peak = ha0[jj] + omega * (t_peak - ts0)
        sin_alt_peak = sin_dec_v[jj] * sin_lat + cos_dec_v[jj] * cos_lat * np.cos(ha_peak)
        peak_alt = np.rad2deg(np.arcsin(np.clip(sin_alt_peak, -1.0, 1.0)))

        old_d = best_duration[jj]
        old_pr = best_peak_rank[jj]
        better = (dur_h > old_d) | ((dur_h == old_d) & (peak_alt > old_pr))
        if np.any(better):
            jbest = jj[better]
            best_duration[jbest] = dur_h[better]
            best_peak_rank[jbest] = peak_alt[better]
            best_night_idx[jbest] = ni
            best_seg_ta[jbest] = seg_ta[jbest]
            best_seg_tb[jbest] = seg_tb[jbest]

    # 5. For objects with a best night: transit time in segment, then rise / set / peak azimuth
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
        # Peak is at transit (HA=0 mod 2π) if it falls in [ta, tb], otherwise at the nearer endpoint.
        # Use per-night omega derived above for each best night to compute the transit time.
        bi = best_night_idx[gid].astype(np.intp, copy=False)
        omega_g = omega_all[bi]
        lst0_g = lst0_all[bi]
        ha0_g = wrap_hour_angle_to_pi_rad(lst0_g - ra_g)
        mid_t = 0.5 * (ta + tb)
        t_cand = ts0_all[bi][:, None] + ((TRANSIT_K[None, :] * TWO_PI) - ha0_g[:, None]) / omega_g[:, None]
        # pick closest to segment midpoint
        pick = np.argmin(np.abs(t_cand - mid_t[:, None]), axis=1)
        t_transit = t_cand[np.arange(t_cand.shape[0]), pick]
        p_ts = np.clip(t_transit, ta, tb)
        n = int(gid.size)
        ra3 = np.concatenate((ra_g, ra_g, ra_g))
        dec3 = np.concatenate((dec_g, dec_g, dec_g))
        ts_all = np.concatenate((ta, tb, p_ts))
        jd_all = unix_timestamp_to_julian_date(ts_all)
        lst_all = local_sidereal_time_rad(jd_all, lon_deg)
        alt_all, az_all = equatorial_to_horizontal(ra3, dec3, lst_all, lat_rad, return_azimuth=True)

        rise_az_arr[gid] = az_all[:n]
        set_az_arr[gid] = az_all[n : 2 * n]
        peak_alt_arr[gid] = alt_all[2 * n :]
        peak_az_arr[gid] = az_all[2 * n :]
        p_ts_arr[gid] = p_ts

    # 6. Local-time strings and output tuples for the caller's DataFrame
    rows = []
    for j in range(n_obj):
        obj_id = object_ids[j]
        name = names[j]
        obj_type = types[j]
        mnum = messier_col[j]
        ra_j = round(float(ra_j2000[j]), 6)
        dec_j = round(float(dec_j2000[j]), 6)
        bi = int(best_night_idx[j])
        if bi < 0:
            rows.append((obj_id, name, obj_type, mnum, ra_j, dec_j) + NO_VIEW_ROW_SUFFIX)
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
        peak_local = datetime.fromtimestamp(p_ts, tz=ctx.local_tz)
        # Best_Date + Best_Time_Local = same local wall-clock instant as peak (may be the calendar
        # day after the astro-dark window's evening label when the peak falls after midnight).
        best_date = peak_local.date()
        rise_hm = datetime.fromtimestamp(seg_ta, tz=ctx.local_tz).strftime('%H:%M')
        set_hm = datetime.fromtimestamp(seg_tb, tz=ctx.local_tz).strftime('%H:%M')
        best_time = peak_local.strftime('%H:%M')
        best_altitude = round(peak_alt, 1)
        best_azimuth = round(peak_az, 1) % 360.0
        duration = round(float(best_duration[j]), 1)
        rise_az_out = round(rise_az, 1) % 360.0
        set_az_out = round(set_az, 1) % 360.0
        rows.append(
            (obj_id, name, obj_type, mnum, ra_j, dec_j, best_date, best_time,
             best_altitude, best_azimuth,
             rise_hm, rise_az_out, set_hm, set_az_out,
             duration, int(total_good[j]),
             dark_start.strftime('%H:%M'), dark_end.strftime('%H:%M'))
        )
    return rows


# --- 5. StarTellerCLI — site, catalog, and the two user-facing pipelines ---


class StarTellerCLI:
    """Observer site + catalog; compute viewing windows."""

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=latitude, lng=longitude)
        self.timezone_name = tz_name
        self.local_tz = ZoneInfo(tz_name) if tz_name else timezone.utc
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
        years_needed = {(start_date + timedelta(days=d)).year for d in range(days)}
        local_tz_str = self.timezone_name or "UTC"
        all_windows = compute_dark_windows_for_years(
            sorted(years_needed), self.latitude, self.longitude, local_tz_str
        )
        if all_windows is None:
            all_windows = []
        result = []
        for date_obj, dark_start, dark_end in all_windows:
            if start_date <= date_obj <= end_date:
                result.append((date_obj, dark_start, dark_end))
        return sorted(result, key=lambda x: x[0])

    def find_optimal_viewing_times(self, min_altitude=20, messier_only=False, use_tqdm=True, dark_windows=None):
        """
        Find the best night and time to view each catalog object

        Takes: minimum altitude (deg), Messier-only flag, tqdm flag, optional precomputed dark windows
        Returns: pandas DataFrame with viewing times and related columns
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
        local_tz_str = self.timezone_name or "UTC"
        if dark_windows:
            mid_idx = len(dark_windows) // 2
            _, ds, de = dark_windows[mid_idx]
            mid_jd = float(unix_timestamp_to_julian_date(0.5 * (ds.timestamp() + de.timestamp())))
        else:
            today = date.today()
            mid_jd = float(
                unix_timestamp_to_julian_date(datetime(today.year, 7, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
            )
        ra_j2000 = df_work["Right_Ascension"].to_numpy(dtype=np.float64, copy=False)
        dec_j2000 = df_work["Declination"].to_numpy(dtype=np.float64, copy=False)
        ra_now, dec_now = precess_equatorial_j2000(ra_j2000, dec_j2000, mid_jd)
        messier_col = df_work["Messier"].fillna("").astype(str).to_numpy()
        object_ids = df_work["Object"].to_numpy()
        display_names = df_work["Name"].to_numpy()
        types = df_work["Type"].to_numpy()
        # Visible_Nights_Per_Year: count of nights with any time above min altitude during astro dark
        columns = [
            'Object', 'Name', 'Type', 'Messier', 'Right_Ascension', 'Declination',
            'Best_Date', 'Best_Time_Local',
            'Max_Altitude_deg', 'Azimuth_deg',
            'Rise_Time_Local', 'Rise_Direction_deg', 'Set_Time_Local', 'Set_Direction_deg',
            'Observing_Duration_Hours', 'Visible_Nights_Per_Year',
            'Dark_Start_Local', 'Dark_End_Local',
        ]
        ctx = ObservationContext(self.latitude, self.longitude, night_dates_tuples, night_dark_start_ts, night_dark_end_ts, local_tz_str)
        results = compute_viewing_rows(
            ViewingInput(
                ctx=ctx,
                object_ids=object_ids,
                names=display_names,
                types=types,
                messier_col=messier_col,
                ra_j2000=ra_j2000,
                dec_j2000=dec_j2000,
                ra_now=ra_now,
                dec_now=dec_now,
                min_altitude=min_altitude,
                progress_nights=use_tqdm,
            )
        )
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
        results_df["never_visible"] = results_df["Max_Altitude_deg"].astype(str).isin(("Never visible", "Error"))
        results_df = results_df.sort_values(
            by=['never_visible', 'Best_Date', 'Object'],
            ascending=[True, True, True]
        ).drop('never_visible', axis=1)
        return results_df
