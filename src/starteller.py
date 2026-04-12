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


def unix_timestamp_to_julian_date(ts_unix):
    return np.asarray(ts_unix, dtype=np.float64) / 86400.0 + 2440587.5


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


def equatorial_to_horizontal_deg(ra_deg, dec_deg, lst_rad, lat_rad, return_azimuth=True):
    """
    Calculate altitude and (optionally) azimuth from equatorial coordinates.
    https://astronomy.stackexchange.com/questions/13067/conversion-from-equatorial-coordinate-to-horizon-coordinates
    https://en.wikipedia.org/wiki/Astronomical_coordinate_systems

    Takes: Right Ascension and Declination in degrees, Local Sidereal Time in radians, latitude in radians
    Returns: alt_deg, and az_deg if return_azimuth is True
    """
    # Convert to radians
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    # Hour angle = LST - RA
    ha_rad = lst_rad - ra_rad

    # sin a = cos(LHA) cos δ cos φ + sin δ sin φ (https://aa.usno.navy.mil/faq/alt_az); ha_rad = LHA
    sin_alt = (
        np.cos(ha_rad) * np.cos(dec_rad) * np.cos(lat_rad)
        + np.sin(dec_rad) * np.sin(lat_rad)
    )
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    alt_deg = np.rad2deg(alt_rad)

    if not return_azimuth:
        return alt_deg

    # Azimuth calculation
    cos_alt = np.cos(alt_rad)
    # Avoid division by zero at zenith
    cos_alt = np.where(np.abs(cos_alt) < 1e-10, np.copysign(1e-10, cos_alt), cos_alt)

    sin_az = -np.cos(dec_rad) * np.sin(ha_rad) / cos_alt
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * np.sin(alt_rad)) / (np.cos(lat_rad) * cos_alt)

    az_rad = np.arctan2(sin_az, cos_az)

    # Convert to degrees
    az_deg = np.rad2deg(az_rad) % 360.0

    return alt_deg, az_deg


# -------------------------------------------------------------------------------------
# 3. Sun position & astronomical-dark windows (smooth alt(ts); roots at −18°)
# -------------------------------------------------------------------------------------

def sun_equatorial_deg(jd):
    """
    Calculate the Sun's RA and Dec.
    Equations from https://aa.usno.navy.mil/faq/sun_approx

    Takes: Julian date (scalar or numpy array)
    Returns: (sun_ra_deg, sun_dec_deg) in degrees (scalar or numpy array)
    """
    # NumPy math works on both scalars and arrays; we only branch at the end for output type.
    is_scalar = np.isscalar(jd)
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
    Calculate Sun altitude.
    Equations from: https://aa.usno.navy.mil/faq/alt_az (but using LST not GAST)

    Takes: Julian date (scalar or numpy array), latitude(deg), longitude(deg)
    Returns: Sun altitude in degrees (scalar or numpy array)
    """
    if np.isscalar(jd):
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
    alt_deg, _ = equatorial_to_horizontal_deg(sun_ra, sun_dec, lst_rad, lat_rad)
    return alt_deg


ASTRO_DARK_ALT_DEG = -18.0

def sun_altitude_minus_target(ts_unix, latitude, longitude, target_alt=ASTRO_DARK_ALT_DEG):
    # Helper for root finding on Sun altitude
    jd = float(unix_timestamp_to_julian_date(ts_unix))
    return float(sun_altitude_deg(jd, latitude, longitude)) - target_alt


def bisect_sun_altitude_crossing(ts_lo, ts_hi, latitude, longitude, target_alt=ASTRO_DARK_ALT_DEG):
    """
    Find the time where Sun altitude == target_alt using bisection.

    Takes: Two Unix timestamps bracketing a crossing, latitude(deg), longitude(deg), target altitude (deg)
    Returns: Unix timestamp (seconds)
    """
    flo = sun_altitude_minus_target(ts_lo, latitude, longitude, target_alt)
    fhi = sun_altitude_minus_target(ts_hi, latitude, longitude, target_alt)
    if flo * fhi > 0:
        return 0.5 * (ts_lo + ts_hi)
    lo, hi = float(ts_lo), float(ts_hi)
    fl, fh = flo, fhi
    for _ in range(32):
        m = 0.5 * (lo + hi)
        fm = sun_altitude_minus_target(m, latitude, longitude, target_alt)
        if fl * fm <= 0:
            hi, fh = m, fm
        else:
            lo, fl = m, fm
    return 0.5 * (lo + hi)


def dark_window_local_noon_day(check_date, latitude, longitude, local_tz, n_scan=100):
    """
    Find the longest astronomical-dark window for one local date. n_scan of 86 is bare minimum to pass built in tests.

    Takes: local date, latitude(deg), longitude(deg), timezone, scan points
    Returns: (dark_start_ts, dark_end_ts) Unix timestamps, or None if no darkness
    """
    t0 = local_tz.localize(datetime.combine(check_date, time(12, 0))).timestamp()
    next_day = check_date + timedelta(days=1)
    t1 = local_tz.localize(datetime.combine(next_day, time(12, 0))).timestamp()
    if t1 <= t0:
        return None

    ts = np.linspace(t0, t1, n_scan)
    jd = unix_timestamp_to_julian_date(ts)
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
            r = bisect_sun_altitude_crossing(float(ts[i]), float(ts[i + 1]), latitude, longitude, ASTRO_DARK_ALT_DEG)
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
        if sun_altitude_minus_target(mid, latitude, longitude, ASTRO_DARK_ALT_DEG) < 0.0:
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
            span = dark_window_local_noon_day(check_date, latitude, longitude, local_tz)
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


def compute_viewing_rows_batch(ctx, object_ids, names, types, messier_col, ra_j2000, dec_j2000, ra_now, dec_now, min_altitude, n_scan=25, progress_nights=False):
    """
    Find the best night/time to view each object.

    Takes: precomputed night context + catalog arrays + minimum altitude (deg)
    Returns: list of output rows (tuples) for the final DataFrame
    """
    # Core idea:
    # - Altitude depends on hour angle: sin(alt) = sin(dec)*sin(lat) + cos(dec)*cos(lat)*cos(HA)
    # - Solve "alt >= min_altitude" as a simple hour-angle limit |HA| <= H
    # - For each astro-dark night, approximate HA(t) as linear using LST at the night endpoints
    # - Pick the longest visible segment; tie-break with peak altitude
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
    h = float(min_altitude)
    h_rad = np.deg2rad(h)
    sin_h = float(np.sin(h_rad))

    best_duration = np.full(n_obj, -1.0, dtype=np.float64)
    best_peak_rank = np.full(n_obj, -999.0, dtype=np.float64)
    best_night_idx = np.full(n_obj, -1, dtype=np.int32)
    total_good = np.zeros(n_obj, dtype=np.int32)
    best_seg_ta = np.zeros(n_obj, dtype=np.float64)
    best_seg_tb = np.zeros(n_obj, dtype=np.float64)

    ts0_all = np.asarray(ctx.night_dark_start_ts, dtype=np.float64)
    ts1_all = np.asarray(ctx.night_dark_end_ts, dtype=np.float64)
    # Precompute LST at night endpoints (per-night linear HA model).
    # We compute LST for all nights in one vectorized call, then derive a per-night sidereal rate.
    jd0 = unix_timestamp_to_julian_date(ts0_all)
    jd1 = unix_timestamp_to_julian_date(ts1_all)
    lst0_all = local_sidereal_time_rad(jd0, lon_deg)
    lst1_all = local_sidereal_time_rad(jd1, lon_deg)
    two_pi = 2.0 * np.pi
    # unwrap forward across the interval (LST is modulo 2π)
    dlst = lst1_all - lst0_all
    dlst = np.where(dlst <= 0.0, dlst + two_pi, dlst)
    dt_all = ts1_all - ts0_all
    # Avoid division by zero; nights with dt<=0 are skipped below.
    omega_all = np.where(dt_all > 0.0, dlst / dt_all, 0.0)

    # Visibility hour-angle half-width per object (H, radians). Handle edge cases:
    # - denom ~ 0 -> altitude is (nearly) constant in time w.r.t HA; either always above or never.
    denom = cos_dec_v * cos_lat
    # For numerical stability near poles/dec=±90.
    denom_safe = np.where(np.abs(denom) < 1e-14, np.nan, denom)
    cosH = (sin_h - sin_dec_v * sin_lat) / denom_safe
    # Always above: cosH <= -1 -> H = π (full circle). Never above: cosH > 1 -> no solutions.
    always_above = np.isfinite(cosH) & (cosH <= -1.0)
    never_above = ~always_above & (np.isnan(cosH) | (cosH > 1.0))
    H = np.empty(n_obj, dtype=np.float64)
    H[always_above] = np.pi
    H[never_above] = 0.0
    mid = ~(always_above | never_above)
    H[mid] = np.arccos(np.clip(cosH[mid], -1.0, 1.0))

    # For denom≈0 cases where cosH is nan, decide by constant-alt test.
    # sin_alt_const = sin(dec)*sin(lat) (since cos(dec)*cos(lat)≈0)
    if np.any(np.isnan(cosH)):
        sin_alt_const = sin_dec_v * sin_lat
        const_above = np.isnan(cosH) & (sin_alt_const >= sin_h)
        always_above |= const_above
        never_above |= (np.isnan(cosH) & ~const_above)
        H[const_above] = np.pi
        H[np.isnan(cosH) & ~const_above] = 0.0

    def _wrap_to_pi(x):
        return (x + np.pi) % two_pi - np.pi

    night_list = list(range(num_nights))
    if progress_nights:
        try:
            from tqdm import tqdm

            night_list = tqdm(night_list, desc="Nights", unit="night", leave=False)
        except ImportError:
            pass

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
        ha0 = _wrap_to_pi(lst0 - ra_rad_vec)
        dt = ts1 - ts0
        ha1 = ha0 + omega * dt  # unwrapped forward

        # Build the best (longest) continuous segment above threshold within the night for each object.
        # If always_above: whole night. If never_above: none.
        seg_len = np.zeros(n_obj, dtype=np.float64)
        seg_ta = np.full(n_obj, np.nan, dtype=np.float64)
        seg_tb = np.full(n_obj, np.nan, dtype=np.float64)
        seg_shift = np.zeros(n_obj, dtype=np.float64)  # which 2π shift interval produced the best segment

        # Whole-night visibility
        if np.any(always_above):
            seg_len[always_above] = dt
            seg_ta[always_above] = ts0
            seg_tb[always_above] = ts1
            seg_shift[always_above] = 0.0

        cand = ~(always_above | never_above)
        if np.any(cand):
            hc0 = ha0[cand]
            hc1 = ha1[cand]
            Hc = H[cand]

            # Consider primary interval and ±2π-shifted intervals to handle wrap across ±π.
            shifts = np.array([-two_pi, 0.0, two_pi], dtype=np.float64)
            # We’ll compute intersection lengths for each shift and pick the best.
            bestL = np.zeros(hc0.shape[0], dtype=np.float64)
            bestTa = np.zeros(hc0.shape[0], dtype=np.float64)
            bestTb = np.zeros(hc0.shape[0], dtype=np.float64)
            bestS = np.zeros(hc0.shape[0], dtype=np.float64)

            for s in shifts:
                lo = -Hc + s
                hi = Hc + s
                a = np.maximum(hc0, lo)
                b = np.minimum(hc1, hi)
                L = np.maximum(0.0, b - a)
                take = L > bestL
                if np.any(take):
                    bestL = np.where(take, L, bestL)
                    bestTa = np.where(take, a, bestTa)
                    bestTb = np.where(take, b, bestTb)
                    bestS = np.where(take, s, bestS)

            # Convert HA bounds back to timestamps.
            # ha(t) = ha0 + omega*(t - ts0) => t = ts0 + (ha - ha0)/omega
            idx = np.flatnonzero(cand)
            Lgood = bestL > 0.0
            if np.any(Lgood):
                ii = idx[Lgood]
                a = bestTa[Lgood]
                b = bestTb[Lgood]
                h0 = ha0[ii]
                ta = ts0 + (a - h0) / omega
                tb = ts0 + (b - h0) / omega
                seg_len[ii] = np.maximum(0.0, tb - ta)
                seg_ta[ii] = ta
                seg_tb[ii] = tb
                seg_shift[ii] = bestS[Lgood]

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
        jd_peak = unix_timestamp_to_julian_date(t_peak)
        lst_peak = local_sidereal_time_rad(jd_peak, lon_deg)
        peak_alt = equatorial_to_horizontal_deg(
            np.rad2deg(ra_rad_vec[jj]),
            np.rad2deg(dec_rad_vec[jj]),
            lst_peak,
            lat_rad,
            return_azimuth=False,
        )

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
        ha0_g = _wrap_to_pi(lst0_g - ra_g)
        # Choose the nearest k (HA=2πk) for the unwrapped segment; because ha0 is in [-π,π],
        # k in {0, ±1} is sufficient. Pick the k that yields a transit time closest to mid-segment.
        mid_t = 0.5 * (ta + tb)
        # Candidate transits for k=-1,0,1
        k = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        t_cand = ts0_all[bi][:, None] + ((k[None, :] * two_pi) - ha0_g[:, None]) / omega_g[:, None]
        # pick closest to segment midpoint
        pick = np.argmin(np.abs(t_cand - mid_t[:, None]), axis=1)
        t_transit = t_cand[np.arange(t_cand.shape[0]), pick]
        p_ts = np.clip(t_transit, ta, tb)
        n = int(gid.size)
        ra3 = np.concatenate([ra_g, ra_g, ra_g])
        dec3 = np.concatenate([dec_g, dec_g, dec_g])
        ts_all = np.concatenate([ta, tb, p_ts])
        jd_all = unix_timestamp_to_julian_date(ts_all)
        lst_all = local_sidereal_time_rad(jd_all, lon_deg)
        alt_all, az_all = equatorial_to_horizontal_deg(
            np.rad2deg(ra3),
            np.rad2deg(dec3),
            lst_all,
            lat_rad,
            return_azimuth=True,
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
        if best_azimuth >= 360.0:
            best_azimuth = 0.0
        duration = round(float(best_duration[j]), 1)
        rise_az_out = round(rise_az, 1)
        if rise_az_out >= 360.0:
            rise_az_out = 0.0
        set_az_out = round(set_az, 1)
        if set_az_out >= 360.0:
            set_az_out = 0.0
        rows.append(
            (obj_id, name, obj_type, mnum, ra_j, dec_j, best_date, best_time,
             best_altitude, best_azimuth,
             rise_hm, rise_az_out, set_hm, set_az_out,
             duration, int(total_good[j]),
             dark_start.strftime('%H:%M'), dark_end.strftime('%H:%M'))
        )
    return rows

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
        Find the best night/time to view each object.

        Takes: min_alt(deg)*, messier_only(bool)*, use_tqdm(bool)*, dark_windows(list)*, time_grid_points(int)*
        Returns: final DataFrame with viewing times
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
            mid_jd = float(unix_timestamp_to_julian_date(0.5 * (ds.timestamp() + de.timestamp())))
        else:
            today = date.today()
            mid_jd = float(
                unix_timestamp_to_julian_date(datetime(today.year, 7, 1, 12, 0, 0, tzinfo=pytz.UTC).timestamp())
            )
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
        results = compute_viewing_rows_batch(ctx, object_ids, display_names, types, messier_col, ra_j2000, dec_j2000, ra_now, dec_now, min_altitude, n_scan=int(time_grid_points), progress_nights=use_tqdm)
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
