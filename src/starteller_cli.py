#!/usr/bin/env python3
"""
StarTeller-CLI - Optimal Deep Sky Object Viewing Time Calculator
A command-line tool to find the best times to observe deep sky objects throughout the year.
"""

import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from skyfield.api import Star, load, wgs84, utc
from timezonefinder import TimezoneFinder
import pytz
try:
    from .catalog_manager import load_ngc_catalog
except ImportError:
    from catalog_manager import load_ngc_catalog
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Get optimal process count
NUM_WORKERS = cpu_count() or 8

# Global variables for worker processes (initialized once per worker)
_worker_ts = None
_worker_observer = None
_worker_eph = None
_worker_t_array = None
_worker_night_dates = None
_worker_night_midpoint_locals = None
_worker_night_dark_starts = None
_worker_night_dark_ends = None
_worker_local_tz = None
_worker_local_tz_str = None

def _init_worker(latitude, longitude, elevation, eph_path, t_array_data, 
                 night_dates_iso, night_midpoint_isos, night_dark_start_isos, 
                 night_dark_end_isos, local_tz_str):
    """Initialize worker process with ALL shared data (called once per worker)."""
    global _worker_ts, _worker_observer, _worker_eph, _worker_t_array
    global _worker_night_dates, _worker_night_midpoint_locals
    global _worker_night_dark_starts, _worker_night_dark_ends
    global _worker_local_tz, _worker_local_tz_str
    
    from skyfield.api import load, wgs84
    from datetime import date
    
    # Initialize Skyfield objects
    _worker_ts = load.timescale()
    if os.path.exists(eph_path):
        _worker_eph = load(eph_path)
    else:
        _worker_eph = load('de421.bsp')
    
    earth = _worker_eph['earth']
    _worker_observer = earth + wgs84.latlon(latitude, longitude, elevation_m=elevation)
    
    # Initialize timezone
    _worker_local_tz_str = local_tz_str
    _worker_local_tz = pytz.timezone(local_tz_str)
    
    # Pre-compute time array ONCE per worker (not per object!)
    midpoint_times_utc = [datetime.fromtimestamp(ts, tz=pytz.UTC) for ts in t_array_data]
    _worker_t_array = _worker_ts.from_datetimes(midpoint_times_utc)
    
    # Pre-parse all night data ONCE per worker
    _worker_night_dates = [date.fromisoformat(s) for s in night_dates_iso]
    _worker_night_midpoint_locals = [datetime.fromisoformat(s) for s in night_midpoint_isos]
    _worker_night_dark_starts = [datetime.fromisoformat(s) for s in night_dark_start_isos]
    _worker_night_dark_ends = [datetime.fromisoformat(s) for s in night_dark_end_isos]

def _process_object_worker(args):
    """Worker function to process a single object (runs in subprocess)."""
    global _worker_ts, _worker_observer, _worker_t_array
    global _worker_night_dates, _worker_night_midpoint_locals
    global _worker_night_dark_starts, _worker_night_dark_ends
    global _worker_local_tz, _worker_local_tz_str
    
    # Only receive minimal object data - night data is already in worker globals!
    obj_id, ra, dec, name, obj_type, min_altitude, direction_filter = args
    
    from skyfield.api import Star
    import numpy as np
    
    try:
        # Create Star object
        star = Star(ra_hours=ra/15.0, dec_degrees=dec)
        
        # VECTORIZED: Calculate alt/az for ALL nights at once
        # Uses pre-computed _worker_t_array
        astrometric = _worker_observer.at(_worker_t_array).observe(star)
        alt, az, _ = astrometric.apparent().altaz()
        
        alt_degrees = alt.degrees
        az_degrees = az.degrees
        
        # Apply filters
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
            return (obj_id, name, obj_type, 'N/A', 'N/A', 'Never visible', 'N/A', 
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 0, 'N/A', 'N/A')
        
        # Find best night
        masked_altitudes = np.where(valid_mask, alt_degrees, -999)
        best_idx = int(np.argmax(masked_altitudes))
        
        best_altitude = round(float(alt_degrees[best_idx]), 1)
        best_azimuth = round(float(az_degrees[best_idx]), 1)
        best_date = _worker_night_dates[best_idx]
        best_midpoint = _worker_night_midpoint_locals[best_idx]
        best_dark_start = _worker_night_dark_starts[best_idx]
        best_dark_end = _worker_night_dark_ends[best_idx]
        
        # Calculate rise/set times with simplified approach
        start_utc = best_dark_start.astimezone(pytz.UTC)
        end_utc = best_dark_end.astimezone(pytz.UTC)
        
        t_endpoints = _worker_ts.from_datetimes([start_utc, end_utc])
        endpoint_astrometric = _worker_observer.at(t_endpoints).observe(star)
        endpoint_alt, endpoint_az, _ = endpoint_astrometric.apparent().altaz()
        
        start_alt, end_alt = endpoint_alt.degrees[0], endpoint_alt.degrees[1]
        start_az, end_az = endpoint_az.degrees[0], endpoint_az.degrees[1]
        
        def meets_dir(az):
            if direction_filter is None:
                return True
            min_az, max_az = direction_filter
            if min_az <= max_az:
                return min_az <= az <= max_az
            return az >= min_az or az <= max_az
        
        start_visible = (start_alt >= min_altitude and meets_dir(start_az))
        end_visible = (end_alt >= min_altitude and meets_dir(end_az))
        
        # Simplified rise/set
        if start_visible and end_visible:
            rise_time = best_dark_start.strftime('%H:%M')
            set_time = best_dark_end.strftime('%H:%M')
            rise_dir = _azimuth_to_cardinal(start_az)
            set_dir = _azimuth_to_cardinal(end_az)
            duration = round((best_dark_end - best_dark_start).total_seconds() / 3600, 1)
        elif start_visible:
            rise_time = best_dark_start.strftime('%H:%M')
            set_time = '~' + best_dark_end.strftime('%H:%M')
            rise_dir = _azimuth_to_cardinal(start_az)
            set_dir = _azimuth_to_cardinal(end_az)
            duration = round((best_dark_end - best_dark_start).total_seconds() / 3600 * 0.5, 1)
        elif end_visible:
            rise_time = '~' + best_dark_start.strftime('%H:%M')
            set_time = best_dark_end.strftime('%H:%M')
            rise_dir = _azimuth_to_cardinal(start_az)
            set_dir = _azimuth_to_cardinal(end_az)
            duration = round((best_dark_end - best_dark_start).total_seconds() / 3600 * 0.5, 1)
        else:
            rise_time = 'N/A'
            set_time = 'N/A'
            rise_dir = 'N/A'
            set_dir = 'N/A'
            duration = 0
        
        # Return as tuple (much faster to serialize than dict!)
        return (obj_id, name, obj_type, best_date, best_midpoint.strftime('%H:%M'),
                best_altitude, best_azimuth, _azimuth_to_cardinal(best_azimuth),
                rise_time, rise_dir, set_time, set_dir, duration,
                total_good_nights, total_good_nights,
                best_dark_start.strftime('%H:%M'), best_dark_end.strftime('%H:%M'))
        
    except Exception as e:
        return (obj_id, name, obj_type, 'N/A', 'N/A', 'Error', 'N/A', 
                'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0, 0, 0, 'N/A', 'N/A')

# Vectorized helper functions for fast computation
def _azimuth_to_cardinal(azimuth):
    """Convert azimuth angle to cardinal direction."""
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
        
        Args:
            latitude (float): Observer latitude in degrees
            longitude (float): Observer longitude in degrees  
            elevation (float): Observer elevation in meters (default: 0)
            catalog_filter (str): Catalog type filter ("messier", "ic", "ngc", "all")
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
        
        # Load timescale and ephemeris data
        self.ts = load.timescale()
        
        # Set up data directory and ephemeris file path
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)  # Ensure data directory exists
        eph_path = os.path.join(data_dir, 'de421.bsp')
        
        if os.path.exists(eph_path):
            self.eph = load(eph_path)  # JPL ephemeris from data folder
        else:
            # Download ephemeris to data folder
            from skyfield.iokit import Loader
            loader = Loader(data_dir)
            self.eph = loader('de421.bsp')  # This will download to data/ folder
        self.earth = self.eph['earth']
        self.sun = self.eph['sun']
        
        # Create observer location
        self.observer = self.earth + wgs84.latlon(latitude, longitude, elevation_m=elevation)
        
        # Load deep sky object catalog
        self.dso_catalog = self._setup_catalog(catalog_filter)
    
    def _generate_location_hash(self):
        """Generate a unique hash for this location for caching purposes."""
        # Round coordinates to 4 decimal places (~11m precision) for caching
        lat_rounded = round(self.latitude, 4)
        lon_rounded = round(self.longitude, 4)
        location_string = f"{lat_rounded},{lon_rounded}"
        return hashlib.md5(location_string.encode()).hexdigest()[:8]
    
    def _setup_catalog(self, catalog_filter):
        """
        Load and setup the deep sky object catalog.
        
        Args:
            catalog_filter (str): Catalog type filter ("messier", "ic", "ngc", "all")
            
        Returns:
            dict: StarTellerCLI-compatible catalog dictionary
        """
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
            
            # Convert to StarTellerCLI format
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
        """Get the cache filepath for night midpoints."""
        if year is None:
            year = datetime.now().year
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'user_data', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"night_midpoints_{self.location_hash}_{year}.pkl")
    
    def _save_cache(self, night_midpoints, year):
        """Save night midpoints to cache file."""
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
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            print(f"Warning: Could not save night midpoints cache: {e}")
            return False
    
    def _load_cache(self, year):
        """Load night midpoints from cache file."""
        try:
            cache_file = self._get_cache_filepath(year)
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify the cache is for the same location and timezone
            if (abs(cache_data['latitude'] - self.latitude) < 0.0001 and 
                abs(cache_data['longitude'] - self.longitude) < 0.0001 and
                cache_data['timezone'] == str(self.local_tz)):
                
                print(f"✓ Using cached night midpoints for {year}: {os.path.basename(cache_file)}")
                return cache_data['night_midpoints']
            else:
                # Cache mismatch - will recalculate silently
                return None
                
        except Exception as e:
            print(f"Warning: Could not load night midpoints cache: {e}")
            return None
    
    def manage_cache_files(self, action="status"):
        """
        Manage cached night midpoints data.
        
        Args:
            action (str): "status" to show cache info, "clear" to delete cache files
        """
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'user_data', 'cache')
        
        if action == "status":
            print(f"\nCache Status for Location: {self.latitude:.4f}°, {self.longitude:.4f}°")
            print(f"Location Hash: {self.location_hash}")
            
            if not os.path.exists(cache_dir):
                print("No cache directory found.")
                return
            
            # Find cache files for this location
            cache_files = []
            for filename in os.listdir(cache_dir):
                if filename.startswith(f"night_midpoints_{self.location_hash}_"):
                    cache_files.append(filename)
            
            if not cache_files:
                print("No cached night midpoints found for this location.")
                return
            
            print(f"Found {len(cache_files)} cached year(s):")
            total_size = 0
            for filename in sorted(cache_files):
                filepath = os.path.join(cache_dir, filename)
                file_size = os.path.getsize(filepath)
                total_size += file_size
                
                # Extract year from filename
                year = filename.split('_')[-1].replace('.pkl', '')
                
                # Get creation date
                try:
                    with open(filepath, 'rb') as f:
                        cache_data = pickle.load(f)
                    created_date = cache_data.get('created_date', 'Unknown')
                    nights_count = len(cache_data.get('night_midpoints', []))
                    print(f"  {year}: {nights_count} nights, {file_size/1024:.1f} KB, created: {created_date}")
                except:
                    print(f"  {year}: {file_size/1024:.1f} KB (corrupted)")
            
            print(f"Total cache size: {total_size/1024:.1f} KB")
            
        elif action == "clear":
            if not os.path.exists(cache_dir):
                print("No cache directory found.")
                return
            
            # Find and delete cache files for this location
            deleted_count = 0
            for filename in os.listdir(cache_dir):
                if filename.startswith(f"night_midpoints_{self.location_hash}_"):
                    filepath = os.path.join(cache_dir, filename)
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        print(f"Deleted: {filename}")
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")
            
            if deleted_count == 0:
                print("No cache files found for this location.")
            else:
                print(f"Deleted {deleted_count} cache file(s).")
        
        else:
            print(f"Unknown action: {action}. Use 'status' or 'clear'.")
    
    # ============================================================================
    # ASTRONOMICAL UTILITIES
    # ============================================================================
    
    def _is_dark_sky(self, times):
        """
        Check if times are during astronomical darkness (sun below -18°).
        
        Astronomical twilight occurs when the sun is 18° or more below the horizon.
        This is the darkest natural condition when even the faintest stars are visible.
        Perfect for astrophotography as there's no interference from scattered sunlight.
        
        Winter = longer dark periods (up to 14+ hours at mid-latitudes)
        Summer = shorter dark periods (as little as 4-6 hours at mid-latitudes)
        
        Args:
            times (list): List of datetime objects (UTC)
            
        Returns:
            numpy.array: Boolean array indicating dark times
        """
        # Convert to Skyfield time objects
        t_array = self.ts.from_datetimes(times)
        
        # Calculate sun position
        sun_astrometric = self.observer.at(t_array).observe(self.sun)
        sun_alt, sun_az, sun_distance = sun_astrometric.apparent().altaz()
        
        # Astronomical twilight: sun below -18 degrees
        # This eliminates all twilight interference for astrophotography
        return sun_alt.degrees < -18.0
    
    def _find_transition_time(self, start_time, end_time, looking_for_dark_start=True):
        """
        Find precise time when sky transitions between light and dark.
        
        Args:
            start_time, end_time: Search window (local time)
            looking_for_dark_start: If True, find light->dark transition. If False, find dark->light.
            
        Returns:
            datetime: Transition time or None if no transition found
        """
        # Early exit if window too small
        if (end_time - start_time).total_seconds() < 120:
            return None
        
        def is_dark_at_time(time_local):
            """Helper to check if a specific time is astronomically dark."""
            time_utc = time_local.astimezone(pytz.UTC)
            return self._is_dark_sky([time_utc])[0]
        
        left, right = start_time, end_time
        
        # Check if transition actually exists in this window
        left_dark = is_dark_at_time(left)
        right_dark = is_dark_at_time(right)
        
        if looking_for_dark_start:
            # Looking for light (False) -> dark (True) transition
            if left_dark or not right_dark:
                return None  # No valid transition in window
        else:
            # Looking for dark (True) -> light (False) transition  
            if not left_dark or right_dark:
                return None  # No valid transition in window
        
        # Binary search until we get 5-minute precision (good enough for midpoint calculation)
        while (right - left).total_seconds() > 300:  # 5-minute precision (reduced from 1-minute)
            mid = left + (right - left) / 2
            mid_dark = is_dark_at_time(mid)
            
            if looking_for_dark_start:
                # Looking for light -> dark transition
                if mid_dark:
                    right = mid  # Transition is before mid
                else:
                    left = mid   # Transition is after mid
            else:
                # Looking for dark -> light transition
                if mid_dark:
                    left = mid   # Transition is after mid
                else:
                    right = mid  # Transition is before mid
        
        # Return the transition point (when darkness changes)
        return right if looking_for_dark_start else left
    
    # ============================================================================
    # NIGHT MIDPOINT CALCULATION
    # ============================================================================
    
    def get_night_midpoints(self, start_date=None, days=365):
        """
        Get night midpoints for the specified period, using cache when available.
        
        Args:
            start_date (date): Start date for calculation (default: today)
            days (int): Number of days to calculate (default: 365)
        
        Returns:
            list: List of (date, midpoint_datetime_local, dark_start_local, dark_end_local) tuples
        """
        from datetime import date
        
        if start_date is None:
            start_date = date.today()
        
        end_date = start_date + timedelta(days=days-1)
        
        # Check if we can use cached data
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
        
        # If we have all the data we need, return it
        if not missing_years and len(all_cached_midpoints) >= days * 0.95:  # Allow 5% missing for edge cases
            print(f"✓ Using cached night midpoints ({len(all_cached_midpoints)} nights)")
            return sorted(all_cached_midpoints, key=lambda x: x[0])
        
        # Calculate missing years efficiently - calculate full years for better caching
        print("Calculating night midpoints with 1-minute precision using binary search...")
        
        if missing_years:
            print(f"Missing cache for years: {missing_years}")
        
        all_calculated_midpoints = []
        
        # Calculate full years for missing data
        for year in sorted(missing_years):
            full_year_start = date(year, 1, 1)
            full_year_days = (date(year + 1, 1, 1) - full_year_start).days
            
            # For current year, don't calculate past today + 365 days for efficiency
            if year == datetime.now().year:
                max_date = date.today() + timedelta(days=365)
                if date(year + 1, 1, 1) > max_date:
                    full_year_days = (max_date - full_year_start).days
            
            # Use the internal calculation method
            year_midpoints = self._calculate_night_midpoints(full_year_start, full_year_days, year)
            if year_midpoints:
                self._save_cache(year_midpoints, year)
                all_calculated_midpoints.extend(year_midpoints)
        
        # Combine cached and calculated data
        all_midpoints = all_cached_midpoints + all_calculated_midpoints
        
        # Filter to requested date range and sort
        result = []
        for date_obj, midpoint, dark_start, dark_end in all_midpoints:
            if start_date <= date_obj <= end_date:
                result.append((date_obj, midpoint, dark_start, dark_end))
        
        return sorted(result, key=lambda x: x[0])
    
    def _calculate_night_midpoints(self, start_date, days, year=None):
        """
        Internal implementation: Calculate night midpoints using vectorized approach.
        Uses coarse grid search + targeted binary search for optimal performance.
        
        Args:
            start_date (date): Start date for calculation
            days (int): Number of days to calculate
            year (int): Year being calculated (for progress display)
        """
        from datetime import datetime, timedelta
        
        # OPTIMIZATION: Use vectorized coarse grid search + targeted binary search
        # This reduces calculations from ~8000 to ~800 for a full year
        
        print(f"Calculating night midpoints for {days} days using optimized vectorized approach...")
        
        # Step 1: Create coarse time grid across all days (every 30 minutes)
        all_times = []
        time_to_date = {}
        
        for day_offset in range(days):
            check_date = start_date + timedelta(days=day_offset)
            
            # Sample every 30 minutes from 3 PM to 11 AM next day
            afternoon = self.local_tz.localize(datetime.combine(check_date, datetime.min.time().replace(hour=15)))
            current_time = afternoon
            end_time = afternoon + timedelta(hours=20)  # 3 PM to 11 AM next day
            
            while current_time <= end_time:
                all_times.append(current_time)
                time_to_date[current_time] = check_date
                current_time += timedelta(minutes=30)
        
        # Step 2: Vectorized darkness calculation for all times at once
        print(f"Calculating sun positions for {len(all_times)} sample times...")
        all_times_utc = [t.astimezone(pytz.UTC) for t in all_times]
        t_array = self.ts.from_datetimes(all_times_utc)
        
        # Single vectorized calculation for all times
        sun_astrometric = self.observer.at(t_array).observe(self.sun)
        sun_alt, sun_az, sun_distance = sun_astrometric.apparent().altaz()
        is_dark = sun_alt.degrees < -18.0
        
        # Step 3: Find dark periods for each day using the coarse grid
        night_midpoints = []
        
        for day_offset in tqdm(range(days), desc=f"Processing nights for {year}" if year else "Processing nights", unit="day"):
            check_date = start_date + timedelta(days=day_offset)
            
            # Find all times for this day
            day_times = []
            day_darkness = []
            
            for i, time_dt in enumerate(all_times):
                if time_to_date.get(time_dt) == check_date:
                    day_times.append(time_dt)
                    day_darkness.append(is_dark[i])
            
            if len(day_times) < 2:
                continue
            
            # Find the dark period from coarse samples
            dark_start_idx = None
            dark_end_idx = None
            
            # Find first dark time (dark start)
            for i in range(len(day_darkness) - 1):
                if not day_darkness[i] and day_darkness[i + 1]:
                    dark_start_idx = i
                    break
            
            # Find last dark time (dark end)
            for i in range(len(day_darkness) - 1, 0, -1):
                if day_darkness[i - 1] and not day_darkness[i]:
                    dark_end_idx = i
                    break
            
            # Handle edge cases
            if dark_start_idx is None:
                # Check if it's already dark at start
                if day_darkness[0]:
                    dark_start_idx = 0
                else:
                    # No dark period found
                    continue
            
            if dark_end_idx is None:
                # Check if it's still dark at end
                if day_darkness[-1]:
                    dark_end_idx = len(day_darkness) - 1
                else:
                    # No dark period found
                    continue
            
            # Step 4: Refine dark start and end times with targeted binary search
            # Only search in small windows around the coarse estimates
            
            coarse_dark_start = day_times[dark_start_idx]
            coarse_dark_end = day_times[dark_end_idx]
            
            # Binary search for precise dark start (search 1 hour around coarse estimate)
            if dark_start_idx > 0:
                search_start = day_times[dark_start_idx - 1]
                search_end = min(day_times[dark_start_idx + 1] if dark_start_idx + 1 < len(day_times) else day_times[dark_start_idx], 
                               day_times[dark_start_idx] + timedelta(hours=1))
                dark_start = self._find_transition_time(search_start, search_end, True)
            else:
                dark_start = coarse_dark_start
            
            # Binary search for precise dark end (search 1 hour around coarse estimate)
            if dark_end_idx < len(day_times) - 1:
                search_start = max(day_times[dark_end_idx - 1] if dark_end_idx > 0 else day_times[dark_end_idx],
                                 day_times[dark_end_idx] - timedelta(hours=1))
                search_end = day_times[dark_end_idx + 1] if dark_end_idx + 1 < len(day_times) else day_times[dark_end_idx]
                dark_end = self._find_transition_time(search_start, search_end, False)
            else:
                dark_end = coarse_dark_end
            
            # Fallback to coarse estimates if binary search fails
            if dark_start is None:
                dark_start = coarse_dark_start
            if dark_end is None:
                dark_end = coarse_dark_end
            
            # Calculate midpoint
            if dark_start and dark_end and dark_end > dark_start:
                dark_duration = dark_end - dark_start
                midpoint = dark_start + dark_duration / 2
                night_midpoints.append((check_date, midpoint, dark_start, dark_end))
        
        print(f"✓ Calculated {len(night_midpoints)} night midpoints using vectorized approach")
        return night_midpoints
    
    # ============================================================================
    # MAIN FUNCTIONALITY
    # ============================================================================
    
    def find_optimal_viewing_times(self, min_altitude=20, direction_filter=None):
        """
        Find optimal viewing times for all objects in the catalog.
        Uses multiprocessing with persistent workers for maximum CPU utilization.
        
        Args:
            min_altitude (float): Minimum altitude in degrees (default: 20)
            direction_filter (tuple): Optional (min_az, max_az) in degrees to filter by direction
            
        Returns:
            pandas.DataFrame: Summary table with optimal viewing information
        """
        print("Calculating optimal viewing times for deep sky objects...")
        print(f"Observer location: {self.latitude:.2f}°, {self.longitude:.2f}°")
        print(f"Local timezone: {self.local_tz}")
        print(f"Minimum altitude: {min_altitude}°")
        print("Dark time criteria: Sun below -18° (astronomical twilight)")
        print(f"Using {NUM_WORKERS} CPU cores for parallel processing...")
        
        if direction_filter:
            print(f"Direction filter: {direction_filter[0]}° to {direction_filter[1]}° azimuth")
        
        # Remove duplicates (prefer Messier names over NGC names)
        unique_objects = {}
        for obj_id, obj_data in self.dso_catalog.items():
            coord_key = (round(obj_data['ra'], 4), round(obj_data['dec'], 4))
            
            if coord_key not in unique_objects:
                unique_objects[coord_key] = (obj_id, obj_data)
            else:
                existing_id, existing_data = unique_objects[coord_key]
                if obj_id.startswith('M') and not existing_id.startswith('M'):
                    unique_objects[coord_key] = (obj_id, obj_data)
        
        print(f"Processing {len(unique_objects)} unique objects (removed {len(self.dso_catalog) - len(unique_objects)} duplicates)")
        
        # Get night midpoints (cached)
        print("Loading night midpoints...")
        night_midpoints = self.get_night_midpoints()
        num_nights = len(night_midpoints)
        
        # Pre-convert data to serializable formats for worker initialization
        print("Preparing data for parallel processing...")
        midpoint_times_utc = [mp[1].astimezone(pytz.UTC) for mp in night_midpoints]
        
        # Convert to timestamps (floats) for efficient serialization
        t_array_data = [t.timestamp() for t in midpoint_times_utc]
        
        # Convert dates/times to ISO strings for worker initialization
        night_dates_iso = [mp[0].isoformat() for mp in night_midpoints]
        night_midpoint_isos = [mp[1].isoformat() for mp in night_midpoints]
        night_dark_start_isos = [mp[2].isoformat() for mp in night_midpoints]
        night_dark_end_isos = [mp[3].isoformat() for mp in night_midpoints]
        
        items = list(unique_objects.values())
        local_tz_str = str(self.local_tz)
        
        # Get ephemeris path for workers
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        eph_path = os.path.join(data_dir, 'de421.bsp')
        
        print(f"Computing altitudes for {len(items)} objects across {num_nights} nights...")
        
        # Prepare work items - MINIMAL data only (night data is in worker globals)
        work_items = [
            (obj_id, obj_data['ra'], obj_data['dec'], obj_data['name'], obj_data['type'],
             min_altitude, direction_filter)
            for obj_id, obj_data in items
        ]
        
        # Process objects in parallel using Pool with initializer
        # Night data is passed ONCE to initializer, not with every task!
        results = []
        
        # Column names for the tuple results
        columns = ['Object', 'Name', 'Type', 'Best_Date', 'Best_Time_Local',
                   'Max_Altitude_deg', 'Azimuth_deg', 'Direction',
                   'Rise_Time_Local', 'Rise_Direction', 'Set_Time_Local', 'Set_Direction',
                   'Observing_Duration_Hours', 'Dark_Nights_Per_Year', 'Good_Viewing_Periods',
                   'Dark_Start_Local', 'Dark_End_Local']
        
        with Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(self.latitude, self.longitude, self.elevation, eph_path,
                      t_array_data, night_dates_iso, night_midpoint_isos,
                      night_dark_start_isos, night_dark_end_isos, local_tz_str)
        ) as pool:
            # Use imap_unordered with large chunksize for minimal IPC overhead
            chunksize = max(100, len(work_items) // NUM_WORKERS)
            for result in tqdm(
                pool.imap_unordered(_process_object_worker, work_items, chunksize=chunksize),
                total=len(items),
                desc="Processing objects",
                unit="obj"
            ):
                results.append(result)
        
        # Convert tuple results to DataFrame
        results_df = pd.DataFrame(results, columns=columns)
        results_df['Timezone'] = local_tz_str
        
        if results_df.empty:
            return results_df
        
        # Sort by maximum altitude (descending)
        def sort_key(x):
            if isinstance(x, str) or x == 'Never visible':
                return -999
            return x
        
        results_df['sort_altitude'] = results_df['Max_Altitude_deg'].apply(sort_key)
        results_df = results_df.sort_values('sort_altitude', ascending=False)
        results_df = results_df.drop('sort_altitude', axis=1)
        
        return results_df


def save_location(latitude, longitude, elevation):
    """Save user location to a file."""
    try:
        # Ensure user_data directory exists
        user_data_dir = os.path.join(os.path.dirname(__file__), '..', 'user_data')
        os.makedirs(user_data_dir, exist_ok=True)
        
        location_file = os.path.join(user_data_dir, 'user_location.txt')
        with open(location_file, 'w') as f:
            f.write(f"{latitude},{longitude},{elevation}")
        print(f"✓ Location saved: {latitude:.2f}°, {longitude:.2f}°, {elevation}m")
    except Exception as e:
        print(f"Warning: Could not save location: {e}")

def load_location():
    """Load user location from file."""
    try:
        location_file = os.path.join(os.path.dirname(__file__), '..', 'user_data', 'user_location.txt')
        with open(location_file, 'r') as f:
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
    """Get user location, either from saved file or manual entry."""
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
    
    # === COLLECT ALL USER INPUT UPFRONT ===
    
    # Get user location (saved or manual input)
    latitude, longitude, elevation = get_user_location()
    
    # Choose catalog type
    print("\nChoose catalog:")
    print("1. Messier Objects (~110 famous deep sky objects)")
    print("2. IC Objects (~5,000 Index Catalog objects)")
    print("3. NGC Objects (~8,000 New General Catalog objects)")
    print("4. All Objects (~13,000 NGC + IC objects)")
    
    catalog_choice = input("Enter choice (1-4, default 1): ").strip() or "1"

    
    # Get viewing preferences
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
    
    # === BEGIN PROCESSING ===
    print("\n" + "=" * 60)
    print("PROCESSING...")
    print("=" * 60)
    
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
    
    # === SAVE RESULTS ===
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"optimal_viewing_times_{datetime.now(utc).strftime('%Y%m%d_%H%M')}.csv")
    results.to_csv(filename, index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"✓ Results saved to: {filename}")
    print(f"✓ Found optimal viewing times for {len(results)} objects")
    
    # Show quick summary
    visible_count = len(results[results['Max_Altitude_deg'] != 'Never visible'])
    print(f"✓ {visible_count} objects visible above {min_alt}°")
    
    print("\nOpen the CSV file to see complete viewing schedule!")
    print("=" * 60)


if __name__ == "__main__":
    main() 