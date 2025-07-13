#!/usr/bin/env python3
"""
StarTeller-CLI - Optimal Deep Sky Object Viewing Time Calculator
A command-line tool to find the best times to observe deep sky objects throughout the year.
"""

import pandas as pd
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from skyfield.api import Star, load, wgs84, utc
from timezonefinder import TimezoneFinder
import pytz
from catalog_manager import load_ngc_catalog
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
warnings.filterwarnings('ignore')

# Global worker function for multiprocessing
def process_batch_worker(batch_data):
    """
    Worker function for multiprocessing batch processing.
    
    Args:
        batch_data: Tuple of (batch, min_altitude, direction_filter, night_midpoints, observer_data)
    
    Returns:
        List of optimal viewing info dictionaries
    """
    batch, min_altitude, direction_filter, night_midpoints, observer_data = batch_data
    
    # Recreate necessary objects in this process
    from skyfield.api import Star, load, wgs84
    import pytz
    
    # Unpack observer data
    latitude, longitude, elevation, local_tz_str = observer_data
    
    # Create Skyfield objects for this process
    ts = load.timescale()
    try:
        # Set up data directory and ephemeris file path
        import os
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)  # Ensure data directory exists
        eph_path = os.path.join(data_dir, 'de421.bsp')
        
        if os.path.exists(eph_path):
            eph = load(eph_path)
        else:
            # Download ephemeris to data folder
            from skyfield.iokit import Loader
            loader = Loader(data_dir)
            eph = loader('de421.bsp')
    except:
        # Fallback if there are any issues
        eph = load('de421.bsp')
    
    earth = eph['earth']
    observer = earth + wgs84.latlon(latitude, longitude, elevation_m=elevation)
    
    
    def _azimuth_to_cardinal_local(azimuth):
        """Local version of azimuth to cardinal conversion."""
        directions = [
            (0, 22.5, "N"), (22.5, 67.5, "NE"), (67.5, 112.5, "E"), (112.5, 157.5, "SE"),
            (157.5, 202.5, "S"), (202.5, 247.5, "SW"), (247.5, 292.5, "W"), (292.5, 337.5, "NW"),
            (337.5, 360, "N")
        ]
        for min_az, max_az, direction in directions:
            if min_az <= azimuth < max_az:
                return direction
        return "N"
    
    def _find_rise_set_times_local(star, dark_start, dark_end, min_altitude, direction_filter):
        """Local version of rise/set time calculation."""
        try:
            def meets_direction_filter(az):
                if direction_filter is None:
                    return True
                min_az, max_az = direction_filter
                if min_az <= max_az:
                    return min_az <= az <= max_az
                else:
                    return az >= min_az or az <= max_az
            
            def binary_search_transition(start_time, end_time, target_state):
                left = start_time
                right = end_time
                
                while (right - left).total_seconds() > 60:
                    mid = left + (right - left) / 2
                    mid_utc = mid.astimezone(pytz.UTC)
                    
                    t = ts.from_datetime(mid_utc)
                    astrometric = observer.at(t).observe(star)
                    alt, az, distance = astrometric.apparent().altaz()
                    
                    current_state = (alt.degrees >= min_altitude and meets_direction_filter(az.degrees))
                    
                    if current_state == target_state:
                        right = mid
                    else:
                        left = mid
                
                return right
            
            # Check if object is visible at start and end of dark period
            start_utc = dark_start.astimezone(pytz.UTC)
            end_utc = dark_end.astimezone(pytz.UTC)
            
            t_start = ts.from_datetime(start_utc)
            t_end = ts.from_datetime(end_utc)
            
            start_astrometric = observer.at(t_start).observe(star)
            end_astrometric = observer.at(t_end).observe(star)
            
            start_alt, start_az, _ = start_astrometric.apparent().altaz()
            end_alt, end_az, _ = end_astrometric.apparent().altaz()
            
            start_visible = (start_alt.degrees >= min_altitude and meets_direction_filter(start_az.degrees))
            end_visible = (end_alt.degrees >= min_altitude and meets_direction_filter(end_az.degrees))
            
            if start_visible and end_visible:
                # Visible entire dark period
                rise_time = dark_start.time().strftime('%H:%M')
                rise_direction = _azimuth_to_cardinal_local(start_az.degrees)
                set_time = dark_end.time().strftime('%H:%M')
                set_direction = _azimuth_to_cardinal_local(end_az.degrees)
                duration = (dark_end - dark_start).total_seconds() / 3600
                
            elif start_visible and not end_visible:
                # Visible at start, sets during dark period
                set_time_dt = binary_search_transition(dark_start, dark_end, False)
                rise_time = dark_start.time().strftime('%H:%M')
                rise_direction = _azimuth_to_cardinal_local(start_az.degrees)
                set_time = set_time_dt.time().strftime('%H:%M')
                
                set_utc = set_time_dt.astimezone(pytz.UTC)
                t_set = ts.from_datetime(set_utc)
                set_astrometric = observer.at(t_set).observe(star)
                _, set_az_calc, _ = set_astrometric.apparent().altaz()
                set_direction = _azimuth_to_cardinal_local(set_az_calc.degrees)
                
                duration = (set_time_dt - dark_start).total_seconds() / 3600
                
            elif not start_visible and end_visible:
                # Rises during dark period, visible at end
                rise_time_dt = binary_search_transition(dark_start, dark_end, True)
                rise_time = rise_time_dt.time().strftime('%H:%M')
                
                rise_utc = rise_time_dt.astimezone(pytz.UTC)
                t_rise = ts.from_datetime(rise_utc)
                rise_astrometric = observer.at(t_rise).observe(star)
                _, rise_az_calc, _ = rise_astrometric.apparent().altaz()
                rise_direction = _azimuth_to_cardinal_local(rise_az_calc.degrees)
                
                set_time = dark_end.time().strftime('%H:%M')
                set_direction = _azimuth_to_cardinal_local(end_az.degrees)
                duration = (dark_end - rise_time_dt).total_seconds() / 3600
                
            else:
                # Check if it rises and sets within the dark period
                rise_time_dt = binary_search_transition(dark_start, dark_end, True)
                
                if rise_time_dt < dark_end:
                    set_time_dt = binary_search_transition(rise_time_dt, dark_end, False)
                    
                    if set_time_dt > rise_time_dt:
                        rise_time = rise_time_dt.time().strftime('%H:%M')
                        set_time = set_time_dt.time().strftime('%H:%M')
                        
                        rise_utc = rise_time_dt.astimezone(pytz.UTC)
                        set_utc = set_time_dt.astimezone(pytz.UTC)
                        
                        t_rise = ts.from_datetime(rise_utc)
                        t_set = ts.from_datetime(set_utc)
                        
                        rise_astrometric = observer.at(t_rise).observe(star)
                        set_astrometric = observer.at(t_set).observe(star)
                        
                        _, rise_az_calc, _ = rise_astrometric.apparent().altaz()
                        _, set_az_calc, _ = set_astrometric.apparent().altaz()
                        
                        rise_direction = _azimuth_to_cardinal_local(rise_az_calc.degrees)
                        set_direction = _azimuth_to_cardinal_local(set_az_calc.degrees)
                        duration = (set_time_dt - rise_time_dt).total_seconds() / 3600
                    else:
                        return {
                            'rise_time': 'N/A', 'rise_direction': 'N/A',
                            'set_time': 'N/A', 'set_direction': 'N/A',
                            'duration_hours': 0
                        }
                else:
                    return {
                        'rise_time': 'N/A', 'rise_direction': 'N/A',
                        'set_time': 'N/A', 'set_direction': 'N/A',
                        'duration_hours': 0
                    }
            
            return {
                'rise_time': rise_time,
                'rise_direction': rise_direction,
                'set_time': set_time,
                'set_direction': set_direction,
                'duration_hours': round(duration, 1)
            }
            
        except Exception as e:
            return {
                'rise_time': 'N/A', 'rise_direction': 'N/A',
                'set_time': 'N/A', 'set_direction': 'N/A',
                'duration_hours': 0
            }
    
    # Process the batch
    batch_results = []
    
    for obj_id, obj_data in batch:
        try:
            star = Star(ra_hours=obj_data['ra']/15.0, dec_degrees=obj_data['dec'])
            
            # Find optimal night efficiently
            best_altitude = -90
            best_date = None
            best_midpoint = None
            best_azimuth = 0
            total_good_nights = 0
            best_dark_start = None
            best_dark_end = None
            
            # Collect all midpoint times for vectorized processing
            midpoint_times_utc = []
            midpoint_data = []
            
            for check_date, midpoint_local, dark_start, dark_end in night_midpoints:
                midpoint_utc = midpoint_local.astimezone(pytz.UTC)
                midpoint_times_utc.append(midpoint_utc)
                midpoint_data.append((check_date, midpoint_local, dark_start, dark_end))
            
            # Vectorized altitude calculation for all midpoints at once
            if midpoint_times_utc:
                t_array = ts.from_datetimes(midpoint_times_utc)
                astrometric = observer.at(t_array).observe(star)
                alt, az, distance = astrometric.apparent().altaz()
                
                # Process results for each night
                for j, (alt_deg, az_deg) in enumerate(zip(alt.degrees, az.degrees)):
                    check_date, midpoint_local, dark_start, dark_end = midpoint_data[j]
                    
                    # Check if it meets criteria at this midpoint
                    above_altitude = alt_deg >= min_altitude
                    meets_direction = True
                    
                    if direction_filter:
                        min_az, max_az = direction_filter
                        if min_az <= max_az:
                            meets_direction = (min_az <= az_deg <= max_az)
                        else:
                            meets_direction = (az_deg >= min_az or az_deg <= max_az)
                    
                    if above_altitude and meets_direction:
                        total_good_nights += 1
                        
                        # Track the night where it's highest at midpoint
                        if alt_deg > best_altitude:
                            best_altitude = alt_deg
                            best_date = check_date
                            best_midpoint = midpoint_local
                            best_azimuth = az_deg
                            best_dark_start = dark_start
                            best_dark_end = dark_end
            
            # Generate result for this object
            if best_date is None:
                optimal_info = {
                    'best_date': None,
                    'best_time_local': 'N/A',
                    'max_altitude': 'Never visible',
                    'max_azimuth': 'N/A',
                    'direction': 'N/A',
                    'dark_start_local': 'N/A',
                    'dark_end_local': 'N/A',
                    'rise_time': 'N/A',
                    'rise_direction': 'N/A',
                    'set_time': 'N/A',
                    'set_direction': 'N/A',
                    'duration_hours': 0,
                    'dark_nights_per_year': 0,
                    'good_viewing_periods': 0
                }
            else:
                # Find precise rise/set times for the optimal night
                rise_set_info = _find_rise_set_times_local(star, best_dark_start, best_dark_end, 
                                                        min_altitude, direction_filter)
                
                optimal_info = {
                    'best_date': best_date,
                    'best_time_local': best_midpoint.time().strftime('%H:%M'),
                    'max_altitude': round(best_altitude, 1),
                    'max_azimuth': round(best_azimuth, 1),
                    'direction': _azimuth_to_cardinal_local(best_azimuth),
                    'dark_start_local': best_dark_start.time().strftime('%H:%M'),
                    'dark_end_local': best_dark_end.time().strftime('%H:%M'),
                    'rise_time': rise_set_info['rise_time'],
                    'rise_direction': rise_set_info['rise_direction'],
                    'set_time': rise_set_info['set_time'],
                    'set_direction': rise_set_info['set_direction'],
                    'duration_hours': rise_set_info['duration_hours'],
                    'dark_nights_per_year': total_good_nights,
                    'good_viewing_periods': total_good_nights
                }
            
            batch_results.append(optimal_info)
            
        except Exception as e:
            batch_results.append({
                'best_date': None,
                'best_time_local': 'N/A',
                'max_altitude': 'Error',
                'max_azimuth': 'N/A',
                'direction': 'N/A',
                'dark_start_local': 'N/A',
                'dark_end_local': 'N/A',
                'rise_time': 'N/A',
                'rise_direction': 'N/A',
                'set_time': 'N/A',
                'set_direction': 'N/A',
                'duration_hours': 0,
                'dark_nights_per_year': 0,
                'good_viewing_periods': 0
            })
    
    return batch_results

class StarTeller:
    # ============================================================================
    # CONSTRUCTOR AND SETUP
    # ============================================================================
    
    def __init__(self, latitude, longitude, elevation=0, limit=None, catalog_filter="all"):
        """
        Initialize StarTeller with observer location.
        
        Args:
            latitude (float): Observer latitude in degrees
            longitude (float): Observer longitude in degrees  
            elevation (float): Observer elevation in meters (default: 0)
            limit (int): Maximum number of objects to load (None for all)
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
        self.dso_catalog = self._setup_catalog(limit, catalog_filter)
    
    def _generate_location_hash(self):
        """Generate a unique hash for this location for caching purposes."""
        # Round coordinates to 4 decimal places (~11m precision) for caching
        lat_rounded = round(self.latitude, 4)
        lon_rounded = round(self.longitude, 4)
        location_string = f"{lat_rounded},{lon_rounded}"
        return hashlib.md5(location_string.encode()).hexdigest()[:8]
    
    def _setup_catalog(self, limit, catalog_filter):
        """
        Load and setup the deep sky object catalog.
        
        Args:
            limit (int): Maximum number of objects to load (None for all)
            catalog_filter (str): Catalog type filter ("messier", "ic", "ngc", "all")
            
        Returns:
            dict: StarTeller-compatible catalog dictionary
        """
        filter_names = {
            "messier": "Messier Objects",
            "ic": "IC Objects", 
            "ngc": "NGC Objects",
            "all": "All Objects"
        }
        
        try:
            # Load NGC catalog with filter
            catalog_df = load_ngc_catalog(limit=limit, catalog_filter=catalog_filter)
            
            if catalog_df.empty:
                print("Failed to load NGC catalog - please ensure NGC.csv file is present")
                return {}
            
            # Convert to StarTeller format
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
        
        Args:
            min_altitude (float): Minimum altitude in degrees (default: 20)
            direction_filter (tuple): Optional (min_az, max_az) in degrees to filter by direction
            
        Returns:
            pandas.DataFrame: Summary table with optimal viewing information
        """
        results = []
        
        print("Calculating optimal viewing times for deep sky objects...")
        print(f"Observer location: {self.latitude:.2f}°, {self.longitude:.2f}°")
        print(f"Local timezone: {self.local_tz}")
        print(f"Minimum altitude: {min_altitude}°")
        print("Dark time criteria: Sun below -18° (astronomical twilight)")
        print("Note: Winter = longer dark periods, Summer = shorter dark periods")
        
        if direction_filter:
            print(f"Direction filter: {direction_filter[0]}° to {direction_filter[1]}° azimuth")
        
        # Remove duplicates (prefer Messier names over NGC names)
        unique_objects = {}
        for obj_id, obj_data in self.dso_catalog.items():
            # Create a key based on coordinates (same coordinates = same object)
            coord_key = (round(obj_data['ra'], 4), round(obj_data['dec'], 4))
            
            if coord_key not in unique_objects:
                unique_objects[coord_key] = (obj_id, obj_data)
            else:
                # If we already have this object, prefer Messier names
                existing_id, existing_data = unique_objects[coord_key]
                if obj_id.startswith('M') and not existing_id.startswith('M'):
                    unique_objects[coord_key] = (obj_id, obj_data)
        
        print(f"Removed {len(self.dso_catalog) - len(unique_objects)} duplicate objects")
        
        # OPTIMIZATION: Calculate night midpoints once for all objects (with caching)
        print("Calculating night midpoints for the year...")
        night_midpoints = self.get_night_midpoints()
        
        # Process unique objects in batches with multithreading for better performance
        items = list(unique_objects.values())
        batch_size = 10  # Process 10 objects at once
        
        # Determine optimal number of processes (use all CPU cores, but cap at reasonable limit)
        available_cores = multiprocessing.cpu_count()
        max_batches = len(items) // batch_size + 1
        
        # Use all cores, but don't exceed number of batches or a reasonable maximum
        max_workers = min(available_cores, max_batches, 64)  # Cap at 64 for sanity
        
        print(f"Detected {available_cores} CPU cores, using {max_workers} processes")
        
        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        print(f"Processing {len(batches)} batches using {max_workers} processes...")
        
        # Prepare observer data for worker processes
        observer_data = (self.latitude, self.longitude, self.elevation, str(self.local_tz))
        
        # Process batches in parallel using multiprocessing
        all_batch_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {}
            for batch in batches:
                batch_data = (batch, min_altitude, direction_filter, night_midpoints, observer_data)
                future = executor.submit(process_batch_worker, batch_data)
                future_to_batch[future] = batch
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches", unit="batch"):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_batch_results.append((batch, batch_results))
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Create error results for this batch
                    error_results = []
                    for _ in range(len(batch)):
                        error_results.append({
                            'best_date': None,
                            'best_time_local': 'N/A',
                            'max_altitude': 'Error',
                            'max_azimuth': 'N/A',
                            'direction': 'N/A',
                            'dark_start_local': 'N/A',
                            'dark_end_local': 'N/A',
                            'rise_time': 'N/A',
                            'rise_direction': 'N/A',
                            'set_time': 'N/A',
                            'set_direction': 'N/A',
                            'duration_hours': 0,
                            'dark_nights_per_year': 0,
                            'good_viewing_periods': 0
                        })
                    all_batch_results.append((batch, error_results))
        
        # Sort results back to original order to maintain consistency
        batch_to_index = {id(batch): i for i, batch in enumerate(batches)}
        sorted_results = sorted(all_batch_results, key=lambda x: batch_to_index[id(x[0])])
        
        # Process results in the original order for consistency
        for batch, batch_results in sorted_results:
            for (obj_id, obj_data), optimal_info in zip(batch, batch_results):
                if optimal_info['best_date'] is not None:
                    result_data = {
                        # Object Information
                        'Object': obj_id,
                        'Name': obj_data['name'],
                        'Type': obj_data['type'],
                        
                        # Best Viewing Information
                        'Best_Date': optimal_info['best_date'],
                        'Best_Time_Local': optimal_info['best_time_local'],
                        'Max_Altitude_deg': optimal_info['max_altitude'],
                        'Azimuth_deg': optimal_info['max_azimuth'],
                        'Direction': optimal_info['direction'],
                        
                        # Object Observing Times
                        'Rise_Time_Local': optimal_info['rise_time'],
                        'Rise_Direction': optimal_info['rise_direction'],
                        'Set_Time_Local': optimal_info['set_time'],
                        'Set_Direction': optimal_info['set_direction'],
                        'Observing_Duration_Hours': optimal_info['duration_hours'],
                        
                        # Statistics & Reference Info
                        'Dark_Nights_Per_Year': optimal_info['dark_nights_per_year'],
                        'Good_Viewing_Periods': optimal_info['good_viewing_periods'],
                        'Dark_Start_Local': optimal_info['dark_start_local'],
                        'Dark_End_Local': optimal_info['dark_end_local'],
                        'Timezone': str(self.local_tz)
                    }
                    results.append(result_data)
                else:
                    results.append({
                        # Object Information
                        'Object': obj_id,
                        'Name': obj_data['name'],
                        'Type': obj_data['type'],
                        
                        # Best Viewing Information
                        'Best_Date': 'N/A',
                        'Best_Time_Local': 'N/A',
                        'Max_Altitude_deg': 'Never visible',
                        'Azimuth_deg': 'N/A',
                        'Direction': 'N/A',
                        
                        # Object Observing Times
                        'Rise_Time_Local': 'N/A',
                        'Rise_Direction': 'N/A',
                        'Set_Time_Local': 'N/A',
                        'Set_Direction': 'N/A',
                        'Observing_Duration_Hours': 0,
                        
                        # Statistics & Reference Info
                        'Dark_Nights_Per_Year': 0,
                        'Good_Viewing_Periods': 0,
                        'Dark_Start_Local': 'N/A',
                        'Dark_End_Local': 'N/A',
                        'Timezone': str(self.local_tz)
                    })
        
        results_df = pd.DataFrame(results)
        
        # Handle empty results gracefully
        if results_df.empty:
            return results_df
        
        # Sort by maximum altitude (descending), handling 'Never visible' entries
        def sort_key(x):
            if isinstance(x, str) or x == 'Never visible':
                return -999  # Put 'Never visible' at the end
            return x
        
        results_df['sort_altitude'] = results_df['Max_Altitude_deg'].apply(sort_key)
        results_df = results_df.sort_values('sort_altitude', ascending=False)
        results_df = results_df.drop('sort_altitude', axis=1)  # Remove helper column
        
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

def create_custom_starteller(latitude, longitude, elevation, object_list):
    """
    Create a StarTeller instance with custom object list.
    
    Args:
        latitude (float): Observer latitude
        longitude (float): Observer longitude  
        elevation (float): Observer elevation
        object_list (list): List of object IDs to search for
        
    Returns:
        StarTeller: Instance with custom catalog
    """
    from catalog_manager import load_ngc_catalog
    
    # Load full catalog
    full_catalog = load_ngc_catalog()
    
    if full_catalog.empty:
        print("Failed to load catalog")
        return None
    
    # Find matching objects
    custom_catalog = []
    found_objects = set()
    found_count = 0
    
    for obj_id in object_list:
        obj_id_upper = obj_id.upper()
        
        # Try exact match first
        matches = full_catalog[full_catalog['object_id'].str.upper() == obj_id_upper]
        
        if matches.empty:
            # Try partial matches (e.g., "M31" matches "M31" object_id)
            matches = full_catalog[full_catalog['object_id'].str.upper().str.contains(obj_id_upper, na=False)]
        
        if matches.empty:
            # Try searching in names
            matches = full_catalog[full_catalog['name'].str.upper().str.contains(obj_id_upper, na=False)]
        
        if not matches.empty:
            # Take the first match
            match = matches.iloc[0]
            if match['object_id'] not in found_objects:
                custom_catalog.append(match)
                found_objects.add(match['object_id'])
                found_count += 1
        else:
            print(f"✗ Object '{obj_id}' not found in catalog")
    
    if not custom_catalog:
        print("No objects found! Using Messier catalog instead.")
        return StarTeller(latitude, longitude, elevation, catalog_filter="messier")
    
    print(f"✓ Found {found_count}/{len(object_list)} custom objects")
    
    # Create a custom StarTeller instance
    st = StarTeller.__new__(StarTeller)
    st.latitude = latitude
    st.longitude = longitude
    st.elevation = elevation
    
    # Create location hash for caching
    st.location_hash = st._generate_location_hash()
    
    # Set up location and timing
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)
    if tz_name:
        st.local_tz = pytz.timezone(tz_name)
        print(f"✓ Timezone: {tz_name}")
    else:
        st.local_tz = pytz.UTC
        print("✓ Timezone: UTC (could not auto-detect)")
    
    # Load astronomical data
    from skyfield.api import load
    from skyfield.api import wgs84
    st.ts = load.timescale()
    
    # Set up data directory and ephemeris file path
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)  # Ensure data directory exists
    eph_path = os.path.join(data_dir, 'de421.bsp')
    
    if os.path.exists(eph_path):
        st.eph = load(eph_path)  # JPL ephemeris from data folder
    else:
        # Download ephemeris to data folder
        from skyfield.iokit import Loader
        loader = Loader(data_dir)
        st.eph = loader('de421.bsp')  # This will download to data/ folder
    st.earth = st.eph['earth'] 
    st.sun = st.eph['sun']
    st.observer = st.earth + wgs84.latlon(latitude, longitude, elevation_m=elevation)
    
    # Create custom catalog
    st.dso_catalog = {}
    for obj_data in custom_catalog:
        obj_id = obj_data['object_id']
        display_name = obj_data.get('common_name', '') or obj_data['name']
        
        st.dso_catalog[obj_id] = {
            'ra': float(obj_data['ra_deg']),
            'dec': float(obj_data['dec_deg']),
            'name': display_name,
            'type': obj_data['type']
        }
    
    print(f"✓ Catalog: {len(st.dso_catalog)} custom objects")
    return st

def main():
    """Main function to run StarTeller."""
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
    print("5. Custom Objects (specify your own list)")
    
    catalog_choice = input("Enter choice (1-5, default 1): ").strip() or "1"
    
    # Handle custom objects
    custom_objects = None
    object_list = None
    if catalog_choice == "5":
        custom_objects = input("Enter object IDs (comma-separated, e.g., M31,NGC224,IC342): ").strip()
        if custom_objects:
            object_list = [obj.strip().upper() for obj in custom_objects.split(',')]
        else:
            print("No objects specified, using Messier catalog instead")
            catalog_choice = "1"
    
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
    
    # Create StarTeller instance with appropriate catalog
    if catalog_choice == "5" and custom_objects:
        st = create_custom_starteller(latitude, longitude, elevation, object_list)
    else:
        catalog_params = {
            "1": ("messier", 150),      # Messier + some extras to ensure all 110 are included
            "2": ("ic", 5000),          # IC objects only
            "3": ("ngc", 8000),         # NGC objects only  
            "4": ("all", None)          # All objects
        }
        
        catalog_type, limit = catalog_params.get(catalog_choice, ("messier", 150))
        st = StarTeller(latitude, longitude, elevation, limit=limit, catalog_filter=catalog_type)
    
    if st is None:
        print("Failed to create StarTeller instance. Exiting.")
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