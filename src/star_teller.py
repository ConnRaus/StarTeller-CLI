#!/usr/bin/env python3
"""
StarTeller - Optimal Deep Sky Object Viewing Time Calculator
A tool to find the best times to observe deep sky objects throughout the year.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from skyfield.api import Star, load, wgs84, utc
from timezonefinder import TimezoneFinder
import pytz
from catalog_manager import load_ngc_catalog
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class StarTeller:
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
        
        # Detect local timezone
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=latitude, lng=longitude)
        if tz_name:
            self.local_tz = pytz.timezone(tz_name)
            print(f"Detected timezone: {tz_name}")
        else:
            self.local_tz = pytz.UTC
            print("Could not detect timezone, using UTC")
        
        # Load timescale and ephemeris data
        self.ts = load.timescale()
        # Look for ephemeris file in data/ folder
        eph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'de421.bsp')
        if os.path.exists(eph_path):
            self.eph = load(eph_path)  # JPL ephemeris from data folder
        else:
            # Fallback to current directory or Skyfield default
            self.eph = load('de421.bsp')  # JPL ephemeris
        self.earth = self.eph['earth']
        self.sun = self.eph['sun']
        
        # Create observer location
        self.observer = self.earth + wgs84.latlon(latitude, longitude, elevation_m=elevation)
        
        # Load deep sky object catalog
        self.dso_catalog = self._load_catalog(limit, catalog_filter)
    
    def _utc_to_local(self, utc_dt):
        """Convert UTC datetime to local timezone."""
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=pytz.UTC)
        return utc_dt.astimezone(self.local_tz)
    
    def _local_to_utc(self, local_dt):
        """Convert local datetime to UTC."""
        if local_dt.tzinfo is None:
            local_dt = self.local_tz.localize(local_dt)
        return local_dt.astimezone(pytz.UTC)
        
    def _load_catalog(self, limit, catalog_filter):
        """
        Load NGC/IC catalog from OpenNGC file.
        
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
        
        print(f"Loading {filter_names.get(catalog_filter, 'All Objects')} from OpenNGC catalog...")
        
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
            
            print(f"✓ Loaded {len(catalog_dict)} objects into StarTeller")
            return catalog_dict
            
        except Exception as e:
            print(f"Error loading catalog: {e}")
            print("Please ensure NGC.csv file is downloaded from OpenNGC")
            return {}
    
    def _is_astronomical_dark(self, times):
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
    
    def calculate_object_altitude(self, ra_deg, dec_deg, date_range_days=365, time_resolution_hours=3):
        """
        Calculate altitude and azimuth for a celestial object during dark times only.
        
        Args:
            ra_deg (float): Right ascension in degrees
            dec_deg (float): Declination in degrees
            date_range_days (int): Number of days to calculate (default: 365)
            time_resolution_hours (int): Time resolution in hours (default: 3)
            
        Returns:
            pandas.DataFrame: DataFrame with datetime, altitude, and azimuth (dark times only)
        """
        # Create the celestial object
        star = Star(ra_hours=ra_deg/15.0, dec_degrees=dec_deg)
        
        # Generate time range with higher resolution for better dark time detection
        start_date = datetime.now(utc)
        times = []
        
        for day in range(0, date_range_days, 1):
            for hour in range(0, 24, time_resolution_hours):
                dt = start_date + timedelta(days=day, hours=hour)
                times.append(dt)
        
        # Filter to only dark times (astronomical twilight)
        dark_mask = self._is_astronomical_dark(times)
        dark_times = [times[i] for i in range(len(times)) if dark_mask[i]]
        
        if len(dark_times) == 0:
            return pd.DataFrame({
                'datetime': [],
                'altitude_deg': [],
                'azimuth_deg': [],
                'date': [],
                'time': []
            })
        
        # Convert to Skyfield time objects
        t_array = self.ts.from_datetimes(dark_times)
        
        # Calculate positions
        astrometric = self.observer.at(t_array).observe(star)
        alt, az, distance = astrometric.apparent().altaz()
        
        # Create DataFrame
        results = pd.DataFrame({
            'datetime': dark_times,
            'altitude_deg': alt.degrees,
            'azimuth_deg': az.degrees,
            'date': [dt.date() for dt in dark_times],
            'time': [dt.time() for dt in dark_times]
        })
        
        return results
    
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
        
        # OPTIMIZATION: Calculate night midpoints once for all objects
        print("Calculating night midpoints for the year...")
        night_midpoints = self._calculate_night_midpoints()
        
        # Process unique objects with progress bar
        items = list(unique_objects.values())
        for obj_id, obj_data in tqdm(items, desc="Processing objects", unit="obj"):
            # Find the optimal night for this object
            optimal_info = self._find_optimal_night_efficient(obj_data['ra'], obj_data['dec'], 
                                                            min_altitude, direction_filter, night_midpoints)
            
            if optimal_info['best_date'] is not None:
                result_data = {
                    'Object': obj_id,
                    'Name': obj_data['name'],
                    'Type': obj_data['type'],
                    'Best_Date': optimal_info['best_date'],
                    'Best_Time_Local': optimal_info['best_time_local'],
                    'Max_Altitude_deg': optimal_info['max_altitude'],
                    'Azimuth_deg': optimal_info['max_azimuth'],
                    'Direction': optimal_info['direction'],
                    'Rise_Time_Local': optimal_info['rise_time'],
                    'Rise_Direction': optimal_info['rise_direction'],
                    'Set_Time_Local': optimal_info['set_time'],
                    'Set_Direction': optimal_info['set_direction'],
                    'Observing_Duration_Hours': optimal_info['duration_hours'],
                    'Dark_Nights_Per_Year': optimal_info['dark_nights_per_year'],
                    'Good_Viewing_Periods': optimal_info['good_viewing_periods'],
                    'Timezone': str(self.local_tz)
                }
                results.append(result_data)
            else:
                results.append({
                    'Object': obj_id,
                    'Name': obj_data['name'],
                    'Type': obj_data['type'],
                    'Best_Date': 'N/A',
                    'Best_Time_Local': 'N/A',
                    'Max_Altitude_deg': 'Never visible',
                    'Azimuth_deg': 'N/A',
                    'Direction': 'N/A',
                    'Rise_Time_Local': 'N/A',
                    'Rise_Direction': 'N/A',
                    'Set_Time_Local': 'N/A',
                    'Set_Direction': 'N/A',
                    'Observing_Duration_Hours': 0,
                    'Dark_Nights_Per_Year': 0,
                    'Good_Viewing_Periods': 0,
                    'Timezone': str(self.local_tz)
                })
        
        results_df = pd.DataFrame(results)
        
        # Sort by maximum altitude (descending), handling 'Never visible' entries
        def sort_key(x):
            if isinstance(x, str) or x == 'Never visible':
                return -999  # Put 'Never visible' at the end
            return x
        
        results_df['sort_altitude'] = results_df['Max_Altitude_deg'].apply(sort_key)
        results_df = results_df.sort_values('sort_altitude', ascending=False)
        results_df = results_df.drop('sort_altitude', axis=1)  # Remove helper column
        
        return results_df

    def _calculate_night_midpoints(self):
        """
        Calculate the midpoint of each night for the entire year.
        This is the same for all objects so we calculate once and cache.
        
        Returns:
            list: List of (date, midpoint_datetime_local, dark_start_local, dark_end_local) tuples
        """
        from datetime import datetime, timedelta, date
        
        night_midpoints = []
        start_date = date.today()
        
        for day_offset in range(365):
            check_date = start_date + timedelta(days=day_offset)
            
            # Find dark period for this date
            # Check from 4 PM to 10 AM next day to capture full possible dark period
            start_check = self.local_tz.localize(datetime.combine(check_date, datetime.min.time().replace(hour=16)))
            end_check = start_check + timedelta(hours=18)  # 4 PM to 10 AM next day
            
            dark_start = None
            dark_end = None
            
            # Find start and end of dark period
            current_time = start_check
            while current_time <= end_check:
                current_utc = current_time.astimezone(pytz.UTC)
                is_dark = self._is_astronomical_dark([current_utc])[0]
                
                if is_dark and dark_start is None:
                    dark_start = current_time
                elif not is_dark and dark_start is not None and dark_end is None:
                    dark_end = current_time
                    break
                    
                current_time += timedelta(minutes=15)  # 15-minute intervals for finding transitions
            
            # If still dark at end of check period, use end time
            if dark_start and dark_end is None:
                dark_end = end_check
            
            if dark_start and dark_end:
                # Calculate midpoint of dark period
                dark_duration = dark_end - dark_start
                midpoint = dark_start + dark_duration / 2
                
                night_midpoints.append((check_date, midpoint, dark_start, dark_end))
        
        return night_midpoints

    def _find_optimal_night_efficient(self, ra_deg, dec_deg, min_altitude, direction_filter, night_midpoints):
        """
        Efficiently find the optimal night by checking object altitude at night midpoints.
        
        Args:
            ra_deg, dec_deg: Object coordinates
            min_altitude: Minimum altitude threshold
            direction_filter: Direction filter tuple or None
            night_midpoints: Pre-calculated night midpoint data
            
        Returns:
            dict: Complete optimal viewing information
        """
        try:
            # Create the celestial object
            star = Star(ra_hours=ra_deg/15.0, dec_degrees=dec_deg)
            
            best_altitude = -90
            best_date = None
            best_midpoint = None
            best_azimuth = 0
            total_good_nights = 0
            
            # Check altitude at each night midpoint
            for check_date, midpoint_local, dark_start, dark_end in night_midpoints:
                midpoint_utc = midpoint_local.astimezone(pytz.UTC)
                
                # Calculate object position at midpoint
                t = self.ts.from_datetime(midpoint_utc)
                astrometric = self.observer.at(t).observe(star)
                alt, az, distance = astrometric.apparent().altaz()
                
                # Check if it meets our criteria at the midpoint
                above_altitude = alt.degrees >= min_altitude
                meets_direction = True
                
                if direction_filter:
                    min_az, max_az = direction_filter
                    if min_az <= max_az:
                        meets_direction = (min_az <= az.degrees <= max_az)
                    else:
                        meets_direction = (az.degrees >= min_az or az.degrees <= max_az)
                
                if above_altitude and meets_direction:
                    total_good_nights += 1
                    
                    # Track the night where it's highest at midpoint
                    if alt.degrees > best_altitude:
                        best_altitude = alt.degrees
                        best_date = check_date
                        best_midpoint = midpoint_local
                        best_azimuth = az.degrees
            
            if best_date is None:
                return {
                    'best_date': None,
                    'best_time_local': 'N/A',
                    'max_altitude': 'Never visible',
                    'max_azimuth': 'N/A',
                    'direction': 'N/A',
                    'rise_time': 'N/A',
                    'rise_direction': 'N/A',
                    'set_time': 'N/A',
                    'set_direction': 'N/A',
                    'duration_hours': 0,
                    'dark_nights_per_year': 0,
                    'good_viewing_periods': 0
                }
            
            # Get the dark period for the optimal night
            optimal_dark_start = None
            optimal_dark_end = None
            for check_date, midpoint_local, dark_start, dark_end in night_midpoints:
                if check_date == best_date:
                    optimal_dark_start = dark_start
                    optimal_dark_end = dark_end
                    break
            
            # Now find precise rise/set times and duration for the optimal night
            rise_set_info = self._find_rise_set_times(star, optimal_dark_start, optimal_dark_end, 
                                                    min_altitude, direction_filter)
            
            return {
                'best_date': best_date,
                'best_time_local': best_midpoint.time().strftime('%H:%M'),
                'max_altitude': round(best_altitude, 1),
                'max_azimuth': round(best_azimuth, 1),
                'direction': self._azimuth_to_cardinal(best_azimuth),
                'rise_time': rise_set_info['rise_time'],
                'rise_direction': rise_set_info['rise_direction'],
                'set_time': rise_set_info['set_time'],
                'set_direction': rise_set_info['set_direction'],
                'duration_hours': rise_set_info['duration_hours'],
                'dark_nights_per_year': total_good_nights,
                'good_viewing_periods': total_good_nights
            }
            
        except Exception as e:
            print(f"Error analyzing object at RA={ra_deg}, Dec={dec_deg}: {e}")
            return {
                'best_date': None,
                'best_time_local': 'N/A',
                'max_altitude': 'Error',
                'max_azimuth': 'N/A',
                'direction': 'N/A',
                'rise_time': 'N/A',
                'rise_direction': 'N/A',
                'set_time': 'N/A',
                'set_direction': 'N/A',
                'duration_hours': 0,
                'dark_nights_per_year': 0,
                'good_viewing_periods': 0
            }

    def _find_rise_set_times(self, star, dark_start, dark_end, min_altitude, direction_filter):
        """
        Use binary search to find precise rise and set times during dark hours.
        
        Args:
            star: Skyfield Star object
            dark_start, dark_end: Dark period boundaries (local time)
            min_altitude: Minimum altitude threshold
            direction_filter: Direction filter or None
            
        Returns:
            dict: Rise/set times and duration info
        """
        from datetime import timedelta
        
        def get_altitude_at_time(time_local):
            """Helper to get object altitude at a specific time."""
            time_utc = time_local.astimezone(pytz.UTC)
            t = self.ts.from_datetime(time_utc)
            astrometric = self.observer.at(t).observe(star)
            alt, az, distance = astrometric.apparent().altaz()
            return alt.degrees, az.degrees
        
        def meets_direction_filter(az):
            """Helper to check direction filter."""
            if not direction_filter:
                return True
            min_az, max_az = direction_filter
            if min_az <= max_az:
                return min_az <= az <= max_az
            else:
                return az >= min_az or az <= max_az
        
        # Binary search for rise time (first time above threshold)
        rise_time = None
        rise_az = None
        search_start = dark_start
        search_end = dark_end
        
        # Check if already above threshold at start of dark period
        start_alt, start_az = get_altitude_at_time(search_start)
        if start_alt >= min_altitude and meets_direction_filter(start_az):
            rise_time = search_start
            rise_az = start_az
        else:
            # Binary search for rise
            while (search_end - search_start).total_seconds() > 300:  # 5-minute precision
                mid_time = search_start + (search_end - search_start) / 2
                mid_alt, mid_az = get_altitude_at_time(mid_time)
                
                if mid_alt >= min_altitude and meets_direction_filter(mid_az):
                    rise_time = mid_time
                    rise_az = mid_az
                    search_end = mid_time
                else:
                    search_start = mid_time
        
        # Binary search for set time (last time above threshold)
        set_time = None
        set_az = None
        search_start = dark_start
        search_end = dark_end
        
        # Check if still above threshold at end of dark period
        end_alt, end_az = get_altitude_at_time(search_end)
        if end_alt >= min_altitude and meets_direction_filter(end_az):
            set_time = search_end
            set_az = end_az
        else:
            # Binary search for set (search backwards)
            while (search_end - search_start).total_seconds() > 300:  # 5-minute precision
                mid_time = search_start + (search_end - search_start) / 2
                mid_alt, mid_az = get_altitude_at_time(mid_time)
                
                if mid_alt >= min_altitude and meets_direction_filter(mid_az):
                    set_time = mid_time
                    set_az = mid_az
                    search_start = mid_time
                else:
                    search_end = mid_time
        
        # Calculate duration
        if rise_time and set_time:
            duration = (set_time - rise_time).total_seconds() / 3600.0  # Convert to hours
            return {
                'rise_time': rise_time.time().strftime('%H:%M'),
                'rise_direction': self._azimuth_to_cardinal(rise_az),
                'set_time': set_time.time().strftime('%H:%M'),
                'set_direction': self._azimuth_to_cardinal(set_az),
                'duration_hours': round(duration, 1)
            }
        else:
            return {
                'rise_time': 'N/A',
                'rise_direction': 'N/A',
                'set_time': 'N/A',
                'set_direction': 'N/A',
                'duration_hours': 0
            }
    
    def _azimuth_to_cardinal(self, azimuth):
        """Convert azimuth to cardinal direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = round(azimuth / 22.5) % 16
        return directions[idx]
    
    def generate_detailed_report(self, obj_id, save_csv=False):
        """
        Generate a detailed altitude/azimuth report for a specific object.
        
        Args:
            obj_id (str): Object identifier (e.g., 'M31')
            save_csv (bool): Whether to save results to CSV
            
        Returns:
            pandas.DataFrame: Detailed altitude/azimuth data
        """
        if obj_id not in self.dso_catalog:
            print(f"Object {obj_id} not found in catalog.")
            return None
        
        obj_data = self.dso_catalog[obj_id]
        print(f"Generating detailed report for {obj_id} - {obj_data['name']}")
        
        # Calculate with higher resolution (every 3 hours)
        df = self.calculate_object_altitude(obj_data['ra'], obj_data['dec'], time_resolution_hours=3)
        
        if len(df) > 0:
            # Convert UTC times to local times
            df['local_datetime'] = df['datetime'].apply(self._utc_to_local)
            df['local_date'] = df['local_datetime'].apply(lambda x: x.date())
            df['local_time'] = df['local_datetime'].apply(lambda x: x.time())
            
            # Add cardinal directions
            df['Direction'] = df['azimuth_deg'].apply(self._azimuth_to_cardinal)
        
        if save_csv:
            filename = f"{obj_id}_detailed_report.csv"
            df.to_csv(filename, index=False)
            print(f"Detailed report saved to {filename}")
        
        return df


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
    
    print(f"Loading custom objects: {', '.join(object_list)}")
    
    # Load full catalog
    full_catalog = load_ngc_catalog()
    
    if full_catalog.empty:
        print("Failed to load catalog")
        return None
    
    # Find matching objects
    custom_catalog = []
    found_objects = set()
    
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
                print(f"✓ Found {obj_id} -> {match['object_id']} ({match['name']})")
        else:
            print(f"✗ Object '{obj_id}' not found in catalog")
    
    if not custom_catalog:
        print("No objects found! Using Messier catalog instead.")
        return StarTeller(latitude, longitude, elevation, catalog_filter="messier")
    
    # Create a custom StarTeller instance
    st = StarTeller.__new__(StarTeller)
    st.latitude = latitude
    st.longitude = longitude
    st.elevation = elevation
    
    # Set up location and timing
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)
    if tz_name:
        st.local_tz = pytz.timezone(tz_name)
        print(f"Detected timezone: {tz_name}")
    else:
        st.local_tz = pytz.UTC
        print("Could not detect timezone, using UTC")
    
    # Load astronomical data
    from skyfield.api import load
    from skyfield.api import wgs84
    st.ts = load.timescale()
    # Look for ephemeris file in data/ folder
    eph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'de421.bsp')
    if os.path.exists(eph_path):
        st.eph = load(eph_path)  # JPL ephemeris from data folder
    else:
        # Fallback to current directory or Skyfield default
        st.eph = load('de421.bsp')  # JPL ephemeris
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
    
    print(f"✓ Created custom catalog with {len(st.dso_catalog)} objects")
    return st

def main():
    """Main function to run StarTeller."""
    print("=" * 60)
    print("                      StarTeller")
    print("        Deep Sky Object Optimal Viewing Calculator")
    print("=" * 60)
    
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
    if catalog_choice == "5":
        custom_objects = input("Enter object IDs (comma-separated, e.g., M31,NGC224,IC342): ").strip()
        if custom_objects:
            object_list = [obj.strip().upper() for obj in custom_objects.split(',')]
            print(f"Custom search for: {', '.join(object_list)}")
        else:
            print("No objects specified, using Messier catalog instead")
            catalog_choice = "1"
    
    # Create StarTeller instance with appropriate catalog
    print("\n" + "=" * 60)
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
    
    # Calculate optimal viewing times
    print("\n" + "=" * 60)
    results = st.find_optimal_viewing_times(min_altitude=min_alt, direction_filter=direction_filter)
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMAL VIEWING TIMES SUMMARY")
    print("=" * 60)
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(results.to_string(index=False))
    
    # Save to CSV
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"optimal_viewing_times_{datetime.now(utc).strftime('%Y%m%d_%H%M')}.csv")
    results.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # Offer detailed report
    print("\n" + "=" * 60)
    obj_choice = input("Generate detailed report for specific object? (Enter object ID like 'M31' or press Enter to skip): ")
    if obj_choice.strip():
        detailed_df = st.generate_detailed_report(obj_choice.upper(), save_csv=True)
        if detailed_df is not None:
            print(f"\nSample of detailed data for {obj_choice.upper()}:")
            print(detailed_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main() 