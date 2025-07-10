#!/usr/bin/env python3
"""
Test script for StarTeller - validates functionality with sample data
"""

import sys
import os
# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from star_teller import StarTeller
import pandas as pd

def test_basic_functionality():
    """Test basic StarTeller functionality with sample location."""
    print("=" * 60)
    print("                 StarTeller Test")
    print("=" * 60)
    
    # Test location: New York City area
    latitude = 40.7
    longitude = -74.0
    elevation = 50
    
    print(f"Testing with sample location: {latitude}°N, {longitude}°W")
    print("Initializing StarTeller...")
    
    try:
        # Initialize StarTeller with small test catalog (just enough to include M31)
        st = StarTeller(latitude, longitude, elevation, limit=35)
        print("✓ StarTeller initialized successfully")
        
        # Test catalog loading
        print(f"✓ Loaded {len(st.dso_catalog)} deep sky objects")
        
        # Show some loaded objects for debugging
        print("First 10 objects loaded:")
        for i, obj_id in enumerate(list(st.dso_catalog.keys())[:10]):
            obj_data = st.dso_catalog[obj_id]
            print(f"  {obj_id}: {obj_data['name']} ({obj_data['type']})")
        
        # Check if we have any Messier objects
        messier_objects = [obj for obj in st.dso_catalog.keys() if obj.startswith('M')]
        print(f"Found {len(messier_objects)} Messier objects: {messier_objects[:5]}...")
        
        # Test individual object calculation
        print("Testing individual object calculation...")
        test_obj = 'M31'  # Andromeda Galaxy
        if test_obj not in st.dso_catalog:
            print(f"❌ ERROR: {test_obj} not found in catalog!")
            print("Available objects (first 10):")
            for i, obj_id in enumerate(list(st.dso_catalog.keys())[:10]):
                print(f"  {obj_id}")
            return False
        
        obj_data = st.dso_catalog[test_obj]
        df = st.calculate_object_altitude(obj_data['ra'], obj_data['dec'], date_range_days=30, time_resolution_hours=12)
        print(f"✓ Calculated {len(df)} data points for {test_obj}")
        
        # Test optimal viewing times calculation
        print("Testing optimal viewing times calculation...")
        results = st.find_optimal_viewing_times(min_altitude=15)
        print(f"✓ Generated optimal viewing times for {len(results)} objects")
        
        # Display sample results
        print("\n" + "=" * 60)
        print("SAMPLE RESULTS (Top 3 Objects)")
        print("=" * 60)
        
        # Configure pandas display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        print(results.head(3).to_string(index=False))
        
        # Test detailed report
        print(f"\nTesting detailed report for {test_obj}...")
        detailed_df = st.generate_detailed_report(test_obj)
        print(f"✓ Generated detailed report with {len(detailed_df)} data points")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("StarTeller is working correctly.")
        print("You can now run: python star_teller.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check your internet connection (Skyfield needs to download ephemeris data)")
        print("3. Ensure you're in the correct virtual environment")
        return False

def test_direction_filtering():
    """Test direction filtering functionality."""
    print("\nTesting direction filtering...")
    
    try:
        st = StarTeller(40.7, -74.0, limit=35)
        
        # Test normal range filter
        results_east = st.find_optimal_viewing_times(min_altitude=15, direction_filter=(45, 135))
        print(f"✓ East filter (45°-135°): {len(results_east)} objects found")
        
        # Test wrapping range filter  
        results_north = st.find_optimal_viewing_times(min_altitude=15, direction_filter=(315, 45))
        print(f"✓ North filter (315°-45°): {len(results_north)} objects found")
        
        return True
        
    except Exception as e:
        print(f"❌ Direction filtering test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting StarTeller validation tests...\n")
    
    basic_test = test_basic_functionality()
    
    if basic_test:
        direction_test = test_direction_filtering()
        
        if direction_test:
            print("\n🎉 All tests completed successfully!")
            print("StarTeller is ready for astrophotography planning!")
        else:
            print("\n⚠️  Basic functionality works, but direction filtering has issues.")
    else:
        print("\n💥 Basic functionality test failed. Please check your installation.")
        sys.exit(1) 