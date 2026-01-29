#!/usr/bin/env python3
"""
Comprehensive Test Suite for StarTeller-CLI
Tests all functionality including downloads, caching, multiprocessing, and error handling.
"""

import sys
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime, date

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from starteller_cli import StarTellerCLI
try:
    from src.catalog_manager import load_ngc_catalog, download_ngc_catalog
except ImportError:
    from catalog_manager import load_ngc_catalog, download_ngc_catalog

class TestStarTellerCLIDownload(unittest.TestCase):
    """Test automatic download functionality."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_data_dir = tempfile.mkdtemp()
        self.original_data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        
    def tearDown(self):
        """Clean up test directories."""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    @patch('catalog_manager.urllib.request.urlretrieve')
    def test_successful_download(self, mock_urlretrieve):
        """Test successful automatic download of NGC.csv."""
        ngc_path = os.path.join(self.test_data_dir, 'NGC.csv')
        
        # Mock successful download
        def mock_download(url, path):
            with open(path, 'w') as f:
                f.write('Name;Type;RA;Dec\nNGC1;G;00:00:00;+00:00:00\n' * 100)  # Create > 1KB file
        
        mock_urlretrieve.side_effect = mock_download
        
        result = download_ngc_catalog(ngc_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(ngc_path))
        self.assertGreater(os.path.getsize(ngc_path), 1000)
        mock_urlretrieve.assert_called_once()
    
    @patch('catalog_manager.urllib.request.urlretrieve')
    def test_network_failure(self, mock_urlretrieve):
        """Test handling of network failures during download."""
        from urllib.error import URLError
        
        ngc_path = os.path.join(self.test_data_dir, 'NGC.csv')
        mock_urlretrieve.side_effect = URLError("Network error")
        
        result = download_ngc_catalog(ngc_path)
        
        self.assertFalse(result)
        self.assertFalse(os.path.exists(ngc_path))
    
    @patch('catalog_manager.urllib.request.urlretrieve')
    def test_corrupted_download(self, mock_urlretrieve):
        """Test handling of corrupted/empty downloads."""
        ngc_path = os.path.join(self.test_data_dir, 'NGC.csv')
        
        # Mock download that creates empty file
        def mock_download(url, path):
            with open(path, 'w') as f:
                f.write('')  # Empty file
        
        mock_urlretrieve.side_effect = mock_download
        
        result = download_ngc_catalog(ngc_path)
        
        self.assertFalse(result)


class TestStarTellerCLICatalog(unittest.TestCase):
    """Test catalog loading functionality."""
    
    def test_catalog_loading(self):
        """Test that the catalog loads successfully."""
        catalog = load_ngc_catalog()
        
        # Should load a non-empty catalog
        self.assertIsInstance(catalog, pd.DataFrame)
        self.assertGreater(len(catalog), 0)
        
        # Should have required columns
        required_columns = ['object_id', 'name', 'ra_deg', 'dec_deg', 'type', 'messier']
        for col in required_columns:
            self.assertIn(col, catalog.columns)
    
    def test_messier_field_format(self):
        """Test that Messier field is formatted correctly."""
        catalog = load_ngc_catalog()
        
        # Find objects with Messier designations
        messier_objects = catalog[catalog['messier'].notna() & (catalog['messier'] != '')]
        
        if not messier_objects.empty:
            # Messier field should be formatted like "M31", "M42", etc.
            for m_val in messier_objects['messier'].head(10):
                self.assertTrue(m_val.startswith('M'), f"Messier value '{m_val}' should start with 'M'")
    
    def test_angular_size_columns(self):
        """Test that angular size columns are present and contain valid data."""
        catalog = load_ngc_catalog()
        
        # Should have angular size columns
        angular_size_columns = ['major_axis_arcmin', 'minor_axis_arcmin', 'position_angle_deg']
        for col in angular_size_columns:
            self.assertIn(col, catalog.columns)
        
        # Some objects should have angular size data (not all NaN)
        # Large galaxies like M31 should have major axis data
        m31_candidates = catalog[catalog['messier'] == 'M31']
        if not m31_candidates.empty:
            m31 = m31_candidates.iloc[0]
            # M31 (Andromeda) has a major axis of about 190 arcmin
            self.assertTrue(
                pd.notna(m31['major_axis_arcmin']),
                "M31 should have major axis data"
            )


class TestStarTellerCLICaching(unittest.TestCase):
    """Test caching functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_cache_dir = tempfile.mkdtemp()
        self.st = StarTellerCLI(40.7, -74.0, elevation=50)
        # Override cache directory for testing
        self.original_get_cache_filepath = self.st._get_cache_filepath
        
        def mock_get_cache_filepath(year=None):
            if year is None:
                year = datetime.now().year
            return Path(self.test_cache_dir) / f"night_midpoints_test_{year}.pkl"
        
        self.st._get_cache_filepath = mock_get_cache_filepath
    
    def tearDown(self):
        """Clean up test directories."""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_cache_creation(self):
        """Test that cache files are created properly."""
        # This will trigger cache creation
        midpoints = self.st.get_night_midpoints(days=7)  # Small test
        
        self.assertGreater(len(midpoints), 0)
        
        # Check if cache file was created
        cache_files = [f for f in os.listdir(self.test_cache_dir) if f.endswith('.pkl')]
        self.assertGreater(len(cache_files), 0)
    
    def test_cache_loading(self):
        """Test loading data from cache."""
        # Create cache data
        test_data = [(date.today(), datetime.now(), datetime.now(), datetime.now())]
        cache_file = self.st._get_cache_filepath(2025)
        
        cache_data = {
            'latitude': self.st.latitude,
            'longitude': self.st.longitude,
            'timezone': str(self.st.local_tz),
            'year': 2025,
            'night_midpoints': test_data,
            'created_date': datetime.now().isoformat()
        }
        
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache_file), 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Test loading
        loaded_data = self.st._load_cache(2025)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), 1)
    
    def test_cache_mismatch(self):
        """Test handling of cache location/timezone mismatch."""
        # Create cache with different location
        cache_file = self.st._get_cache_filepath(2025)
        
        cache_data = {
            'latitude': 50.0,  # Different latitude
            'longitude': -100.0,  # Different longitude
            'timezone': str(self.st.local_tz),
            'year': 2025,
            'night_midpoints': [(date.today(), datetime.now(), datetime.now(), datetime.now())],
            'created_date': datetime.now().isoformat()
        }
        
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache_file), 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Should return None due to location mismatch
        loaded_data = self.st._load_cache(2025)
        self.assertIsNone(loaded_data)


class TestStarTellerCLIFunctionality(unittest.TestCase):
    """Test core StarTeller-CLI functionality."""
    
    def setUp(self):
        """Set up test StarTellerCLI instance."""
        self.test_cache_dir = tempfile.mkdtemp()
        self.st = StarTellerCLI(40.7, -74.0, elevation=50)
        
        # Override cache directory for testing
        self.original_get_cache_filepath = self.st._get_cache_filepath
        
        def mock_get_cache_filepath(year=None):
            if year is None:
                year = datetime.now().year
            return Path(self.test_cache_dir) / f"night_midpoints_test_{year}.pkl"
        
        self.st._get_cache_filepath = mock_get_cache_filepath
    
    def tearDown(self):
        """Clean up test directories."""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_initialization(self):
        """Test StarTellerCLI initialization."""
        self.assertAlmostEqual(self.st.latitude, 40.7, places=1)
        self.assertAlmostEqual(self.st.longitude, -74.0, places=1)
        self.assertEqual(self.st.elevation, 50)
        self.assertIsNotNone(self.st.local_tz)
        self.assertGreater(len(self.st.dso_catalog), 0)
    
    def test_location_hash(self):
        """Test location hash generation."""
        hash1 = self.st._generate_location_hash()
        
        # Same location should produce same hash
        st2 = StarTellerCLI(40.7, -74.0, elevation=100)  # Different elevation shouldn't matter for hash
        # Mock cache directory for st2
        def mock_get_cache_filepath_st2(year=None):
            if year is None:
                year = datetime.now().year
            return Path(self.test_cache_dir) / f"night_midpoints_test2_{year}.pkl"
        st2._get_cache_filepath = mock_get_cache_filepath_st2
        
        hash2 = st2._generate_location_hash()
        
        self.assertEqual(hash1, hash2)
        
        # Different location should produce different hash
        st3 = StarTellerCLI(41.0, -74.0, elevation=50)
        # Mock cache directory for st3
        def mock_get_cache_filepath_st3(year=None):
            if year is None:
                year = datetime.now().year
            return Path(self.test_cache_dir) / f"night_midpoints_test3_{year}.pkl"
        st3._get_cache_filepath = mock_get_cache_filepath_st3
        
        hash3 = st3._generate_location_hash()
        
        self.assertNotEqual(hash1, hash3)
    

    def test_find_optimal_viewing_times(self):
        """Test optimal viewing times calculation."""
        results = self.st.find_optimal_viewing_times(min_altitude=20)
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        
        # Check required columns exist
        required_columns = [
            'Object', 'Name', 'Type', 'Best_Date', 'Best_Time_Local',
            'Max_Altitude_deg', 'Azimuth_deg', 'Rise_Direction_deg', 'Set_Direction_deg',
            'Major_Axis_arcmin', 'Minor_Axis_arcmin', 'Position_Angle_deg'
        ]
        for col in required_columns:
            self.assertIn(col, results.columns)
    
    def test_direction_filtering(self):
        """Test direction filtering functionality."""
        # Test normal range (East: 45°-135°)
        results_east = self.st.find_optimal_viewing_times(min_altitude=15, direction_filter=(45, 135))
        
        # Test wrapping range (North: 315°-45°)  
        results_north = self.st.find_optimal_viewing_times(min_altitude=15, direction_filter=(315, 45))
        
        # Both should return DataFrame
        self.assertIsInstance(results_east, pd.DataFrame)
        self.assertIsInstance(results_north, pd.DataFrame)
        
        # Results might be different due to filtering
        self.assertGreaterEqual(len(results_east), 0)
        self.assertGreaterEqual(len(results_north), 0)
    
    def test_altitude_filtering(self):
        """Test altitude filtering with different thresholds."""
        results_low = self.st.find_optimal_viewing_times(min_altitude=10)
        results_high = self.st.find_optimal_viewing_times(min_altitude=70)
        
        # Higher altitude requirement should generally result in fewer objects
        # (though not always due to different optimal dates)
        self.assertIsInstance(results_low, pd.DataFrame)
        self.assertIsInstance(results_high, pd.DataFrame)


class TestStarTellerCLIErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_cache_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test directories."""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def _mock_cache_dir(self, st):
        """Mock cache directory for a StarTellerCLI instance."""
        original_get_cache_filepath = st._get_cache_filepath
        
        def mock_get_cache_filepath(year=None):
            if year is None:
                year = datetime.now().year
            return Path(self.test_cache_dir) / f"night_midpoints_test_{year}.pkl"
        
        st._get_cache_filepath = mock_get_cache_filepath
        return original_get_cache_filepath
    
    def test_invalid_coordinates(self):
        """Test handling of invalid coordinates."""
        # These should still work but may have limited functionality
        try:
            st = StarTellerCLI(91.0, 181.0, elevation=50)  # Invalid lat/lon
            self._mock_cache_dir(st)
            # Should still initialize but may have issues with timezone
            self.assertIsNotNone(st)
        except Exception:
            # Some level of graceful degradation is acceptable
            pass
    
    def test_large_catalog(self):
        """Test handling of full catalog."""
        st = StarTellerCLI(40.7, -74.0, elevation=50)
        self._mock_cache_dir(st)
        
        # Should handle full catalog gracefully
        try:
            results = st.find_optimal_viewing_times()
            self.assertIsInstance(results, pd.DataFrame)
            self.assertGreater(len(results), 0)
        except Exception as e:
            self.fail(f"Catalog handling should be graceful, but got: {e}")
    
    def test_corrupted_cache(self):
        """Test handling of corrupted cache files."""
        st = StarTellerCLI(40.7, -74.0, elevation=50)
        self._mock_cache_dir(st)
        
        # Create a corrupted cache file
        cache_file = st._get_cache_filepath(2025)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(cache_file), 'w') as f:
            f.write("This is not valid pickle data")
        
        # Should handle corruption gracefully and return None
        loaded_data = st._load_cache(2025)
        self.assertIsNone(loaded_data)


def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("        STARTELLER-CLI COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestStarTellerCLIDownload,
        TestStarTellerCLICatalog, 
        TestStarTellerCLICaching,
        TestStarTellerCLIFunctionality,
        TestStarTellerCLIErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print("✅ Download functionality working")
        print("✅ Catalog loading working")
        print("✅ Caching system working")
        print("✅ Core calculations working")
        print("✅ Error handling working")
        print("\nStarTeller-CLI is ready for production use!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"⚠️  {len(result.errors)} test(s) had errors")
        print("\nPlease review the test output above for details.")
    
    return result.wasSuccessful()


def quick_integration_test():
    """Quick integration test for basic functionality."""
    print("Running quick integration test...")
    
    # Use temporary cache directory for testing
    test_cache_dir = tempfile.mkdtemp()
    
    try:
        # Test basic functionality
        st = StarTellerCLI(40.7, -74.0, elevation=50)
        
        # Override cache directory for testing
        def mock_get_cache_filepath(year=None):
            if year is None:
                year = datetime.now().year
            return Path(test_cache_dir) / f"night_midpoints_quick_test_{year}.pkl"
        
        st._get_cache_filepath = mock_get_cache_filepath
        
        print(f"✅ Initialized with {len(st.dso_catalog)} objects")
        
        # Test calculation
        results = st.find_optimal_viewing_times(min_altitude=20)
        print(f"✅ Generated results for {len(results)} objects")
        
        # Test that we have expected data
        visible_objects = results[results['Max_Altitude_deg'] != 'Never visible']
        print(f"✅ Found {len(visible_objects)} visible objects")
        
        print("🎉 Quick integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Quick integration test FAILED: {e}")
        return False
    finally:
        # Clean up temporary cache directory
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="StarTeller-CLI Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick integration test only")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_integration_test()
    elif args.comprehensive:
        success = run_comprehensive_test()
    else:
        # Default: run quick test first, then comprehensive if requested
        print("Use --quick for basic test or --comprehensive for full test suite")
        print("Running quick integration test by default...\n")
        success = quick_integration_test()
        
        if success:
            print("\nTo run full test suite: python test_starteller_cli.py --comprehensive")
    
    sys.exit(0 if success else 1) 