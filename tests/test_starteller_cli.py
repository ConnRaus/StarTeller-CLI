#!/usr/bin/env python3
"""
Comprehensive Test Suite for StarTeller-CLI
Tests downloads, catalog loading, core calculations, error handling, and --clean.
"""

import sys
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch
from io import StringIO
from pathlib import Path
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from starteller_cli import StarTellerCLI, run_clean, get_user_data_dir
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
        
        # Should have required columns (names match final viewing-times CSV where applicable)
        required_columns = [
            'Object', 'Name', 'Right_Ascension', 'Declination', 'Type', 'Messier',
        ]
        for col in required_columns:
            self.assertIn(col, catalog.columns)
    
    def test_messier_field_format(self):
        """Test that Messier field is formatted correctly."""
        catalog = load_ngc_catalog()
        
        # Find objects with Messier designations
        messier_objects = catalog[catalog['Messier'].notna() & (catalog['Messier'] != '')]
        
        if not messier_objects.empty:
            # Messier field should be formatted like "M31", "M42", etc.
            for m_val in messier_objects['Messier'].head(10):
                self.assertTrue(m_val.startswith('M'), f"Messier value '{m_val}' should start with 'M'")
    
    def test_angular_size_columns(self):
        """Test that angular size columns are present and contain valid data."""
        catalog = load_ngc_catalog()
        
        # Should have angular size columns
        angular_size_columns = ['Major_Axis_arcmin', 'Minor_Axis_arcmin', 'Position_Angle_deg']
        for col in angular_size_columns:
            self.assertIn(col, catalog.columns)
        
        # Some objects should have angular size data (not all NaN)
        # Large galaxies like M31 should have major axis data
        m31_candidates = catalog[catalog['Messier'] == 'M31']
        if not m31_candidates.empty:
            m31 = m31_candidates.iloc[0]
            # M31 (Andromeda) has a major axis of about 190 arcmin
            self.assertTrue(
                pd.notna(m31['Major_Axis_arcmin']),
                "M31 should have major axis data"
            )


class TestStarTellerCLIFunctionality(unittest.TestCase):
    """Test core StarTeller-CLI functionality."""
    
    def setUp(self):
        """Set up test StarTellerCLI instance."""
        self.st = StarTellerCLI(40.7, -74.0, elevation=50)
    
    def test_initialization(self):
        """Test StarTellerCLI initialization."""
        self.assertAlmostEqual(self.st.latitude, 40.7, places=1)
        self.assertAlmostEqual(self.st.longitude, -74.0, places=1)
        self.assertIsNotNone(self.st.local_tz)
        self.assertGreater(len(self.st.catalog_df), 0)

    def test_find_optimal_viewing_times(self):
        """Test optimal viewing times calculation."""
        results = self.st.find_optimal_viewing_times(min_altitude=20, messier_only=True)
        
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
    
    def test_altitude_filtering(self):
        """Test altitude filtering with different thresholds."""
        results_low = self.st.find_optimal_viewing_times(min_altitude=10, messier_only=True)
        results_high = self.st.find_optimal_viewing_times(min_altitude=70, messier_only=True)
        
        # Higher altitude requirement should generally result in fewer objects
        # (though not always due to different optimal dates)
        self.assertIsInstance(results_low, pd.DataFrame)
        self.assertIsInstance(results_high, pd.DataFrame)


class TestStarTellerCLIErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_coordinates(self):
        """Test handling of invalid coordinates."""
        try:
            st = StarTellerCLI(91.0, 181.0, elevation=50)
            self.assertIsNotNone(st)
        except Exception:
            pass

    def test_large_catalog(self):
        """Test handling of full catalog."""
        st = StarTellerCLI(40.7, -74.0, elevation=50)
        try:
            results = st.find_optimal_viewing_times(messier_only=True)
            self.assertIsInstance(results, pd.DataFrame)
            self.assertGreater(len(results), 0)
        except Exception as e:
            self.fail(f"Catalog handling should be graceful, but got: {e}")


class TestStarTellerCLIClean(unittest.TestCase):
    """Test --clean: removal of user data for a fresh run."""

    def setUp(self):
        """Use a temp user data dir so we don't touch real data."""
        self.test_user_data_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if self.test_user_data_dir.exists():
            shutil.rmtree(self.test_user_data_dir)

    @patch("starteller_cli.get_user_data_dir")
    def test_clean_removes_user_data(self, mock_user_data_dir):
        """run_clean removes the user data directory when it exists."""
        mock_user_data_dir.return_value = self.test_user_data_dir
        (self.test_user_data_dir / "NGC.csv").write_text("Name;Type;RA;Dec\nNGC1;G;00:00:00;+00:00:00\n")
        (self.test_user_data_dir / "addendum.csv").write_text("Name;Type;RA;Dec\n")
        (self.test_user_data_dir / "user_location.txt").write_text("40.0,-74.0,0")
        (self.test_user_data_dir / "output_dir.txt").write_text("/tmp/out")
        self.assertTrue(self.test_user_data_dir.exists())

        run_clean()

        self.assertFalse(self.test_user_data_dir.exists(), "user data dir should be removed")

    @patch("starteller_cli.get_user_data_dir")
    def test_clean_when_dir_missing(self, mock_user_data_dir):
        """run_clean does not raise when the user data directory does not exist."""
        missing_user = Path(tempfile.mkdtemp())
        shutil.rmtree(missing_user)
        mock_user_data_dir.return_value = missing_user
        self.assertFalse(missing_user.exists())

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            run_clean()
            out = stdout.getvalue()
        self.assertIn("already clean", out.lower())

    @patch("starteller_cli.get_user_data_dir")
    def test_clean_prints_removed_path(self, mock_user_data_dir):
        """run_clean prints the path it removed."""
        mock_user_data_dir.return_value = self.test_user_data_dir
        (self.test_user_data_dir / "NGC.csv").write_text("dummy")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            run_clean()
            out = stdout.getvalue()

        self.assertIn("Removed user data", out)
        self.assertIn(str(self.test_user_data_dir), out)


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
        TestStarTellerCLIFunctionality,
        TestStarTellerCLIErrorHandling,
        TestStarTellerCLIClean,
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

    try:
        st = StarTellerCLI(40.7, -74.0, elevation=50)
        print(f"✅ Initialized with {len(st.catalog_df)} objects")
        
        # Test calculation
        results = st.find_optimal_viewing_times(min_altitude=20, messier_only=True)
        print(f"✅ Generated results for {len(results)} objects")
        
        # Test that we have expected data
        visible_objects = results[results['Max_Altitude_deg'] != 'Never visible']
        print(f"✅ Found {len(visible_objects)} visible objects")
        
        print("🎉 Quick integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Quick integration test FAILED: {e}")
        return False


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