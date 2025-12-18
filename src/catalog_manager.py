#!/usr/bin/env python3
"""
Simple NGC Catalog Loader for StarTeller-CLI
Handles automatic download and loading of OpenNGC catalog data.
"""

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def get_user_data_dir():
    """Get platform-specific user data directory for StarTeller-CLI."""
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return Path(base) / 'StarTeller-CLI'
    elif sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'StarTeller-CLI'
    else:
        return Path.home() / '.local' / 'share' / 'starteller-cli'

def download_ngc_catalog(ngc_path):
    """
    Automatically download the NGC.csv file from OpenNGC GitHub repository.
    
    Args:
        ngc_path (str): Path where the file should be saved
        
    Returns:
        bool: True if download successful, False otherwise
    """
    url = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/refs/heads/master/database_files/NGC.csv"
    
    try:
        print("📥 NGC.csv not found - downloading from OpenNGC repository...")
        print(f"   Downloading from: {url}")
        
        # Ensure the directory exists
        ngc_file = Path(ngc_path)
        ngc_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, str(ngc_path))
    
        # Verify the file was downloaded and has content
        if ngc_file.exists() and ngc_file.stat().st_size > 1000:  # At least 1KB
            print(f"✅ Successfully downloaded NGC.csv ({ngc_file.stat().st_size/1024:.0f} KB)")
            return True
        else:
            print("❌ Download failed - file is empty or corrupted")
            return False
            
    except urllib.error.URLError as e:
        print(f"❌ Network error downloading NGC.csv: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading NGC.csv: {e}")
        return False

def download_addendum_catalog(addendum_path):
    """
    Automatically download the addendum.csv file from OpenNGC GitHub repository.
    
    Args:
        addendum_path (str): Path where the file should be saved
        
    Returns:
        bool: True if download successful, False otherwise
    """
    url = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/refs/heads/master/database_files/addendum.csv"
    
    try:
        print("📥 addendum.csv not found - downloading from OpenNGC repository...")
        print(f"   Downloading from: {url}")
        
        # Ensure the directory exists
        addendum_file = Path(addendum_path)
        addendum_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, str(addendum_path))
    
        # Verify the file was downloaded and has content
        if addendum_file.exists() and addendum_file.stat().st_size > 100:  # At least 100 bytes (smaller file)
            print(f"✅ Successfully downloaded addendum.csv ({addendum_file.stat().st_size/1024:.1f} KB)")
            return True
        else:
            print("❌ Download failed - file is empty or corrupted")
            return False
            
    except urllib.error.URLError as e:
        print(f"❌ Network error downloading addendum.csv: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading addendum.csv: {e}")
        return False

def load_ngc_catalog(catalog_filter="all"):
    """
    Load NGC/IC catalog from local OpenNGC file, including addendum objects.
    
    The addendum includes additional objects from various catalogs:
    - Caldwell (C), Barnard (B), UGC, PGC, ESO, Harvard (H), Melotte (Mel),
      MWSC, HCG, and Cl objects. When filter is "all", these are included.
    
    Args:
        catalog_filter (str): Filter catalog by type ("messier", "ic", "ngc", "all")
        
    Returns:
        pandas.DataFrame: NGC catalog data or empty DataFrame if file not found
    """
    filter_names = {
        "messier": "Messier",
        "ic": "IC", 
        "ngc": "NGC",
        "all": "All NGC/IC"
    }
    try:
        # Check for the OpenNGC catalog file in user data directory
        user_data_dir = get_user_data_dir()
        user_data_dir.mkdir(parents=True, exist_ok=True)
        ngc_path = user_data_dir / 'NGC.csv'
        
        # If file doesn't exist, try to download it automatically
        if not ngc_path.exists():
            if not download_ngc_catalog(str(ngc_path)):
                # Download failed - provide manual instructions
                print("\n" + "=" * 60)
                print("MANUAL DOWNLOAD REQUIRED")
                print("=" * 60)
                print("❌ Automatic download failed. Please download manually:")
                print("   1. Go to: https://github.com/mattiaverga/OpenNGC/blob/master/database_files/NGC.csv")
                print("   2. Click 'Raw' button to download the file")
                print("   3. Save it as 'NGC.csv' in the user data directory")
                print(f"   4. Full path should be: {ngc_path}")
                print("=" * 60)
                return pd.DataFrame()
        
        # Load and filter catalog quietly
        df = pd.read_csv(str(ngc_path), sep=';', low_memory=False)
        
        # Filter for NGC and IC objects with coordinates
        df = df[df['Name'].str.match(r'^(NGC|IC)\d+$', na=False)]
        df = df.dropna(subset=['RA', 'Dec'])
        
        # Load addendum catalog if available
        addendum_path = user_data_dir / 'addendum.csv'
        if not addendum_path.exists():
            download_addendum_catalog(str(addendum_path))
        
        if addendum_path.exists():
            try:
                addendum_df = pd.read_csv(str(addendum_path), sep=';', low_memory=False)
                # Filter for objects with coordinates
                addendum_df = addendum_df.dropna(subset=['RA', 'Dec'])
                # Concatenate with main catalog
                df = pd.concat([df, addendum_df], ignore_index=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not load addendum.csv: {e}")
                print("   Continuing with NGC/IC catalog only...")
        
        # Apply catalog filter
        if catalog_filter == "messier":
            # Only objects with Messier designations
            df = df[df['M'].notna() & (df['M'] != '')].copy()
        elif catalog_filter == "ic":
            # Only IC objects
            df = df[df['Name'].str.startswith('IC')].copy()
        elif catalog_filter == "ngc":
            # Only NGC objects
            df = df[df['Name'].str.startswith('NGC')].copy()
        # else: All objects (no filter needed)

        
        # Parse coordinates from HMS/DMS to decimal degrees
        def parse_coordinate(coord_str, is_ra=False):
            """Parse HMS/DMS coordinate to decimal degrees."""
            if pd.isna(coord_str) or coord_str == '':
                return np.nan
            try:
                coord_str = str(coord_str).strip()
                negative = coord_str.startswith('-')
                coord_str = coord_str.lstrip('+-')
                
                parts = coord_str.split(':')
                if len(parts) == 3:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    
                    decimal = hours + minutes/60 + seconds/3600
                    
                    if is_ra:
                        decimal *= 15  # Convert hours to degrees for RA
                    if negative:
                        decimal *= -1
                        
                    return decimal
                else:
                    return float(coord_str)
            except:
                return np.nan
        
        df['ra_deg'] = df['RA'].apply(lambda x: parse_coordinate(x, is_ra=True))
        df['dec_deg'] = df['Dec'].apply(lambda x: parse_coordinate(x, is_ra=False))
        
        # Remove entries with failed coordinate conversion
        df = df.dropna(subset=['ra_deg', 'dec_deg'])
        
        # Expand object types (based on actual NGC.csv data and addendum)
        type_expansions = {
            'G': 'Galaxy',
            'SNR': 'Supernova remnant',
            'GCl': 'Globular cluster',
            'GCI': 'Globular cluster',
            'OCl': 'Open cluster',
            'Neb': 'Nebula',
            'HII': 'HII region',
            'PN': 'Planetary nebula',
            'RfN': 'Reflection nebula',
            'DrkN': 'Dark nebula',
            '**': 'Double star',
            '*': 'Star',
            '*Ass': 'Stellar association',
            'GPair': 'Galaxy pair',
            'GGroup': 'Galaxy group',
            'GTrpl': 'Galaxy triplet',
            'EmN': 'Emission nebula',
            'Nova': 'Nova',
            'Dup': 'Duplicate object',
            'Other': 'Other object',
            'Cl+N': 'Cluster with nebula',
            'NonEx': 'Non-existent object'
        }
        
        df['expanded_type'] = df['Type'].map(type_expansions).fillna(df['Type'])
        
        # Use V-Mag if available, otherwise B-Mag
        df['magnitude'] = df['V-Mag'].fillna(df['B-Mag'])
        
        # Create standardized catalog with clean names
        def clean_name(name):
            """Convert NGC0221 to NGC 221 (remove leading zeros), handle other catalog prefixes"""
            import re
            # Handle NGC/IC objects
            match = re.match(r'^(NGC|IC)(\d+)$', name)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))  # Convert to int to remove leading zeros
                return f"{prefix} {number}"
            # Handle Caldwell objects (C009 -> C 9)
            match = re.match(r'^C(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"C {number}"
            # Handle Barnard objects (B033 -> B 33)
            match = re.match(r'^B(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"B {number}"
            # Handle Harvard objects (H05 -> H 5)
            match = re.match(r'^H(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"H {number}"
            # Handle Melotte objects (Mel022 -> Mel 22)
            match = re.match(r'^Mel(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"Mel {number}"
            # Handle UGC objects (UGC04305 -> UGC 4305)
            match = re.match(r'^UGC(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"UGC {number}"
            # Handle PGC objects (PGC000143 -> PGC 143)
            match = re.match(r'^PGC(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"PGC {number}"
            # Handle MWSC objects (MWSC3156 -> MWSC 3156)
            match = re.match(r'^MWSC(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"MWSC {number}"
            # Handle HCG objects (HCG079 -> HCG 79)
            match = re.match(r'^HCG(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"HCG {number}"
            # Handle ESO objects (ESO056-115 -> ESO 056-115)
            match = re.match(r'^ESO(\d+)-(\d+)$', name)
            if match:
                return f"ESO {match.group(1)}-{match.group(2)}"
            # Handle Cl objects (Cl399 -> Cl 399)
            match = re.match(r'^Cl(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"Cl {number}"
            # Handle M objects (M040 -> M 40)
            match = re.match(r'^M(\d+)$', name)
            if match:
                number = int(match.group(1))
                return f"M {number}"
            return name
        
        # Fix M040 -> M40 (only case in addendum that needs fixing)
        def normalize_messier_id(name):
            """Fix M040 to M40"""
            if name == 'M040':
                return 'M40'
            return name
        
        catalog_df = pd.DataFrame({
            'object_id': df['Name'].apply(normalize_messier_id),
            'name': df['Name'].apply(clean_name),
            'ra_deg': df['ra_deg'],
            'dec_deg': df['dec_deg'],
            'type': df['expanded_type'],
            'magnitude': df['magnitude'],
            'common_name': df['Common names'].fillna(''),
            'messier': df['M'].fillna('')
        })
        
        # Add Messier cross-references
        messier_objects = []
        for _, row in catalog_df.iterrows():
            if row['messier'] and str(row['messier']).strip():
                # Convert to int to remove decimals and leading zeros (033.0 -> 33)
                try:
                    messier_num = int(float(str(row['messier']).strip()))
                    messier_id = f"M{messier_num}"
                    messier_name = row['common_name'] if row['common_name'] else row['name']
                except (ValueError, TypeError):
                    continue  # Skip invalid Messier numbers
                messier_objects.append({
                    'object_id': messier_id,
                    'name': messier_name,
                    'ra_deg': row['ra_deg'],
                    'dec_deg': row['dec_deg'],
                    'type': row['type'],
                    'magnitude': row['magnitude'],
                    'common_name': row['common_name'],
                    'messier': row['messier']
                })
        
        # Add Messier entries to catalog
        if messier_objects:
            messier_df = pd.DataFrame(messier_objects)
            catalog_df = pd.concat([catalog_df, messier_df], ignore_index=True)
        
        # Filter to reasonable coordinate ranges
        catalog_df = catalog_df[
            (catalog_df['ra_deg'] >= 0) & (catalog_df['ra_deg'] <= 360) &
            (catalog_df['dec_deg'] >= -90) & (catalog_df['dec_deg'] <= 90)
        ]
        
        return catalog_df
        
    except Exception as e:
        print(f"✗ Error loading OpenNGC catalog: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    catalog = load_ngc_catalog()
    if not catalog.empty:
        print(f"\nSuccessfully loaded {len(catalog)} objects from OpenNGC!")
        print("This catalog is ready for StarTeller-CLI.")
    else:
        print("\nFailed to load catalog") 