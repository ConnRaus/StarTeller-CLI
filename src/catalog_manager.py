#!/usr/bin/env python3
"""
Simple NGC Catalog Loader for StarTeller-CLI
Handles automatic download and loading of OpenNGC catalog data.
"""

import pandas as pd
import numpy as np
import os
import urllib.request
import urllib.error

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
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(ngc_path), exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, ngc_path)
        
        # Verify the file was downloaded and has content
        if os.path.exists(ngc_path) and os.path.getsize(ngc_path) > 1000:  # At least 1KB
            print(f"✅ Successfully downloaded NGC.csv ({os.path.getsize(ngc_path)/1024:.0f} KB)")
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

def load_ngc_catalog(catalog_filter="all"):
    """
    Load NGC/IC catalog from local OpenNGC file.
    
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
        # Check for the OpenNGC catalog file (in data/ folder)
        ngc_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'NGC.csv')
        
        # If file doesn't exist, try to download it automatically
        if not os.path.exists(ngc_path):
            if not download_ngc_catalog(ngc_path):
                # Download failed - provide manual instructions
                print("\n" + "=" * 60)
                print("MANUAL DOWNLOAD REQUIRED")
                print("=" * 60)
                print("❌ Automatic download failed. Please download manually:")
                print("   1. Go to: https://github.com/mattiaverga/OpenNGC/blob/master/database_files/NGC.csv")
                print("   2. Click 'Raw' button to download the file")
                print("   3. Save it as 'NGC.csv' in the data/ folder")
                print(f"   4. Full path should be: {ngc_path}")
                print("=" * 60)
                return pd.DataFrame()
        
        # Load and filter catalog quietly
        df = pd.read_csv(ngc_path, sep=';', low_memory=False)
        
        # Filter for NGC and IC objects with coordinates
        df = df[df['Name'].str.match(r'^(NGC|IC)\d+$', na=False)]
        df = df.dropna(subset=['RA', 'Dec'])
        
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
        
        # Expand object types (based on actual NGC.csv data)
        type_expansions = {
            'G': 'Galaxy',
            'SNR': 'Supernova remnant',
            'GCl': 'Globular cluster',
            'OCl': 'Open cluster',
            'Neb': 'Nebula',
            'HII': 'HII region',
            'PN': 'Planetary nebula',
            'RfN': 'Reflection nebula',
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
            """Convert NGC0221 to NGC 221 (remove leading zeros)"""
            import re
            match = re.match(r'^(NGC|IC)(\d+)$', name)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))  # Convert to int to remove leading zeros
                return f"{prefix} {number}"
            return name
        
        catalog_df = pd.DataFrame({
            'object_id': df['Name'],
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
        
        # Sort by catalog and number
        def sort_key(name):
            import re
            # Handle Messier objects first
            messier_match = re.match(r'M(\d+)', name)
            if messier_match:
                return (0, int(messier_match.group(1)))  # Messier objects first
            
            # Handle NGC/IC objects
            ngc_match = re.match(r'(NGC|IC)(\d+)', name)
            if ngc_match:
                prefix = ngc_match.group(1)
                number = int(ngc_match.group(2))
                return (1 if prefix == 'NGC' else 2, number)
            
            return (3, 99999)  # Everything else last
        
        catalog_df['sort_key'] = catalog_df['object_id'].apply(sort_key)
        catalog_df = catalog_df.sort_values('sort_key').drop('sort_key', axis=1)
        
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