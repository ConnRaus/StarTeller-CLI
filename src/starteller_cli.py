#!/usr/bin/env python3
"""
StarTeller-CLI — interactive prompts, paths, and status messages.

Core algorithms live in starteller.py.
"""

import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pytz

try:
    from .starteller import StarTellerCLI
except ImportError:
    from starteller import StarTellerCLI

__all__ = [
    "StarTellerCLI",
    "get_user_data_dir",
    "get_output_dir",
    "load_output_dir",
    "save_output_dir",
    "load_location",
    "save_location",
    "get_user_output_dir",
    "get_user_location",
    "run_clean",
    "main",
    "find_optimal_viewing_times_with_messages",
]


def get_user_data_dir():
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return Path(base) / 'StarTeller-CLI'
    elif sys.platform == 'darwin':
        return Path.home() / 'Library/Application Support/StarTeller-CLI'
    else:
        return Path.home() / '.local/share/starteller-cli'


def load_output_dir():
    try:
        output_file = get_user_data_dir() / 'output_dir.txt'
        if not output_file.exists():
            return None
        with open(str(output_file), 'r') as f:
            path = f.read().strip()
            if path:
                return path
    except Exception:
        pass
    return None


def save_output_dir(output_path):
    try:
        user_data_dir = get_user_data_dir()
        user_data_dir.mkdir(parents=True, exist_ok=True)
        output_file = user_data_dir / 'output_dir.txt'
        with open(str(output_file), 'w') as f:
            f.write(str(output_path))
        print(f"✓ Output directory saved: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save output directory: {e}")


def get_output_dir():
    saved_dir = load_output_dir()
    if saved_dir:
        return Path(saved_dir)
    return Path.cwd() / 'starteller_output'


def load_location():
    try:
        location_file = get_user_data_dir() / 'user_location.txt'
        if not location_file.exists():
            return None
        with open(str(location_file), 'r') as f:
            data = f.read().strip().split(',')
            if len(data) >= 2:
                return float(data[0]), float(data[1])
    except Exception:
        pass
    return None


def save_location(latitude, longitude):
    try:
        user_data_dir = get_user_data_dir()
        user_data_dir.mkdir(parents=True, exist_ok=True)
        location_file = user_data_dir / 'user_location.txt'
        with open(str(location_file), 'w') as f:
            f.write(f"{latitude},{longitude}")
        print(f"✓ Location saved: {latitude:.2f}°, {longitude:.2f}°")
    except Exception as e:
        print(f"Warning: Could not save location: {e}")


def get_user_output_dir():
    saved_dir = load_output_dir()
    if saved_dir:
        print(f"\nSaved output directory: {saved_dir}")
        use_saved = input("Use saved output directory? (y/n, default y): ").strip().lower()
        if use_saved in ['', 'y', 'yes']:
            return Path(saved_dir)
    print("\nOutput directory setup:")
    print("  Enter a path for output files, or press Enter for default (./starteller_output)")
    output_input = input("Output directory: ").strip()
    if output_input:
        output_path = Path(output_input).expanduser().resolve()
    else:
        output_path = Path.cwd() / 'starteller_output'
        print(f"  Using default: {output_path}")
    save_choice = input("Save this output directory for future use? (y/n, default y): ").strip().lower()
    if save_choice in ['', 'y', 'yes']:
        save_output_dir(output_path)
    return output_path


def get_user_location():
    saved_location = load_location()
    if saved_location:
        latitude, longitude = saved_location
        print(f"\nSaved location found: {latitude:.2f}°, {longitude:.2f}°")
        use_saved = input("Use saved location? (y/n, default y): ").strip().lower()
        if use_saved in ['', 'y', 'yes']:
            return latitude, longitude
    print("\nEnter your observing location:")
    latitude = float(input("Latitude (degrees, positive for North): "))
    longitude = float(input("Longitude (degrees, positive for East): "))
    save_choice = input("Save this location for future use? (y/n, default y): ").strip().lower()
    if save_choice in ['', 'y', 'yes']:
        save_location(latitude, longitude)
    return latitude, longitude


def years_spanned(start_date, days):
    return {(start_date + timedelta(days=i)).year for i in range(days)}


def find_optimal_viewing_times_with_messages(st, min_altitude=20, messier_only=False):
    """Print status lines, then run the core pipeline once (nights computed a single time)."""
    print("Calculating optimal viewing times for deep sky objects...")
    print(f"Observer location: {st.latitude:.2f}°, {st.longitude:.2f}°")
    print(f"Local timezone: {st.local_tz}")
    print(f"Minimum altitude: {min_altitude}°")
    print("Dark time criteria: Sun below -18° (astronomical twilight)")
    print("Best night: longest time above your altitude limit during that dark period (tie → higher peak altitude)")
    print("Visible nights: count of nights with any usable time in astro dark (not “only at dark midpoint”)")

    df_work = st.catalog_df
    if messier_only:
        m = df_work["Messier"].fillna("").astype(str).str.strip()
        print(f"Processing {len(df_work[m != ''])} objects (Messier only)")
    else:
        print(f"Processing {len(df_work)} objects")

    t_total_start = time.perf_counter()
    start_date = date.today()
    days = 365
    print(f"Calculating night darkness times for {len(years_spanned(start_date, days))} year(s)...")
    t_night = time.perf_counter()
    dark_windows = st.get_dark_windows(start_date=start_date, days=days)
    print(f"✓ Night calculations completed in {time.perf_counter() - t_night:.2f}s")

    if dark_windows:
        _, ds, de = dark_windows[len(dark_windows) // 2]
        mid_ts = (ds.timestamp() + de.timestamp()) * 0.5
        epoch_date_str = datetime.fromtimestamp(mid_ts, tz=st.local_tz).strftime('%Y-%m-%d')
    else:
        epoch_date_str = f"{date.today().year}-07-01"
    print(f"Precessing coordinates from J2000.0 to epoch {epoch_date_str} (accounting for Earth's precession)...")

    results = st.find_optimal_viewing_times(
        min_altitude=min_altitude,
        messier_only=messier_only,
        use_tqdm=True,
        dark_windows=dark_windows,
    )
    print(f"✓ Processing completed in {time.perf_counter() - t_total_start:.2f}s")
    return results


def run_clean():
    user_data_dir = get_user_data_dir()
    if user_data_dir.exists():
        shutil.rmtree(user_data_dir)
        print("StarTeller-CLI --clean: Removed user data:")
        print(f"  ✓ {user_data_dir}")
        print("Next run will prompt for location and re-download NGC.csv and addendum.csv as needed.")
    else:
        print("StarTeller-CLI --clean: No user data directory found (already clean).")


def main():
    messier_only = "--messier-only" in sys.argv

    print("=" * 60)
    print("                   StarTeller-CLI")
    print("        Deep Sky Object Optimal Viewing Calculator")
    print("=" * 60)

    latitude, longitude = get_user_location()
    output_dir = get_user_output_dir()

    print("\nViewing preferences:")
    min_alt = float(input("Minimum altitude (degrees, default 20): ") or 20)

    print("\n" + "=" * 60)
    print("PROCESSING...")
    print("=" * 60)

    st = StarTellerCLI(latitude, longitude)
    if st.timezone_name:
        print(f"✓ Timezone: {st.timezone_name}")
    else:
        print("✓ Timezone: UTC (could not auto-detect)")
    if st.catalog_df.empty:
        print("Failed to load NGC catalog - please ensure NGC.csv file is present")
        return
    print(f"✓ Catalog: {len(st.catalog_df)} objects loaded")

    results = find_optimal_viewing_times_with_messages(
        st, min_altitude=min_alt, messier_only=messier_only
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"optimal_viewing_times_{datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M')}.csv"
    results.to_csv(str(filename), index=False)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"✓ Results saved to: {filename}")
    print(f"✓ Found optimal viewing times for {len(results)} objects")
    visible_count = len(results[results['Max_Altitude_deg'] != 'Never visible'])
    print(f"✓ {visible_count} objects visible above {min_alt}°")
    print("\nOpen the CSV file to see complete viewing schedule!")
    print("=" * 60)


if __name__ == "__main__":
    if "--clean" in sys.argv:
        run_clean()
        sys.exit(0)
    main()
