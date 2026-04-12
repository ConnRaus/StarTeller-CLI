# StarTeller-CLI

A comprehensive command-line tool for astrophotographers and telescope enthusiasts to find optimal viewing times for deep sky objects throughout the year.

Given your location, StarTeller scans about one year of local nights. For each object it picks the best night: the one with the longest stretch of time where the object is above your minimum altitude during astronomical darkness (Sun more than 18° below the horizon). If two nights tie on that duration, the night with the higher peak altitude within the visible segment wins. Coordinates are precessed from J2000 to an epoch near the middle of the night list before the scan.

## Installation

### Install from PyPI (Recommended)

```bash
pip install starteller-cli
starteller
```

That's it! The `starteller` command will be available in your terminal.

### Install from Source (Development)

If you want to modify the code or install the latest development version:

```bash
git clone https://github.com/ConnRaus/StarTeller-CLI.git
cd StarTeller-CLI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install .
starteller
```

Or run directly without installing:

```bash
git clone https://github.com/ConnRaus/StarTeller-CLI.git
cd StarTeller-CLI
pip install -r requirements.txt
python src/starteller_cli.py
```

## How it works

1. Enter your coordinates (or use a saved location)
2. Set your output directory (or use a saved preference)
3. Set minimum altitude
4. Get a CSV with optimal viewing times for ~13,000 deep sky objects

The first run downloads the NGC catalog and addendum (~4MB). Night darkness times are computed each run from your location and a rolling ~365-day window from today’s date.

## Output

Results go to `./starteller_output/` by default, or a custom directory you set on first run. CSV columns are written in this order:

| Column                   | Description                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Object                   | NGC/IC/Messier ID                                                                                          |
| Name                     | Common name if available                                                                                   |
| Type                     | Galaxy, Nebula, Cluster, etc.                                                                              |
| Messier                  | Messier number if applicable (e.g., M31)                                                                   |
| Constellation            | IAU constellation (from catalog)                                                                           |
| Right_Ascension          | Right ascension in degrees (J2000)                                                                         |
| Declination              | Declination in degrees (J2000)                                                                             |
| Major_Axis_arcmin        | Major axis angular size in arcminutes                                                                      |
| Minor_Axis_arcmin        | Minor axis angular size in arcminutes                                                                      |
| Position_Angle_deg       | Position angle of major axis (N through E)                                                                 |
| V_Mag                    | Visual magnitude (catalog)                                                                                 |
| SurfBr                   | Surface brightness (catalog)                                                                               |
| Best_Date                | Local calendar date of the **best night** (longest qualifying dark-time span, not “at midnight”)           |
| Best_Time_Local          | Local time of the **peak** used for that night (within the visible segment, toward transit when possible)  |
| Max_Altitude_deg         | Approximate altitude at that peak on the best night, or `Never visible` if no night qualifies              |
| Azimuth_deg              | Azimuth at peak (0°=N, 90°=E, etc.)                                                                        |
| Rise_Time_Local          | Local time the object crosses **up** through your minimum altitude on that best night’s segment (HH:MM)    |
| Rise_Direction_deg       | Azimuth at rise                                                                                            |
| Set_Time_Local           | Local time it crosses **down** through minimum altitude on that segment (HH:MM)                            |
| Set_Direction_deg        | Azimuth at set                                                                                             |
| Observing_Duration_Hours | Hours above your minimum altitude **during astro dark on that best night only** (not summed over the year) |
| Visible_Nights_Per_Year  | Count of nights in the scan with **any** time above min altitude during astro dark                         |
| Dark_Start_Local         | Start of astronomical darkness on the best night (HH:MM local)                                             |
| Dark_End_Local           | End of astronomical darkness on the best night (HH:MM local)                                               |
| Timezone                 | IANA timezone name used for local times                                                                    |

## Options

**Filters and flags:**

- **Minimum altitude** (prompt, default 20°): the object must reach this altitude during astro dark for a night to count; the best night maximizes the length of that interval.
- **`starteller --messier-only`**: only Messier catalog entries are processed (smaller CSV).

**Included catalogs:**

The output includes all ~13,000 objects from NGC, IC, Messier, Caldwell, and other catalogs from OpenNGC (subject to the Messier-only flag above). The full catalog can be found on my fork of OpenNGC [here](https://github.com/ConnRaus/Modified_OpenNGC).

## File locations

Data and settings are stored in platform-specific directories:

**Windows:** `%LOCALAPPDATA%\StarTeller-CLI\`  
**Linux:** `~/.local/share/starteller-cli/`  
**macOS:** `~/Library/Application Support/StarTeller-CLI/`

This includes the NGC catalog, your saved location, and output directory preference. Output CSVs go to your configured output directory (default: `./starteller_output/`).

## Requirements

- Python 3.10+
- Internet connection (first run only, to download catalog)

Dependencies: pandas, numpy, tzdata (IANA DB for `zoneinfo`), timezonefinder, tqdm

## Data source

Catalog data comes from [OpenNGC](https://github.com/mattiaverga/OpenNGC) by Mattia Verga, licensed under CC-BY-SA-4.0.

## License

AGPL-3.0-or-later (GNU Affero General Public License v3). See [LICENSE](LICENSE).

The NGC catalog data is CC-BY-SA-4.0.
