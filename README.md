# StarTeller-CLI

A comprehensive command-line tool for astrophotographers and telescope enthusiasts to find optimal viewing times for deep sky objects throughout the year.

Given your location, StarTeller calculates when each object in the NGC/IC/Messier catalogs reaches its highest point during astronomical darkness. It accounts for altitude, direction, and dark sky conditions to help you plan observation sessions.

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
2. Pick a catalog (Messier, NGC, IC, or all ~13,000 objects)
3. Set minimum altitude and optional direction filter
4. Get a CSV with optimal viewing times for each object

The first run downloads the NGC catalog and Addendum (~4MB) and calculates night darkness times for the year. Both are cached, so subsequent runs are fast.

## Output

Results go to `starteller_output/` in your current directory. The CSV includes:

| Column                   | Description                                 |
| ------------------------ | ------------------------------------------- |
| Object                   | NGC/IC/Messier ID                           |
| Name                     | Common name if available                    |
| Type                     | Galaxy, Nebula, Cluster, etc.               |
| RA                       | Right Ascension in degrees                  |
| Dec                      | Declination in degrees                      |
| Best_Date                | Date when object is highest at midnight     |
| Best_Time_Local          | Time of peak altitude                       |
| Max_Altitude_deg         | Maximum altitude reached                    |
| Azimuth_deg              | Azimuth angle at peak altitude              |
| Direction                | Cardinal direction (N, NE, E, etc.)         |
| Rise_Time_Local          | When it rises above your minimum altitude   |
| Rise_Direction           | Direction it rises from                     |
| Set_Time_Local           | When it drops below minimum altitude        |
| Set_Direction            | Direction it sets toward                    |
| Observing_Duration_Hours | Total time above minimum altitude           |
| Dark_Nights_Per_Year     | Number of nights with astronomical darkness |
| Good_Viewing_Periods     | Number of good viewing periods              |
| Dark_Start_Local         | Start of astronomical darkness              |
| Dark_End_Local           | End of astronomical darkness                |
| Timezone                 | Timezone used for local times               |

## Options

**Catalogs:**

- Messier (~110 objects)
- NGC (~8,000 objects)
- IC (~5,000 objects)
- All (~13,000 objects)

**Filters:**

- Minimum altitude (default 20°)
- Direction filter - e.g., `90,180` for objects in the East to South

## Python API

You can also use StarTeller programmatically:

```python
from src.starteller_cli import StarTellerCLI

st = StarTellerCLI(
    latitude=40.7,
    longitude=-74.0,
    elevation=10,
    catalog_filter='messier'
)

results = st.find_optimal_viewing_times(min_altitude=25)
results = st.find_optimal_viewing_times(direction_filter=(90, 180))  # East to South
```

## File locations

Data is stored in platform-specific directories:

**Windows:** `%LOCALAPPDATA%\StarTeller-CLI\`  
**Linux:** `~/.local/share/starteller-cli/`  
**macOS:** `~/Library/Application Support/StarTeller-CLI/`

Cache goes to the platform's cache directory. Output CSVs go to `./starteller_output/`.

## Requirements

- Python 3.8+
- Internet connection (first run only, to download catalog)

Dependencies: pandas, numpy, pytz, timezonefinder, tqdm

## Data source

Catalog data comes from [OpenNGC](https://github.com/mattiaverga/OpenNGC) by Mattia Verga, licensed under CC-BY-SA-4.0.

## License

MIT. See [LICENSE](LICENSE).

The NGC catalog data is CC-BY-SA-4.0.
