# StarTeller-CLI

A comprehensive command-line tool for astrophotographers and telescope enthusiasts to find optimal viewing times for deep sky objects throughout the year.

Given your location, StarTeller calculates when each object in the NGC/IC/Messier catalogs reaches its highest point during astronomical darkness. It accounts for altitude, direction, and dark sky conditions to help you plan observation sessions.

## Installation

### Install as a package

```bash
git clone https://github.com/ConnRaus/StarTeller-CLI.git
cd StarTeller-CLI
python -m venv venv
venv\Scripts\activate  # On Linux/Mac: source venv/bin/activate
pip install .
starteller
```

### Or run directly

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

The first run downloads the NGC catalog (~2MB) and calculates night darkness times for the year. Both are cached, so subsequent runs are fast.

## Output

Results go to `starteller_output/` in your current directory. The CSV includes:

| Column | Description |
|--------|-------------|
| Object | NGC/IC/Messier ID |
| Name | Common name if available |
| Type | Galaxy, Nebula, Cluster, etc. |
| Best_Date | Date when object is highest at midnight |
| Best_Time_Local | Time of peak altitude |
| Max_Altitude_deg | How high it gets |
| Direction | Cardinal direction (N, NE, E, etc.) |
| Rise_Time_Local | When it rises above your minimum altitude |
| Set_Time_Local | When it drops below |
| Observing_Duration_Hours | Total time above minimum altitude |

## Options

**Catalogs:**
- Messier (~110 objects) - the famous ones
- NGC (~8,000 objects)
- IC (~5,000 objects)
- All (~13,000 objects) - default

**Filters:**
- Minimum altitude (default 20°)
- Direction filter - e.g., `90,180` for objects in the East to South

## Python API

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
