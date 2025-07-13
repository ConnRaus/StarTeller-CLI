# StarTeller-CLI

A comprehensive command-line tool for astrophotographers and telescope enthusiasts to find optimal viewing times for deep sky objects throughout the year.

StarTeller-CLI calculates the absolute best viewing times for celestial objects, taking into account altitude, azimuth, dark sky conditions, and seasonal visibility to help you plan perfect observation sessions.

## Features

- **Automatic Data Management**: Downloads and caches astronomical catalogs automatically
- **Comprehensive Catalogs**: Access to 13,000+ deep sky objects (NGC, IC, Messier)
- **Optimal Timing**: Finds the best viewing times considering dark sky conditions and object altitude
- **Location Intelligence**: Automatic location detection or manual coordinate input
- **Direction Filtering**: Filter objects by viewing direction (North, South, East, West)
- **Performance Optimized**: Multiprocessing support and intelligent caching
- **Export Ready**: Detailed CSV output with all viewing information
- **Custom Object Lists**: Target specific objects of interest

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/StarTeller-CLI.git
cd StarTeller-CLI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run StarTeller-CLI

```bash
python src/star_teller.py
```

The program will:

- Automatically download the NGC catalog if needed
- Detect your location (or prompt for manual input)
- Guide you through catalog and viewing preferences
- Generate a comprehensive CSV with optimal viewing times

### 3. Output

Results are saved as CSV files in the `output/` directory with columns including:

- **Object**: NGC/IC/Messier designation
- **Name**: Common name (e.g., "Andromeda Galaxy")
- **Type**: Object type (Galaxy, Nebula, etc.)
- **Best_Date**: Optimal viewing date
- **Best_Time_Local**: Best viewing time in your timezone
- **Max_Altitude_deg**: Maximum altitude reached
- **Azimuth_deg**: Viewing direction
- **Direction**: Cardinal direction (N, NE, E, etc.)

## Usage Options

### Catalog Selection

- **Messier Objects** (~110 famous deep sky objects)
- **NGC Objects** (~8,000 New General Catalog objects)
- **IC Objects** (~5,000 Index Catalog objects)
- **All Objects** (~13,000 combined NGC + IC objects)
- **Custom Objects** (specify your own list)

### Viewing Preferences

- **Minimum Altitude**: Set minimum viewing angle (default: 20°)
- **Direction Filter**: Target specific sky regions (e.g., East-South)
- **Location**: Automatic detection or manual coordinates

### Example Custom Objects

```
M31,NGC224,IC342,M42,NGC2024
```

## Advanced Usage

### Location Management

The program automatically detects and saves your location. To manually set coordinates:

```python
from src.star_teller import StarTeller

# Create instance with specific coordinates
st = StarTeller(latitude=40.7589, longitude=-73.9851, elevation=50)
results = st.find_optimal_viewing_times(min_altitude=30)
```

### Direction Filtering

Filter objects by viewing direction using azimuth ranges:

```python
# North: 315° to 45°
results = st.find_optimal_viewing_times(direction_filter=(315, 45))

# East to South: 90° to 180°
results = st.find_optimal_viewing_times(direction_filter=(90, 180))
```

### Cache Management

StarTeller-CLI uses intelligent caching for performance:

- Astronomical calculations are cached by location and year
- Catalog data is cached locally after first download
- Cache automatically invalidates when location changes

## Requirements

- Python 3.8+
- Internet connection (for initial catalog download)
- ~50MB disk space for catalog data and cache

### Dependencies

- **skyfield**: Astronomical calculations
- **pandas**: Data processing and CSV output
- **numpy**: Numerical computations
- **pytz**: Timezone handling
- **timezonefinder**: Automatic timezone detection
- **tqdm**: Progress bars

## Data Sources and Credits

### OpenNGC Catalog

This project uses the comprehensive OpenNGC catalog maintained by **Mattia Verga**:

- **Source**: https://github.com/mattiaverga/OpenNGC
- **License**: CC-BY-SA-4.0
- **Usage**: The OpenNGC catalog is used under the Creative Commons Attribution-ShareAlike 4.0 International License

The OpenNGC catalog provides:

- 13,000+ deep sky objects from NGC and IC catalogs
- Accurate coordinates and physical data
- Common names familiar to amateur astronomers
- Maintained specifically for the astronomy community

### Additional Credits

- **Skyfield**: Astronomical calculations by Brandon Rhodes
- **Pandas**: Data manipulation library
- **NASA JPL**: Ephemeris data for astronomical calculations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The NGC catalog data is licensed under CC-BY-SA-4.0 by the OpenNGC project.

## Performance Notes

- **Initial Run**: May take 2-3 minutes while downloading and processing catalogs
- **Subsequent Runs**: Typically complete in 10-30 seconds thanks to caching
- **Memory Usage**: ~100-200MB during calculation
- **Multiprocessing**: Automatically uses all available CPU cores for calculations

## Troubleshooting

### Common Issues

1. **Network Error**: If catalog download fails, check internet connection
2. **Location Detection**: If automatic location fails, you'll be prompted for manual input
3. **Cache Issues**: Delete cache files in `user_data/` to force regeneration
4. **Memory Issues**: Reduce catalog size by using filters (Messier vs All)

### Testing

Run the comprehensive test suite:

```bash
python tests/test_star_teller.py --comprehensive
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- New features
- Bug fixes
- Performance improvements
- Documentation updates

## Acknowledgments

Special thanks to the OpenNGC project for providing the comprehensive catalog data that makes StarTeller-CLI possible. The OpenNGC project represents years of effort to create a clean, accurate, and amateur-friendly deep sky object database.

---

**Perfect for**: Astrophotographers planning imaging sessions, visual observers with telescopes, astronomy enthusiasts learning the night sky, and anyone wanting to optimize their stargazing experience.
