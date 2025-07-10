# StarTeller

An astrophotography planning tool that finds optimal viewing times for deep sky objects throughout the year.

## Quick Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the NGC catalog:**

   - Go to https://github.com/mattiaverga/OpenNGC/blob/master/database_files/NGC.csv
   - Download the file and save it as `NGC.csv` in the StarTeller directory

3. **Start StarTeller:**
   ```bash
   python star_teller.py
   ```

## Catalog

StarTeller uses the comprehensive OpenNGC catalog which includes both NGC and IC objects (thousands of objects with proper common names like "Andromeda Galaxy" and "Crab Nebula").

The OpenNGC catalog is specifically maintained for amateur astronomers and includes all the objects and names that stargazers actually use.

## Usage

StarTeller calculates when deep sky objects are:

- Above the horizon
- In dark sky conditions (after astronomical twilight)
- At optimal viewing altitudes
- Clear of obstructions in specified directions

Perfect for planning astrophotography sessions!
