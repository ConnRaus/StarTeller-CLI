#!/usr/bin/env python3
"""
Regenerate tests/fixtures/<lat>_<lon>_<min_alt>.csv files.

Must match TestStarTellerFixtureCSVs: pinned “today” and the same patches as
test_starteller_cli._run_starteller_csv.

Run from repo root:
  StarTeller-CLI/.venv/bin/python tests/regenerate_fixtures.py
"""

from __future__ import annotations

import datetime as dt_std
import os
import sys
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from starteller_cli import StarTellerCLI  # noqa: E402

# Keep in sync with tests/test_starteller_cli.py
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_PINNED_TODAY = date(2026, 4, 11)
_PINNED_NOW = datetime(2026, 4, 11, 12, 0, 0)

_CASES: list[tuple[float, float, float, str]] = [
    (40.0, -74.0, 20.0, "40_-74_20.csv"),
    (40.0, -74.0, 0.0, "40_-74_0.csv"),
    (0.0, 0.0, 20.0, "0_0_20.csv"),
    (0.0, 0.0, 0.0, "0_0_0.csv"),
    (-34.0, 151.0, 20.0, "-34_151_20.csv"),
    (-34.0, 151.0, 0.0, "-34_151_0.csv"),
    (80.0, -40.0, 20.0, "80_-40_20.csv"),
    (80.0, -40.0, 0.0, "80_-40_0.csv"),
]


def _run_pinned(latitude: float, longitude: float, min_altitude: float):
    with patch("starteller.date") as mock_date, patch("starteller.datetime") as mock_dt:
        mock_date.today.return_value = _PINNED_TODAY
        mock_date.side_effect = lambda *a, **k: date(*a, **k)
        mock_dt.now.return_value = _PINNED_NOW
        mock_dt.side_effect = lambda *a, **k: dt_std.datetime(*a, **k)
        mock_dt.fromtimestamp = dt_std.datetime.fromtimestamp
        mock_dt.combine = dt_std.datetime.combine
        st = StarTellerCLI(latitude, longitude)
        return st.find_optimal_viewing_times(
            min_altitude=min_altitude,
            messier_only=False,
            use_tqdm=False,
        )


def main() -> None:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for lat, lon, min_alt, name in _CASES:
        actual = _run_pinned(lat, lon, min_alt)
        buf = StringIO()
        actual.to_csv(buf, index=False, lineterminator="\n")
        out = _FIXTURES_DIR / name
        out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Wrote {out} ({len(actual)} rows)")


if __name__ == "__main__":
    main()
