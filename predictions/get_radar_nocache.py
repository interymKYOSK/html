#!/usr/bin/env python
"""
Wrapper script to disable wetterdienst caching before any imports.
This avoids the SQLite dependency issue by mocking diskcache.
"""
import os
import sys


# Mock sqlite3 and diskcache BEFORE any imports
class MockSQLite:
    """Mock sqlite3 to prevent import errors"""

    pass


class MockCache:
    """Mock diskcache.Cache to avoid SQLite dependency"""

    def __init__(self, *args, **kwargs):
        self._cache = {}

    def get(self, key, default=None):
        return self._cache.get(key, default)

    def set(self, key, value, **kwargs):
        self._cache[key] = value

    def delete(self, key):
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Inject mocks into sys.modules BEFORE any imports
sys.modules["sqlite3"] = MockSQLite()
sys.modules["diskcache"] = type("diskcache", (), {"Cache": MockCache})()

# Disable wetterdienst cache
os.environ["WETTERDIENST_CACHE_DISABLE"] = "true"

# Now run the actual radar script
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from pyproj import Proj, Transformer
from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import imageio.v2 as imageio

from wetterdienst.provider.dwd.radar import (
    DwdRadarParameter,
    DwdRadarPeriod,
    DwdRadarResolution,
    DwdRadarValues,
)

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# RADOLAN projection
# -------------------------------------------------------------------
radolan_proj = Proj(
    proj="stere",
    lat_0=90,
    lat_ts=60,
    lon_0=10,
    a=6370040,
    b=6370040,
)

wgs84 = Proj("epsg:4326")
transformer = Transformer.from_proj(radolan_proj, wgs84, always_xy=True)

# Default RADOLAN grid parameters
PIXEL_SIZE = 1000
NX = 900
NY = 900
X_ORIGIN = -523462
Y_ORIGIN = -4658645


# -------------------------------------------------------------------
# RADOLAN lon/lat grid
# -------------------------------------------------------------------
def radolan_lonlat_grid(nx=None, ny=None):
    if nx is None:
        nx = NX
    if ny is None:
        ny = NY

    x = X_ORIGIN + np.arange(nx) * PIXEL_SIZE
    y = Y_ORIGIN + np.arange(ny) * PIXEL_SIZE
    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)
    return lon, lat


# -------------------------------------------------------------------
# Distance mask (km)
# -------------------------------------------------------------------
def mask_radius_km(da, lon, lat, lon0, lat0, radius_km):
    R = 6371.0
    lon1 = np.radians(lon)
    lat1 = np.radians(lat)
    lon2 = np.radians(lon0)
    lat2 = np.radians(lat0)

    dlon = lon1 - lon2
    dlat = lat1 - lat2

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    dist_km = R * c
    return da.where(dist_km <= radius_km)


# -------------------------------------------------------------------
# Colormap (white = 0 mm)
# -------------------------------------------------------------------
def rainfall_colormap():
    base = plt.cm.viridis(np.linspace(0, 1, 256))
    base[0] = [1, 1, 1, 1]
    return ListedColormap(base)


# -------------------------------------------------------------------
# Plot + save
# -------------------------------------------------------------------
def save_radolan_png(
    da, lon, lat, timestamp, index, lat_center, lon_center, radius_km, name, output_dir
):
    # Calculate map extent
    map_extent_degrees = (radius_km / 111) * 1.2  # Add 20% padding

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Zoom based on radius
    ax.set_extent(
        [
            lon_center - map_extent_degrees,
            lon_center + map_extent_degrees,
            lat_center - map_extent_degrees,
            lat_center + map_extent_degrees,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Add topography/terrain
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#e8e8e8", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#d4f1f9", zorder=0)

    # Add lakes
    ax.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="#d4f1f9",
        edgecolor="blue",
        linewidth=0.3,
        alpha=0.7,
        zorder=1,
    )

    # Add rivers
    ax.add_feature(
        cfeature.RIVERS.with_scale("10m"),
        edgecolor="#4da6ff",
        linewidth=0.7,
        alpha=0.8,
        zorder=2,
    )

    # Add borders and coastline
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, zorder=3)

    # Add state/province boundaries
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(states, edgecolor="black", linewidth=0.5, alpha=0.6, zorder=3)

    # Plot precipitation data
    pcm = ax.pcolormesh(
        lon,
        lat,
        da,
        cmap=rainfall_colormap(),
        shading="auto",
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=20,
        alpha=0.7,
        zorder=4,
    )

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Precipitation [mm/h]", fontsize=9)

    # Mark center location
    ax.plot(
        lon_center,
        lat_center,
        "ro",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    ax.set_title(f"{timestamp} @{name}", fontsize=16)

    outfile = output_dir / f"frame-{index:03d}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved %s", outfile)
    return outfile


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def radolan_last_2h_to_png(lat, lon, radius, name):
    """
    Generate radar precipitation maps for the last 2 hours.

    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius: Radius in kilometers
        name: Location name for the title
    """
    output_dir = Path("radar_png")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_mp4 = output_dir / "radar_inverted.mp4"

    # Use timezone-aware datetime
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)

    radolan = DwdRadarValues(
        parameter=DwdRadarParameter.RADOLAN_CDC,
        resolution=DwdRadarResolution.HOURLY,
        period=DwdRadarPeriod.RECENT,
        start_date=start.isoformat(),
        end_date=now.isoformat(),
    )

    items = sorted(radolan.query(), key=lambda i: i.timestamp, reverse=True)

    if not items:
        log.error("No radar data returned from query")
        log.info("This might be due to network issues or data availability")
        return False

    log.info(f"Found {len(items)} radar items")
    frame_files = []

    for idx, item in enumerate(items):
        log.info("Processing %s", item.timestamp)

        try:
            ds = xr.open_dataset(item.data, engine="radolan")
            product = next(iter(ds.data_vars))
            da = ds[product].astype("float32")

            log.info(
                "Applying precision scaling (x10) - values appear to be in 0.1 mm/h units"
            )
            da = da * 10

            log.info("Product: %s", product)
            log.info("Data shape: %s", da.shape)
            log.info(
                "Data range: min=%s, max=%s",
                float(da.min(skipna=True)),
                float(da.max(skipna=True)),
            )

            ny, nx = da.shape
            log.info("Creating lon/lat grid for %dx%d", ny, nx)
            grid_lon, grid_lat = radolan_lonlat_grid(nx, ny)

            da_masked = mask_radius_km(da, grid_lon, grid_lat, lon, lat, radius)

            frame_file = save_radolan_png(
                da_masked,
                grid_lon,
                grid_lat,
                item.timestamp,
                idx,
                lat,
                lon,
                radius,
                name,
                output_dir,
            )
            frame_files.append(frame_file)

        except Exception as e:
            log.error("Error processing %s: %s", item.timestamp, e)
            import traceback

            traceback.print_exc()
            continue

    if not frame_files:
        log.error("No frames were successfully processed")
        return False

    # Reverse to show oldest first
    frame_files = list(reversed(frame_files))

    # Create video with inverted colors
    with imageio.get_writer(
        output_mp4,
        fps=2,
        codec="libx264",
        quality=8,
    ) as writer:
        for f in frame_files:
            img = imageio.imread(f)

            # Ensure RGB
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Invert colors
            img = 255 - img

            writer.append_data(img)

    log.info("Finished! Video saved to %s", output_mp4)
    return True


# -------------------------------------------------------------------
# CLI argument parsing
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate radar precipitation maps from DWD RADOLAN data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-lat",
        "--latitude",
        type=float,
        required=True,
        help="Center latitude (decimal degrees)",
    )

    parser.add_argument(
        "-lon",
        "--longitude",
        type=float,
        required=True,
        help="Center longitude (decimal degrees)",
    )

    parser.add_argument(
        "-rad", "--radius", type=float, required=True, help="Radius in kilometers"
    )

    parser.add_argument(
        "-name",
        "--name",
        type=str,
        required=True,
        help="Location name (displayed in plot title)",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    try:
        success = radolan_last_2h_to_png(
            lat=args.latitude, lon=args.longitude, radius=args.radius, name=args.name
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
