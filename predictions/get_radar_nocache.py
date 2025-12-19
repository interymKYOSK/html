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
from zoneinfo import ZoneInfo
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

# Simple RADOLAN binary parser (no external dependencies)
import struct
import gzip


def parse_radolan_binary(data_bytes):
    """
    Simple RADOLAN binary format parser
    Returns: (data_array, metadata_dict)
    """
    # Check if gzipped
    if data_bytes[:2] == b"\x1f\x8b":
        data_bytes = gzip.decompress(data_bytes)

    # RADOLAN has ASCII header followed by binary data
    # Header ends with ETX (0x03)
    header_end = data_bytes.find(b"\x03")
    if header_end == -1:
        raise ValueError("Could not find RADOLAN header end marker")

    header = data_bytes[:header_end].decode("latin-1", errors="ignore")
    binary_data = data_bytes[header_end + 1 :]

    # Parse dimensions from header (typical RW product is 900x900)
    # Default to 900x900 if not found
    nx = ny = 900

    # Read binary data as uint16 (2 bytes per pixel)
    data_size = nx * ny * 2
    if len(binary_data) < data_size:
        # Try to infer dimensions from data size
        total_pixels = len(binary_data) // 2
        nx = ny = int(np.sqrt(total_pixels))

    # Parse as big-endian uint16
    try:
        data = np.frombuffer(binary_data[: nx * ny * 2], dtype=">u2")
        data = data.reshape((ny, nx))
    except ValueError as e:
        log.error(f"Error reshaping data: {e}, trying 1200x1100")
        # Some products are 1200x1100
        nx, ny = 1200, 1100
        data = np.frombuffer(binary_data[: nx * ny * 2], dtype=">u2")
        data = data.reshape((ny, nx))

    # Convert to float and handle special values
    data = data.astype("float32")

    # RADOLAN special values:
    # 250 = clutter/noise
    # 249 = no data
    # 0 = no precipitation (will be masked during visualization)
    data[data >= 249] = np.nan

    # RADOLAN RW values are in 0.1 mm/h
    # Already will be scaled by 10 in main code

    metadata = {"header": header, "nx": nx, "ny": ny}

    return data, metadata


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
# Colormap (typical weather radar colors)
# -------------------------------------------------------------------
def rainfall_colormap():
    """
    Create a colormap for radar precipitation following typical weather radar standards:
    """
    colors = [
        "#FFFFFF",  # 0: White (no rain)
        "#00CCFF",  # 0.1-1: Light cyan
        "#7167FF",  # 1-2: Blue
        "#00FF00",  # 2-3: Green
        "#FFFF00",  # 3-5: Yellow
        "#FFA500",  # 5-10: Orange
        "#FF0000",  # 10-20: Red
        "#8B008B",  # 20+: Dark magenta
    ]

    return ListedColormap(colors, N=len(colors))


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
        alpha=1,
        zorder=1,
    )

    # Add rivers
    ax.add_feature(
        cfeature.RIVERS.with_scale("10m"),
        edgecolor="#4da6ff",
        linewidth=0.7,
        alpha=1,
        zorder=2,
    )

    # Add borders and coastline
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=1, zorder=3)

    # Add state/province boundaries
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(states, edgecolor="black", linewidth=0.5, alpha=0.8, zorder=3)

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
        alpha=0.9,
        zorder=4,
    )

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Precipitation [mm/h]", fontsize=9)

    # Set custom colorbar ticks and labels for precipitation ranges
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    cbar.set_ticklabels(
        ["0", "0.1-1", "1-2", "2-3", "3-5", "5-10", "10-20", "20+"], fontsize=8
    )

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

    # convert timestamp from UTC to local time in Berlin (assumed CET/CEST)
    timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
    timestamp_local = timestamp_utc.astimezone(ZoneInfo("Europe/Berlin"))
    timestamp_str = timestamp_local.strftime("%Y-%m-%d %H:%M:%S")

    ax.set_title(f"{timestamp_str} @{name}", fontsize=16)

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

    # Format dates as strings without timezone info to avoid type mismatch
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = now.strftime("%Y-%m-%d %H:%M:%S")

    radolan = DwdRadarValues(
        parameter=DwdRadarParameter.RADOLAN_CDC,
        resolution=DwdRadarResolution.HOURLY,
        period=DwdRadarPeriod.RECENT,
        start_date=start_str,
        end_date=end_str,
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
            # Read RADOLAN data directly from bytes
            if isinstance(item.data, bytes):
                data_bytes = item.data
            else:
                # item.data is likely a file-like object
                data_bytes = item.data.read()

            # Parse RADOLAN binary format
            data, metadata = parse_radolan_binary(data_bytes)

            # Debug: log raw data statistics
            log.info(
                "Raw data (before scaling): min=%s, max=%s, mean=%s",
                float(np.nanmin(data)),
                float(np.nanmax(data)),
                float(np.nanmean(data)),
            )

            # Create xarray DataArray
            da = xr.DataArray(data, dims=["y", "x"], attrs=metadata).astype("float32")

            log.info(
                "Applying precision scaling (x10) - values appear to be in 0.1 mm/h units"
            )
            da = da * 10

            # log.info("Product: %s", product)
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

            # Mask zero precipitation for visualization (no rain = transparent)
            da_masked = da_masked.where(da_masked > 0)

            # Debug: log masked data statistics
            log.info(
                "After masking: min=%s, max=%s, valid_pixels=%s",
                float(da_masked.min(skipna=True)),
                float(da_masked.max(skipna=True)),
                int(np.sum(~np.isnan(da_masked.values))),
            )

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
            # img = 255 - img

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
        required=False,
        default=47.993794,
        help="Center latitude (decimal degrees)",
    )

    parser.add_argument(
        "-lon",
        "--longitude",
        type=float,
        required=False,
        default=7.84082,
        help="Center longitude (decimal degrees)",
    )

    parser.add_argument(
        "-rad",
        "--radius",
        type=float,
        required=False,
        default=300,
        help="Radius in kilometers",
    )

    parser.add_argument(
        "-name",
        "--name",
        type=str,
        required=False,
        default="CCCfr",
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
