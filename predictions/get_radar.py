import argparse
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
from pyproj import Proj, Transformer
from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import imageio.v2 as imageio
import sys

# CRITICAL FIX: Disable cache BEFORE importing wetterdienst
import os

os.environ["WETTERDIENST_CACHE_DISABLE"] = "True"

# Mock diskcache to prevent import errors
from unittest.mock import MagicMock

mock_diskcache = MagicMock()
sys.modules["diskcache"] = mock_diskcache

from wetterdienst.provider.dwd.radar import (
    DwdRadarParameter,
    DwdRadarPeriod,
    DwdRadarResolution,
    DwdRadarValues,
)
from wetterdienst import Settings

# Import wradlib for RADOLAN reading
try:
    import wradlib as wrl

    HAS_WRADLIB = True
except ImportError:
    HAS_WRADLIB = False
    logging.warning("wradlib not available, will try alternative methods")

# Also disable via Settings object
Settings.cache_disable = True

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
# Read RADOLAN data
# -------------------------------------------------------------------
def read_radolan_data(data_source):
    """
    Read RADOLAN data from a file or bytes object.
    Tries multiple methods in order of preference.

    Returns: numpy array of precipitation data
    """
    # Method 1: Try wradlib if available
    if HAS_WRADLIB:
        try:
            log.info("Attempting to read with wradlib...")
            if isinstance(data_source, bytes):
                data_source = io.BytesIO(data_source)
            elif isinstance(data_source, str):
                # It's a file path
                pass

            # Read RADOLAN data
            data, metadata = wrl.io.read_radolan_composite(data_source)
            log.info(f"Successfully read with wradlib. Shape: {data.shape}")
            return data
        except Exception as e:
            log.warning(f"wradlib read failed: {e}")

    # Method 2: Try xarray with h5netcdf
    try:
        log.info("Attempting to read with xarray...")
        if isinstance(data_source, bytes):
            data_source = io.BytesIO(data_source)

        ds = xr.open_dataset(data_source, engine="h5netcdf")
        product = next(iter(ds.data_vars))
        data = ds[product].values
        log.info(f"Successfully read with xarray. Shape: {data.shape}")
        return data
    except Exception as e:
        log.warning(f"xarray read failed: {e}")

    # Method 3: Raw binary read (RADOLAN format)
    try:
        log.info("Attempting raw RADOLAN binary read...")
        if isinstance(data_source, bytes):
            raw_data = data_source
        else:
            with open(data_source, "rb") as f:
                raw_data = f.read()

        # Find the header end (marked by ETX, 0x03)
        header_end = raw_data.find(b"\x03")
        if header_end == -1:
            raise ValueError("Could not find RADOLAN header end marker")

        # Data starts after header
        binary_data = raw_data[header_end + 1 :]

        # RADOLAN RW product is typically 900x900 with 2 bytes per pixel
        expected_size = 900 * 900 * 2
        if len(binary_data) >= expected_size:
            # Read as 16-bit integers, big-endian
            data = np.frombuffer(binary_data[:expected_size], dtype=">u2")
            data = data.reshape(900, 900)

            # Apply RADOLAN scaling: value = (raw - 4096) / 2.0 / 10.0
            # But for RW product, simpler: raw value * 0.1 mm/h
            data = data.astype(float)
            # Handle missing data (marked as 65535 or other special values)
            data[data >= 4095] = np.nan

            log.info(f"Successfully read raw RADOLAN. Shape: {data.shape}")
            return data
        else:
            raise ValueError(
                f"Binary data size mismatch. Got {len(binary_data)}, expected ~{expected_size}"
            )

    except Exception as e:
        log.error(f"Raw RADOLAN read failed: {e}")
        raise ValueError("All methods to read RADOLAN data failed")


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
def mask_radius_km(data, lon, lat, lon0, lat0, radius_km):
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

    # Create masked array
    mask = dist_km > radius_km
    masked_data = np.ma.masked_where(mask, data)

    return masked_data


# -------------------------------------------------------------------
# Colormap (typical weather radar colors)
# -------------------------------------------------------------------
def rainfall_colormap_with_norm():
    """
    Create a colormap and normalization for radar precipitation.
    Returns: (colormap, norm) tuple
    Ranges: 0, 0.1-1, 1-2, 2-3, 3-5, 5-10, 10-20, 20+
    """
    colors = [
        "#FFFFFF",  # White: no rain
        "#00CCFF",  # Light cyan: 0.1-1
        "#7167FF",  # Blue: 1-2
        "#00FF00",  # Green: 2-3
        "#FFFF00",  # Yellow: 3-5
        "#FFA500",  # Orange: 5-10
        "#FF0000",  # Red: 10-20
        "#8B008B",  # Dark magenta: 20+
    ]

    # Define boundaries for precipitation ranges (mm/h)
    boundaries = [0, 0.1, 1, 2, 3, 5, 10, 20, 25]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    return cmap, norm


# -------------------------------------------------------------------
# Plot + save
# -------------------------------------------------------------------
def save_radolan_png(
    data,
    lon,
    lat,
    timestamp,
    index,
    lat_center,
    lon_center,
    radius_km,
    name,
    output_dir,
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

    # Get colormap and normalization
    cmap, norm = rainfall_colormap_with_norm()

    # Plot precipitation data
    pcm = ax.pcolormesh(
        lon,
        lat,
        data,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        alpha=0.9,
        zorder=4,
    )

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Precipitation [mm/h]", fontsize=9)

    # Set custom colorbar ticks and labels for precipitation ranges
    cbar.set_ticks([0.05, 0.55, 1.5, 2.5, 4.0, 7.5, 15.0, 22.5])
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

    # convert timestamp from UTC to local time in Berlin
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

    # Format dates as strings
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
            # Read data using our flexible reader
            data = read_radolan_data(item.data)

            # Convert to float32 for processing
            data = data.astype("float32")

            log.info(
                "Applying precision scaling (x10) - values appear to be in 0.1 mm/h units"
            )
            data = data * 10

            log.info("Data shape: %s", data.shape)
            log.info(
                "Data range: min=%s, max=%s",
                np.nanmin(data),
                np.nanmax(data),
            )

            ny, nx = data.shape
            log.info("Creating lon/lat grid for %dx%d", ny, nx)
            grid_lon, grid_lat = radolan_lonlat_grid(nx, ny)

            data_masked = mask_radius_km(data, grid_lon, grid_lat, lon, lat, radius)

            frame_file = save_radolan_png(
                data_masked,
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

    # Create video
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
