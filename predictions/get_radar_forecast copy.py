import argparse
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from pyproj import Proj, Transformer

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import imageio.v2 as imageio
import sys
from PIL import Image

from wetterdienst.provider.dwd.radar import (
    DwdRadarParameter,
    DwdRadarPeriod,
    DwdRadarResolution,
    DwdRadarValues,
)
from wetterdienst import Settings

# Disable SQLite cache to avoid sqlite3 dependency
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
# Colormap
# -------------------------------------------------------------------
def rainfall_colormap_with_norm():
    """
    Create a colormap and normalization for radar precipitation.
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

    boundaries = [0, 0.1, 1, 2, 3, 5, 10, 20, 25]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    return cmap, norm


# -------------------------------------------------------------------
# Plot + save (with forecast indicator)
# -------------------------------------------------------------------
def save_radolan_png(
    da,
    lon,
    lat,
    timestamp,
    index,
    lat_center,
    lon_center,
    radius_km,
    name,
    output_dir,
    is_forecast=False,
):
    """
    Save radar frame with optional "FORECAST" label.

    Args:
        is_forecast: If True, marks frame as forecast
    """
    map_extent_degrees = (radius_km / 111) * 1.2

    # Fixed figure size to ensure consistency
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [
            lon_center - map_extent_degrees,
            lon_center + map_extent_degrees,
            lat_center - map_extent_degrees,
            lat_center + map_extent_degrees,
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#e8e8e8", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#d4f1f9", zorder=0)
    ax.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="#d4f1f9",
        edgecolor="blue",
        linewidth=0.3,
        alpha=1,
        zorder=1,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("10m"),
        edgecolor="#4da6ff",
        linewidth=0.7,
        alpha=1,
        zorder=2,
    )
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=1, zorder=3)

    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(states, edgecolor="black", linewidth=0.5, alpha=0.8, zorder=3)

    cmap, norm = rainfall_colormap_with_norm()

    pcm = ax.pcolormesh(
        lon,
        lat,
        da,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        alpha=0.99 if not is_forecast else 0.9,
        zorder=4,
    )

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Precipitation [mm/h]", fontsize=9)
    cbar.set_ticks([0.05, 0.55, 1.5, 2.5, 4.0, 7.5, 15.0, 22.5])
    cbar.set_ticklabels(
        ["0", "0.1-1", "1-2", "2-3", "3-5", "5-10", "10-20", "20+"], fontsize=8
    )

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

    # Convert timestamp to local time
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    if timestamp.tzinfo is None:
        timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp_utc = timestamp

    timestamp_local = timestamp_utc.astimezone(ZoneInfo("Europe/Berlin"))

    # For forecast frames, show HH:MM with offset
    if is_forecast:
        now_local = datetime.now(ZoneInfo("Europe/Berlin"))
        time_offset = timestamp_local - now_local
        total_minutes = int(round(time_offset.total_seconds() / 60 / 15) * 15)
        hours = total_minutes // 60
        minutes = abs(total_minutes % 60)

        if hours > 0:
            offset_str = f"+{hours}h {minutes}m"
        else:
            offset_str = f"+{minutes}m"

        time_str = timestamp_local.strftime("%Y-%m-%d %H:%M")
        title = f"{time_str} ({offset_str}) @{name}"
    else:
        # Show full timestamp for observed frames
        timestamp_str = timestamp_local.strftime("%Y-%m-%d %H:%M")
        title = f"{timestamp_str} @{name}"

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Add ZUKUNFT label for forecast frames on top of the plot
    if is_forecast:
        ax.text(
            0.5,
            0.95,
            "ZUKUNFT",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="red",
            ha="center",
            va="center",
        )

    outfile = output_dir / f"frame-{index:03d}.png"
    plt.savefig(outfile, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log.info("Saved %s (%s)", outfile.name, "FORECAST" if is_forecast else "OBSERVED")
    return outfile


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def radar_with_forecast_to_video(lat, lon, radius, name):
    """
    Generate radar video combining RADOLAN and RADVOR (radar-based nowcast).

    RADVOR provides 5-minute granularity forecasts for the same historical period.
    This creates a combined video showing high-resolution precipitation patterns.

    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius: Radius in kilometers
        name: Location name for the title
    """
    output_dir = Path("radar_png")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_mp4 = output_dir / "radar_forecast.mp4"

    now = datetime.now(timezone.utc)

    # === Query Historical RADOLAN (past 2 hours) ===
    log.info("=" * 60)
    log.info("Querying RADOLAN (historical, past 2 hours)...")
    log.info("=" * 60)

    hist_start = now - timedelta(hours=2)
    hist_start_str = hist_start.strftime("%Y-%m-%d %H:%M:%S")
    hist_end_str = now.strftime("%Y-%m-%d %H:%M:%S")

    historical_items = []
    try:
        radolan = DwdRadarValues(
            parameter=DwdRadarParameter.RADOLAN_CDC,
            resolution=DwdRadarResolution.HOURLY,
            period=DwdRadarPeriod.RECENT,
            start_date=hist_start_str,
            end_date=hist_end_str,
        )
        historical_items = sorted(
            radolan.query(), key=lambda i: i.timestamp, reverse=True
        )
    except Exception as e:
        log.error(f"Failed to query RADOLAN: {e}")

    if not historical_items:
        log.warning("No RADOLAN historical data found")
    else:
        log.info(f"Found {len(historical_items)} RADOLAN timesteps")

    # === Query RADVOR Forecast (next 1-2 hours) ===
    log.info("=" * 60)
    log.info("Querying RADVOR (radar-based forecast, next 1-2 hours)...")
    log.info("=" * 60)

    # RADVOR provides 1-2 hour ahead forecasts
    forecast_start = now
    forecast_end = now + timedelta(hours=2)
    forecast_start_str = forecast_start.strftime("%Y-%m-%d %H:%M:%S")
    forecast_end_str = forecast_end.strftime("%Y-%m-%d %H:%M:%S")

    forecast_items = []
    try:
        log.info(
            f"  Trying RQ_REFLECTIVITY from {forecast_start_str} to {forecast_end_str}..."
        )
        radvor = DwdRadarValues(
            parameter=DwdRadarParameter.RQ_REFLECTIVITY,
            start_date=forecast_start_str,
            end_date=forecast_end_str,
        )
        forecast_items = sorted(radvor.query(), key=lambda i: i.timestamp)
        if forecast_items:
            log.info(f"✓ Found {len(forecast_items)} RQ_REFLECTIVITY timesteps")
        else:
            log.warning("RQ_REFLECTIVITY query returned no results")
    except Exception as e:
        log.warning(f"RQ_REFLECTIVITY failed: {e}")
        log.info("Trying RE_REFLECTIVITY...")
        try:
            radvor = DwdRadarValues(
                parameter=DwdRadarParameter.RE_REFLECTIVITY,
                start_date=forecast_start_str,
                end_date=forecast_end_str,
            )
            forecast_items = sorted(radvor.query(), key=lambda i: i.timestamp)
            if forecast_items:
                log.info(f"✓ Found {len(forecast_items)} RE_REFLECTIVITY timesteps")
            else:
                log.warning("RE_REFLECTIVITY query returned no results")
        except Exception as e2:
            log.warning(f"RE_REFLECTIVITY also failed: {e2}")
        log.warning("No RADVOR forecast data found")
    else:
        log.info(f"Found {len(forecast_items)} RADVOR timesteps")

    # === Combine and sort all items ===
    log.info("=" * 60)
    log.info("Combining historical + forecast...")
    log.info("=" * 60)

    all_items = []

    # Add historical items (marked as observed)
    for item in reversed(historical_items):
        if item.timestamp:
            all_items.append(
                {"item": item, "is_forecast": False, "timestamp": item.timestamp}
            )

    # Add forecast items (marked as forecast)
    for item in forecast_items:
        if item.timestamp:
            all_items.append(
                {"item": item, "is_forecast": True, "timestamp": item.timestamp}
            )

    # Sort all by timestamp
    all_items = sorted(all_items, key=lambda x: x["timestamp"])

    if not all_items:
        log.error("No data (historical or forecast) found")
        return False

    log.info(f"Total frames to generate: {len(all_items)}")

    # === Generate frames ===
    frame_files = []

    for idx, entry in enumerate(all_items):
        item = entry["item"]
        is_forecast = entry["is_forecast"]
        data_type = "FORECAST" if is_forecast else "OBSERVED"

        log.info(
            f"[{idx+1}/{len(all_items)}] Processing {item.timestamp} ({data_type})"
        )

        try:
            ds = xr.open_dataset(item.data, engine="radolan")
            product = next(iter(ds.data_vars))
            da = ds[product].astype("float32")

            # Scale RADOLAN values (0.1 mm/h -> mm/h)
            da = da * 10

            ny, nx = da.shape
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
                is_forecast=is_forecast,
            )
            frame_files.append(frame_file)

        except Exception as e:
            log.error(f"  Error processing {item.timestamp}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not frame_files:
        log.error("No frames were successfully processed")
        return False

    # === Create video ===
    log.info("=" * 60)
    log.info(f"Creating video from {len(frame_files)} frames...")
    log.info("=" * 60)

    # First pass: detect target frame size (from first frame)
    target_size = None
    if frame_files:
        first_img = imageio.imread(frame_files[0])
        target_size = first_img.shape[:2]  # (height, width)
        log.info(f"Target frame size: {target_size}")

    # Second pass: create video with frame size normalization
    with imageio.get_writer(output_mp4, fps=2, codec="libx264", quality=8) as writer:
        for f in frame_files:
            img = imageio.imread(f)

            # Ensure RGB
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Resize if needed
            current_size = img.shape[:2]
            if current_size != target_size:
                log.warning(
                    f"  Frame size mismatch: {current_size} vs {target_size}, resizing..."
                )
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize(
                    (target_size[1], target_size[0]), Image.Resampling.LANCZOS
                )
                img = np.array(pil_img)

            writer.append_data(img)

    log.info(f"✓ Video saved: {output_mp4}")
    log.info(f"  Duration: {len(frame_files)/2:.1f} seconds")
    log.info(
        f"  Historical frames: {sum(1 for e in all_items if not e['is_forecast'])}"
    )
    log.info(f"  Forecast frames: {sum(1 for e in all_items if e['is_forecast'])}")

    return True


# -------------------------------------------------------------------
# CLI argument parsing
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate radar video combining RADOLAN (historical) + RADVOR (forecast)",
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
        default=150,
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
        success = radar_with_forecast_to_video(
            lat=args.latitude, lon=args.longitude, radius=args.radius, name=args.name
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
