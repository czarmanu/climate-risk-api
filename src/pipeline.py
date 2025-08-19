# *****************************************************************************
# pipeline.py
# *****************************************************************************

# Purpose: Pipeline: hazard -> exposure -> vulnerability -> EAL aggregation

# Orchestrates:
# - Read hazard grid (NetCDF), sample depths at asset points
# - Interpolate vulnerability curves (depth → damage ratio)
# - Compute per-event losses and Expected Annual Loss (EAL)
# - Aggregate by asset/country/global
# - Write CSV outputs consumed by the FastAPI service

# Author(s):
# Manu Tom, 2025-

# *****************************************************************************
# Example invocation
# *****************************************************************************
# python src/pipeline.py --data-dir ./data --out ./results

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr


# a helper/private function inside the module, not for public use
def _load_inputs(data_dir: Path) -> Tuple[xr.Dataset, pd.DataFrame,
                                          dict, pd.DataFrame]:
    """
    Load all required input datasets for the risk pipeline.

    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the input files:
            - hazard.nc
            - assets.csv
            - vulnerability.json
            - event_rates.csv

    Returns
    -------
    ds : xr.Dataset
        Hazard dataset (NetCDF), e.g. flood depths for multiple events
        across a spatial grid.
    assets : pd.DataFrame
        Table of exposed assets (location, value, country, etc.).
    vuln : dict
        Vulnerability curve definition mapping hazard intensity
        (e.g. flood depth)
        to damage ratio (fraction of loss).
    rates : pd.DataFrame
        Table of event occurrence rates (event_id → annual rate).
"""
    # Load hazard data from NetCDF file as an xarray Dataset.
    # - "hazard.nc" stores hazard fields (e.g., flood depth per event).
    # - chunks={"event": 1, "y": 60} enables lazy/dask-backed reading
    #   for scalability: process one event at a time and 60 rows in Y dimension
    ds = xr.open_dataset(data_dir / "hazard.nc", chunks={"event": 1, "y": 60})

    # Load asset exposure data from CSV into a Pandas DataFrame.
    # Columns typically: asset_id, lon, lat, country, tiv (total insured value)
    assets = pd.read_csv(data_dir / "assets.csv")

    # Load vulnerability function from JSON.
    # - Maps hazard intensity (e.g. water depth in meters) → damage ratio (0–1)
    # - Example: {"depth_m": [0.0, 1.0], "damage_ratio": [0.0, 1.0]}
    with open(data_dir / "vulnerability.json", "r", encoding="utf-8") as f:
        vuln = json.load(f)

    # Load event occurrence rates from CSV into a DataFrame.
    # - Columns: event_id, annual_rate (probability of event per year).
    # - Used to weight event losses when computing Expected Annual Loss (EAL)
    rates = pd.read_csv(data_dir / "event_rates.csv")

    # Return all loaded objects as a tuple for downstream processing
    return ds, assets, vuln, rates


def _lonlat_to_idx(lon: np.ndarray, lat: np.ndarray, xs: np.ndarray,
                   ys: np.ndarray):
    """Map (lon, lat) points to bounding grid indices and bilinear weights.

    Parameters
    ----------
    lon : np.ndarray
        Longitudes of points (deg), shape (N,).
    lat : np.ndarray
        Latitudes of points (deg), shape (N,).
    xs : np.ndarray
        Sorted longitudes of grid cell centers, shape (nx,).
    ys : np.ndarray
        Sorted latitudes of grid cell centers, shape (ny,).

    Returns
    -------
    i0, i1, j0, j1 : np.ndarray
        Row (y) and column (x) indices of the bounding cells, shape (N,).
    wx, wy : np.ndarray
        Normalized bilinear weights along x and y in [0, 1], shape (N,).

    Notes
    -----
    Indices are clipped to grid bounds. If a point lies on a grid line,
    the delta along that axis can be zero; we guard against division by
    zero by substituting 1.0 for the spacing, which yields a weight of 0.0.
    """
    # Find insertion indices in the sorted x (lon) and y (lat) arrays
    # - j1: index of the grid column just to the right of lon
    # - i1: index of the grid row just above lat
    j1 = np.searchsorted(xs, lon, side="left")
    i1 = np.searchsorted(ys, lat, side="left")

    # Get left (j0) and bottom (i0) neighbor indices
    # Clip to ensure they stay within the grid boundaries
    j0 = np.clip(j1 - 1, 0, len(xs) - 1)
    i0 = np.clip(i1 - 1, 0, len(ys) - 1)

    # Clip right/top indices as well
    j1 = np.clip(j1, 0, len(xs) - 1)
    i1 = np.clip(i1, 0, len(ys) - 1)

    # Extract the bounding grid coordinates
    x0, x1 = xs[j0], xs[j1]  # longitudes bounding the asset
    y0, y1 = ys[i0], ys[i1]  # latitudes bounding the asset

    # Compute grid spacing in x and y (avoid division by zero if same cell)
    dx = np.where((x1 - x0) == 0, 1.0, (x1 - x0))
    dy = np.where((y1 - y0) == 0, 1.0, (y1 - y0))

    # Compute bilinear interpolation weights (0–1 fraction within cell)
    wx = np.clip((lon - x0) / dx, 0.0, 1.0)  # weight along longitude
    wy = np.clip((lat - y0) / dy, 0.0, 1.0)  # weight along latitude

    return i0, i1, j0, j1, wx, wy


def _bilinear_sample(depth: np.ndarray, i0, i1, j0, j1, wx, wy) -> np.ndarray:
    """Vectorized bilinear sampling of a 2-D grid at N locations.

    Parameters
    ----------
    depth : np.ndarray
        2-D array (ny, nx) to sample (e.g., flood depth for one event).
    i0, i1, j0, j1 : np.ndarray
        Row/column indices of the four neighboring cells, shape (N,).
    wx, wy : np.ndarray
        Interpolation weights along x and y in [0, 1], shape (N,).

    Returns
    -------
    np.ndarray
        Sampled values at the N query locations, shape (N,).
    """
    # Depth at four corners of the bounding grid cell
    v00 = depth[i0, j0]  # bottom-left
    v01 = depth[i0, j1]  # bottom-right
    v10 = depth[i1, j0]  # top-left
    v11 = depth[i1, j1]  # top-right

    # Bilinear interpolation formula:
    # Weighted average of the four corners using wx, wy
    return ((1 - wx) * (1 - wy) * v00  # bottom-left
            + wx * (1 - wy) * v01      # bottom-right
            + (1 - wx) * wy * v10      # top-left
            + wx * wy * v11)           # top-right


def _interp_damage_ratio(depths: np.ndarray, depth_knots: np.ndarray,
                         dr_knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear mapping depth→damage ratio with clamping.

    Parameters
    ----------
    depths : np.ndarray
        Flood depths (m) at assets for each event; any shape.
    depth_knots : np.ndarray
        1-D strictly increasing depths (m), shape (K,).
    dr_knots : np.ndarray
        1-D damage ratios in [0, 1] at those depths, shape (K,).

    Returns
    -------
    np.ndarray
        Damage ratios with the same shape as `depths`.

    Notes
    -----
    Values outside the knot range are clamped to the end knots before
    linear interpolation.
    """
    # Clamp depths to within the vulnerability curve’s domain
    depths_clamped = np.clip(depths, depth_knots[0], depth_knots[-1])

    # Interpolate linearly between (depth_m, damage_ratio) points
    return np.interp(depths_clamped, depth_knots, dr_knots)


def compute_eal(ds: xr.Dataset, assets: pd.DataFrame, vuln: dict,
                rates: pd.DataFrame):
    """
    Compute Expected Annual Loss (EAL) at asset, country, and global levels.

    Parameters
    ----------
    ds : xr.Dataset
        Hazard dataset (NetCDF) with dimensions:
        - event × y × x grid
        - variable: "flood_depth" [m].
    assets : pd.DataFrame
        Asset inventory with columns:
        - asset_id : str
        - lon, lat : float (geographic coordinates)
        - country : str (ISO3 code)
        - tiv : float (total insured value, USD).
    vuln : dict
        Vulnerability curve stored as JSON-like dict:
        - "depth_m" : list of breakpoints [m]
        - "damage_ratio" : list of fractional losses [0–1].
    rates : pd.DataFrame
        Event occurrence rates with columns:
        - event_id : int (0..n_events-1)
        - annual_rate : float (events/year).

    Returns
    -------
    out_asset : pd.DataFrame
        Per-asset results with columns:
        - asset_id, country, tiv, eal_usd.
    out_country : pd.DataFrame
        Per-country aggregates:
        - country, asset_count, tiv_sum, eal_usd.
    out_global : pd.DataFrame
        Global totals (single row):
        - tiv_sum, eal_usd.

    Notes
    -----
    Workflow:
    1. Map each asset to its surrounding hazard grid cells.
    2. Bilinearly interpolate flood depths at asset coordinates.
    3. Convert depths to fractional losses using vulnerability curve.
    4. Multiply by asset TIV → per-event losses.
    5. Weight by event annual rates → Expected Annual Loss.
    6. Aggregate to asset, country, and global summaries.
    """
    # Extract the longitude (x) and latitude (y) grid coordinates from
    # the hazard dataset. These represent the raster grid cell centers
    # for which flood depths are defined.
    xs = ds["x"].values
    ys = ds["y"].values

    # Work on a copy of the asset table to avoid mutating the caller’s df.
    assets = assets.copy()

    # Map each asset’s (lon, lat) coordinates to indices in the hazard grid.
    # This returns surrounding cell indices (i0, i1, j0, j1)
    # and bilinear interpolation weights (wx, wy).
    i0, i1, j0, j1, wx, wy = _lonlat_to_idx(
        assets["lon"].to_numpy(), assets["lat"].to_numpy(), xs, ys
    )

    # Initialize an array to hold interpolated flood depth at each asset,
    # for each event.
    # Shape: (#assets, #events).
    depth_m = np.zeros((len(assets), ds.sizes["event"]), dtype=np.float32)

    # Loop over all events in the hazard dataset.
    # For each event, extract the flood depth grid
    # and sample it at each asset location.
    for e in range(ds.sizes["event"]):
        # Select the flood depth grid for event e.
        depth = ds["flood_depth"].isel(event=e).values
        # Perform bilinear sampling at each asset coordinate and
        # store results in the depth matrix.
        depth_m[:, e] = _bilinear_sample(depth, i0, i1, j0, j1, wx, wy)

    # Convert vulnerability curve (depth → damage ratio) into NumPy arrays.
    # depth_knots: depth breakpoints (e.g., 0.0 m, 0.5 m, 1.0 m).
    # dr_knots: damage ratios at those breakpoints (0.0–1.0 fraction
    # of asset value lost).
    depth_knots = np.asarray(vuln["depth_m"], dtype=float)
    dr_knots = np.asarray(vuln["damage_ratio"], dtype=float)

    # Interpolate the damage ratio for every (asset, event) depth value.
    # Result: matrix of same shape as depth_m, with fractional losses (0–1).
    dr = _interp_damage_ratio(depth_m, depth_knots, dr_knots)

    # Expand each asset’s total insured value (TIV) to match event dimension.
    # Shape becomes (#assets, 1), so broadcasting works in multiplication.
    tiv = assets["tiv"].to_numpy()[:, None]

    # Compute per-event dollar losses for each asset:
    # loss = damage ratio × insured value.
    loss = dr * tiv

    # Build a dictionary mapping event_id → annual occurrence rate
    # from the rates dataframe.
    rate_map = rates.set_index("event_id")["annual_rate"].to_dict()

    # Convert this dictionary into a vector aligned
    # with event indices [0..n_events-1].
    # Shape (1, #events) so broadcasting with loss works cleanly.
    rates_vec = np.array([rate_map[e] for e in range(ds.sizes["event"])],
                         dtype=float)[None, :]

    # Compute Expected Annual Loss (EAL) per asset:
    # For each asset, sum across all events:
    # (loss per event × annual occurrence rate).
    eal_asset = (loss * rates_vec).sum(axis=1)

    # Assemble output dataframe at the asset level:
    # Keep asset_id, country, tiv, and computed EAL in USD.
    out_asset = assets[["asset_id", "country", "tiv"]].copy()
    out_asset["eal_usd"] = eal_asset

    # Aggregate to country-level results:
    # - Count of assets
    # - Total insured value
    # - Total expected annual loss
    out_country = (
        out_asset.groupby("country")
        .agg(asset_count=("asset_id", "count"),
             tiv_sum=("tiv", "sum"),
             eal_usd=("eal_usd", "sum"))
        .reset_index()
    )

    # Aggregate to global totals (single-row dataframe).
    out_global = pd.DataFrame({
        "tiv_sum": [out_asset["tiv"].sum()],
        "eal_usd": [out_asset["eal_usd"].sum()],
    })

    # Return all three outputs as a tuple:
    # - per-asset EAL
    # - per-country EAL
    # - global total EAL
    return out_asset, out_country, out_global


def main() -> None:
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()

    # Add argument for input data  and results directories (default: ./data)
    # type=Path ensures the value is automatically converted
    # into a pathlib.Path object
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out", type=Path, default=Path("./results"))

    # Parse command-line arguments into a Namespace object
    # Example: args.data_dir → Path("./data"), args.out → Path("./results")
    args = parser.parse_args()

    # Load required inputs:
    # - Hazard dataset (NetCDF file with flood depths)
    # - Asset table (CSV with asset locations, Total Insured Value (TIV),
    # country, etc.)
    # - Vulnerability curve (JSON: depth vs damage ratio)
    # - Event occurrence rates (CSV: event_id → annual rate)
    ds, assets, vuln, rates = _load_inputs(args.data_dir)

    # Run the core computation:
    # - Sample hazard depths at asset coordinates
    # - Apply vulnerability curve (depth → damage ratio)
    # - Compute Expected Annual Loss (EAL) for each asset
    # - Aggregate EAL by country and globally
    out_asset, out_country, out_global = compute_eal(ds, assets, vuln, rates)

    # Ensure the output directory exists (create if missing)
    # parents=True → create intermediate directories if needed
    # exist_ok=True → do nothing if the directory already exists
    args.out.mkdir(parents=True, exist_ok=True)

    # Save results to CSV files
    # (to_csv writes the Pandas DataFrame (out_asset etc) contents
    # to a CSV file.):
    # - Asset-level losses
    # - Country-level aggregated losses
    # - Global aggregated losses
    # index=False → don’t write the DataFrame’s row index as an extra column
    out_asset.to_csv(args.out / "eal_by_asset.csv", index=False)
    out_country.to_csv(args.out / "eal_by_country.csv", index=False)
    out_global.to_csv(args.out / "eal_global.csv", index=False)

    # Sanity checks:
    # 1. No duplicate asset IDs in asset-level results
    assert not out_asset["asset_id"].duplicated().any()
    # 2. Country-level EAL sums must match global EAL within tolerance
    assert abs(out_country["eal_usd"].sum() - out_global["eal_usd"][0]) < 1e-6
    # 3. Country-level TIV sums must match global TIV within tolerance
    assert abs(out_country["tiv_sum"].sum() - out_global["tiv_sum"][0]) < 1e-6


# Entry point check:
# When this script is run directly (not imported), execute main()
if __name__ == "__main__":
    main()
