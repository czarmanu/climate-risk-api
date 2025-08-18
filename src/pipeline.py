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


def _load_inputs(data_dir: Path) -> Tuple[xr.Dataset, pd.DataFrame,
                                          dict, pd.DataFrame]:
    ds = xr.open_dataset(data_dir / "hazard.nc", chunks={"event": 1, "y": 60})
    assets = pd.read_csv(data_dir / "assets.csv")
    with open(data_dir / "vulnerability.json", "r", encoding="utf-8") as f:
        vuln = json.load(f)
    rates = pd.read_csv(data_dir / "event_rates.csv")
    return ds, assets, vuln, rates


def _lonlat_to_idx(lon: np.ndarray, lat: np.ndarray, xs: np.ndarray,
                   ys: np.ndarray):
    j1 = np.searchsorted(xs, lon, side="left")
    i1 = np.searchsorted(ys, lat, side="left")
    j0 = np.clip(j1 - 1, 0, len(xs) - 1)
    i0 = np.clip(i1 - 1, 0, len(ys) - 1)
    j1 = np.clip(j1, 0, len(xs) - 1)
    i1 = np.clip(i1, 0, len(ys) - 1)
    x0, x1 = xs[j0], xs[j1]
    y0, y1 = ys[i0], ys[i1]
    dx = np.where((x1 - x0) == 0, 1.0, (x1 - x0))
    dy = np.where((y1 - y0) == 0, 1.0, (y1 - y0))
    wx = np.clip((lon - x0) / dx, 0.0, 1.0)
    wy = np.clip((lat - y0) / dy, 0.0, 1.0)
    return i0, i1, j0, j1, wx, wy


def _bilinear_sample(depth: np.ndarray, i0, i1, j0, j1, wx, wy) -> np.ndarray:
    v00 = depth[i0, j0]
    v01 = depth[i0, j1]
    v10 = depth[i1, j0]
    v11 = depth[i1, j1]
    return ((1 - wx) * (1 - wy) * v00
            + wx * (1 - wy) * v01
            + (1 - wx) * wy * v10
            + wx * wy * v11)


def _interp_damage_ratio(depths: np.ndarray, depth_knots: np.ndarray,
                         dr_knots: np.ndarray) -> np.ndarray:
    depths_clamped = np.clip(depths, depth_knots[0], depth_knots[-1])
    return np.interp(depths_clamped, depth_knots, dr_knots)


def compute_eal(ds: xr.Dataset, assets: pd.DataFrame, vuln: dict,
                rates: pd.DataFrame):
    xs = ds["x"].values
    ys = ds["y"].values
    assets = assets.copy()

    i0, i1, j0, j1, wx, wy = _lonlat_to_idx(
        assets["lon"].to_numpy(), assets["lat"].to_numpy(), xs, ys
    )

    depth_m = np.zeros((len(assets), ds.sizes["event"]), dtype=np.float32)
    for e in range(ds.sizes["event"]):
        depth = ds["flood_depth"].isel(event=e).values
        depth_m[:, e] = _bilinear_sample(depth, i0, i1, j0, j1, wx, wy)

    depth_knots = np.asarray(vuln["depth_m"], dtype=float)
    dr_knots = np.asarray(vuln["damage_ratio"], dtype=float)

    dr = _interp_damage_ratio(depth_m, depth_knots, dr_knots)
    tiv = assets["tiv"].to_numpy()[:, None]
    loss = dr * tiv

    rate_map = rates.set_index("event_id")["annual_rate"].to_dict()
    rates_vec = np.array([rate_map[e] for e in range(ds.sizes["event"])],
                         dtype=float)[None, :]
    eal_asset = (loss * rates_vec).sum(axis=1)

    out_asset = assets[["asset_id", "country", "tiv"]].copy()
    out_asset["eal_usd"] = eal_asset

    out_country = (
        out_asset.groupby("country")
        .agg(asset_count=("asset_id", "count"),
             tiv_sum=("tiv", "sum"),
             eal_usd=("eal_usd", "sum"))
        .reset_index()
    )

    out_global = pd.DataFrame({
        "tiv_sum": [out_asset["tiv"].sum()],
        "eal_usd": [out_asset["eal_usd"].sum()],
    })

    return out_asset, out_country, out_global


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out", type=Path, default=Path("./results"))
    args = parser.parse_args()

    ds, assets, vuln, rates = _load_inputs(args.data_dir)
    out_asset, out_country, out_global = compute_eal(ds, assets, vuln, rates)

    args.out.mkdir(parents=True, exist_ok=True)
    out_asset.to_csv(args.out / "eal_by_asset.csv", index=False)
    out_country.to_csv(args.out / "eal_by_country.csv", index=False)
    out_global.to_csv(args.out / "eal_global.csv", index=False)

    assert not out_asset["asset_id"].duplicated().any()
    assert abs(out_country["eal_usd"].sum() - out_global["eal_usd"][0]) < 1e-6
    assert abs(out_country["tiv_sum"].sum() - out_global["tiv_sum"][0]) < 1e-6


if __name__ == "__main__":
    main()
