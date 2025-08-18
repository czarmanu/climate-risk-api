# *****************************************************************************
# app.py
# *****************************************************************************

# Purpose: FastAPI service exposing risk analytics computed by the pipeline

# Orchestrates:
# - Read project-root data as needed for on-demand single-asset queries
# - Read results CSVs produced by the pipeline
# - Serve REST endpoints for country-level and ad-hoc asset EAL

# Author(s):
# Manu Tom, 2025-

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP = FastAPI(title="Risk API")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class AssetIn(BaseModel):
    lon: float = Field(..., ge=-180.0, le=180.0)
    lat: float = Field(..., ge=-90.0, le=90.0)
    tiv: float = Field(..., gt=0.0)


def _load_on_demand():
    ds = xr.open_dataset(DATA_DIR / "hazard.nc", chunks={"event": 1, "y": 60})
    with open(DATA_DIR / "vulnerability.json", "r", encoding="utf-8") as f:
        vuln = json.load(f)
    rates = pd.read_csv(DATA_DIR / "event_rates.csv")
    return ds, vuln, rates


def _lonlat_to_idx(lon, lat, xs, ys):
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


def _bilinear_sample(depth, i0, i1, j0, j1, wx, wy):
    v00 = depth[i0, j0]
    v01 = depth[i0, j1]
    v10 = depth[i1, j0]
    v11 = depth[i1, j1]
    return ((1 - wx) * (1 - wy) * v00
            + wx * (1 - wy) * v01
            + (1 - wx) * wy * v10
            + wx * wy * v11)


@APP.get("/risk/country/{iso3}")
def risk_by_country(iso3: str):
    path = RESULTS_DIR / "eal_by_country.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Results not found.")
    df = pd.read_csv(path)
    row = df[df["country"] == iso3.upper()]
    if row.empty:
        raise HTTPException(status_code=404, detail="Country not found.")
    rec = row.iloc[0].to_dict()
    return {
        "country": iso3.upper(),
        "asset_count": int(rec["asset_count"]),
        "tiv_sum": float(rec["tiv_sum"]),
        "eal_usd": float(rec["eal_usd"]),
    }


@APP.post("/risk/asset")
def risk_for_asset(asset: AssetIn):
    ds, vuln, rates = _load_on_demand()
    xs = ds["x"].values
    ys = ds["y"].values

    i0, i1, j0, j1, wx, wy = _lonlat_to_idx(
        np.array([asset.lon]), np.array([asset.lat]), xs, ys
    )

    ne = ds.sizes["event"]
    depths = np.zeros((ne,), dtype=np.float32)
    for e in range(ne):
        depth = ds["flood_depth"].isel(event=e).values
        depths[e] = _bilinear_sample(depth, i0, i1, j0, j1, wx, wy)[0]

    depth_knots = np.asarray(vuln["depth_m"], dtype=float)
    dr_knots = np.asarray(vuln["damage_ratio"], dtype=float)
    depths_clamped = np.clip(depths, depth_knots[0], depth_knots[-1])
    dr = np.interp(depths_clamped, depth_knots, dr_knots)

    rate_map = rates.set_index("event_id")["annual_rate"].to_dict()
    rates_vec = np.array([rate_map[e] for e in range(ne)], dtype=float)

    loss = dr * asset.tiv
    eal = float(np.sum(loss * rates_vec))
    return {"eal_usd": eal}
