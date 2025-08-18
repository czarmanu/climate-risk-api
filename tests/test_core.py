# *****************************************************************************
# test_core.py
# *****************************************************************************
# Purpose:
# Unit test for compute_eal(): verifies vulnerability interpolation,
# bilinear hazard sampling, and correct aggregation of Expected Annual Loss
# (asset → country → global) on synthetic data.
#
# Author(s):
# Manu Tom, 2025-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from pipeline import compute_eal


def test_vuln_interp_and_aggregation(tmp_path: Path) -> None:
    ys = np.array([0.0, 1.0])
    xs = np.array([0.0, 1.0])
    depth = np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype="float32")
    ds = xr.Dataset(
        {"flood_depth": (("event", "y", "x"), depth)},
        coords={"event": [0], "y": ys, "x": xs},
    )

    assets = pd.DataFrame({
        "asset_id": ["A"],
        "lon": [0.5],
        "lat": [0.5],
        "country": ["CHE"],
        "tiv": [1000.0],
    })

    vuln = {"depth_m": [0.0, 1.0], "damage_ratio": [0.0, 1.0]}
    rates = pd.DataFrame({"event_id": [0], "annual_rate": [0.1]})

    out_asset, out_country, out_global = compute_eal(ds, assets, vuln, rates)

    assert out_asset.shape[0] == 1
    assert abs(out_asset["eal_usd"].iloc[0] - 50.0) < 1e-6
    assert abs(out_country["eal_usd"].sum() - 50.0) < 1e-6
    assert abs(out_global["eal_usd"].iloc[0] - 50.0) < 1e-6
