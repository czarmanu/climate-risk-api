![CI](https://github.com/czarmanu/climate-risk-api/actions/workflows/github_actions_CI.yml/badge.svg)  
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


# Hazard → Exposure → Risk (EAL)

## Quick start
```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run pipeline (writes results to ./results)
python src/pipeline.py --data-dir ./data --out ./results

# 3. Start API server
uvicorn app:APP --app-dir src --reload --host 0.0.0.0 --port 8000

# 4. Load API (in a browser, upon successful server setup)
http://localhost:8000/docs
```

## Example queries (CMD)
```bash
# Country-level EAL
curl -s "http://localhost:8000/risk/country/CHE" | jq
# Sample results
{
  "country": "CHE",
  "asset_count": 105,
  "tiv_sum": 25275335.05,
  "eal_usd": 1253154.10210936
}

# Single asset EAL
curl -s -X POST "http://localhost:8000/risk/asset" \
  -H "Content-Type: application/json" \
  -d '{"lon":8.55,"lat":47.37,"tiv":1000000}' | jq
# Sample results
{
  "eal_usd": 44560.049396665396
}
```

# Test
```bash
PYTHONPATH=src pytest -q
```

## Endpoints
- `GET /risk/country/{iso3}` → returns `asset_count, tiv_sum, eal_usd`
- `POST /risk/asset` with JSON `{"lon": 8.5, "lat": 47.4, "tiv": 1e6}` → returns `{"eal_usd": ...}`

## Notes
- Synthetic data are in `./data`:
  - `hazard.nc`: 20 events, 60×120 grid in EPSG:4326
  - `assets.csv`: 500 random assets
  - `vulnerability.json`: depth→damage ratio
  - `event_rates.csv`: annual rates per event
- Results CSVs written to `./results`.
- Code is vectorized; bilinear sampling is used; EAL = Σ(loss × rate).
