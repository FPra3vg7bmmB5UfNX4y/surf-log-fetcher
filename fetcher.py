"""
Peniche Surf Log — CMEMS + ECMWF Data Fetcher
==============================================
Runs every 3 hours via Railway cron (or any scheduler).
Pulls CMEMS swell data + ECMWF wind + WorldTides, upserts to Supabase.

Data sources:
  - CMEMS GLOBAL_ANALYSISFORECAST_WAV_001_027 (swell1, swell2, wind wave)
  - ECMWF Open Data (wind speed, direction, gusts)
  - WorldTides API (tide height, phase) — optional

Requirements: see requirements.txt
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta

import copernicusmarine
import numpy as np
import requests
from ecmwf.opendata import Client as ECMWFClient
import cfgrib
import xarray as xr
import tempfile

# xarray ≥ 2024.x changed the default compat for merge from 'no_conflicts' to
# 'override'. cfgrib still uses the old default and emits a FutureWarning that
# becomes a hard error in some xarray builds. Opt in to the new behaviour now.
xr.set_options(use_new_combine_kwarg_defaults=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
# These come from Railway environment variables (or your .env file locally)

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
WORLDTIDES_KEY = os.environ.get("WORLDTIDES_KEY", "")  # optional

CMEMS_USER = os.environ["COPERNICUSMARINE_SERVICE_USERNAME"]
CMEMS_PASS = os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"]

# Peniche area bounding box — slightly wider for interpolation accuracy
LAT_PT  = 39.3557
LON_PT  = -9.3808
LAT_MIN, LAT_MAX = 38.5, 40.5
LON_MIN, LON_MAX = -10.5, -8.0

# ─── Supabase helpers ─────────────────────────────────────────────────────────
def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

def sb_upsert(table: str, rows: list[dict]):
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}?on_conflict=valid_at",
        headers={**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows,
        timeout=30,
    )
    r.raise_for_status()
    log.info(f"  → upserted {len(rows)} rows into {table}")

# ─── CMEMS ───────────────────────────────────────────────────────────────────
CMEMS_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
CMEMS_VARS = [
    "VHM0_SW1",    # significant height of primary swell (m)
    "VTM01_SW1",   # mean period of primary swell (s) — VTPK_SW1 not in this dataset
    "VMDR_SW1",    # mean direction of primary swell (degrees)
    "VHM0_SW2",    # significant height of secondary swell (m)
    "VTM01_SW2",   # mean period of secondary swell (s)
    "VMDR_SW2",
    "VHM0_WW",    # significant height of wind waves (m)
    "VTM01_WW",   # mean wave period of wind waves (s)
    "VHM0",       # combined significant wave height (m)
]

def fetch_cmems() -> list[dict]:
    """Pull next 10 days of CMEMS wave forecast, return list of dicts."""
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=10)

    log.info("Fetching CMEMS wave forecast…")
    ds = copernicusmarine.open_dataset(
        dataset_id=CMEMS_DATASET,
        variables=CMEMS_VARS,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=now.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        username=CMEMS_USER,
        password=CMEMS_PASS,
    )

    # Nearest grid point to Peniche
    pt = ds.sel(longitude=LON_PT, latitude=LAT_PT, method="nearest")
    df = pt.to_dataframe().reset_index()

    rows = []
    for _, r in df.iterrows():
        valid_at = r["time"]
        if hasattr(valid_at, "to_pydatetime"):
            valid_at = valid_at.to_pydatetime()
        if valid_at.tzinfo is None:
            valid_at = valid_at.replace(tzinfo=timezone.utc)

        def v(col):
            val = r.get(col)
            if val is None or (hasattr(val, '__float__') and np.isnan(float(val))):
                return None
            return round(float(val), 3)

        # Swell energy proxy: Hs² × T
        e1 = round(v("VHM0_SW1")**2 * v("VTM01_SW1"), 3) if v("VHM0_SW1") and v("VTM01_SW1") else None
        e2 = round(v("VHM0_SW2")**2 * v("VTM01_SW2"), 3) if v("VHM0_SW2") and v("VTM01_SW2") else None

        rows.append({
            "valid_at": valid_at.isoformat(),
            "fetched_at": now.isoformat(),
            "source": "cmems",
            "swell1_height":    v("VHM0_SW1"),
            "swell1_period":    v("VTM01_SW1"),
            "swell1_direction": v("VMDR_SW1"),
            "swell1_energy":    e1,
            "swell2_height":    v("VHM0_SW2"),
            "swell2_period":    v("VTM01_SW2"),
            "swell2_direction": v("VMDR_SW2"),
            "swell2_energy":    e2,
            "wind_wave_height": v("VHM0_WW"),
            "wind_wave_period": v("VTM01_WW"),
            "wave_height_total":v("VHM0"),
        })

    log.info(f"  → {len(rows)} CMEMS time steps from {rows[0]['valid_at']} to {rows[-1]['valid_at']}")
    return rows

# ─── ECMWF wind ──────────────────────────────────────────────────────────────
def fetch_ecmwf_wind() -> dict[str, dict]:
    """
    Pull ECMWF Open Data wind forecast.
    Returns dict keyed by ISO timestamp → {wind_speed, wind_direction, wind_gusts}
    """
    log.info("Fetching ECMWF Open Data wind…")
    client = ECMWFClient()
    wind_by_time = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "ecmwf_wind.grib2")
        # ECMWF: 00Z/12Z runs go to 240h; 06Z/18Z only to 90h.
        # Use 3h steps for 0-90h, 6h steps for 96-240h — both resolutions always available.
        short_steps = list(range(0, 91, 3))
        long_steps  = list(range(96, 241, 6))
        client.retrieve(
            type="fc",
            step=short_steps + long_steps,
            param=["10u", "10v", "fg10"],  # u-wind, v-wind, gust at 10m
            target=outfile,
        )

        datasets = cfgrib.open_datasets(outfile)
        now = datetime.now(timezone.utc)

        for ds_w in datasets:
            try:
                df = ds_w.to_dataframe().reset_index()
                for _, row in df.iterrows():
                    # Build valid time
                    step = row.get("step")
                    ref = row.get("time")
                    if step is None or ref is None:
                        continue
                    if hasattr(step, "total_seconds"):
                        step_h = step.total_seconds() / 3600
                    else:
                        step_h = float(step)
                    valid = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=step_h)
                    key = valid.replace(minute=0, second=0, microsecond=0).isoformat()

                    if key not in wind_by_time:
                        wind_by_time[key] = {}
                    for col in ["u10","v10","i10fg"]:
                        val = row.get(col)
                        if val is not None and not np.isnan(float(val)):
                            wind_by_time[key][col] = float(val)
            except Exception as e:
                log.warning(f"  ECMWF parse warning: {e}")

    # Convert u/v to speed + direction, convert m/s → km/h
    result = {}
    for ts, w in wind_by_time.items():
        u = w.get("u10")
        v_val = w.get("v10")
        gust = w.get("i10fg")
        if u is None or v_val is None:
            continue
        speed = round(np.sqrt(u**2 + v_val**2) * 3.6, 1)
        direction = round((270 - np.degrees(np.arctan2(v_val, u))) % 360, 1)
        result[ts] = {
            "wind_speed": speed,
            "wind_direction": direction,
            "wind_gusts": round(gust * 3.6, 1) if gust else None,
        }

    log.info(f"  → {len(result)} ECMWF wind time steps")
    return result

# ─── WorldTides ──────────────────────────────────────────────────────────────
def fetch_tides() -> list[dict]:
    """Fetch 2 days of tide predictions from WorldTides, return list of dicts."""
    if not WORLDTIDES_KEY:
        log.info("No WorldTides key set — skipping tides")
        return []

    log.info("Fetching WorldTides…")
    r = requests.get(
        "https://www.worldtides.info/api/v3",
        params={
            "heights": True, "extremes": True,
            "lat": LAT_PT, "lon": LON_PT,
            "days": 2, "step": 900,
            "datum": "LAT",   # heights above Lowest Astronomical Tide (≥ 0, matches tide tables)
            "key": WORLDTIDES_KEY,
        },
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    if d.get("status") != 200:
        raise RuntimeError(f"WorldTides error: {d.get('error')}")

    heights = d.get("heights", [])
    extremes = d.get("extremes", [])
    now_ts = datetime.now(timezone.utc).timestamp()

    tide_rows = []
    for h in heights:
        ts = h["dt"]
        valid = datetime.fromtimestamp(ts, tz=timezone.utc)

        # State: rising or falling
        idx = heights.index(h)
        state = None
        if 0 < idx < len(heights)-1:
            state = "rising" if heights[idx+1]["height"] > heights[idx-1]["height"] else "falling"

        # Phase: low / mid / high (within 1.5h of an extreme)
        phase = "mid"
        for ex in extremes:
            if abs(ex["dt"] - ts) < 5400:
                phase = ex["type"].lower()
                break

        # Next extreme
        next_ex = next((e for e in extremes if e["dt"] > ts), None)

        tide_rows.append({
            "valid_at": valid.isoformat(),
            "tide_height":    round(h["height"], 3),
            "tide_state":     state,
            "tide_phase":     phase,
            "tide_next_type": next_ex["type"] if next_ex else None,
            "tide_next_height": round(next_ex["height"], 3) if next_ex else None,
        })

    log.info(f"  → {len(tide_rows)} tide data points")
    return tide_rows

# ─── Merge & upsert ──────────────────────────────────────────────────────────
def merge_and_upsert(cmems_rows: list[dict], wind_map: dict, tide_rows: list[dict]):
    """Merge CMEMS + ECMWF wind + tides by valid_at, upsert to conditions table."""

    # Build tide lookup: nearest tide height to each timestamp
    def nearest_tide(ts_iso: str) -> dict:
        if not tide_rows:
            return {}
        target = datetime.fromisoformat(ts_iso).timestamp()
        best = min(tide_rows, key=lambda r: abs(datetime.fromisoformat(r["valid_at"]).timestamp() - target))
        return {k: best[k] for k in ["tide_height","tide_state","tide_phase","tide_next_type","tide_next_height"]}

    merged = []
    for row in cmems_rows:
        ts = row["valid_at"]
        # Snap to nearest 3h for ECMWF lookup
        ts_snap = ts[:14] + "00:00+00:00" if "+" in ts else ts[:14] + "00:00"
        wind = wind_map.get(ts_snap, wind_map.get(ts[:13]+":00:00+00:00", {}))
        tide = nearest_tide(ts)

        merged.append({
            **row,
            **wind,
            **tide,
        })

    log.info(f"Upserting {len(merged)} rows to Supabase conditions table…")
    # Upsert in batches of 50 to stay within Supabase request limits
    for i in range(0, len(merged), 50):
        sb_upsert("conditions", merged[i:i+50])

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Peniche Surf Log — Fetcher starting")
    log.info(f"Run time: {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 60)

    errors = []

    try:
        cmems_rows = fetch_cmems()
    except Exception as e:
        log.error(f"CMEMS fetch failed: {e}")
        errors.append(f"CMEMS: {e}")
        cmems_rows = []

    wind_map = {}
    try:
        wind_map = fetch_ecmwf_wind()
    except Exception as e:
        log.error(f"ECMWF fetch failed: {e}")
        errors.append(f"ECMWF: {e}")

    tide_rows = []
    try:
        tide_rows = fetch_tides()
    except Exception as e:
        log.error(f"WorldTides fetch failed: {e}")
        # Not critical — log and continue

    if cmems_rows:
        try:
            merge_and_upsert(cmems_rows, wind_map, tide_rows)
        except Exception as e:
            log.error(f"Supabase upsert failed: {e}")
            errors.append(f"Supabase: {e}")

    if errors:
        log.error("Completed with errors: " + "; ".join(errors))
        sys.exit(1)

    log.info("✓ Fetcher completed successfully")

if __name__ == "__main__":
    main()
