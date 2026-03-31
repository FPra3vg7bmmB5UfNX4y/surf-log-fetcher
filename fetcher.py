"""
Peniche Surf Log — CMEMS + Open-Meteo Data Fetcher
===================================================
Runs every 3 hours via Railway cron (or any scheduler).
Pulls CMEMS swell data + Open-Meteo wind forecast, upserts to Supabase.

Data sources:
  - CMEMS GLOBAL_ANALYSISFORECAST_WAV_001_027 (swell1, swell2, wind wave)
  - Open-Meteo forecast API (wind_speed, wind_direction, wind_gusts — no auth needed)
  - WorldTides API (tide height, phase) — pre-fetched annually into Supabase tides table

Requirements: see requirements.txt
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta

import copernicusmarine
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

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
    if not r.ok:
        log.error(f"  Supabase {r.status_code} on {table}: {r.text[:500]}")
    r.raise_for_status()
    log.info(f"  → upserted {len(rows)} rows into {table}")

# ─── CMEMS ───────────────────────────────────────────────────────────────────
CMEMS_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
CMEMS_VARS = [
    "VHM0_SW1",    # significant height of primary swell (m)
    "VTM01_SW1",   # mean period of primary swell (s)
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
    start = now - timedelta(hours=3)   # include one past row so conditions table always has a current entry
    end = now + timedelta(days=10)

    log.info("Fetching CMEMS wave forecast…")
    ds = copernicusmarine.open_dataset(
        dataset_id=CMEMS_DATASET,
        variables=CMEMS_VARS,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
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

# ─── Open-Meteo wind ─────────────────────────────────────────────────────────
def fetch_openmeteo_wind() -> dict[str, dict]:
    """
    Fetch 10-day hourly wind forecast from Open-Meteo (no auth required).
    Returns dict keyed by UTC ISO hour string → {wind_speed, wind_direction, wind_gusts}.
    """
    log.info("Fetching Open-Meteo wind forecast…")
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":        LAT_PT,
            "longitude":       LON_PT,
            "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "wind_speed_unit": "kmh",
            "forecast_days":   10,
            "timezone":        "UTC",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    hourly = data["hourly"]
    times  = hourly["time"]           # "2026-03-31T06:00" — no tz suffix
    speeds = hourly["wind_speed_10m"]
    dirs   = hourly["wind_direction_10m"]
    gusts  = hourly["wind_gusts_10m"]

    wind_map = {}
    for t, spd, d, g in zip(times, speeds, dirs, gusts):
        # Normalise to UTC ISO with +00:00 so it matches CMEMS valid_at keys
        key = t + ":00+00:00"
        wind_map[key] = {
            "wind_speed":     round(float(spd), 1) if spd is not None else None,
            "wind_direction": round(float(d), 1)   if d   is not None else None,
            "wind_gusts":     round(float(g), 1)   if g   is not None else None,
        }

    log.info(f"  → {len(wind_map)} Open-Meteo wind time steps")
    if wind_map:
        sample_key = next(iter(wind_map))
        log.info(f"  Sample wind entry: {sample_key} → {wind_map[sample_key]}")
    return wind_map

# ─── Tides from Supabase ─────────────────────────────────────────────────────
def fetch_tides_from_db(start_iso: str, end_iso: str) -> list[dict]:
    """
    Read pre-computed tide rows from the Supabase tides table.
    Fetches the window covering the CMEMS forecast range plus a 2h buffer
    so the nearest-match in merge_and_upsert always has data to pick from.
    """
    start_dt = (datetime.fromisoformat(start_iso) - timedelta(hours=2)).isoformat()
    end_dt   = (datetime.fromisoformat(end_iso)   + timedelta(hours=2)).isoformat()

    # URL-encode the + in timezone offset (+00:00 → %2B00:00) so PostgREST
    # doesn't interpret it as a space and reject the timestamp filter.
    start_enc = start_dt.replace("+", "%2B")
    end_enc   = end_dt.replace("+", "%2B")

    log.info("Reading tides from Supabase…")
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/tides"
        f"?select=valid_at,tide_height,tide_state,tide_phase,tide_next_type,tide_next_height"
        f"&valid_at=gte.{start_enc}&valid_at=lte.{end_enc}"
        f"&order=valid_at.asc&limit=5000",
        headers=sb_headers(),
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()
    log.info(f"  → {len(rows)} tide rows from DB")
    return rows

# ─── Sanitise ────────────────────────────────────────────────────────────────
def sanitise_row(row: dict) -> dict:
    """
    Clamp and round wind fields to fit numeric(5,1) (max 9999.9).
    Guards against occasional Open-Meteo outlier values.
    """
    out = dict(row)
    if out.get("wind_speed") is not None:
        out["wind_speed"]     = round(min(float(out["wind_speed"]),     200.0), 1)
    if out.get("wind_gusts") is not None:
        out["wind_gusts"]     = round(min(float(out["wind_gusts"]),     200.0), 1)
    if out.get("wind_direction") is not None:
        out["wind_direction"] = round(min(float(out["wind_direction"]), 360.0), 1)
    return out

# ─── Merge & upsert ──────────────────────────────────────────────────────────
def merge_and_upsert(cmems_rows: list[dict], wind_map: dict, tide_rows: list[dict]):
    """Merge CMEMS + Open-Meteo wind + tides by valid_at, upsert to conditions table."""

    # Build tide lookup: nearest tide to each timestamp
    def nearest_tide(ts_iso: str) -> dict:
        if not tide_rows:
            return {}
        target = datetime.fromisoformat(ts_iso).timestamp()
        best = min(tide_rows, key=lambda r: abs(datetime.fromisoformat(r["valid_at"]).timestamp() - target))
        return {k: best[k] for k in ["tide_height","tide_state","tide_phase","tide_next_type","tide_next_height"]}

    merged = []
    wind_hits = 0
    for row in cmems_rows:
        ts = row["valid_at"]  # e.g. "2026-03-31T06:00:00+00:00"

        # Match to Open-Meteo hour: CMEMS steps are 3h, Open-Meteo is 1h,
        # so exact key hit is expected; fall back to truncating minutes/seconds.
        wind = wind_map.get(ts) or wind_map.get(ts[:13] + ":00:00+00:00", {})
        if wind:
            wind_hits += 1

        tide = nearest_tide(ts)
        merged.append({**row, **wind, **tide})

    log.info(f"  Wind matched {wind_hits}/{len(cmems_rows)} CMEMS rows")
    if merged:
        first = merged[0]
        log.info(f"  Sample merged row wind fields: speed={first.get('wind_speed')}, "
                 f"dir={first.get('wind_direction')}, gusts={first.get('wind_gusts')}")

    merged = [sanitise_row(r) for r in merged]

    # Ensure every row has identical keys (PGRST102 requires uniform shape).
    # Take the union of all keys and fill gaps with None.
    all_keys = set().union(*[r.keys() for r in merged])
    merged = [{k: r.get(k) for k in all_keys} for r in merged]

    log.info(f"Upserting {len(merged)} rows to Supabase conditions table…")
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
        wind_map = fetch_openmeteo_wind()
    except Exception as e:
        log.error(f"Open-Meteo fetch failed: {e}")
        # Not critical — conditions will be upserted without wind columns

    tide_rows = []
    if cmems_rows:
        try:
            tide_rows = fetch_tides_from_db(cmems_rows[0]["valid_at"], cmems_rows[-1]["valid_at"])
        except Exception as e:
            log.error(f"Tides DB fetch failed: {e}")
            # Not critical — conditions will be upserted without tide columns

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
