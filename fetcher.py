"""
Peniche Surf Log — CMEMS + Open-Meteo + IPMA Data Fetcher
==========================================================
Runs every 3 hours via Railway cron (or any scheduler).
Pulls CMEMS swell data + Open-Meteo wind forecast + IPMA station obs,
upserts to Supabase.

Data sources:
  - CMEMS GLOBAL_ANALYSISFORECAST_WAV_001_027 (swell1, swell2, wind wave)
  - Open-Meteo forecast API (wind_speed, wind_direction, wind_gusts — no auth needed)
  - IPMA station 1200535 Cabo Carvoeiro (wind_speed_obs, wind_dir_obs — observed, recent hours only)
  - WorldTides API (tide height, phase) — pre-fetched annually into Supabase tides table

Requirements: see requirements.txt
"""

import os
import sys
import logging
import time
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
    "VMDR_WW",    # mean direction of wind waves (degrees)
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
            "wind_wave_height":     v("VHM0_WW"),
            "wind_wave_period":     v("VTM01_WW"),
            "wind_wave_direction":  v("VMDR_WW"),
            "wave_height_total":v("VHM0"),
        })

    log.info(f"  → {len(rows)} CMEMS time steps from {rows[0]['valid_at']} to {rows[-1]['valid_at']}")
    return rows

# ─── IPMA station observations ───────────────────────────────────────────────
# Station 1200535 = Cabo Carvoeiro (nearest coastal station to Peniche, ~10 km N)
IPMA_STATION_ID = "1200535"

# IPMA forecast ddVento string labels → degrees
IPMA_FORECAST_DIR_DEG: dict[str, float] = {
    "N":  0.0,  "NE": 45.0,  "E":  90.0,  "SE": 135.0,
    "S": 180.0,  "SW": 225.0,  "W": 270.0,  "NW": 315.0,
    # Portuguese compass aliases used in some IPMA responses
    "SO": 225.0,  "O": 270.0,  "NO": 315.0,
}

# IPMA idDireccVento codes → degrees
# 0 = calm/variable (no meaningful direction), 1–8 = N/NE/E/SE/S/SW/W/NW
# 9 = N (confirmed: most common code in station data, correlates with northerly forecasts)
IPMA_DIR_DEG: dict[int, float | None] = {
    0: None,    # calm / variable
    1: 0.0,     # N
    2: 45.0,    # NE
    3: 90.0,    # E
    4: 135.0,   # SE
    5: 180.0,   # S
    6: 225.0,   # SW
    7: 270.0,   # W
    8: 315.0,   # NW
    9: 0.0,     # N
}

def fetch_ipma_obs() -> dict[str, dict]:
    """
    Fetch recent IPMA station observations for Cabo Carvoeiro (station 1200535).
    Returns dict keyed by UTC ISO timestamp → {wind_speed_obs, wind_dir_obs, wind_gusts_obs}.
    IPMA publishes only the last ~3 observation hours so this only populates
    the most recent rows in the conditions table.
    """
    log.info("Fetching IPMA station observations…")
    r = requests.get(
        "https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json",
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    obs_map = {}
    for ts_str, stations in data.items():
        station = stations.get(IPMA_STATION_ID)
        if not station:
            continue
        # IPMA timestamps are "YYYY-MM-DDTHH:MM" — no tz suffix; treat as UTC
        try:
            dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except ValueError:
            log.warning(f"  IPMA: unrecognised timestamp {ts_str!r}, skipping")
            continue

        speed_raw = station.get("intensidadeVentoKM")  # already km/h
        dir_code  = station.get("idDireccVento")

        obs_map[dt.isoformat()] = {
            "wind_speed_obs": round(float(speed_raw), 1) if speed_raw is not None else None,
            "wind_dir_obs":   IPMA_DIR_DEG.get(int(dir_code)) if dir_code is not None else None,
            "wind_gusts_obs": None,  # IPMA hourly obs don't include a gust field
        }

    log.info(f"  → {len(obs_map)} IPMA observation time steps")
    return obs_map

# ─── station_observations ────────────────────────────────────────────────────
def _nm(station: dict, key: str, decimals: int = 1):
    """Get a numeric field from a station dict; convert -99 / -99.0 → None."""
    raw = station.get(key)
    if raw is None:
        return None
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    return None if f == -99.0 else round(f, decimals)

def fetch_and_store_station_obs() -> int:
    """
    Fetch IPMA observations.json, keep only station 1200535, convert -99/-99.0
    to NULL in every field, store raw wind_dir_code and computed wind_dir_deg,
    then upsert into station_observations on (station_id, observed_at).
    Returns number of rows upserted.
    """
    log.info("Fetching IPMA observations → station_observations table…")
    resp = requests.get(
        "https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json",
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    now = datetime.now(timezone.utc)
    rows = []
    for ts_str, stations in data.items():
        # Skip null/empty timestamp buckets
        if not stations:
            continue
        station = stations.get(IPMA_STATION_ID)
        # Skip null station entries
        if not station:
            continue

        try:
            dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except ValueError:
            log.warning(f"  station_obs: bad timestamp {ts_str!r}, skipping")
            continue

        # Wind direction code: apply -99 → None, keep 0 as valid code
        dir_raw = station.get("idDireccVento")
        dir_code = None
        if dir_raw is not None:
            try:
                dc = int(dir_raw)
                dir_code = None if dc == -99 else dc
            except (TypeError, ValueError):
                pass

        # wind_dir_deg: use shared IPMA_DIR_DEG (0 → NULL, 1-8 → compass, 9 → N)
        wind_dir_deg = IPMA_DIR_DEG.get(dir_code) if dir_code is not None else None

        rows.append({
            "station_id":     IPMA_STATION_ID,
            "observed_at":    dt.isoformat(),
            "wind_speed_kmh": _nm(station, "intensidadeVentoKM"),
            "wind_dir_code":  dir_code,
            "wind_dir_deg":   wind_dir_deg,
            "temp_c":         _nm(station, "temperatura"),
            "humidity_pct":   _nm(station, "humidade"),
            "pressure_hpa":   _nm(station, "pressao"),
            "radiation":      _nm(station, "radiacao"),
            "fetched_at":     now.isoformat(),
        })

    if not rows:
        log.info("  station_observations: no data for station 1200535")
        return 0

    total = 0
    for i in range(0, len(rows), 50):
        chunk = rows[i:i+50]
        r2 = requests.post(
            f"{SUPABASE_URL}/rest/v1/station_observations?on_conflict=station_id,observed_at",
            headers={**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=chunk,
            timeout=30,
        )
        if not r2.ok:
            log.error(f"  Supabase {r2.status_code} on station_observations: {r2.text[:500]}")
        r2.raise_for_status()
        total += len(chunk)

    log.info(f"  station_observations: upserted {total} rows for station {IPMA_STATION_ID}")
    return total

# ─── Open-Meteo wind ─────────────────────────────────────────────────────────
_OPENMETEO_BACKOFF = [20, 60]   # seconds before attempt 2, then attempt 3

def fetch_openmeteo_wind() -> dict[str, dict]:
    """
    Fetch 10-day hourly wind forecast from Open-Meteo (no auth required).
    Returns dict keyed by UTC ISO hour string → {wind_speed, wind_direction, wind_gusts}.
    Retries up to 3 times on HTTP 429 / 5xx, honouring Retry-After if present.
    """
    log.info("Fetching Open-Meteo wind forecast…")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        LAT_PT,
        "longitude":       LON_PT,
        "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "kmh",
        "forecast_days":   10,
        "timezone":        "UTC",
    }

    resp = None
    for attempt in range(1, len(_OPENMETEO_BACKOFF) + 2):   # attempts 1, 2, 3
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt > len(_OPENMETEO_BACKOFF):
                raise
            wait = _OPENMETEO_BACKOFF[attempt - 1]
            log.warning(f"  Open-Meteo attempt {attempt} network error: {exc}; retrying in {wait}s")
            time.sleep(wait)
            continue

        if resp.ok:
            break

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt > len(_OPENMETEO_BACKOFF):
                resp.raise_for_status()   # exhausted retries
            ra = resp.headers.get("Retry-After", "")
            wait = min(int(ra), 90) if ra.isdigit() else _OPENMETEO_BACKOFF[attempt - 1]
            log.warning(f"  Open-Meteo attempt {attempt} → HTTP {resp.status_code}; retrying in {wait}s")
            time.sleep(wait)
        else:
            resp.raise_for_status()   # non-retryable 4xx — fail immediately

    data = resp.json()

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

# ─── wind_forecast_history ────────────────────────────────────────────────────
def store_wind_forecast_history(wind_map: dict, issued_at: datetime) -> int:
    """
    Insert the next 48 h of Open-Meteo wind forecast into wind_forecast_history.

    issued_at is the **fetch time** — Open-Meteo does not expose its model-run
    time in the API response, so we use the time this script fetched the data.

    Uses ON CONFLICT (issued_at, valid_at) DO NOTHING so re-runs are safe.
    Returns number of rows attempted (some may be no-ops due to the conflict rule).
    """
    log.info("Writing wind forecast history…")
    cutoff = issued_at + timedelta(hours=48)

    rows = []
    for ts_str, w in wind_map.items():
        try:
            valid_at = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if valid_at.tzinfo is None:
            valid_at = valid_at.replace(tzinfo=timezone.utc)
        if valid_at > cutoff:
            continue

        lead_h = round((valid_at.timestamp() - issued_at.timestamp()) / 3600, 1)

        rows.append({
            "issued_at":      issued_at.isoformat(),
            "valid_at":       valid_at.isoformat(),
            "lead_hours":     lead_h,
            "wind_speed_kmh": w.get("wind_speed"),
            "wind_dir_deg":   w.get("wind_direction"),
            "wind_gusts_kmh": w.get("wind_gusts"),
            "source":         "open-meteo",
        })

    if not rows:
        log.info("  wind_forecast_history: no rows to insert")
        return 0

    total = 0
    for i in range(0, len(rows), 50):
        chunk = rows[i:i+50]
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/wind_forecast_history?on_conflict=source,issued_at,valid_at",
            headers={**sb_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
            json=chunk,
            timeout=30,
        )
        if not r.ok:
            log.error(f"  Supabase {r.status_code} on wind_forecast_history: {r.text[:500]}")
        r.raise_for_status()
        total += len(chunk)

    log.info(
        f"  wind_forecast_history: attempted {total} rows "
        f"(issued_at={issued_at.isoformat()[:16]}Z, ON CONFLICT DO NOTHING)"
    )
    return total

# ─── IPMA wind forecast ──────────────────────────────────────────────────────
def fetch_and_store_ipma_forecast() -> int:
    """
    Fetch IPMA hourly / 3-hourly wind forecast for Peniche (globalIdLocal 1101400).

    idPeriodo values in the feed:
      1  → hourly entries  (ffVento present)
      3  → 3-hourly entries (ffVento present)
      24 → daily summary   (no ffVento — skipped)

    Inserts into wind_forecast_history with source='ipma'.
    Uses ON CONFLICT (source, issued_at, valid_at) DO NOTHING.
    Returns number of rows attempted.
    """
    log.info("Fetching IPMA wind forecast → wind_forecast_history…")
    resp = requests.get(
        "https://api.ipma.pt/public-data/forecast/aggregate/1101400.json",
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    rows = []
    unknown_dirs: set[str] = set()
    for entry in data:
        if entry.get("idPeriodo") not in (1, 3):
            continue

        try:
            valid_at  = datetime.fromisoformat(entry["dataPrev"]).replace(tzinfo=timezone.utc)
            issued_at = datetime.fromisoformat(entry["dataUpdate"]).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue

        # Wind speed — stored as a string; -99 / "-99.0" → NULL
        ff_raw = entry.get("ffVento")
        wind_speed = None
        if ff_raw is not None:
            try:
                f = float(ff_raw)
                wind_speed = None if f == -99.0 else round(f, 1)
            except (TypeError, ValueError):
                pass

        # Wind direction — string compass label
        dd = entry.get("ddVento")
        if dd is None:
            wind_dir = None
        elif dd in IPMA_FORECAST_DIR_DEG:
            wind_dir = IPMA_FORECAST_DIR_DEG[dd]
        else:
            if dd not in unknown_dirs:
                log.warning(f"  IPMA forecast: unknown ddVento {dd!r} → NULL")
                unknown_dirs.add(dd)
            wind_dir = None

        lead_h = round((valid_at.timestamp() - issued_at.timestamp()) / 3600, 1)

        rows.append({
            "source":         "ipma",
            "issued_at":      issued_at.isoformat(),
            "valid_at":       valid_at.isoformat(),
            "lead_hours":     lead_h,
            "wind_speed_kmh": wind_speed,
            "wind_dir_deg":   wind_dir,
            "wind_gusts_kmh": None,
        })

    if not rows:
        log.info("  IPMA forecast: no rows to insert")
        return 0

    total = 0
    for i in range(0, len(rows), 50):
        chunk = rows[i:i+50]
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/wind_forecast_history?on_conflict=source,issued_at,valid_at",
            headers={**sb_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
            json=chunk,
            timeout=30,
        )
        if not r.ok:
            log.error(f"  Supabase {r.status_code} on wind_forecast_history (ipma): {r.text[:500]}")
        r.raise_for_status()
        total += len(chunk)

    log.info(f"  IPMA forecast: attempted {total} rows into wind_forecast_history")
    return total

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
    if out.get("wind_speed_obs") is not None:
        out["wind_speed_obs"] = round(min(float(out["wind_speed_obs"]), 200.0), 1)
    if out.get("wind_gusts_obs") is not None:
        out["wind_gusts_obs"] = round(min(float(out["wind_gusts_obs"]), 200.0), 1)
    if out.get("wind_dir_obs") is not None:
        out["wind_dir_obs"]   = round(min(float(out["wind_dir_obs"]),   360.0), 1)
    return out

# ─── Merge & upsert ──────────────────────────────────────────────────────────
def merge_and_upsert(cmems_rows: list[dict], wind_map: dict, tide_rows: list[dict],
                     ipma_obs: dict):
    """Merge CMEMS + Open-Meteo wind + IPMA obs + tides by valid_at, upsert to conditions table."""

    # Build tide lookup: nearest tide to each timestamp
    def nearest_tide(ts_iso: str) -> dict:
        if not tide_rows:
            return {}
        target = datetime.fromisoformat(ts_iso).timestamp()
        best = min(tide_rows, key=lambda r: abs(datetime.fromisoformat(r["valid_at"]).timestamp() - target))
        return {k: best[k] for k in ["tide_height","tide_state","tide_phase","tide_next_type","tide_next_height"]}

    def nearest_ipma(ts_iso: str) -> dict:
        """Return IPMA obs for ts_iso only if within 1.5 h; else empty (no obs for future rows)."""
        if not ipma_obs:
            return {}
        target = datetime.fromisoformat(ts_iso).timestamp()
        best_key = min(ipma_obs.keys(), key=lambda k: abs(datetime.fromisoformat(k).timestamp() - target))
        delta_h = abs(datetime.fromisoformat(best_key).timestamp() - target) / 3600
        return ipma_obs[best_key] if delta_h <= 1.5 else {}

    merged = []
    wind_hits = 0
    ipma_hits = 0
    for row in cmems_rows:
        ts = row["valid_at"]  # e.g. "2026-03-31T06:00:00+00:00"

        # Match to Open-Meteo hour: CMEMS steps are 3h, Open-Meteo is 1h,
        # so exact key hit is expected; fall back to truncating minutes/seconds.
        wind = wind_map.get(ts) or wind_map.get(ts[:13] + ":00:00+00:00", {})
        if wind:
            wind_hits += 1

        obs = nearest_ipma(ts)
        if obs:
            ipma_hits += 1

        tide = nearest_tide(ts)
        merged.append({**row, **wind, **obs, **tide})

    log.info(f"  Wind matched {wind_hits}/{len(cmems_rows)} CMEMS rows")
    log.info(f"  IPMA obs matched {ipma_hits}/{len(cmems_rows)} CMEMS rows")
    if merged:
        first = merged[0]
        log.info(f"  Sample merged row wind fields: speed={first.get('wind_speed')}, "
                 f"dir={first.get('wind_direction')}, gusts={first.get('wind_gusts')}, "
                 f"obs_speed={first.get('wind_speed_obs')}, obs_dir={first.get('wind_dir_obs')}")

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
    now = datetime.now(timezone.utc)

    log.info("=" * 60)
    log.info("Peniche Surf Log — Fetcher starting")
    log.info(f"Run time: {now.isoformat()}")
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

    if not wind_map:
        log.warning(
            "Open-Meteo wind data unavailable — wind_speed / wind_direction / wind_gusts "
            "will be omitted from the conditions upsert; existing DB values are preserved"
        )

    ipma_obs = {}
    try:
        ipma_obs = fetch_ipma_obs()
    except Exception as e:
        log.error(f"IPMA fetch failed: {e}")
        # Not critical — conditions will be upserted without observed wind columns

    tide_rows = []
    if cmems_rows:
        try:
            tide_rows = fetch_tides_from_db(cmems_rows[0]["valid_at"], cmems_rows[-1]["valid_at"])
        except Exception as e:
            log.error(f"Tides DB fetch failed: {e}")
            # Not critical — conditions will be upserted without tide columns

    if cmems_rows:
        try:
            merge_and_upsert(cmems_rows, wind_map, tide_rows, ipma_obs)
        except Exception as e:
            log.error(f"Supabase upsert failed: {e}")
            errors.append(f"Supabase: {e}")

    # ── New tables ────────────────────────────────────────────────────────────
    try:
        fetch_and_store_station_obs()
    except Exception as e:
        log.error(f"station_observations failed: {e}")
        # Not critical — conditions and tide writes are already complete above

    if wind_map:
        try:
            store_wind_forecast_history(wind_map, now)
        except Exception as e:
            log.error(f"wind_forecast_history (open-meteo) failed: {e}")
            # Not critical — conditions and tide writes are already complete above

    try:
        fetch_and_store_ipma_forecast()
    except Exception as e:
        log.error(f"wind_forecast_history (ipma) failed: {e}")
        # Not critical — conditions and tide writes are already complete above

    if errors:
        log.error("Completed with errors: " + "; ".join(errors))
        sys.exit(1)

    log.info("✓ Fetcher completed successfully")

if __name__ == "__main__":
    main()
