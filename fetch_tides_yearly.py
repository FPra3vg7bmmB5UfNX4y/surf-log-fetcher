"""
Peniche Surf Log — WorldTides Yearly Bulk Fetcher
=================================================
Run once per year (or whenever you want to refresh tide predictions).
Fetches 365 days of tide heights and extremes from WorldTides at
15-min resolution and upserts everything to the Supabase `tides` table.

The main fetcher.py reads from this table instead of calling WorldTides
on every 3h cron run, so the WorldTides API key is only needed here.

Usage:
    python fetch_tides_yearly.py

Required env vars:
    SUPABASE_URL, SUPABASE_KEY, WORLDTIDES_KEY
"""

import os
import sys
import logging
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
WORLDTIDES_KEY = os.environ["WORLDTIDES_KEY"]

LAT_PT = 39.3557
LON_PT = -9.3808


def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def sb_upsert(rows: list[dict]):
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/tides?on_conflict=valid_at",
        headers={**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows,
        timeout=30,
    )
    r.raise_for_status()
    log.info(f"  → upserted {len(rows)} rows")


def main():
    log.info("=" * 60)
    log.info("Peniche Surf Log — Yearly tide fetch")
    log.info(f"Run time: {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 60)

    log.info("Fetching WorldTides — 365 days at 15-min resolution, datum=LAT…")
    r = requests.get(
        "https://www.worldtides.info/api/v3",
        params={
            "heights":  True,
            "extremes": True,
            "lat":  LAT_PT,
            "lon":  LON_PT,
            "days": 365,
            "step": 900,
            "datum": "LAT",
            "key":  WORLDTIDES_KEY,
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    if d.get("status") != 200:
        raise RuntimeError(f"WorldTides error: {d.get('error')}")

    heights  = d.get("heights", [])
    extremes = d.get("extremes", [])
    log.info(f"  → {len(heights)} height points, {len(extremes)} extremes, callCount={d.get('callCount')}")

    rows = []
    for idx, h in enumerate(heights):
        ts    = h["dt"]
        valid = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Rising / falling from neighbours
        state = None
        if 0 < idx < len(heights) - 1:
            state = "rising" if heights[idx + 1]["height"] > heights[idx - 1]["height"] else "falling"

        # Phase: within 1.5h of an extreme → label it low/high, else mid
        phase = "mid"
        for ex in extremes:
            if abs(ex["dt"] - ts) < 5400:
                phase = ex["type"].lower()
                break

        next_ex = next((e for e in extremes if e["dt"] > ts), None)

        rows.append({
            "valid_at":    valid.isoformat(),
            "height":      round(h["height"], 3),
            "state":       state,
            "phase":       phase,
            "next_type":   next_ex["type"] if next_ex else None,
            "next_height": round(next_ex["height"], 3) if next_ex else None,
        })

    log.info(f"Upserting {len(rows)} tide rows to Supabase in batches of 500…")
    for i in range(0, len(rows), 500):
        sb_upsert(rows[i:i + 500])

    log.info("✓ Yearly tide fetch complete")


if __name__ == "__main__":
    main()
