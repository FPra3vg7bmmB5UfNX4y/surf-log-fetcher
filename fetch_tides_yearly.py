"""
Peniche Surf Log — WorldTides Bulk Tide Fetcher
================================================
Fetches tide heights and extremes from WorldTides at 15-min resolution
and upserts everything to the Supabase `tides` table.

The main fetcher.py reads from this table instead of calling WorldTides
on every 3h cron run, so the WorldTides API key is only needed here.

Usage:
    python fetch_tides_yearly.py           # default: 30 days
    python fetch_tides_yearly.py --days 90 # more days (needs paid plan)

Credit cost: ~2 credits per day requested (heights + extremes).
Free tier:   10 credits/day  → max ~5 days per run
Paid plan:   10 000 credits/day → 365 days in one call

Required env vars:
    SUPABASE_URL, SUPABASE_KEY, WORLDTIDES_KEY
"""

import os
import sys
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30,
                        help="Days of tide data to fetch (default 30; 365 needs paid WorldTides plan)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"Peniche Surf Log — Bulk tide fetch ({args.days} days)")
    log.info(f"Run time: {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 60)

    log.info(f"Fetching WorldTides — {args.days} days at 15-min resolution, datum=LAT…")
    r = requests.get(
        "https://www.worldtides.info/api/v3",
        params={
            "heights":  True,
            "extremes": True,
            "lat":  LAT_PT,
            "lon":  LON_PT,
            "days": args.days,
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
            "valid_at":        valid.isoformat(),
            "tide_height":     round(h["height"], 3),
            "tide_state":      state,
            "tide_phase":      phase,
            "tide_next_type":  next_ex["type"] if next_ex else None,
            "tide_next_height":round(next_ex["height"], 3) if next_ex else None,
        })

    log.info(f"Upserting {len(rows)} tide rows to Supabase in batches of 500…")
    for i in range(0, len(rows), 500):
        sb_upsert(rows[i:i + 500])

    log.info("✓ Yearly tide fetch complete")


if __name__ == "__main__":
    main()
