"""
Ambil kalender ekonomi USD dari Finnhub dan tulis ke docs/calendar.json.
Dijalankan otomatis oleh .github/workflows/update-calendar.yml lewat GitHub Actions.

CATATAN PENTING:
Bot Telegram kamu sudah punya kode yang memanggil Finnhub economic calendar
untuk filter fundamental. Cek field response yang sebenarnya dari kode itu
(print(response.json()) sekali saat development) dan sesuaikan nama field di
bawah ini (event/impact/actual/estimate/prev) kalau berbeda dari asumsi di sini.
Finnhub bisa mengubah nama field tanpa pemberitahuan besar.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import requests

FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "calendar.json")

# Ambil 7 hari ke depan
today = datetime.now(timezone.utc).date()
end_date = today + timedelta(days=7)

URL = (
    "https://finnhub.io/api/v1/calendar/economic"
    f"?from={today.isoformat()}&to={end_date.isoformat()}&token={FINNHUB_API_KEY}"
)

# WIB = UTC+7
WIB_OFFSET = timedelta(hours=7)
DAY_NAMES_ID = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]


def to_wib_label(date_str, time_str):
    """Finnhub biasanya kirim date 'YYYY-MM-DD' dan time 'HH:MM:SS' dalam UTC."""
    try:
        dt_utc = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_wib = dt_utc + WIB_OFFSET
        day_name = DAY_NAMES_ID[dt_wib.weekday()]
        return f"{day_name} {dt_wib.strftime('%H:%M')}"
    except (ValueError, TypeError):
        return date_str or "-"


def map_impact(raw_impact):
    """Normalisasi nilai impact dari Finnhub ke 'high' / 'med' / 'low'."""
    if raw_impact is None:
        return "low"
    val = str(raw_impact).lower()
    if val in ("3", "high"):
        return "high"
    if val in ("2", "medium", "med"):
        return "med"
    return "low"


def main():
    resp = requests.get(URL, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    events = payload.get("economicCalendar", [])

    filtered = []
    for e in events:
        if e.get("country") != "US":
            continue

        impact = map_impact(e.get("impact"))
        if impact == "low":
            continue  # dashboard cuma butuh high/medium biar ringkas

        forecast = e.get("estimate", "-")
        previous = e.get("prev", "-")
        name = f"{e.get('event', 'Unknown event')} — forecast {forecast}, prev {previous}"

        filtered.append(
            {
                "when": to_wib_label(e.get("date"), e.get("time")),
                "name": name,
                "impact": impact,
            }
        )

    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "events": filtered,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(filtered)} events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
