import time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from curl_cffi import requests

PLAYER_ID = 259117  # Joshua Kimmich on Sofascore
START_DATE = "2024-08-01"
END_DATE = "2025-06-01"
TOURNAMENT_NAME = "Bundesliga"

API = "https://www.sofascore.com/api/v1"

session = requests.Session(impersonate="chrome")

def get_json(url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def iso_date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

# 1) Get Kimmich's match history, newest page first.
events = []
page = 0

while True:
    url = f"{API}/player/{PLAYER_ID}/events/last/{page}"
    data = get_json(url)
    batch = data.get("events", [])

    if not batch:
        break

    events.extend(batch)
    print(f"Đã lấy trang {page}: {len(batch)} trận")

    # Once the oldest event on this page is before our period,
    # we can stop after collecting this page.
    oldest = min(e.get("startTimestamp", 0) for e in batch)
    if iso_date(oldest) < START_DATE:
        break

    if not data.get("hasNextPage", True):
        break

    page += 1
    time.sleep(0.3)

# 2) Keep Bundesliga matches in the requested season.
selected = []
for e in events:
    d = iso_date(e["startTimestamp"])
    tournament = e.get("tournament", {}).get("name", "")

    if START_DATE <= d < END_DATE and tournament == TOURNAMENT_NAME:
        selected.append(e)

print(f"\nTìm thấy {len(selected)} trận Bundesliga trong khoảng {START_DATE} -> {END_DATE}.")

# 3) Read player statistics from each match lineup.
rows = []

for i, event in enumerate(sorted(selected, key=lambda x: x["startTimestamp"])):
    event_id = event["id"]
    d = iso_date(event["startTimestamp"])

    try:
        data = get_json(f"{API}/event/{event_id}/lineups")

        player_stats = None

        for side in ("home", "away"):
            for p in data.get(side, {}).get("players", []):
                player = p.get("player", {})
                if player.get("id") == PLAYER_ID:
                    player_stats = p.get("statistics", {})
                    break
            if player_stats is not None:
                break

        # Player did not play / no statistics available.
        if not player_stats:
            print(f"[{i+1}/{len(selected)}] {d}: không có stats")
            continue

        cmp_ = player_stats.get("accuratePass")
        att = player_stats.get("totalPass")
        kp = player_stats.get("keyPass")

        # Some Sofascore responses use plural field names.
        if cmp_ is None:
            cmp_ = player_stats.get("accuratePasses")
        if att is None:
            att = player_stats.get("totalPasses")
        if kp is None:
            kp = player_stats.get("keyPasses")

        if cmp_ is None or att is None or kp is None:
            print(f"[{i+1}/{len(selected)}] {d}: thiếu Cmp/Att/KP")
            continue

        cmp_pct = (float(cmp_) / float(att) * 100) if float(att) else 0

        rows.append({
            "Date": d,
            "Cmp": int(cmp_),
            "Att": int(att),
            "Cmp%": round(cmp_pct, 2),
            "KP": int(kp)
        })

        print(
            f"[{i+1}/{len(selected)}] {d}: "
            f"Cmp={cmp_}, Att={att}, Cmp%={cmp_pct:.2f}, KP={kp}"
        )

    except Exception as ex:
        print(f"[{i+1}/{len(selected)}] {d}: lỗi -> {ex}")

    time.sleep(0.25)

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(
        "Không lấy được dữ liệu. Kiểm tra kết nối hoặc Sofascore có chặn request."
    )

df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
OUTPUT_FILE = Path(__file__).resolve().parent / "kimmich.csv"
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n===== HOÀN TẤT =====")
print(df)
print(f"\nĐã lưu {len(df)} dòng vào: {OUTPUT_FILE}")
