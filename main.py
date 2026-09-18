"""
XAU-AI Bot — Sinyal Trading + Prediksi News untuk XAU/USD
============================================================
Fitur:
1. Deteksi FVG (Fair Value Gap) dari candle 5 menit -> kirim SINYAL buy/sell
   lengkap dengan FOTO chart yang menandai area FVG-nya, DAN sekarang
   dilengkapi SL/TP otomatis (berbasis ATR + kekuatan trend EMA).
2. Cek kalender ekonomi (Finnhub) -> kalau ada news yang baru rilis dan
   hasil "actual" menyimpang jauh dari "previous"/"estimate" (indikasi
   market akan SPIKE), kirim alert duluan.
3. Ringkasan fundamental harian (jadwal news hari ini) dikirim 1x/hari.

Cara pakai (lihat juga .github/workflows/bot.yml):
    python main.py signal   # dicek tiap 5 menit
    python main.py news     # dicek tiap 5-15 menit (cek news yang baru rilis)
    python main.py daily    # dicek 1x/hari (ringkasan kalender)

Environment variables (isi lewat GitHub Secrets):
    TELEGRAM_BOT_TOKEN   -> token bot Telegram
    TELEGRAM_CHAT_ID     -> chat_id tujuan kirim pesan
    TWELVEDATA_API_KEY   -> API key twelvedata.com (gratis)
    FINNHUB_API_KEY      -> API key finnhub.io (gratis)
"""

import os
import sys
import json
import math
import datetime as dt
from io import BytesIO

import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

SYMBOL = "XAU/USD"
INTERVAL = "5min"
CANDLE_COUNT = 150          # jumlah candle yang diambil tiap run
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14             # untuk buffer SL & pengukur kekuatan trend
FVG_LOOKBACK = 20           # cari FVG di N candle terakhir
SPIKE_THRESHOLD_PCT = 15.0  # deviasi actual vs previous/forecast dianggap "spike" kalau > 15%

# --- Parameter SL/TP ---
SL_BUFFER_ATR_MULT = 0.3    # buffer SL di luar zona FVG = 0.3x ATR
RR_STRONG_TREND = 3.0       # risk:reward kalau trend kuat
RR_MEDIUM_TREND = 2.0       # risk:reward kalau trend sedang
RR_WEAK_TREND = 1.5         # risk:reward kalau trend lemah/baru mulai
TREND_STRONG_THRESHOLD = 1.5   # ema_gap / atr >= ini -> trend kuat
TREND_MEDIUM_THRESHOLD = 0.7   # ema_gap / atr >= ini -> trend sedang

STATE_FILE = "state.json"   # menyimpan sinyal/berita terakhir supaya tidak kirim ulang


# ----------------------------------------------------------------------
# UTIL: STATE (biar tidak kirim sinyal / news yang sama berulang-ulang)
# ----------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_signal_time": None, "last_signal_type": None, "sent_news_ids": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ----------------------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------------------
def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diset.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }, timeout=20)
    if resp.status_code != 200:
        print(f"[ERROR] Gagal kirim pesan Telegram: {resp.status_code} {resp.text}")
        return False
    return True


def send_telegram_photo(image_bytes: BytesIO, caption: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diset.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    image_bytes.seek(0)
    resp = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("fvg_chart.png", image_bytes, "image/png")},
        timeout=30
    )
    if resp.status_code != 200:
        print(f"[ERROR] Gagal kirim foto Telegram: {resp.status_code} {resp.text}")
        return False
    return True


# ----------------------------------------------------------------------
# DATA HARGA (Twelve Data)
# ----------------------------------------------------------------------
def get_price_data() -> pd.DataFrame:
    """Ambil candle OHLC terbaru. Ganti fungsi ini kalau mau pakai sumber
    data lain (mis. Alpha Vantage, exchange lain, dll)."""
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY belum diset.")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": CANDLE_COUNT,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()

    if "values" not in data:
        raise RuntimeError(f"Gagal ambil data harga: {data}")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "time", "open": "open", "high": "high",
        "low": "low", "close": "close"
    })
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# INDIKATOR: EMA, RSI & ATR
# ----------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # ATR (Average True Range) -> dipakai untuk buffer SL & kekuatan trend
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(ATR_PERIOD).mean()
    # fallback biar tidak NaN di awal data
    df["atr"] = df["atr"].fillna(true_range.expanding().mean())

    return df


# ----------------------------------------------------------------------
# DETEKSI FVG (Fair Value Gap) — pola 3 candle
# ----------------------------------------------------------------------
def detect_fvg(df: pd.DataFrame) -> list:
    """
    Bullish FVG (celah BUY): low candle ke-3 > high candle ke-1
    Bearish FVG (celah SELL): high candle ke-3 < low candle ke-1
    Mengembalikan list dict: {index, type, top, bottom, time}
    """
    fvgs = []
    start = max(2, len(df) - FVG_LOOKBACK)
    for i in range(start, len(df)):
        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]

        if c3["low"] > c1["high"]:
            fvgs.append({
                "index": i,
                "type": "buy",
                "top": c3["low"],
                "bottom": c1["high"],
                "time": df.iloc[i]["time"],
            })
        elif c3["high"] < c1["low"]:
            fvgs.append({
                "index": i,
                "type": "sell",
                "top": c1["low"],
                "bottom": c3["high"],
                "time": df.iloc[i]["time"],
            })
    return fvgs


# ----------------------------------------------------------------------
# SL/TP: berbasis ATR (buffer) + kekuatan trend EMA (rasio RR)
# ----------------------------------------------------------------------
def calculate_sl_tp(df: pd.DataFrame, signal_type: str, fvg: dict) -> dict:
    """
    Menentukan SL & TP untuk sinyal yang sudah dikonfirmasi FVG + EMA cross.

    Logika:
    - SL ditaruh di LUAR zona FVG (bukan di harga entry), dengan buffer
      berbasis ATR supaya tidak kena stop-hunt/noise wick tipis.
      * BUY  -> SL = FVG bottom - buffer
      * SELL -> SL = FVG top    + buffer
      (Kalau harga balik masuk & menembus FVG, artinya setup-nya invalid.)
    - TP ditentukan dari rasio risk:reward yang MENYESUAIKAN kekuatan trend:
      semakin lebar jarak EMA9/EMA21 relatif terhadap ATR (trend kuat),
      semakin jauh target TP-nya. Trend lemah/baru mulai -> TP konservatif.
    """
    last = df.iloc[-1]
    atr = last["atr"] if not math.isnan(last["atr"]) else (last["high"] - last["low"])
    if atr <= 0:
        atr = last["close"] * 0.001  # fallback kecil kalau ATR 0/invalid

    buffer = atr * SL_BUFFER_ATR_MULT
    entry = last["close"]

    # kekuatan trend = jarak EMA fast-slow relatif terhadap volatilitas (ATR)
    ema_gap = abs(last["ema_fast"] - last["ema_slow"])
    trend_strength = ema_gap / atr if atr > 0 else 0.0

    if trend_strength >= TREND_STRONG_THRESHOLD:
        rr = RR_STRONG_TREND
        trend_label = "kuat"
    elif trend_strength >= TREND_MEDIUM_THRESHOLD:
        rr = RR_MEDIUM_TREND
        trend_label = "sedang"
    else:
        rr = RR_WEAK_TREND
        trend_label = "lemah"

    if signal_type == "buy":
        sl = fvg["bottom"] - buffer
        risk = entry - sl
        tp = entry + risk * rr
    else:  # sell
        sl = fvg["top"] + buffer
        risk = sl - entry
        tp = entry - risk * rr

    return {
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "risk": round(risk, 2),
        "rr": rr,
        "trend_strength": round(trend_strength, 2),
        "trend_label": trend_label,
    }


# ----------------------------------------------------------------------
# LOGIKA SINYAL: gabungan EMA cross + RSI + konfirmasi FVG
# ----------------------------------------------------------------------
def check_signal(df: pd.DataFrame, fvgs: list):
    if len(df) < max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD) + 3:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema_cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    ema_cross_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    recent_fvg_buy = next((f for f in reversed(fvgs) if f["type"] == "buy"), None)
    recent_fvg_sell = next((f for f in reversed(fvgs) if f["type"] == "sell"), None)

    # BUY: EMA cross up + RSI belum overbought + ada FVG buy yang belum terlalu lama
    if ema_cross_up and last["rsi"] < 70 and recent_fvg_buy and \
            (len(df) - 1 - recent_fvg_buy["index"]) <= 10:
        sltp = calculate_sl_tp(df, "buy", recent_fvg_buy)
        return {"type": "buy", "fvg": recent_fvg_buy, "price": last["close"],
                "time": last["time"], "sltp": sltp}

    # SELL: EMA cross down + RSI belum oversold + ada FVG sell yang belum terlalu lama
    if ema_cross_down and last["rsi"] > 30 and recent_fvg_sell and \
            (len(df) - 1 - recent_fvg_sell["index"]) <= 10:
        sltp = calculate_sl_tp(df, "sell", recent_fvg_sell)
        return {"type": "sell", "fvg": recent_fvg_sell, "price": last["close"],
                "time": last["time"], "sltp": sltp}

    return None


# ----------------------------------------------------------------------
# CHART: candlestick sederhana + area FVG yang di-highlight + garis SL/TP
# ----------------------------------------------------------------------
def plot_fvg_chart(df: pd.DataFrame, fvg: dict, sltp: dict = None, n_candles: int = 40) -> BytesIO:
    plot_df = df.tail(n_candles).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in plot_df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1)
        ax.add_patch(patches.Rectangle(
            (i - 0.3, min(row["open"], row["close"])),
            0.6, abs(row["close"] - row["open"]) or 0.01,
            color=color
        ))

    # highlight area FVG
    fvg_color = "#2196f3" if fvg["type"] == "buy" else "#ff9800"
    ax.axhspan(fvg["bottom"], fvg["top"], color=fvg_color, alpha=0.25,
               label=f"FVG {fvg['type'].upper()} zone")

    # garis SL & TP kalau tersedia
    if sltp:
        ax.axhline(sltp["sl"], color="#d32f2f", linestyle="--", linewidth=1.2,
                    label=f"SL {sltp['sl']}")
        ax.axhline(sltp["tp"], color="#2e7d32", linestyle="--", linewidth=1.2,
                    label=f"TP {sltp['tp']}")

    ax.set_title(f"XAU/USD — Sinyal {fvg['type'].upper()} (FVG zone + SL/TP)")
    ax.set_xlabel("Candle (5 menit)")
    ax.set_ylabel("Harga")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------
# KALENDER EKONOMI (Finnhub) — untuk ringkasan harian & deteksi spike
# ----------------------------------------------------------------------
def get_economic_calendar(date_from: str, date_to: str) -> list:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY belum diset.")
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {"from": date_from, "to": date_to, "token": FINNHUB_API_KEY}
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()
    events = data.get("economicCalendar", [])
    # fokus event yang relevan ke USD/Gold (bisa disesuaikan)
    relevant = [e for e in events if e.get("country") in ("US",) and e.get("impact") in ("high", "medium")]
    return relevant


def pct_deviation(actual, reference):
    try:
        actual = float(actual)
        reference = float(reference)
        if reference == 0:
            return None
        return abs(actual - reference) / abs(reference) * 100
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# MODE: signal — cek FVG + EMA/RSI, kirim sinyal + foto + SL/TP
# ----------------------------------------------------------------------
def run_signal_mode():
    state = load_state()
    df = get_price_data()
    df = add_indicators(df)
    fvgs = detect_fvg(df)
    signal = check_signal(df, fvgs)

    if not signal:
        print("Tidak ada sinyal baru saat ini.")
        return

    sig_key = f"{signal['time']}_{signal['type']}"
    if state.get("last_signal_time") == sig_key:
        print("Sinyal ini sudah pernah dikirim, skip.")
        return

    sltp = signal["sltp"]
    chart = plot_fvg_chart(df, signal["fvg"], sltp=sltp)
    emoji = "🟢" if signal["type"] == "buy" else "🔴"
    caption = (
        f"{emoji} <b>SINYAL {signal['type'].upper()}</b> — XAU/USD\n"
        f"Harga: {signal['price']:.2f}\n"
        f"Waktu: {signal['time']}\n"
        f"Area FVG: {signal['fvg']['bottom']:.2f} - {signal['fvg']['top']:.2f}\n"
        f"🛑 SL: {sltp['sl']} | 🎯 TP: {sltp['tp']} (RR 1:{sltp['rr']})\n"
        f"Trend: {sltp['trend_label']} (gap EMA/ATR = {sltp['trend_strength']})\n"
        f"(EMA{EMA_FAST}/{EMA_SLOW} cross + konfirmasi FVG)"
    )
    send_telegram_photo(chart, caption)

    state["last_signal_time"] = sig_key
    state["last_signal_type"] = signal["type"]
    save_state(state)


# ----------------------------------------------------------------------
# MODE: news — cek event yang baru rilis, deteksi potensi spike
# ----------------------------------------------------------------------
def run_news_mode():
    state = load_state()
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    events = get_economic_calendar(today, today)

    now = dt.datetime.utcnow()
    sent_ids = set(state.get("sent_news_ids", []))

    for e in events:
        event_id = f"{e.get('event')}_{e.get('time')}"
        actual = e.get("actual")
        if actual in (None, "", "NA") or event_id in sent_ids:
            continue  # belum rilis, atau sudah pernah dikirim

        # bandingkan actual vs previous & vs estimate
        dev_prev = pct_deviation(actual, e.get("prev"))
        dev_est = pct_deviation(actual, e.get("estimate"))
        max_dev = max([d for d in (dev_prev, dev_est) if d is not None], default=None)

        if max_dev is not None and max_dev >= SPIKE_THRESHOLD_PCT:
            msg = (
                f"⚡ <b>POTENSI SPIKE — {e.get('event')}</b>\n"
                f"Negara: {e.get('country')}\n"
                f"Actual: {actual} | Forecast: {e.get('estimate')} | Sebelumnya: {e.get('prev')}\n"
                f"Deviasi: {max_dev:.1f}% dari data sebelumnya\n"
                f"⚠️ Waspada pergerakan tajam di XAU/USD, cek chart sekarang."
            )
            send_telegram_message(msg)

        sent_ids.add(event_id)

    state["sent_news_ids"] = list(sent_ids)[-200:]  # jaga file tidak membengkak
    save_state(state)


# ----------------------------------------------------------------------
# MODE: daily — ringkasan kalender fundamental hari ini
# ----------------------------------------------------------------------
def run_daily_mode():
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    events = get_economic_calendar(today, today)

    if not events:
        send_telegram_message("📅 <b>Kalender Fundamental Hari Ini</b>\nTidak ada news high/medium impact untuk USD hari ini.")
        return

    lines = ["📅 <b>Kalender Fundamental Hari Ini (USD)</b>"]
    for e in sorted(events, key=lambda x: x.get("time", "")):
        lines.append(
            f"• {e.get('time', '')[-8:-3]} — {e.get('event')} "
            f"(impact: {e.get('impact')}, forecast: {e.get('estimate')}, sebelumnya: {e.get('prev')})"
        )
    send_telegram_message("\n".join(lines))


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "signal"
    try:
        if mode == "signal":
            run_signal_mode()
        elif mode == "news":
            run_news_mode()
        elif mode == "daily":
            run_daily_mode()
        else:
            print(f"Mode tidak dikenal: {mode}. Gunakan: signal | news | daily")
    except Exception as e:
        # kalau error, kirim juga ke Telegram biar ketahuan tanpa buka log Actions
        send_telegram_message(f"❌ Bot error ({mode}): {e}")
        raise
