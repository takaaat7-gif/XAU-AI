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
4. Narasi AI (opsional) -> menambahkan konteks H4 + catatan status posisi
   yang ditulis oleh LLM (Claude) sebagai pelengkap sinyal FVG, tanpa
   mengubah logic sinyal itu sendiri. Kalau ANTHROPIC_API_KEY tidak diisi,
   fitur ini otomatis dilewati (caption tetap terkirim seperti biasa).

Cara pakai (lihat juga .github/workflows/bot.yml):
    python main.py signal   # dicek tiap 5 menit
    python main.py news     # dicek tiap 5-15 menit (cek news yang baru rilis)
    python main.py daily    # dicek 1x/hari (ringkasan kalender)

Environment variables (isi lewat GitHub Secrets):
    TELEGRAM_BOT_TOKEN   -> token bot Telegram
    TELEGRAM_CHAT_ID     -> chat_id tujuan kirim pesan
    TWELVEDATA_API_KEY   -> API key twelvedata.com (gratis)
    FINNHUB_API_KEY      -> API key finnhub.io (gratis)
    ANTHROPIC_API_KEY    -> API key console.anthropic.com (opsional, untuk narasi AI)
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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AI_NARRATIVE_MODEL = "claude-sonnet-4-6"

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

# --- Parameter Support & Resistance (S/R) ---
H1_CANDLES_PER_BAR = 12               # 12 x candle 5 menit = 1 candle H1 (resample, tanpa API call baru)
SR_PIVOT_WINDOW = 2                   # fractal: bandingkan 2 candle kiri & 2 candle kanan
SR_CLUSTER_TOLERANCE_PCT = 0.12       # gabung swing high/low yang jaraknya < 0.12% jadi satu zona
SR_MIN_TOUCHES = 2                    # minimal disentuh berapa kali biar dianggap level valid
SR_MAX_LEVELS = 3                     # ambil berapa level teratas tiap sisi (Resistance & Support)
SR_TP_BUFFER_ATR_MULT = 0.15          # jarak aman dari level S/R saat TP di-cap ke level tsb
SR_PROXIMITY_ATR_MULT = 1.0           # kalau entry sedekat ini (dalam satuan ATR) ke S/R lawan arah -> beri warning

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
# SUPPORT & RESISTANCE — dihitung sendiri oleh bot (bukan LLM), dipakai
# untuk memvalidasi sinyal FVG (SL/TP disesuaikan, warning kalau entry
# terlalu dekat level lawan arah) dan sebagai bahan narasi AI.
# ----------------------------------------------------------------------
def resample_ohlc(df: pd.DataFrame, candles_per_bar: int) -> pd.DataFrame:
    """Gabungkan candle kecil (5 menit) jadi candle lebih besar (H1) tanpa
    API call tambahan -> swing high/low lebih stabil, tidak kebanyakan noise
    dibanding kalau dideteksi langsung dari candle 5 menit."""
    if len(df) < candles_per_bar * (SR_PIVOT_WINDOW * 2 + 5):
        return df.copy()
    n_bars = len(df) // candles_per_bar
    trimmed = df.iloc[-(n_bars * candles_per_bar):].reset_index(drop=True)
    rows = []
    for i in range(n_bars):
        chunk = trimmed.iloc[i * candles_per_bar:(i + 1) * candles_per_bar]
        rows.append({
            "time": chunk.iloc[-1]["time"],
            "open": chunk.iloc[0]["open"],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk.iloc[-1]["close"],
        })
    return pd.DataFrame(rows)


def detect_support_resistance(df: pd.DataFrame) -> dict:
    """Deteksi level Support & Resistance dari swing high/low (fractal) di
    candle H1 hasil resample, dikelompokkan jadi zona + dihitung berapa kali
    disentuh (touches). Mengembalikan {"resistance": [...], "support": [...]},
    masing-masing diurutkan dari yang paling dekat harga sekarang."""
    df_h1 = resample_ohlc(df, H1_CANDLES_PER_BAR)
    w = SR_PIVOT_WINDOW
    if len(df_h1) < (w * 2 + 5):
        return {"resistance": [], "support": []}

    pivots = []
    for i in range(w, len(df_h1) - w):
        window_high = df_h1["high"].iloc[i - w:i + w + 1]
        window_low = df_h1["low"].iloc[i - w:i + w + 1]
        if df_h1["high"].iloc[i] == window_high.max():
            pivots.append(("high", float(df_h1["high"].iloc[i])))
        if df_h1["low"].iloc[i] == window_low.min():
            pivots.append(("low", float(df_h1["low"].iloc[i])))

    # gabungkan pivot yang berdekatan (dalam toleransi %) jadi satu zona/level
    clusters = []
    for kind, price in sorted(pivots, key=lambda x: x[1]):
        merged = False
        for c in clusters:
            if c["kind"] == kind and abs(price - c["level"]) / c["level"] * 100 <= SR_CLUSTER_TOLERANCE_PCT:
                c["level"] = (c["level"] * c["touches"] + price) / (c["touches"] + 1)
                c["touches"] += 1
                merged = True
                break
        if not merged:
            clusters.append({"kind": kind, "level": price, "touches": 1})

    current_price = float(df_h1.iloc[-1]["close"])
    resistance = sorted(
        [c for c in clusters
         if c["kind"] == "high" and c["level"] > current_price and c["touches"] >= SR_MIN_TOUCHES],
        key=lambda c: c["level"]
    )[:SR_MAX_LEVELS]
    support = sorted(
        [c for c in clusters
         if c["kind"] == "low" and c["level"] < current_price and c["touches"] >= SR_MIN_TOUCHES],
        key=lambda c: -c["level"]
    )[:SR_MAX_LEVELS]

    return {
        "resistance": [{"level": round(c["level"], 2), "touches": c["touches"]} for c in resistance],
        "support": [{"level": round(c["level"], 2), "touches": c["touches"]} for c in support],
    }


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
def calculate_sl_tp(df: pd.DataFrame, signal_type: str, fvg: dict, sr: dict = None) -> dict:
    """
    Menentukan SL & TP untuk sinyal yang sudah dikonfirmasi FVG + EMA cross,
    lalu divalidasi terhadap Support/Resistance (sr) hasil detect_support_resistance().

    Logika:
    - SL ditaruh di LUAR zona FVG (bukan di harga entry), dengan buffer
      berbasis ATR supaya tidak kena stop-hunt/noise wick tipis.
      * BUY  -> SL = FVG bottom - buffer
      * SELL -> SL = FVG top    + buffer
      (Kalau harga balik masuk & menembus FVG, artinya setup-nya invalid.)
    - TP awal dari rasio risk:reward yang MENYESUAIKAN kekuatan trend.
    - TP lalu di-cap kalau ada level Resistance (buy) / Support (sell) yang
      valid (touches >= SR_MIN_TOUCHES) berada SEBELUM target RR -> TP
      dipindah ke level itu (minus buffer kecil), supaya TP realistis dan
      tidak "menembus" zona S/R kuat.
    - Kalau entry sendiri sudah terlalu dekat dengan level S/R lawan arah
      (mis. BUY tapi resistance kuat cuma sejengkal di atas entry), sinyal
      tetap dikirim tapi diberi tanda "warning" -> validitasnya lebih rendah.
    """
    last = df.iloc[-1]
    atr = last["atr"] if not math.isnan(last["atr"]) else (last["high"] - last["low"])
    if atr <= 0:
        atr = last["close"] * 0.001  # fallback kecil kalau ATR 0/invalid

    buffer = atr * SL_BUFFER_ATR_MULT
    sr_buffer = atr * SR_TP_BUFFER_ATR_MULT
    entry = last["close"]
    sr = sr or {}

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

    capped_by = None
    warning = None

    if signal_type == "buy":
        sl = fvg["bottom"] - buffer
        risk = entry - sl
        tp = entry + risk * rr

        # cap TP kalau ada resistance valid di antara entry dan target RR
        for r in sr.get("resistance", []):
            if entry < r["level"] <= tp:
                tp = round(r["level"] - sr_buffer, 2)
                capped_by = r
                break

        nearest_res = sr.get("resistance", [None])[0] if sr.get("resistance") else None
        if nearest_res and (nearest_res["level"] - entry) <= atr * SR_PROXIMITY_ATR_MULT:
            warning = f"entry dekat resistance {nearest_res['level']} ({nearest_res['touches']}x sentuh)"

    else:  # sell
        sl = fvg["top"] + buffer
        risk = sl - entry
        tp = entry - risk * rr

        # cap TP kalau ada support valid di antara entry dan target RR
        for s in sr.get("support", []):
            if tp <= s["level"] < entry:
                tp = round(s["level"] + sr_buffer, 2)
                capped_by = s
                break

        nearest_sup = sr.get("support", [None])[0] if sr.get("support") else None
        if nearest_sup and (entry - nearest_sup["level"]) <= atr * SR_PROXIMITY_ATR_MULT:
            warning = f"entry dekat support {nearest_sup['level']} ({nearest_sup['touches']}x sentuh)"

    effective_rr = round(abs(tp - entry) / risk, 2) if risk > 0 else rr

    return {
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "risk": round(risk, 2),
        "rr": effective_rr,
        "rr_target": rr,
        "trend_strength": round(trend_strength, 2),
        "trend_label": trend_label,
        "capped_by": capped_by,
        "warning": warning,
    }


# ----------------------------------------------------------------------
# KONTEKS H4 — bahan mentah (angka) untuk narasi AI, dihitung sendiri
# oleh bot (bukan oleh LLM), supaya LLM tidak mengarang angka.
# ----------------------------------------------------------------------
def get_h4_context(candle_count: int = 60) -> dict:
    """Ambil candle H4 dan hitung struktur + posisi harga dalam rentang H4.
    Kalau gagal (API key kosong / error), kembalikan dict kosong -> narasi
    AI otomatis dilewati, tidak mengganggu jalannya sinyal FVG."""
    if not TWELVEDATA_API_KEY:
        return {}
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": "4h",
        "outputsize": candle_count,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        data = resp.json()
        if "values" not in data:
            return {}
        df_h4 = pd.DataFrame(data["values"])
        for col in ["open", "high", "low", "close"]:
            df_h4[col] = df_h4[col].astype(float)

        recent = df_h4.tail(20)  # ~20 candle H4 terakhir sebagai lookback swing
        h4_high = recent["high"].max()
        h4_low = recent["low"].min()
        last_close = df_h4.iloc[-1]["close"]

        rng = h4_high - h4_low
        posisi_pct = ((last_close - h4_low) / rng * 100) if rng > 0 else 50.0

        # struktur kasar: bandingkan swing high/low separuh awal vs separuh akhir
        half = max(1, len(recent) // 2)
        first_high, first_low = recent.iloc[:half]["high"].max(), recent.iloc[:half]["low"].min()
        second_high, second_low = recent.iloc[half:]["high"].max(), recent.iloc[half:]["low"].min()
        if second_high > first_high and second_low > first_low:
            struktur = "uptrend"
        elif second_high < first_high and second_low < first_low:
            struktur = "downtrend"
        else:
            struktur = "campuran"

        return {
            "h4_range_high": round(h4_high, 2),
            "h4_range_low": round(h4_low, 2),
            "h4_posisi_pct": round(posisi_pct, 1),
            "h4_struktur": struktur,
            "h4_last_close": round(last_close, 2),
        }
    except Exception as e:
        print(f"[WARN] Gagal ambil konteks H4: {e}")
        return {}


# ----------------------------------------------------------------------
# NARASI AI — prompt untuk LLM yang menulis konteks H4 + status posisi
# dalam Bahasa Indonesia, sebagai TAMBAHAN caption (bukan pengganti logic
# sinyal FVG/EMA/RSI/SL/TP, yang semuanya masih dihitung oleh bot sendiri).
# ----------------------------------------------------------------------
AI_NARRATIVE_SYSTEM_PROMPT = """Kamu adalah analis teknikal forex yang menulis catatan singkat untuk trader retail dalam Bahasa Indonesia.

Tugasmu: berdasarkan data JSON yang diberikan user (harga, level entry/SL/TP, konteks H4, status posisi), tulis narasi analisa singkat dengan struktur berikut, HANYA dalam format JSON (tanpa teks lain, tanpa markdown code fence, tanpa penjelasan tambahan):

{
  "konteks_h4": ["poin 1", "poin 2", "poin 3 (opsional)"],
  "catatan_status": "satu kalimat status posisi saat ini"
}

Aturan ketat:
- Bahasa Indonesia, ringkas, gaya profesional analis teknikal (bukan hype, tanpa emoji, tanpa janji profit).
- konteks_h4: maksimal 3 bullet pendek — mencakup struktur H4 (uptrend/downtrend/campuran), posisi harga dalam rentang H4 (dalam %), dan level Resistance/Support terdekat beserta jumlah "touches"-nya kalau data resistance/support tersedia di JSON.
- catatan_status: 1 kalimat, sesuaikan dengan status_posisi yang diberikan (contoh: kalau "entry_tersentuh" -> ingatkan untuk memantau TP dan SL serta jangan entry ulang; kalau "menunggu_entry" -> jelaskan bahwa harga masih menunggu mendekati level entry). Kalau ada field "warning" di data, sebutkan juga sebagai catatan kehati-hatian singkat.
- SEMUA angka yang kamu sebut harus PERSIS berasal dari data JSON yang diberikan. Jangan mengarang angka, level, atau persentase baru.
- Jangan berikan saran keuangan personal, jaminan profit, atau ajakan entry/exit eksplisit — ini catatan observasi pasar, bukan rekomendasi."""


def generate_ai_narrative(signal: dict, sltp: dict, h4_context: dict, sr: dict = None) -> dict:
    """Panggil Claude untuk menulis narasi konteks H4 + status posisi.
    Kalau ANTHROPIC_API_KEY kosong atau ada error apa pun, kembalikan dict
    kosong -> caption tetap terkirim normal tanpa narasi tambahan."""
    if not ANTHROPIC_API_KEY or not h4_context:
        return {}

    entry = signal["price"]
    current_price = h4_context.get("h4_last_close", entry)
    if signal["type"] == "buy":
        status_posisi = "entry_tersentuh" if current_price >= entry else "menunggu_entry"
    else:
        status_posisi = "entry_tersentuh" if current_price <= entry else "menunggu_entry"

    payload_data = {
        "pair": SYMBOL,
        "arah": signal["type"],
        "harga_sekarang": current_price,
        "entry": entry,
        "sl": sltp["sl"],
        "tp": sltp["tp"],
        "rr": sltp["rr"],
        "status_posisi": status_posisi,
        "resistance": (sr or {}).get("resistance", []),
        "support": (sr or {}).get("support", []),
        "warning": sltp.get("warning"),
        **h4_context,
    }

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": AI_NARRATIVE_MODEL,
                "max_tokens": 300,
                "system": AI_NARRATIVE_SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": json.dumps(payload_data, ensure_ascii=False)}
                ],
            },
            timeout=20,
        )
        data = resp.json()
        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        ).strip()
        # jaga-jaga kalau LLM tetap membungkus dengan ```json ... ```
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        return json.loads(text)
    except Exception as e:
        print(f"[WARN] Gagal generate narasi AI: {e}")
        return {}


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

    sr = detect_support_resistance(df)

    # BUY: EMA cross up + RSI belum overbought + ada FVG buy yang belum terlalu lama
    if ema_cross_up and last["rsi"] < 70 and recent_fvg_buy and \
            (len(df) - 1 - recent_fvg_buy["index"]) <= 10:
        sltp = calculate_sl_tp(df, "buy", recent_fvg_buy, sr=sr)
        return {"type": "buy", "fvg": recent_fvg_buy, "price": last["close"],
                "time": last["time"], "sltp": sltp, "sr": sr}

    # SELL: EMA cross down + RSI belum oversold + ada FVG sell yang belum terlalu lama
    if ema_cross_down and last["rsi"] > 30 and recent_fvg_sell and \
            (len(df) - 1 - recent_fvg_sell["index"]) <= 10:
        sltp = calculate_sl_tp(df, "sell", recent_fvg_sell, sr=sr)
        return {"type": "sell", "fvg": recent_fvg_sell, "price": last["close"],
                "time": last["time"], "sltp": sltp, "sr": sr}

    return None


# ----------------------------------------------------------------------
# CHART: candlestick sederhana + area FVG yang di-highlight + garis SL/TP
# ----------------------------------------------------------------------
def plot_fvg_chart(df: pd.DataFrame, fvg: dict, sltp: dict = None, sr: dict = None, n_candles: int = 40) -> BytesIO:
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

    # garis Resistance & Support (tipis, di belakang, dengan label jumlah touches)
    if sr:
        for r in sr.get("resistance", []):
            ax.axhline(r["level"], color="#c62828", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.text(len(plot_df) - 1, r["level"], f"  R {r['touches']}x", color="#c62828",
                    fontsize=7, va="center")
        for s in sr.get("support", []):
            ax.axhline(s["level"], color="#1565c0", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.text(len(plot_df) - 1, s["level"], f"  S {s['touches']}x", color="#1565c0",
                    fontsize=7, va="center")

    ax.set_title(f"XAU/USD — Sinyal {fvg['type'].upper()} (FVG zone + SL/TP + S/R)")
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
    sr = signal.get("sr", {})
    chart = plot_fvg_chart(df, signal["fvg"], sltp=sltp, sr=sr)
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

    if sltp.get("capped_by"):
        c = sltp["capped_by"]
        caption += f"\n📐 TP disesuaikan ke level S/R {c['level']} ({c['touches']}x sentuh)"

    nearest_res = sr.get("resistance", [])
    nearest_sup = sr.get("support", [])
    if nearest_res or nearest_sup:
        caption += "\n\n<b>S/R terdekat:</b>"
        for r in nearest_res:
            caption += f"\n🔺 R {r['level']} ({r['touches']}x sentuh)"
        for s in nearest_sup:
            caption += f"\n🔻 S {s['level']} ({s['touches']}x sentuh)"

    if sltp.get("warning"):
        caption += f"\n\n⚠️ {sltp['warning']} — validitas sinyal lebih rendah, pertimbangkan dengan hati-hati."

    # --- Narasi AI (opsional, pelengkap) ---
    h4_context = get_h4_context()
    narrative = generate_ai_narrative(signal, sltp, h4_context, sr=sr)
    if narrative.get("konteks_h4"):
        caption += "\n\n<b>Konteks H4:</b>\n" + "\n".join(f"• {p}" for p in narrative["konteks_h4"])
    if narrative.get("catatan_status"):
        caption += f"\n\n📌 {narrative['catatan_status']}"

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
