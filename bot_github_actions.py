"""
Bot Telegram - Sinyal Buy/Sell Forex (XAU/USD, timeframe 15 menit)
Strategi: EMA Crossover (EMA9 & EMA21) + konfirmasi RSI(14)

Versi ini didesain untuk GitHub Actions: skrip jalan SEKALI setiap kali
dipanggil (dijadwalkan oleh GitHub Actions tiap beberapa menit), bukan loop
selamanya. Status sinyal terakhir disimpan di file `last_signal.txt` supaya
tidak mengirim sinyal yang sama berulang-ulang.

PENTING: Ini HANYA memberi sinyal, TIDAK melakukan eksekusi order/transaksi.
"""

import os
import logging
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

SYMBOL = "XAU/USD"
INTERVAL = "15min"
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
STATE_FILE = "last_signal.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def kirim_pesan_telegram(pesan: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": pesan,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("Pesan terkirim ke Telegram.")
    except Exception as e:
        log.error(f"Gagal mengirim pesan Telegram: {e}")


def ambil_data_harga():
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 100,
        "apikey": TWELVE_DATA_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if "values" not in data:
            log.error(f"Respons API tidak berisi data harga: {data}")
            return None

        candles = list(reversed(data["values"]))
        closes = [float(c["close"]) for c in candles]
        return closes

    except Exception as e:
        log.error(f"Gagal mengambil data harga: {e}")
        return None


def hitung_ema(harga_list, periode):
    if len(harga_list) < periode:
        return []

    ema_values = []
    multiplier = 2 / (periode + 1)
    sma_awal = sum(harga_list[:periode]) / periode
    ema_values.append(sma_awal)

    for harga in harga_list[periode:]:
        ema_baru = (harga - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(ema_baru)

    return ema_values


def hitung_rsi(harga_list, periode=14):
    if len(harga_list) < periode + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(harga_list)):
        selisih = harga_list[i] - harga_list[i - 1]
        if selisih > 0:
            gains.append(selisih)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(selisih))

    avg_gain = sum(gains[:periode]) / periode
    avg_loss = sum(losses[:periode]) / periode

    for i in range(periode, len(gains)):
        avg_gain = (avg_gain * (periode - 1) + gains[i]) / periode
        avg_loss = (avg_loss * (periode - 1) + losses[i]) / periode

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def analisa_sinyal(closes):
    ema_fast_values = hitung_ema(closes, EMA_FAST)
    ema_slow_values = hitung_ema(closes, EMA_SLOW)

    if len(ema_fast_values) < 2 or len(ema_slow_values) < 2:
        return None, None

    selisih_panjang = len(ema_fast_values) - len(ema_slow_values)
    ema_fast_aligned = ema_fast_values[selisih_panjang:]

    ema_fast_sekarang = ema_fast_aligned[-1]
    ema_fast_sebelumnya = ema_fast_aligned[-2]
    ema_slow_sekarang = ema_slow_values[-1]
    ema_slow_sebelumnya = ema_slow_values[-2]

    rsi = hitung_rsi(closes, RSI_PERIOD)
    harga_sekarang = closes[-1]

    sinyal = None

    if ema_fast_sebelumnya <= ema_slow_sebelumnya and ema_fast_sekarang > ema_slow_sekarang:
        if rsi is not None and rsi < 70:
            sinyal = "BUY"
    elif ema_fast_sebelumnya >= ema_slow_sebelumnya and ema_fast_sekarang < ema_slow_sekarang:
        if rsi is not None and rsi > 30:
            sinyal = "SELL"

    info = {
        "harga": harga_sekarang,
        "ema_fast": ema_fast_sekarang,
        "ema_slow": ema_slow_sekarang,
        "rsi": rsi,
    }
    return sinyal, info


def format_pesan_sinyal(sinyal, info):
    emoji = "🟢" if sinyal == "BUY" else "🔴"
    return (
        f"{emoji} *SINYAL {sinyal}* - {SYMBOL} ({INTERVAL})\n\n"
        f"Harga saat ini: `{info['harga']:.3f}`\n"
        f"EMA{EMA_FAST}: `{info['ema_fast']:.3f}`\n"
        f"EMA{EMA_SLOW}: `{info['ema_slow']:.3f}`\n"
        f"RSI({RSI_PERIOD}): `{info['rsi']:.2f}`\n\n"
        f"⚠️ Ini sinyal otomatis, bukan saran keuangan. "
        f"Selalu lakukan analisa & manajemen risiko sendiri."
    )


def baca_sinyal_terakhir():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return f.read().strip()
    return None


def simpan_sinyal_terakhir(sinyal):
    with open(STATE_FILE, "w") as f:
        f.write(sinyal)


def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not TWELVE_DATA_API_KEY:
        log.error(
            "Environment variable belum lengkap. "
            "Pastikan TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, dan TWELVE_DATA_API_KEY sudah diisi."
        )
        return

    closes = ambil_data_harga()
    if not closes:
        log.warning("Data harga tidak tersedia, lewati pengecekan kali ini.")
        return

    sinyal, info = analisa_sinyal(closes)
    last_signal = baca_sinyal_terakhir()

    if sinyal and sinyal != last_signal:
        pesan = format_pesan_sinyal(sinyal, info)
        kirim_pesan_telegram(pesan)
        simpan_sinyal_terakhir(sinyal)
        log.info(f"Sinyal baru terkirim: {sinyal}")
    else:
        rsi_text = f"{info['rsi']:.2f}" if info and info.get("rsi") is not None else "N/A"
        log.info(f"Tidak ada sinyal baru. (RSI saat ini: {rsi_text})")


if __name__ == "__main__":
    main()
