# Setup kalender live di GitHub Pages

Langkah supaya kalender di dashboard jadi live (bukan manual lagi):

## 1. Tambahkan file ke repo `XAU-AI` kamu
Struktur folder yang perlu ada:

```
XAU-AI/
├── .github/workflows/update-calendar.yml
├── scripts/update_calendar_json.py
└── docs/
    └── index.html
```

## 2. Tambahkan secret Finnhub
Repo → Settings → Secrets and variables → Actions → New repository secret
- Name: `FINNHUB_API_KEY`
- Value: API key Finnhub yang sudah dipakai bot Telegram kamu

## 3. Cek nama field response Finnhub
Sebelum dipakai serius, jalankan sekali secara lokal untuk lihat field asli:

```python
import requests
r = requests.get("https://finnhub.io/api/v1/calendar/economic?from=2026-09-16&to=2026-09-23&token=API_KEY_KAMU")
print(r.json())
```

Cocokkan nama field (`event`, `impact`, `estimate`, `prev`, `country`, `date`, `time`) di
`scripts/update_calendar_json.py` dengan yang benar-benar dikembalikan Finnhub — field ini
bisa beda dari asumsi di skrip kalau Finnhub mengubah skema mereka.

## 4. Aktifkan GitHub Pages
Repo → Settings → Pages → Source: "Deploy from a branch" → Branch: `main` (atau branch
default kamu) → Folder: `/docs` → Save.

Setelah itu dashboard live di:
`https://<username-github-kamu>.github.io/XAU-AI/`

## 5. Jalankan workflow pertama kali
Repo → Actions → "Update Economic Calendar" → Run workflow (klik manual sekali untuk
mengisi `docs/calendar.json` pertama kalinya, tidak perlu tunggu jadwal cron 6 jam).

## Yang perlu diketahui
- Dashboard yang saya publish di Claude (link artifact sebelumnya) **tetap memakai data
  fallback statis** — itu cuma untuk pratinjau, bukan versi live. Versi live ada di GitHub
  Pages kamu sendiri, di luar sandbox Claude.
- `docs/index.html` di sini sama persis dengan yang di Claude, tapi otomatis mencoba
  `fetch('calendar.json')` dulu. Kalau berhasil (di GitHub Pages), datanya live. Kalau
  gagal (dibuka di tempat lain / calendar.json belum ada), otomatis balik ke data fallback
  di dalam kode.
- Kalau nanti mau ubah tampilan dashboard, edit `docs/index.html` langsung — tidak perlu
  lewat Claude lagi kecuali mau bantuan desain ulang.
