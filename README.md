# AegisVision — Indian ANPR (YOLOv8 + ByteTrack + FastAPI)

A license plate recognition pipeline built specifically for **Indian plates**, not
adapted from a US/EU template. It detects vehicles, tracks them across frames,
reads the plate, and corrects OCR misreads using the actual structure of an
Indian registration plate — then logs confirmed reads to a database with a
live Streamlit dashboard to watch it happen.

This is a working prototype, not a hardened production system. See
[Limitations](#limitations) below for what that means concretely.

## Demo

There's no permanently hosted demo — everything currently runs as three
local processes (see [Running it](#running-it)). To get a shareable link
for a quick walkthrough you can tunnel the dashboard with localtunnel or ngrok:

```bash
# after starting uvicorn and streamlit locally
npx localtunnel --port 8501     # tunnels the Streamlit dashboard
npx localtunnel --port 8000     # optional: tunnels the API directly
```

That gives you a temporary public URL good for a demo call; it's not
meant to stay up (see [Limitations](#limitations) — no auth on the API).
In video-file mode the pipeline also writes an annotated `output.mp4` you can
share directly.

## Why Indian plates need special handling

Most open-source ANPR projects assume clean, high-contrast Western plates.
Indian plates follow a specific grammar — `SS DD L(1-3) NNNN`
(state code, RTO district code, 1–3 letter series, 4-digit number) — and
OCR models routinely confuse `O/0`, `S/5`, `I/1`, `B/8` in the wrong
positions. Rather than trusting the raw OCR output, `plate_rules.py`:

- Validates a read against the real list of Indian state/UT codes
- Rejects structurally impossible reads (letters where digits must be, etc.)
- Auto-corrects near-misses by rebuilding the plate position-by-position
  (state → letters, district → digits, series → letters, number → digits)
  and re-validating

It can't decide between two equally *valid* reads (e.g. `MH03DY5705` vs
`MH03DZ5705`) — that's handled separately by a voting step, not by rules.

## How it works

```
RTSP live camera  OR  local video file          (chosen by USE_RTSP in config.py)
     │
     ▼
YOLOv8 (vehicle detection) + ByteTrack (persistent vehicle IDs)
     │
     ▼
YOLOv8 (plate detection) on the same frame
     │
     ▼
PaddleOCR on the plate crop → plate_rules.py correction
     │
     ▼
Per-vehicle-ID vote counter (needs MIN_VOTES matching reads before it's trusted)
     │
     ▼
POST /api/v1/upload-evidence (crop image) + POST /api/v1/detections/add
     │
     ▼
FastAPI  →  SQLite
     │
     ▼
Streamlit dashboard (polls the API every 30s, shows detections + evidence crops)
```

**This loop is single-threaded and synchronous.** Detection, OCR, and the
API calls all happen in sequence on one thread per frame. It's simple and
easy to reason about, but a slow or unreachable API will stall the frame
loop. If you need real throughput, the detection loop and the network I/O
should run on separate threads/processes with a queue between them — that
refactor hasn't been done yet.

### Two run modes (`USE_RTSP` in `config.py`)

- `USE_RTSP = False` → reads the local test video (`videos/<file>.mp4`),
  draws annotations, and writes `output.mp4`. If run inside Google Colab it
  auto-triggers a download of that file.
- `USE_RTSP = True` → opens the live RTSP stream via the FFMPEG backend with
  a 1-frame buffer (grabs the latest frame, not a backlog), shows a live
  `ANPR Live` window (press `q` to quit), and **auto-reconnects** on dropped
  frames — giving up only after ~1 minute of continuous failure.

A live IP camera lives on your **local network**, so the pipeline must run on
a machine on that same network. It will **not** work from Google Colab in RTSP
mode (a cloud VM can't reach `192.168.x.x`, and there's no display).

## How FastAPI and Streamlit talk to each other

They're not integrated in a tight sense — no shared imports, no direct
function calls. They're two separate processes that only communicate over
plain HTTP:

- FastAPI (`main.py` + `routes.py`) runs on `localhost:8000` and exposes the
  REST endpoints in the [API](#api) table below.
- Streamlit (`dashboard.py`) runs on its own port. Every time it needs data
  it calls `requests.get("http://localhost:8000/api/v1/detections")`, parses
  the JSON into a pandas DataFrame, and renders it as an HTML table.
- Every 30 seconds the dashboard calls `st.rerun()`, which re-executes the
  whole script and hits the API fresh. That's what makes the table look
  "live" — it's **polling**, not a persistent connection.
- `anpr_colab.py` never talks to Streamlit directly. It POSTs confirmed
  plate reads to FastAPI, which writes to SQLite. Streamlit just reads
  whatever's in that database via the API.

Mental model: **FastAPI is the only thing that touches the database.
Streamlit is a thin client that polls FastAPI over REST.**

## Project structure

```
anpr_colab.py     — detection + tracking + OCR + voting; live-window or output.mp4
plate_rules.py    — Indian plate validation/correction (pure logic, no I/O)
config.py         — USE_RTSP toggle, RTSP URL, video source, model paths, API base URL
main.py           — FastAPI app: CORS, static mount, table creation, upload-evidence
routes.py         — API endpoints (/api/v1 router)
crud.py           — SQLAlchemy queries
models.py         — DetectionLog table definition
db.py             — engine/session setup (SQLite default, DATABASE_URL override)
dashboard.py      — Streamlit live dashboard + manual entry form
models/           — YOLO weights: best.pt (plate), yolov8n.pt (vehicle)   [not committed]
videos/           — local test clips used when USE_RTSP = False           [not committed]
static/crops/     — saved evidence crops served at /static                [not committed]
```

## Hardware requirements

This runs two YOLOv8 models plus PaddleOCR per relevant frame — it needs a
GPU to keep up with a live stream. Tested on:

- **Google Colab** (Colab GPU runtime, video-file mode), or
- **Any NVIDIA GPU with 16–32GB+ VRAM** for local/on-prem RTSP deployment

`DEVICE` in `anpr_colab.py` selects the compute device:

- `0` → NVIDIA GPU (current default)
- `"mps"` → Apple Silicon Mac (M1/M2/M3)
- `"cpu"` → CPU (works everywhere, too slow for live RTSP; fine for a short clip)

## Dataset & Training

The plate-detection model (`best.pt`) was trained on the
[License Plate Recognition dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
(v4, `resized640_aug3x-ACCURATE`) from Roboflow Universe, by Roboflow
Universe Projects, licensed CC BY 4.0.

- **24,242 images total** (21,174 train / 2,048 valid / 1,020 test — an
  87/8/4 split), single class: license plate, object detection format
- **Preprocessing:** auto-orient, resized to 640×640
- **Augmentation (3x per source image):** horizontal flip, up to 15% zoom
  crop, ±10° rotation, ±2° shear, 10% grayscale, ±15% hue/saturation/
  brightness/exposure jitter, slight blur, cutout

*(Add your own fine-tuning details here once finalized — base checkpoint,
epochs, batch size, resulting mAP — so the training run is reproducible.)*

## Setup

```bash
pip install -r requirements.txt
```

You need two model weight files placed in a `models/` folder (the paths are
built in `config.py` from `MODELS_DIR`):

- `models/yolov8n.pt` — standard YOLOv8-nano vehicle detector
- `models/best.pt` — your own trained plate-detection model (not included;
  see [Dataset & Training](#dataset--training) above)

Configure the source in `config.py`:

```python
USE_RTSP = True                                      # True = live camera, False = test video
RTSP_URL = "rtsp://<camera-ip>:554/<stream-path>"    # your camera
# when USE_RTSP = False, it reads videos/bike1.mp4
```

> ⚠️ **Do not commit a real camera IP/credentials.** Move `RTSP_URL` to an
> environment variable or `.env` file before pushing. The value currently in
> `config.py` points at a real internal camera and should be replaced with a
> placeholder.

Optional: use Postgres instead of the default local SQLite file:

```bash
# .env
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

## Running it

Three separate processes:

```bash
# 1. Backend API (creates tables on first run, serves evidence images at /static)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. AI pipeline (reads the source configured in config.py, POSTs confirmed plates)
python anpr_colab.py

# 3. Dashboard
streamlit run dashboard.py
```

## API

| Method | Endpoint                       | Purpose                                          |
|--------|--------------------------------|--------------------------------------------------|
| GET    | `/`                            | Health check                                     |
| GET    | `/api/v1/detections`           | List recent detections                           |
| GET    | `/api/v1/detections/search?q=` | Partial plate search                             |
| GET    | `/api/v1/detections/{id}`      | Get one detection by ID                          |
| POST   | `/api/v1/detections/add`       | Ingest a new detection (used by the AI pipeline) |
| POST   | `/api/v1/upload-evidence`      | Upload a crop image, returns its `/static` URL   |
| GET    | `/api/v1/analytics`            | Total/today counts, vehicle breakdown            |

None of these endpoints require authentication — see [Limitations](#limitations).

## Known gaps / bugs

Things that are half-wired and worth cleaning up:

- **`vehicle_color` isn't persisted.** The dashboard sends a `vehicle_color`
  field and renders a Color column, but the `DetectionLog` model, the
  `AIDetectionCreate` schema, and `crud.create_detection` have no such field —
  so it's dropped on ingest and the column always shows "—". Either add the
  column end-to-end or remove it from the UI.
- **Evidence crop filename mismatch.** The pipeline saves crops as
  `static/crops/{plate}.jpg`, but the dashboard's "🏷️ Plate" link points at
  `{plate}_plate.jpg` (and falls back to `{plate}_vehicle.jpg`), which don't
  exist → broken thumbnails. Align the filenames.
- **Tunnel header mismatch.** The pipeline sends a localtunnel header
  (`Bypass-Tunnel-Reminder`) while the dashboard sends an ngrok header
  (`ngrok-skip-browser-warning`). Pick one tunnel and standardize.

## Limitations

Being upfront about what this is *not*:

- **No authentication on any endpoint.** `/detections/add` and
  `/upload-evidence` accept requests from anyone who can reach them. Fine on
  `localhost`; not fine exposed over a tunnel or the public internet as-is.
- **Single-threaded pipeline.** No async/queue decoupling between capture,
  inference, and the network calls.
- **SQLite by default.** Works for a single-camera prototype; not meant for
  concurrent writers or high query volume.
- **CORS is wide open** (`allow_origins=["*"]`) — acceptable for local dev, not
  for a public deployment.
- **No automated tests.** `plate_rules.py` in particular is pure logic and
  would be easy to unit-test — that's the first thing worth adding (it already
  has a `__main__` self-test block to build on).
- **Camera credentials / IPs belong in environment variables**, not committed
  config files.

## Roadmap

- [ ] Move capture/inference/API-call onto separate threads with a queue
- [ ] Add API key auth on the ingest + upload endpoints
- [ ] Unit tests for `plate_rules.py`
- [ ] Persist `vehicle_color` end-to-end (or drop the UI column)
- [ ] Align evidence crop filenames between pipeline and dashboard
- [ ] Swap default SQLite for Postgres in a docker-compose setup
- [ ] Fill in fine-tuning details (base checkpoint, epochs, mAP)
