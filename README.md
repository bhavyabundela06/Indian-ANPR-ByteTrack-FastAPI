# Indian ANPR — YOLOv8 + ByteTrack + FastAPI

A license plate recognition pipeline built specifically for **Indian plates**, not
adapted from a US/EU template. Detects vehicles, tracks them across frames,
reads the plate, and corrects OCR misreads using the actual structure of an
Indian registration plate — then logs confirmed reads to a database with a
live dashboard to watch it happen.

This is a working prototype, not a hardened production system. See
[Limitations](#limitations) below for what that means concretely.

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
RTSP / video feed
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
Per-vehicle-ID vote counter (needs N matching reads before it's trusted)
     │
     ▼
POST /api/v1/detections/add  →  FastAPI  →  SQLite
     │
     ▼
Streamlit dashboard (polls the API, shows live detections + evidence crops)
```

**This loop is single-threaded and synchronous.** Detection, OCR, and the
API call all happen in sequence on one thread per frame. It's simple and
easy to reason about, but a slow or unreachable API will stall the frame
loop. If you need real throughput, the detection loop and the network I/O
should run on separate threads/processes with a queue between them — that
refactor hasn't been done yet.

## Project structure

```
anpr_colab.py     — detection + tracking + OCR + voting, POSTs to the API
plate_rules.py    — Indian plate validation/correction (pure logic, no I/O)
config.py         — RTSP URL, camera location, API base URL
main.py           — FastAPI app: CORS, static file mount, table creation
routes.py         — API endpoints
crud.py           — SQLAlchemy queries
models.py         — DetectionLog table definition
db.py             — engine/session setup
dashboard.py      — Streamlit live dashboard + manual entry form
```

## Setup

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
*Configure Your Environment
•	Download your custom plate detection model and place it in the models/ folder as best.pt.
•	Ensure yolov8n.pt is also present in the models/ folder.
•	Place your test video in the videos/ folder.

*Run the System
1.	Backend: uvicorn main:app --reload
2.	AI Pipeline: python anpr_colab.py
3.	Dashboard: streamlit run dashboard.py

📊 Dataset & Training
The object detection models utilized in this pipeline were trained and evaluated using high-quality Indian traffic data sourced from Kaggle.
•	Dataset Link: [Insert your Kaggle dataset link here]
=======
You need two model weight files in the project root:
- `yolov8n.pt` — auto-downloads on first run via ultralytics
- `best.pt` — your own trained plate-detection model (not included; train
  or source this yourself)

Set your camera/stream source in `config.py`:

```python
RTSP_URL = "rtsp://<user>:<pass>@<camera-ip>:554/<stream-path>"
```
>>>>>>> d37dc33 (Modified README.md)

⚠️ Don't commit real camera credentials. Move `RTSP_URL` to an environment
variable or `.env` file before pushing config changes — the current
`config.py` has a placeholder for this reason.

<<<<<<< HEAD

=======
Optional: set a real database instead of the default local SQLite file:

```bash
# .env
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

## Running it

Three separate processes:

```bash
# 1. Backend API (creates tables on first run, serves evidence images at /static)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. Dashboard
streamlit run dashboard.py

# 3. AI pipeline (reads from RTSP_URL in config.py, POSTs confirmed plates)
python anpr_colab.py
```

## API

| Method | Endpoint                     | Purpose                              |
|--------|-------------------------------|---------------------------------------|
| GET    | `/api/v1/detections`          | List recent detections                |
| GET    | `/api/v1/detections/{id}`     | Get one detection by ID               |
| GET    | `/api/v1/detections/search?q=`| Partial plate search                  |
| POST   | `/api/v1/detections/add`      | Ingest a new detection (used by the AI pipeline) |
| GET    | `/api/v1/analytics`           | Total/today counts, vehicle breakdown |

None of these endpoints require authentication — see Limitations.

## Limitations

Being upfront about what this is *not*, so nobody's surprised:

- **No authentication on any endpoint.** `/detections/add` will accept a
  POST from anyone who can reach it. Fine on `localhost`; not fine exposed
  over ngrok or the public internet as-is.
- **Single-threaded pipeline.** See the architecture note above — no
  async/queue decoupling between capture, inference, and the network call.
- **SQLite by default.** Works for a single-camera prototype; not meant
  for concurrent writers or high query volume.
- **No automated tests.** `plate_rules.py` in particular is pure logic
  with no I/O and would be easy to unit-test — that's the first thing
  worth adding.
- **Camera credentials belong in environment variables**, not committed
  config files.

## Roadmap

- [ ] Move capture/inference/API-call onto separate threads with a queue
- [ ] Add API key auth on the ingest endpoint
- [ ] Unit tests for `plate_rules.py`
- [ ] Swap default SQLite for Postgres in the docker-compose setup
>>>>>>> d37dc33 (Modified README.md)
