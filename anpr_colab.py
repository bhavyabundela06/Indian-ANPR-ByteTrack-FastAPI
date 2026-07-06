
"""
anpr_colab.py — AI pipeline: detect vehicles + plates, OCR, vote, POST to backend.

FIXES vs. previous version:
1. read_plate() only handled the LEGACY PaddleOCR list output. If _ocr.predict()
   succeeded (PaddleOCR 3.x returns a dict with rec_texts/rec_scores), iterating
   result[0] as `line[1][1]` crashed with KeyError/TypeError and silently killed
   every read. Restored the dual-format handling from the desktop version.
2. DEVICE was defined but never passed to the model calls — GPU was unused.
3. preprocess() lost the 200–320px upscale branch; restored.
4. requests.post() had no timeout — a dead backend froze the whole video loop.
5. evidence_url sent a local disk path ("static/crops/X.jpg"). The dashboard
   builds links as API_BASE_URL + url, so we now send a URL path with a
   leading slash ("/static/crops/X.jpg") that main.py's StaticFiles mount serves.
6. RTSP URL and API base now come from config.py (single source of truth) —
   the old hardcoded RTSP URL here was missing the admin:admin credentials.
7. Votes for already-saved track IDs are cleared so the per-frame loop doesn't
   re-scan them forever.
"""

import os
from collections import defaultdict, Counter

import cv2
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR

from plate_rules import validate_or_correct
import config

# Use the dynamic paths from config.py
SOURCE = config.VIDEO_SOURCE             
PLATE_MODEL = config.PLATE_MODEL
VEHICLE_MODEL = config.VEHICLE_MODEL

VEHICLE_CLASSES = [2, 3, 5, 7]       
V_MAP = {2: "car", 3: "bike", 5: "bus", 7: "truck"}
# --- DEVICE SWITCH ---
# Use 0 for NVIDIA GPU, 'cpu' for standard processor.
# If you are on a Mac with M1/M2/M3, you can use 'mps'.
DEVICE = 0                           
VEHICLE_CONF = 0.35
PLATE_CONF = 0.35
OCR_MIN_CONF = 0.55
MIN_VOTES = 3
OCR_EVERY = 2

API_URL = f"{config.API_BASE_URL}/api/v1/detections/add"
CROPS_DIR = "static/crops"
os.makedirs(CROPS_DIR, exist_ok=True)

_ocr = PaddleOCR(lang="en")


def preprocess(crop):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if max(h, w) < 200:
        crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) < 320:  # FIXED: this branch had been dropped
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def read_plate(crop):
    """OCR a plate crop. Handles BOTH PaddleOCR 3.x dict output and the
    legacy nested-list output — the previous version only handled legacy,
    so every read failed on newer PaddleOCR installs."""
    img = preprocess(crop)
    if img is None:
        return None, 0.0
    try:
        result = _ocr.predict(img)
    except AttributeError:
        result = _ocr.ocr(img)
    if not result or not result[0]:
        return None, 0.0

    res = result[0]
    pairs = []
    # New API: dict-like with "rec_texts" / "rec_scores"
    try:
        for t, s in zip(res["rec_texts"], res["rec_scores"]):
            if t and t.strip() and float(s) >= OCR_MIN_CONF:
                pairs.append((t, float(s)))
    except (KeyError, TypeError, IndexError):
        # Legacy API: list of [box, (text, score)]
        if isinstance(res, (list, tuple)):
            for line in res:
                try:
                    if float(line[1][1]) >= OCR_MIN_CONF:
                        pairs.append((line[1][0], float(line[1][1])))
                except (IndexError, TypeError):
                    continue

    if not pairs:
        return None, 0.0

    cands = [max(pairs, key=lambda p: p[1])]
    if len(pairs) > 1:  # joined multi-line read as a second candidate
        cands.append(("".join(t for t, _ in pairs),
                      sum(s for _, s in pairs) / len(pairs)))
    fixed = [(validate_or_correct(t), c) for t, c in cands]
    fixed = [(p, c) for p, c in fixed if p]
    if not fixed:
        return None, 0.0
    return max(fixed, key=lambda x: x[1])


def send_to_api(plate, conf, v_class, evidence_url):
    """POST a confirmed plate to the FastAPI backend."""
    payload = {
        "camera_id": "CAM-01-MAIN",
        "plate_number": plate,
        "vehicle_type": v_class,
        "confidence": float(conf),
        "evidence_url": evidence_url,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)  # FIXED: timeout
        if r.status_code == 200:
            print(f"📡 Sent to API: {plate}")
        else:
            print(f"⚠️ API returned {r.status_code} for {plate}: {r.text[:200]}")
    except requests.RequestException as e:
        print(f"❌ Failed to reach API: {e}")


def main():
    print("⏳ Loading models...")
    vehicle_model = YOLO(VEHICLE_MODEL)
    plate_model = YOLO(PLATE_MODEL)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {SOURCE}")
        return

    votes = defaultdict(Counter)
    plate_meta = {}
    saved_ids = set()
    saved_plates = set()
    frame_idx = 0

    print("🚀 Processing stream...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        vres = vehicle_model.track(
            frame, persist=True, classes=VEHICLE_CLASSES, conf=VEHICLE_CONF,
            tracker="bytetrack.yaml", device=DEVICE, verbose=False,  # FIXED: device
        )[0]

        tracks = []
        if vres.boxes is not None and vres.boxes.id is not None:
            ids = [int(i) for i in vres.boxes.id.tolist()]
            cls_ids = [int(c) for c in vres.boxes.cls.tolist()]
            xyxy = vres.boxes.xyxy.tolist()
            for (x1, y1, x2, y2), cid, cls_id in zip(xyxy, ids, cls_ids):
                tracks.append((x1, y1, x2, y2, cid, V_MAP.get(cls_id, "vehicle")))

        if frame_idx % OCR_EVERY == 0 and tracks:
            pres = plate_model(frame, verbose=False, conf=PLATE_CONF, device=DEVICE)[0]
            for pb in pres.boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                matched = next(
                    (t for t in tracks
                     if px1 > t[0] and py1 > t[1] and px2 < t[2] and py2 < t[3]),
                    None,
                )
                if not matched:
                    continue
                cid, v_class = matched[4], matched[5]
                if cid in saved_ids:
                    continue

                crop = frame[max(0, py1 - 5):py2 + 5, max(0, px1 - 5):px2 + 5].copy()
                if crop.size == 0:
                    continue
                plate, conf = read_plate(crop)
                if not plate:
                    continue

                votes[cid][plate] += 1
                key = (cid, plate)
                if key not in plate_meta or conf > plate_meta[key]["conf"]:
                    plate_meta[key] = {"conf": conf, "crop": crop, "v_class": v_class}

        for cid, counter in list(votes.items()):
            if cid in saved_ids:
                continue
            winner, count = counter.most_common(1)[0]
            if count < MIN_VOTES:
                continue

            saved_ids.add(cid)
            del votes[cid]  # FIXED: stop re-scanning finished tracks
            if winner in saved_plates:
                continue
            saved_plates.add(winner)

            meta = plate_meta[(cid, winner)]
            crop_file = os.path.join(CROPS_DIR, f"{winner}.jpg")
            cv2.imwrite(crop_file, meta["crop"])
            print(f"✅ CONFIRMED {winner} ({meta['conf']:.2f})")

            # FIXED: send a URL path (served by main.py's /static mount),
            # not a local disk path
            send_to_api(winner, meta["conf"], meta["v_class"],
                        f"/static/crops/{winner}.jpg")

    cap.release()
    print(f"\n📊 Unique plates saved: {len(saved_plates)}")
    print("✅ AI processing complete.")


if __name__ == "__main__":
    main()