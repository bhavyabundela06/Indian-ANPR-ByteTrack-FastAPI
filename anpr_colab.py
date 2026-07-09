# """
# anpr_colab.py — AI pipeline: detect vehicles + plates, OCR, vote, POST to backend.

# CHANGES IN THIS VERSION:
# 1. NEW: writes an annotated output.mp4 (Colab has no live display, so this is
#    how you actually see the detections). Draws:
#      - a green box + track ID + vehicle type on every tracked vehicle
#      - a yellow box on each detected plate region
#      - once a plate is CONFIRMED (passed the vote threshold), that plate's
#        text is drawn on its vehicle for the rest of the video, not just the
#        one frame it was confirmed on
#    At the end, if running inside Colab, it auto-triggers a download of
#    output.mp4. If not on Colab, it just tells you the file path.
# 2. FIXED: DEVICE was set to "mps" (Apple Silicon). Colab's GPU is NVIDIA,
#    so "mps" would raise a runtime error there. Set to 0 for Colab's GPU —
#    change back to "mps" if you're testing locally on a Mac instead.
# """

# import os
# os.environ["FLAGS_enable_pir_api"] = "0"
# from collections import defaultdict, Counter

# import cv2
# import requests
# from ultralytics import YOLO
# from paddleocr import PaddleOCR

# from plate_rules import validate_or_correct
# import config

# # Use the dynamic paths from config.py
# SOURCE = config.VIDEO_SOURCE
# PLATE_MODEL = config.PLATE_MODEL
# VEHICLE_MODEL = config.VEHICLE_MODEL

# VEHICLE_CLASSES = [2, 3, 5, 7]
# V_MAP = {2: "car", 3: "bike", 5: "bus", 7: "truck"}

# # --- DEVICE SWITCH ---
# # Use 0 for NVIDIA GPU (Colab), 'cpu' for standard processor.
# # If you are on a Mac with M1/M2/M3, use 'mps' instead.
# DEVICE = 0  # FIXED: was "mps" — wrong for Colab's NVIDIA GPU

# VEHICLE_CONF = 0.35
# PLATE_CONF = 0.35
# OCR_MIN_CONF = 0.55
# MIN_VOTES = 3
# OCR_EVERY = 2

# API_URL = f"{config.API_BASE_URL}/api/v1/detections/add"
# CROPS_DIR = "static/crops"
# os.makedirs(CROPS_DIR, exist_ok=True)

# # NEW: output video settings
# OUTPUT_VIDEO = "output.mp4"
# BOX_COLOR_VEHICLE = (0, 255, 0)      # green
# BOX_COLOR_PLATE = (0, 220, 255)      # yellow
# BOX_COLOR_CONFIRMED = (255, 120, 0)  # blue-ish, marks a confirmed plate

# _ocr = PaddleOCR(lang="en")


# def preprocess(crop):
#     if crop is None or crop.size == 0:
#         return None
#     h, w = crop.shape[:2]
#     if max(h, w) < 200:
#         crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
#     elif max(h, w) < 320:
#         crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
#     lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
#     return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# def read_plate(crop):
#     """OCR a plate crop. Handles BOTH PaddleOCR 3.x dict output and the
#     legacy nested-list output."""
#     img = preprocess(crop)
#     if img is None:
#         return None, 0.0
#     try:
#         result = _ocr.predict(img)
#     except AttributeError:
#         result = _ocr.ocr(img)
#     if not result or not result[0]:
#         return None, 0.0

#     res = result[0]
#     pairs = []
#     try:
#         for t, s in zip(res["rec_texts"], res["rec_scores"]):
#             if t and t.strip() and float(s) >= OCR_MIN_CONF:
#                 pairs.append((t, float(s)))
#     except (KeyError, TypeError, IndexError):
#         if isinstance(res, (list, tuple)):
#             for line in res:
#                 try:
#                     if float(line[1][1]) >= OCR_MIN_CONF:
#                         pairs.append((line[1][0], float(line[1][1])))
#                 except (IndexError, TypeError):
#                     continue

#     if not pairs:
#         return None, 0.0

#     cands = [max(pairs, key=lambda p: p[1])]
#     if len(pairs) > 1:
#         cands.append(("".join(t for t, _ in pairs),
#                       sum(s for _, s in pairs) / len(pairs)))
#     fixed = [(validate_or_correct(t), c) for t, c in cands]
#     fixed = [(p, c) for p, c in fixed if p]
#     if not fixed:
#         return None, 0.0
#     return max(fixed, key=lambda x: x[1])


# def send_to_api(plate, conf, v_class, crop_img):
#     base_url = config.API_BASE_URL.rstrip('/')
#     upload_url = f"{base_url}/api/v1/upload-evidence"
#     detect_url = f"{base_url}/api/v1/detections/add"

#     _, img_encoded = cv2.imencode('.jpg', crop_img)
#     img_bytes = img_encoded.tobytes()

#     headers = {"Bypass-Tunnel-Reminder": "true"}
#     evidence_url = None

#     try:
#         files = {"file": (f"{plate}.jpg", img_bytes, "image/jpeg")}
#         img_res = requests.post(upload_url, files=files, headers=headers, timeout=15)

#         if img_res.status_code == 200:
#             evidence_url = img_res.json().get("evidence_url")
#         else:
#             print(f"⚠️ Image Rejected (Code {img_res.status_code}) at {upload_url}")
#             print(f"Server Response: {img_res.text[:100]}")
#     except Exception as e:
#         print(f"❌ Image upload failed: {e}")

#     payload = {
#         "camera_id": "CAM-01-MAIN",
#         "plate_number": plate,
#         "vehicle_type": v_class,
#         "confidence": float(conf),
#         "evidence_url": evidence_url,
#     }

#     try:
#         r = requests.post(detect_url, json=payload, headers=headers, timeout=10)
#         if r.status_code == 200:
#             print(f"📡 Uploaded image & logged to database: {plate}")
#         else:
#             print(f"⚠️ Database API returned {r.status_code} for {plate}: {r.text[:200]}")
#     except requests.RequestException as e:
#         print(f"❌ Failed to reach Database API: {e}")


# def draw_label(frame, x1, y1, text, color, above=True):
#     """Small helper: a filled label strip so text stays readable over any background."""
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     scale, thickness = 0.6, 2
#     (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
#     y_text = y1 - 8 if above else y1 + th + 8
#     cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), color, -1)
#     cv2.putText(frame, text, (x1 + 3, y_text), font, scale, (0, 0, 0), thickness)


# def main():
#     print("⏳ Loading models...")
#     vehicle_model = YOLO(VEHICLE_MODEL)
#     plate_model = YOLO(PLATE_MODEL)

#     cap = cv2.VideoCapture(SOURCE)
#     if not cap.isOpened():
#         print(f"❌ Cannot open source: {SOURCE}")
#         return

#     # NEW: set up the annotated output video writer
#     fps = cap.get(cv2.CAP_PROP_FPS) or 20
#     frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

#     votes = defaultdict(Counter)
#     plate_meta = {}
#     saved_ids = set()
#     saved_plates = set()
#     confirmed_label = {}  # NEW: cid -> confirmed plate text, drawn every frame after confirmation
#     frame_idx = 0

#     print("🚀 Processing stream... (writing annotated frames to output.mp4)")
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         frame_idx += 1

#         vres = vehicle_model.track(
#             frame, persist=True, classes=VEHICLE_CLASSES, conf=VEHICLE_CONF,
#             tracker="bytetrack.yaml", device=DEVICE, verbose=False,
#         )[0]

#         tracks = []
#         if vres.boxes is not None and vres.boxes.id is not None:
#             ids = [int(i) for i in vres.boxes.id.tolist()]
#             cls_ids = [int(c) for c in vres.boxes.cls.tolist()]
#             xyxy = vres.boxes.xyxy.tolist()
#             for (x1, y1, x2, y2), cid, cls_id in zip(xyxy, ids, cls_ids):
#                 v_class = V_MAP.get(cls_id, "vehicle")
#                 tracks.append((x1, y1, x2, y2, cid, v_class))

#                 # NEW: draw the vehicle box + label every frame
#                 ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
#                 cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), BOX_COLOR_VEHICLE, 2)
#                 if cid in confirmed_label:
#                     label = f"{confirmed_label[cid]} ({v_class})"
#                     draw_label(frame, ix1, iy1, label, BOX_COLOR_CONFIRMED)
#                 else:
#                     draw_label(frame, ix1, iy1, f"ID {cid} {v_class}", BOX_COLOR_VEHICLE)

#         if frame_idx % OCR_EVERY == 0 and tracks:
#             pres = plate_model(frame, verbose=False, conf=PLATE_CONF, device=DEVICE)[0]
#             for pb in pres.boxes:
#                 px1, py1, px2, py2 = map(int, pb.xyxy[0])

#                 # NEW: draw every detected plate region, even before OCR/voting
#                 cv2.rectangle(frame, (px1, py1), (px2, py2), BOX_COLOR_PLATE, 2)

#                 matched = next(
#                     (t for t in tracks
#                      if px1 > t[0] and py1 > t[1] and px2 < t[2] and py2 < t[3]),
#                     None,
#                 )
#                 if not matched:
#                     continue
#                 cid, v_class = matched[4], matched[5]
#                 if cid in saved_ids:
#                     continue

#                 crop = frame[max(0, py1 - 5):py2 + 5, max(0, px1 - 5):px2 + 5].copy()
#                 if crop.size == 0:
#                     continue
#                 plate, conf = read_plate(crop)
#                 if not plate:
#                     continue

#                 votes[cid][plate] += 1
#                 key = (cid, plate)
#                 if key not in plate_meta or conf > plate_meta[key]["conf"]:
#                     plate_meta[key] = {"conf": conf, "crop": crop, "v_class": v_class}

#         for cid, counter in list(votes.items()):
#             if cid in saved_ids:
#                 continue
#             winner, count = counter.most_common(1)[0]
#             if count < MIN_VOTES:
#                 continue

#             saved_ids.add(cid)
#             del votes[cid]
#             confirmed_label[cid] = winner  # NEW: keep labeling this vehicle from now on
#             if winner in saved_plates:
#                 continue
#             saved_plates.add(winner)

#             meta = plate_meta[(cid, winner)]
#             crop_file = os.path.join(CROPS_DIR, f"{winner}.jpg")
#             cv2.imwrite(crop_file, meta["crop"])
#             print(f"✅ CONFIRMED {winner} ({meta['conf']:.2f})")

#             send_to_api(winner, meta["conf"], meta["v_class"], meta["crop"])

#         out.write(frame)  # NEW: write the annotated frame

#     cap.release()
#     out.release()  # NEW
#     print(f"\n📊 Unique plates saved: {len(saved_plates)}")
#     print(f"📹 Annotated video written to {OUTPUT_VIDEO}")

#     # NEW: auto-download if running in Colab; otherwise just point at the file
#     try:
#         from google.colab import files as colab_files
#         print("⬇️ Triggering download of the annotated video...")
#         colab_files.download(OUTPUT_VIDEO)
#     except ImportError:
#         print(f"Not running in Colab — find the video at ./{OUTPUT_VIDEO}")

#     print("✅ AI processing complete.")


# if __name__ == "__main__":
#     main()




"""
anpr_colab.py — AI pipeline: detect vehicles + plates, OCR, vote, POST to backend.

RUNS LOCALLY (VS Code) on a video file OR a live IP camera (RTSP).

SOURCE is chosen in config.py:
  - USE_RTSP = False  -> reads the local test video, writes annotated output.mp4
  - USE_RTSP = True   -> reads the live RTSP stream, shows a live window
                         (press 'q' to quit) and auto-reconnects on drops.

NOTE: a live IP camera lives on your LOCAL network, so run this on a machine
on the SAME network as the camera. It will NOT work in Google Colab (its cloud
VM can't reach 192.168.x.x, and there's no display for the live window).
"""

import os
os.environ["FLAGS_enable_pir_api"] = "0"
import time
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
# 'mps'  -> Mac with Apple Silicon (M1/M2/M3)  <-- local default
# 0      -> NVIDIA GPU
# 'cpu'  -> standard processor (works everywhere, slower)
DEVICE = 0

VEHICLE_CONF = 0.35
PLATE_CONF = 0.35
OCR_MIN_CONF = 0.55
MIN_VOTES = 3
OCR_EVERY = 2

API_URL = f"{config.API_BASE_URL}/api/v1/detections/add"
CROPS_DIR = "static/crops"
os.makedirs(CROPS_DIR, exist_ok=True)

# Output video settings (used only in video-file mode)
OUTPUT_VIDEO = "output.mp4"
BOX_COLOR_VEHICLE = (0, 255, 0)      # green
BOX_COLOR_PLATE = (0, 220, 255)      # yellow
BOX_COLOR_CONFIRMED = (255, 120, 0)  # blue-ish, marks a confirmed plate

_ocr = PaddleOCR(lang="en")


def preprocess(crop):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if max(h, w) < 200:
        crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) < 320:
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def read_plate(crop):
    """OCR a plate crop. Handles BOTH PaddleOCR 3.x dict output and the
    legacy nested-list output."""
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
    try:
        for t, s in zip(res["rec_texts"], res["rec_scores"]):
            if t and t.strip() and float(s) >= OCR_MIN_CONF:
                pairs.append((t, float(s)))
    except (KeyError, TypeError, IndexError):
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
    if len(pairs) > 1:
        cands.append(("".join(t for t, _ in pairs),
                      sum(s for _, s in pairs) / len(pairs)))
    fixed = [(validate_or_correct(t), c) for t, c in cands]
    fixed = [(p, c) for p, c in fixed if p]
    if not fixed:
        return None, 0.0
    return max(fixed, key=lambda x: x[1])


def send_to_api(plate, conf, v_class, crop_img):
    base_url = config.API_BASE_URL.rstrip('/')
    upload_url = f"{base_url}/api/v1/upload-evidence"
    detect_url = f"{base_url}/api/v1/detections/add"

    _, img_encoded = cv2.imencode('.jpg', crop_img)
    img_bytes = img_encoded.tobytes()

    headers = {"Bypass-Tunnel-Reminder": "true"}
    evidence_url = None

    try:
        files = {"file": (f"{plate}.jpg", img_bytes, "image/jpeg")}
        img_res = requests.post(upload_url, files=files, headers=headers, timeout=15)

        if img_res.status_code == 200:
            evidence_url = img_res.json().get("evidence_url")
        else:
            print(f"⚠️ Image Rejected (Code {img_res.status_code}) at {upload_url}")
            print(f"Server Response: {img_res.text[:100]}")
    except Exception as e:
        print(f"❌ Image upload failed: {e}")

    payload = {
        "camera_id": "CAM-01-MAIN",
        "plate_number": plate,
        "vehicle_type": v_class,
        "confidence": float(conf),
        "evidence_url": evidence_url,
    }

    try:
        r = requests.post(detect_url, json=payload, headers=headers, timeout=10)
        if r.status_code == 200:
            print(f"📡 Uploaded image & logged to database: {plate}")
        else:
            print(f"⚠️ Database API returned {r.status_code} for {plate}: {r.text[:200]}")
    except requests.RequestException as e:
        print(f"❌ Failed to reach Database API: {e}")


def draw_label(frame, x1, y1, text, color, above=True):
    """Small helper: a filled label strip so text stays readable over any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y_text = y1 - 8 if above else y1 + th + 8
    cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), color, -1)
    cv2.putText(frame, text, (x1 + 3, y_text), font, scale, (0, 0, 0), thickness)


def main():
    print("⏳ Loading models...")
    vehicle_model = YOLO(VEHICLE_MODEL)
    plate_model = YOLO(PLATE_MODEL)

    is_rtsp = str(SOURCE).lower().startswith(("rtsp://", "http://", "https://"))

    def open_capture():
        # For RTSP use the FFMPEG backend + a 1-frame buffer so we always
        # grab the LATEST frame instead of a growing backlog (kills latency).
        if is_rtsp:
            c = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            c = cv2.VideoCapture(SOURCE)
        return c

    cap = open_capture()
    if not cap.isOpened():
        print(f"❌ Cannot open source: {SOURCE}")
        return

    # RTSP often reports fps as 0 — fall back to a sane default.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 20
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Record a file only for finite video files. A live feed is endless, so
    # we show a live window instead of writing an ever-growing output.mp4.
    out = None
    if not is_rtsp:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

    votes = defaultdict(Counter)
    plate_meta = {}
    saved_ids = set()
    saved_plates = set()
    confirmed_label = {}  # cid -> confirmed plate text, drawn every frame after confirmation
    frame_idx = 0

    if is_rtsp:
        print("🚀 Processing LIVE RTSP stream... (press 'q' in the window to stop)")
    else:
        print("🚀 Processing stream... (writing annotated frames to output.mp4)")

    consecutive_fails = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_rtsp:
                # A dropped packet must NOT kill the program — reconnect.
                consecutive_fails += 1
                print(f"⚠️ Frame read failed ({consecutive_fails}). Reconnecting…")
                cap.release()
                time.sleep(2)
                cap = open_capture()
                if consecutive_fails > 30:   # ~1 min of nothing → give up
                    print("❌ Stream unrecoverable. Exiting.")
                    break
                continue
            break  # video file simply ended
        consecutive_fails = 0
        frame_idx += 1

        vres = vehicle_model.track(
            frame, persist=True, classes=VEHICLE_CLASSES, conf=VEHICLE_CONF,
            tracker="bytetrack.yaml", device=DEVICE, verbose=False,
        )[0]

        tracks = []
        if vres.boxes is not None and vres.boxes.id is not None:
            ids = [int(i) for i in vres.boxes.id.tolist()]
            cls_ids = [int(c) for c in vres.boxes.cls.tolist()]
            xyxy = vres.boxes.xyxy.tolist()
            for (x1, y1, x2, y2), cid, cls_id in zip(xyxy, ids, cls_ids):
                v_class = V_MAP.get(cls_id, "vehicle")
                tracks.append((x1, y1, x2, y2, cid, v_class))

                # Draw the vehicle box + label every frame
                ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), BOX_COLOR_VEHICLE, 2)
                if cid in confirmed_label:
                    label = f"{confirmed_label[cid]} ({v_class})"
                    draw_label(frame, ix1, iy1, label, BOX_COLOR_CONFIRMED)
                else:
                    draw_label(frame, ix1, iy1, f"ID {cid} {v_class}", BOX_COLOR_VEHICLE)

        if frame_idx % OCR_EVERY == 0 and tracks:
            pres = plate_model(frame, verbose=False, conf=PLATE_CONF, device=DEVICE)[0]
            for pb in pres.boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0])

                # Draw every detected plate region, even before OCR/voting
                cv2.rectangle(frame, (px1, py1), (px2, py2), BOX_COLOR_PLATE, 2)

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
            del votes[cid]
            confirmed_label[cid] = winner  # keep labeling this vehicle from now on
            if winner in saved_plates:
                continue
            saved_plates.add(winner)

            meta = plate_meta[(cid, winner)]
            crop_file = os.path.join(CROPS_DIR, f"{winner}.jpg")
            cv2.imwrite(crop_file, meta["crop"])
            print(f"✅ CONFIRMED {winner} ({meta['conf']:.2f})")

            send_to_api(winner, meta["conf"], meta["v_class"], meta["crop"])

        if out is not None:
            out.write(frame)          # record only in video-file mode
        if is_rtsp:
            cv2.imshow("ANPR Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Unique plates saved: {len(saved_plates)}")

    if out is not None:
        print(f"📹 Annotated video written to {OUTPUT_VIDEO}")
        try:
            from google.colab import files as colab_files
            print("⬇️ Triggering download of the annotated video...")
            colab_files.download(OUTPUT_VIDEO)
        except ImportError:
            print(f"Not running in Colab — find the video at ./{OUTPUT_VIDEO}")

    print("✅ AI processing complete.")


if __name__ == "__main__":
    main()