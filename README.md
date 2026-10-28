# AI Vision Tracker

A complete local Python + Flask application for object detection and multi-object tracking using YOLO, ByteTrack, OpenCV, and SQLite.

## Features
- Live webcam detection with labeled boxes, IDs, movement trails, live class counts, and FPS.
- Video upload (MP4/MOV/AVI/MKV/WEBM), processing, and MP4 download.
- Detection session history stored locally in SQLite.
- Responsive dark dashboard; no API key required.

## Windows quick start
1. Install Python 3.10–3.12 (64-bit) and check **Add Python to PATH**.
2. Extract the ZIP, open `AI_Vision_Tracker`, double-click `start.bat`.
3. Wait for dependencies to install (first launch requires internet). A browser opens at `http://127.0.0.1:5000`.
4. Click **Start camera**, or upload a short video. First use downloads the pretrained YOLO weights automatically.

## Manual startup (Windows PowerShell)
```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

## Linux/macOS
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```
Open `http://127.0.0.1:5000`.

## Configuration
- `YOLO_MODEL` (default `yolo11n.pt`) — lightweight pretrained model.
- `YOLO_CONF` (default `0.35`) — minimum detection confidence.
- `CAMERA_INDEX` (default `0`) — webcam device index.

## Important notes
- Video processing is synchronous: the upload request waits until inference finishes. Begin with a short 10–30 second clip; CPU-only processing may be slow.
- Video outputs are written using OpenCV's `mp4v` codec. If a browser cannot play them, download and open them with a desktop video player.
- `counts` for a processed video represent **frame-level detections accumulated across frames**, not unique object totals. `unique_objects` counts distinct tracking IDs (an object may get a new ID if tracking is lost).
- Webcam sessions are saved to history when the stream stops. Only one camera stream is supported at a time.
- The pretrained model supports its training categories (commonly COCO categories), not arbitrary custom objects.
- This project is intended to run locally. Do not expose Flask's development server to the public internet.
- Check the licensing terms of Ultralytics and model weights before commercial use.

## Project layout
- `app.py` — API, inference, tracking, video, history
- `templates/index.html` — UI
- `static/css/style.css` — styling
- `static/js/script.js` — UI behavior
- `instance/history.sqlite3` — created automatically
- `outputs/` — saved tracked videos

## Troubleshooting
- **Camera unavailable:** Allow desktop apps to use the camera in Windows Privacy settings and close Zoom/Teams.
- **Slow detection:** Use short videos, lower-resolution input, and the default nano model. CPU inference may be slow.
- **Model download fails:** Check internet connection on first run.
- **Port already in use:** Close another program on port 5000 or edit `app.run(port=5000)` and the browser URL.
- **Installation fails:** Try Python 3.11 64-bit and a fresh `.venv`.
