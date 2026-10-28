"""AI Vision Tracker — local Flask application."""
from __future__ import annotations
import atexit
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / 'uploads'
OUTPUTS = BASE / 'outputs'
INSTANCE = BASE / 'instance'
for directory in (UPLOADS, OUTPUTS, INSTANCE):
    directory.mkdir(exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024
DB_PATH = INSTANCE / 'history.sqlite3'
MODEL_NAME = os.environ.get('YOLO_MODEL', 'yolo11n.pt')
CONFIDENCE = float(os.environ.get('YOLO_CONF', '0.35'))
_model = None
_model_lock = threading.Lock()
_camera_lock = threading.Lock()
_camera = None
_camera_running = False
_camera_stats = {'objects': 0, 'counts': {}, 'fps': 0, 'active': False}


def database():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with database() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL,
            filename TEXT, output_file TEXT, frames INTEGER NOT NULL,
            unique_objects INTEGER NOT NULL, counts_json TEXT NOT NULL
        )''')


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(MODEL_NAME)
        return _model


def draw_tracks(frame, result, trail_history=None):
    """Draw labeled tracks and return per-frame class counts + IDs."""
    boxes = result.boxes
    counts = Counter()
    seen = set()
    if boxes is None or len(boxes) == 0:
        return frame, counts, seen
    names = result.names
    xyxy = boxes.xyxy.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(classes)
    for rect, cls, conf, track_id in zip(xyxy, classes, confidences, ids):
        name = names[int(cls)]
        counts[name] += 1
        if track_id is not None:
            seen.add((int(cls), track_id))
        x1, y1, x2, y2 = rect
        color = (75, 210, 155) if track_id is not None else (235, 170, 85)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{name}  #{track_id}  {conf:.0%}' if track_id is not None else f'{name}  {conf:.0%}'
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - height - 13)), (x1 + width + 12, y1), color, -1)
        cv2.putText(frame, label, (x1 + 6, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (13, 23, 34), 1, cv2.LINE_AA)
        if trail_history is not None and track_id is not None:
            key = (int(cls), track_id)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            trail_history.setdefault(key, []).append(center)
            trail_history[key] = trail_history[key][-28:]
            points = trail_history[key]
            for start, end in zip(points, points[1:]):
                cv2.line(frame, start, end, color, 2)
    return frame, counts, seen


def track_frame(frame, tracker_state, trails):
    # persist=True retains ByteTrack identities between consecutive frames.
    # Calls are serialized because Ultralytics tracker state is not thread-safe.
    with _model_lock:
        model = _model
        result = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=CONFIDENCE, verbose=False)[0]
    return draw_tracks(frame, result, trails)


def store_session(source, filename, output_file, frames, seen, counts):
    sid = uuid.uuid4().hex[:12]
    with database() as conn:
        conn.execute('INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (
            sid, datetime.now(timezone.utc).isoformat(timespec='seconds'), source,
            filename, output_file, frames, len(seen), json.dumps(dict(counts))))
    return sid


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'model': MODEL_NAME})


@app.get('/api/stats')
def stats():
    with _camera_lock:
        return jsonify(dict(_camera_stats))


@app.get('/api/history')
def history():
    with database() as conn:
        rows = conn.execute('SELECT * FROM sessions ORDER BY created_at DESC LIMIT 40').fetchall()
    return jsonify([dict(row, counts=json.loads(row['counts_json'])) for row in rows])


def camera_frames():
    global _camera_running
    trails = {}
    seen = set()
    totals = Counter()
    frames = 0
    last = time.monotonic()
    # Use a fresh tracker instance for each camera session by resetting trackers.
    try:
        get_model()
        with _model_lock:
            if getattr(_model, 'predictor', None) is not None:
                _model.predictor.trackers = []
        while True:
            with _camera_lock:
                if not _camera_running or _camera is None:
                    break
                ok, frame = _camera.read()
            if not ok:
                break
            frame, counts, ids = track_frame(frame, None, trails)
            seen.update(ids)
            totals.update(counts)
            frames += 1
            now = time.monotonic()
            fps = 1 / max(now - last, .001)
            last = now
            with _camera_lock:
                _camera_stats.update(objects=sum(counts.values()), counts=dict(counts), fps=round(fps, 1), active=True, unique=len(seen))
            cv2.putText(frame, f'LIVE  |  {fps:.1f} FPS', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (75, 210, 155), 2)
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n'
    except Exception as exc:
        app.logger.exception('Camera stream failed: %s', exc)
    finally:
        stop_camera()
        if frames:
            store_session('camera', None, None, frames, seen, totals)


def stop_camera():
    global _camera, _camera_running
    with _camera_lock:
        _camera_running = False
        _camera_stats.update(active=False, fps=0, objects=0, counts={})
        if _camera is not None:
            _camera.release()
            _camera = None


@app.post('/api/camera/start')
def start_camera():
    global _camera, _camera_running
    with _camera_lock:
        if _camera_running:
            return jsonify({'ok': True})
        camera_index = int(os.environ.get('CAMERA_INDEX', '0'))
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if not cap.isOpened():
            cap.release()
            return jsonify({'error': 'Could not open camera. Check camera permissions and close other camera apps.'}), 503
        _camera = cap
        _camera_running = True
    return jsonify({'ok': True})


@app.post('/api/camera/stop')
def camera_stop():
    stop_camera()
    return jsonify({'ok': True})


@app.get('/api/camera/stream')
def stream():
    with _camera_lock:
        if not _camera_running:
            return jsonify({'error': 'Start the camera first.'}), 409
    return Response(camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})


@app.post('/api/video')
def process_video():
    if _camera_running:
        return jsonify({'error': 'Stop the live camera before processing a video.'}), 409
    upload = request.files.get('video')
    if not upload or not upload.filename:
        return jsonify({'error': 'Choose a video file first.'}), 400
    ext = Path(upload.filename).suffix.lower()
    if ext not in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
        return jsonify({'error': 'Supported formats: MP4, MOV, AVI, MKV, WEBM.'}), 400
    token = uuid.uuid4().hex[:12]
    filename = secure_filename(upload.filename)
    source_path = UPLOADS / f'{token}_{filename}'
    output_name = f'tracked_{token}.mp4'
    output_path = OUTPUTS / output_name
    upload.save(source_path)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        source_path.unlink(missing_ok=True)
        return jsonify({'error': 'This video cannot be opened by OpenCV.'}), 400
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps < 1 or fps > 240:
        fps = 25
    if width < 1 or height < 1:
        capture.release()
        source_path.unlink(missing_ok=True)
        return jsonify({'error': 'Video dimensions are invalid.'}), 400
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        source_path.unlink(missing_ok=True)
        return jsonify({'error': 'Could not create output video. Check OpenCV codec support.'}), 500
    frames = 0
    seen = set()
    totals = Counter()
    trails = {}
    try:
        get_model()
        with _model_lock:
            if getattr(_model, 'predictor', None) is not None:
                _model.predictor.trackers = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame, counts, ids = track_frame(frame, None, trails)
            seen.update(ids)
            totals.update(counts)
            writer.write(frame)
            frames += 1
        if frames == 0:
            raise ValueError('No readable frames were found in the video.')
        sid = store_session('video', filename, output_name, frames, seen, totals)
        return jsonify({'ok': True, 'id': sid, 'frames': frames,
                        'unique_objects': len(seen), 'counts': dict(totals),
                        'download_url': f'/api/download/{output_name}'})
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        app.logger.exception('Video processing failed')
        return jsonify({'error': f'Video processing failed: {exc}'}), 500
    finally:
        capture.release()
        writer.release()
        source_path.unlink(missing_ok=True)


@app.get('/api/download/<path:filename>')
def download(filename):
    if filename != secure_filename(filename) or not filename.startswith('tracked_') or not filename.endswith('.mp4'):
        return jsonify({'error': 'Invalid file.'}), 400
    return send_from_directory(OUTPUTS, filename, as_attachment=True)


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'Video is too large. Maximum size: 250 MB.'}), 413


init_db()
atexit.register(stop_camera)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
