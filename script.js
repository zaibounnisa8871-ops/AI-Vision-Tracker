const $ = (id) => document.getElementById(id);
let mode = 'camera';
let running = false;
let busy = false;
let statsTimer = null;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server error (${response.status}).`); }
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  return data;
}
function showCounts(counts) {
  const entries = Object.entries(counts || {}).sort((a,b) => b[1]-a[1]);
  $('class-count').textContent = `${entries.length} TYPES`;
  $('categories').textContent = entries.length;
  $('class-list').replaceChildren();
  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'no-objects';
    empty.textContent = 'No objects detected yet.';
    $('class-list').append(empty);
    return;
  }
  for (const [name, count] of entries) {
    const row = document.createElement('div'); row.className = 'class-row';
    const label = document.createElement('span'); label.textContent = `◉  ${name}`;
    const amount = document.createElement('b'); amount.textContent = count;
    row.append(label, amount); $('class-list').append(row);
  }
}
function resetStats() {
  $('objects').textContent = '0'; $('unique').textContent = '0'; $('fps').textContent = '0'; showCounts({});
}
function setMode(next) {
  if (running || busy) { status('Stop the camera or wait for video processing before switching.', true); return; }
  mode = next;
  $('camera-tab').classList.toggle('active', next === 'camera');
  $('upload-tab').classList.toggle('active', next === 'upload');
  $('camera-controls').hidden = next !== 'camera';
  $('upload-controls').hidden = next !== 'upload';
  $('empty-view').querySelector('p').textContent = next === 'camera'
    ? 'Start your camera to detect and track objects with persistent IDs.'
    : 'Choose a video and let AI track the objects in every frame.';
  $('result').hidden = true;
  status(next === 'camera' ? 'Camera mode ready.' : 'Video upload mode ready.');
}
async function pollStats() {
  try {
    const data = await jsonFetch('/api/stats');
    if (!running) return;
    if (!data.active) return;
    $('objects').textContent = data.objects || 0;
    $('unique').textContent = data.unique || 0;
    $('fps').textContent = data.fps || 0;
    showCounts(data.counts);
    $('detection-status').textContent = `${data.objects || 0} objects in frame`;
    $('detection-caption').textContent = 'Tracking and classifying live objects.';
  } catch { /* A transient stats failure should not stop video. */ }
}
async function startCamera() {
  if (running || busy) return;
  $('start-btn').disabled = true;
  status('Starting camera and loading AI model. First run may download model weights...');
  try {
    await jsonFetch('/api/camera/start', {method:'POST'});
    running = true;
    $('camera-feed').src = `/api/camera/stream?t=${Date.now()}`;
    $('camera-feed').hidden = false;
    $('empty-view').hidden = true;
    $('stop-btn').disabled = false;
    $('mode-pill').textContent = '● LIVE'; $('mode-pill').classList.add('on');
    status('Live camera running. Objects are detected and tracked automatically.');
    statsTimer = setInterval(pollStats, 700);
  } catch (err) { $('start-btn').disabled = false; status(err.message, true); }
}
async function stopCamera() {
  if (!running) return;
  running = false;
  clearInterval(statsTimer);
  $('camera-feed').removeAttribute('src'); $('camera-feed').hidden = true;
  $('empty-view').hidden = false;
  $('start-btn').disabled = false; $('stop-btn').disabled = true;
  $('mode-pill').textContent = '● STANDBY'; $('mode-pill').classList.remove('on');
  $('detection-status').textContent = 'Camera stopped';
  $('detection-caption').textContent = 'Start another session whenever you are ready.';
  resetStats();
  try { await jsonFetch('/api/camera/stop', {method:'POST'}); status('Camera stopped. Session saved to history.'); await loadHistory(); }
  catch (err) { status(err.message, true); }
}
async function processVideo() {
  const file = $('video-file').files[0];
  if (!file) { status('Choose a video file first.', true); return; }
  if (file.size > 250 * 1024 * 1024) { status('Maximum file size is 250 MB.', true); return; }
  if (busy) return;
  busy = true;
  $('process-btn').disabled = true; $('result').hidden = true;
  $('mode-pill').textContent = '● PROCESSING'; $('mode-pill').classList.add('on');
  $('detection-status').textContent = 'Analyzing video…';
  $('detection-caption').textContent = 'Please wait; long videos can take several minutes on CPU.';
  status('Uploading and processing video. Keep this tab open until analysis finishes.');
  const form = new FormData(); form.append('video', file);
  try {
    const data = await jsonFetch('/api/video', {method:'POST', body:form});
    $('objects').textContent = data.unique_objects;
    $('unique').textContent = data.unique_objects;
    $('fps').textContent = '—';
    showCounts(data.counts);
    $('detection-status').textContent = 'Video analysis complete';
    $('detection-caption').textContent = `${data.frames} frames processed successfully.`;
    $('result-summary').textContent = `${data.frames} frames · ${data.unique_objects} unique track IDs`;
    $('download-link').href = data.download_url;
    $('result').hidden = false;
    status('Analysis finished. Your annotated MP4 is ready to download.');
    await loadHistory();
  } catch (err) { status(err.message, true); $('detection-status').textContent = 'Processing failed'; }
  finally { busy = false; $('process-btn').disabled = false; $('mode-pill').textContent = '● STANDBY'; $('mode-pill').classList.remove('on'); }
}
async function loadHistory() {
  try {
    const rows = await jsonFetch('/api/history');
    $('history-body').replaceChildren();
    if (!rows.length) {
      const tr = document.createElement('tr'), td = document.createElement('td');
      td.colSpan = 6; td.className = 'no-history'; td.textContent = 'No sessions yet. Run your first analysis!';
      tr.append(td); $('history-body').append(tr); return;
    }
    for (const row of rows) {
      const tr = document.createElement('tr');
      for (const value of [row.id, row.created_at.replace('T',' ').slice(0,19), row.source, row.frames, row.unique_objects]) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      }
      const td = document.createElement('td');
      if (row.output_file) { const link = document.createElement('a'); link.href = `/api/download/${encodeURIComponent(row.output_file)}`; link.textContent = '↓ Download'; td.append(link); }
      else td.textContent = '—';
      tr.append(td); $('history-body').append(tr);
    }
  } catch (err) { status(`History unavailable: ${err.message}`, true); }
}
$('camera-tab').addEventListener('click', () => setMode('camera'));
$('upload-tab').addEventListener('click', () => setMode('upload'));
$('start-btn').addEventListener('click', startCamera);
$('stop-btn').addEventListener('click', stopCamera);
$('process-btn').addEventListener('click', processVideo);
$('refresh-btn').addEventListener('click', loadHistory);
$('video-file').addEventListener('change', () => { $('file-name').textContent = $('video-file').files[0]?.name || 'No file selected'; });
window.addEventListener('pagehide', () => { if (running) navigator.sendBeacon('/api/camera/stop', new Blob([], {type:'text/plain'})); });
loadHistory();
