#!/usr/bin/env python3
"""Web dashboard for AI Documentary Agent with Telegram auth."""

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from jose import jwt

import config
from pipeline import run_pipeline, read_status, cancel_pipeline, write_status

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Documentary Agent")

SECRET_KEY = hashlib.sha256(config.TELEGRAM_BOT_TOKEN.encode()).hexdigest() if config.TELEGRAM_BOT_TOKEN else "dev-secret"
ALGORITHM = "HS256"
COOKIE_NAME = "session"
DATA_DIR = Path(config.DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)
ADMIN_FILE = DATA_DIR / "admin.json"

_pipeline_thread: threading.Thread | None = None


# --- Auth helpers ---

def verify_telegram_auth(data: dict) -> bool:
    check_hash = data.pop("hash", None)
    if not check_hash:
        return False
    sorted_items = sorted(data.items())
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted_items)
    secret = hashlib.sha256(config.TELEGRAM_BOT_TOKEN.encode()).digest()
    computed = hmac.new(secret, data_check_string.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, check_hash)


def get_admin_id() -> int | None:
    if ADMIN_FILE.exists():
        try:
            return json.loads(ADMIN_FILE.read_text()).get("telegram_id")
        except Exception:
            return None
    return None


def set_admin_id(telegram_id: int, username: str = ""):
    ADMIN_FILE.write_text(json.dumps({"telegram_id": telegram_id, "username": username}))


def create_session_token(telegram_id: int, username: str = "") -> str:
    payload = {"sub": str(telegram_id), "username": username,
               "iat": int(time.time()), "exp": int(time.time()) + 86400 * 7}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        telegram_id = int(payload["sub"])
        admin_id = get_admin_id()
        if admin_id and telegram_id != admin_id:
            return None
        return {"telegram_id": telegram_id, "username": payload.get("username", "")}
    except Exception:
        return None


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = get_current_user(request)
    if not user:
        return HTMLResponse(LOGIN_HTML)
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/auth/telegram")
async def telegram_auth(request: Request):
    params = dict(request.query_params)
    if not params.get("hash"):
        raise HTTPException(400, "Missing auth data")

    auth_data = {k: v for k, v in params.items()}
    hash_val = auth_data.get("hash", "")

    check_data = {k: v for k, v in auth_data.items() if k != "hash"}
    if not verify_telegram_auth({**check_data, "hash": hash_val}):
        raise HTTPException(403, "Invalid Telegram auth")

    telegram_id = int(auth_data["id"])
    username = auth_data.get("username", auth_data.get("first_name", ""))

    admin_id = get_admin_id()
    if admin_id is None:
        set_admin_id(telegram_id, username)
        logger.info("Admin set: %s (ID: %d)", username, telegram_id)
    elif telegram_id != admin_id:
        raise HTTPException(403, "Access denied. You are not the admin.")

    token = create_session_token(telegram_id, username)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(COOKIE_NAME, token, max_age=86400 * 7, httponly=True, samesite="lax")
    return response


@app.get("/auth/logout")
async def logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(COOKIE_NAME)
    return response


@app.get("/api/status")
async def api_status(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)
    return JSONResponse(read_status())


@app.post("/api/transcript")
async def api_save_transcript(request: Request):
    """Save user-pasted transcript for a video."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)
    body = await request.json()
    text = body.get("transcript", "").strip()
    url = body.get("url", "").strip()
    if not text:
        raise HTTPException(400, "No transcript text provided")
    if not url:
        raise HTTPException(400, "No URL provided")

    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    video_id = match.group(1) if match else "unknown"

    out_dir = Path(config.OUTPUT_DIR) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_file = out_dir / "user_transcript.txt"
    transcript_file.write_text(text, encoding="utf-8")

    logger.info("User transcript saved: %s (%d chars)", video_id, len(text))
    return JSONResponse({"ok": True, "message": f"Transcript saved ({len(text)} chars)"})


@app.post("/api/generate")
async def api_generate(request: Request):
    global _pipeline_thread
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)

    body = await request.json()
    url = body.get("url", "").strip()
    if not url:
        raise HTTPException(400, "Missing URL")

    status = read_status()
    if status.get("state") == "running":
        raise HTTPException(409, "Pipeline already running")

    def _run():
        try:
            run_pipeline(url)
        except Exception as e:
            logger.exception("Pipeline thread failed: %s", e)

    _pipeline_thread = threading.Thread(target=_run, daemon=True)
    _pipeline_thread.start()
    return JSONResponse({"ok": True, "message": "Pipeline started"})


@app.post("/api/cancel")
async def api_cancel(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)
    cancel_pipeline()
    return JSONResponse({"ok": True, "message": "Cancel signal sent"})


@app.get("/api/outputs")
async def api_outputs(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)

    output_base = Path(config.OUTPUT_DIR)
    outputs = []
    if output_base.exists():
        for d in sorted(output_base.iterdir(), reverse=True):
            if not d.is_dir() or d.name == "status.json":
                continue
            entry = {"id": d.name, "files": {}}
            for fname in ["final_video.mp4", "thumbnail.png", "metadata.json", "subtitles.srt", "analysis.md"]:
                fpath = d / fname
                if fpath.exists():
                    entry["files"][fname] = {"size_mb": round(fpath.stat().st_size / 1048576, 1)}
            checkpoint = d / "checkpoint.json"
            if checkpoint.exists():
                try:
                    entry["last_step"] = json.loads(checkpoint.read_text()).get("last_step", "")
                except Exception:
                    pass
            if entry.get("files") or entry.get("last_step"):
                outputs.append(entry)
    return JSONResponse(outputs[:20])


@app.get("/download/{video_id}/{filename}")
async def download_file(video_id: str, filename: str, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)
    if ".." in video_id or ".." in filename or "/" in video_id:
        raise HTTPException(400)
    fpath = Path(config.OUTPUT_DIR) / video_id / filename
    if not fpath.exists():
        raise HTTPException(404)
    return FileResponse(fpath, filename=filename)


# --- HTML Templates ---

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Documentary Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  display:flex;align-items:center;justify-content:center;min-height:100vh}
.login-card{background:#13131a;border:1px solid #2a2a3a;border-radius:16px;padding:48px 40px;
  text-align:center;max-width:400px;width:90%}
h1{font-size:24px;font-weight:700;margin-bottom:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;font-size:14px;margin-bottom:32px}
.tg-login{margin:24px 0}
.footer{color:#555;font-size:12px;margin-top:24px}
</style>
</head>
<body>
<div class="login-card">
  <h1>AI Documentary Agent</h1>
  <p class="subtitle">Sign in to access the dashboard</p>
  <div class="tg-login">
    <script async src="https://telegram.org/js/telegram-widget.js?22"
      data-telegram-login="{bot_username}"
      data-size="large"
      data-radius="8"
      data-auth-url="/auth/telegram"
      data-request-access="write"></script>
    <noscript><p style="color:#f87171">Enable JavaScript for Telegram Login</p></noscript>
  </div>
  <p class="footer">First login becomes admin</p>
</div>
</body>
</html>""".replace("{bot_username}", config.TELEGRAM_BOT_USERNAME)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard — AI Documentary Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  padding:20px;max-width:960px;margin:0 auto}
a{color:#818cf8;text-decoration:none}
a:hover{text-decoration:underline}

.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;
  padding-bottom:16px;border-bottom:1px solid #1e1e2e}
.header h1{font-size:20px;font-weight:700;background:linear-gradient(135deg,#6366f1,#8b5cf6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}

.card{background:#13131a;border:1px solid #2a2a3a;border-radius:12px;padding:20px;margin-bottom:16px}
.card h2{font-size:15px;font-weight:600;color:#a0a0b0;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}

.status-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600}
.status-idle{background:#1e1e2e;color:#888}
.status-running{background:#1e293b;color:#38bdf8;animation:pulse 2s infinite}
.status-completed{background:#052e16;color:#4ade80}
.status-error{background:#2d0a0a;color:#f87171}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}

.progress-wrap{background:#1e1e2e;border-radius:8px;height:28px;overflow:hidden;margin:12px 0;position:relative}
.progress-bar{height:100%;background:linear-gradient(90deg,#6366f1,#8b5cf6);border-radius:8px;
  transition:width .5s ease}
.progress-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:12px;
  font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}

.logs{background:#0c0c14;border:1px solid #1e1e2e;border-radius:8px;padding:12px;
  font-family:'SF Mono',Monaco,Consolas,monospace;font-size:12px;line-height:1.6;
  max-height:300px;overflow-y:auto;color:#9ca3af;white-space:pre-wrap;word-break:break-all}

.input-row{display:flex;gap:8px;margin-bottom:8px}
.input-row input{flex:1;background:#0c0c14;border:1px solid #2a2a3a;border-radius:8px;
  padding:10px 14px;color:#e0e0e0;font-size:14px;outline:none}
.input-row input:focus{border-color:#6366f1}

textarea{width:100%;background:#0c0c14;border:1px solid #2a2a3a;border-radius:8px;
  padding:10px 14px;color:#e0e0e0;font-size:13px;font-family:inherit;outline:none;resize:vertical;
  min-height:60px}
textarea:focus{border-color:#6366f1}
.transcript-toggle{color:#818cf8;cursor:pointer;font-size:13px;margin-bottom:8px;display:inline-block}
.transcript-toggle:hover{text-decoration:underline}

.btn{padding:10px 20px;border-radius:8px;border:none;font-size:14px;font-weight:600;cursor:pointer;
  transition:all .2s}
.btn-primary{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff}
.btn-primary:hover{opacity:.9}
.btn-danger{background:#7f1d1d;color:#fca5a5}
.btn-danger:hover{background:#991b1b}
.btn-sm{padding:6px 14px;font-size:12px}
.btn:disabled{opacity:.4;cursor:not-allowed}

.outputs-list{list-style:none}
.outputs-list li{padding:12px;border-bottom:1px solid #1e1e2e;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:8px}
.outputs-list li:last-child{border-bottom:none}
.output-id{font-weight:600;font-family:monospace;color:#c4b5fd}
.output-files{display:flex;gap:6px;flex-wrap:wrap}
.output-files a{background:#1e1e2e;padding:4px 10px;border-radius:6px;font-size:12px;color:#818cf8}
.output-files a:hover{background:#2a2a3a;text-decoration:none}

.empty{color:#555;text-align:center;padding:20px;font-size:14px}
.hint{color:#666;font-size:12px;margin-top:4px}
.success{color:#4ade80}
.error-text{color:#f87171}
</style>
</head>
<body>

<div class="header">
  <h1>AI Documentary Agent</h1>
  <a href="/auth/logout">Logout</a>
</div>

<!-- Status -->
<div class="card">
  <h2>Pipeline Status</h2>
  <div>
    <span id="badge" class="status-badge status-idle">idle</span>
    <span id="step-label" style="margin-left:12px;font-size:14px;color:#ccc"></span>
  </div>
  <div class="progress-wrap">
    <div id="bar" class="progress-bar" style="width:0%"></div>
    <div id="bar-text" class="progress-text">0%</div>
  </div>
</div>

<!-- Generate -->
<div class="card">
  <h2>Generate New Video</h2>
  <div class="input-row">
    <input id="url-input" type="text" placeholder="YouTube URL (e.g. https://youtube.com/watch?v=...)">
    <button id="gen-btn" class="btn btn-primary" onclick="generate()">Generate</button>
    <button id="cancel-btn" class="btn btn-danger" onclick="cancelPipeline()" disabled>Cancel</button>
  </div>

  <div style="margin-top:12px">
    <span class="transcript-toggle" onclick="toggleTranscript()">
      &#9660; Manual Transcript (paste if auto-download fails)
    </span>
    <div id="transcript-section" style="display:none;margin-top:8px">
      <textarea id="transcript-input" rows="6"
        placeholder="Paste the video transcript here if automatic download fails.&#10;&#10;You can get it from YouTube (click ... under video > Show transcript) or Google the transcript."></textarea>
      <div style="margin-top:6px;display:flex;gap:8px;align-items:center">
        <button class="btn btn-primary btn-sm" onclick="saveTranscript()">Save Transcript</button>
        <span id="transcript-msg" class="hint"></span>
      </div>
      <p class="hint" style="margin-top:6px">
        Save the transcript first, then click Generate. The pipeline will use your pasted text instead of downloading.
      </p>
    </div>
  </div>

  <div id="gen-msg" style="font-size:13px;margin-top:8px;color:#888"></div>
</div>

<!-- Logs -->
<div class="card">
  <h2>Logs</h2>
  <div id="logs" class="logs">Waiting for data...</div>
</div>

<!-- Outputs -->
<div class="card">
  <h2>Completed Videos</h2>
  <ul id="outputs" class="outputs-list">
    <li class="empty">Loading...</li>
  </ul>
</div>

<script>
const $ = s => document.querySelector(s);

function toggleTranscript() {
  const sec = $('#transcript-section');
  sec.style.display = sec.style.display === 'none' ? 'block' : 'none';
}

async function saveTranscript() {
  const url = $('#url-input').value.trim();
  const text = $('#transcript-input').value.trim();
  const msg = $('#transcript-msg');

  if (!url) { msg.textContent = 'Enter a YouTube URL first'; msg.className = 'hint error-text'; return; }
  if (!text) { msg.textContent = 'Paste transcript text'; msg.className = 'hint error-text'; return; }

  msg.textContent = 'Saving...';
  msg.className = 'hint';
  try {
    const r = await fetch('/api/transcript', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url, transcript: text})
    });
    const d = await r.json();
    if (r.ok) {
      msg.textContent = '✓ ' + d.message;
      msg.className = 'hint success';
    } else {
      msg.textContent = d.detail || 'Error saving';
      msg.className = 'hint error-text';
    }
  } catch(e) {
    msg.textContent = 'Error: ' + e.message;
    msg.className = 'hint error-text';
  }
}

async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    if (r.status === 401) { location.reload(); return; }
    const d = await r.json();

    const badge = $('#badge');
    badge.textContent = d.state;
    badge.className = 'status-badge status-' + d.state;

    $('#step-label').textContent = d.step_name ? ('Step ' + d.step_index + '/' + d.step_total + ': ' + d.step_name) : '';
    $('#bar').style.width = d.progress_pct + '%';
    $('#bar-text').textContent = d.progress_pct + '%';

    const running = d.state === 'running';
    $('#gen-btn').disabled = running;
    $('#cancel-btn').disabled = !running;

    if (d.logs && d.logs.length) {
      const logsEl = $('#logs');
      logsEl.textContent = d.logs.join('\\n');
      logsEl.scrollTop = logsEl.scrollHeight;
    }
  } catch(e) {}
}

async function fetchOutputs() {
  try {
    const r = await fetch('/api/outputs');
    if (!r.ok) return;
    const items = await r.json();
    const ul = $('#outputs');
    if (!items.length) { ul.innerHTML = '<li class="empty">No outputs yet</li>'; return; }
    ul.innerHTML = items.map(o => {
      const files = Object.keys(o.files || {}).map(f =>
        `<a href="/download/${o.id}/${f}">${f} (${o.files[f].size_mb}MB)</a>`
      ).join('');
      const step = o.last_step ? ` — step: ${o.last_step}` : '';
      return `<li><span class="output-id">${o.id}${step}</span><div class="output-files">${files}</div></li>`;
    }).join('');
  } catch(e) {}
}

async function generate() {
  const url = $('#url-input').value.trim();
  if (!url) { $('#gen-msg').textContent = 'Enter a YouTube URL'; return; }
  $('#gen-msg').textContent = 'Starting...';
  try {
    const r = await fetch('/api/generate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url})
    });
    const d = await r.json();
    $('#gen-msg').textContent = d.message || d.detail || 'Started';
    if (r.ok) $('#url-input').value = '';
  } catch(e) { $('#gen-msg').textContent = 'Error: ' + e.message; }
}

async function cancelPipeline() {
  try {
    await fetch('/api/cancel', {method:'POST'});
    $('#gen-msg').textContent = 'Cancel signal sent';
  } catch(e) {}
}

setInterval(fetchStatus, 3000);
setInterval(fetchOutputs, 10000);
fetchStatus();
fetchOutputs();
</script>
</body>
</html>"""
