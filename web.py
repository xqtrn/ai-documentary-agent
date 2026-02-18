"""
AI Documentary Agent — FastAPI Web Application
Enhanced dashboard with project detail view, scene management, and generation controls.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel

import config
from pipeline import (
    run_pipeline,
    run_analysis,
    generate_scene_batch,
    generate_project_audio,
    assemble_project,
    get_project_data,
    get_scene_status,
    regenerate_scene,
    edit_scene_prompt,
    estimate_cost,
    read_status,
    write_status,
    cancel_pipeline,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Documentary Agent", version="2.0.0")

# ---------------------------------------------------------------------------
# Telegram Auth Helpers
# ---------------------------------------------------------------------------

BOT_TOKEN = getattr(config, "TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
ADMIN_IDS = getattr(config, "ADMIN_IDS", [])
TEST_SECRET = getattr(config, "TEST_SECRET", os.getenv("TEST_SECRET", ""))
OUTPUT_DIR = getattr(config, "OUTPUT_DIR", "outputs")
SESSION_COOKIE = "docu_session"
SESSION_STORE: dict = {}  # token -> {"user_id": int, "first_name": str, ...}


def get_admin_id() -> Optional[int]:
    if ADMIN_IDS:
        return ADMIN_IDS[0]
    return None


def verify_telegram_auth(params: dict) -> bool:
    """Verify data received from Telegram Login Widget."""
    if not BOT_TOKEN:
        return False
    check_hash = params.pop("hash", None)
    if not check_hash:
        return False
    sorted_params = sorted(params.items())
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted_params)
    secret_key = hashlib.sha256(BOT_TOKEN.encode()).digest()
    h = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    return h == check_hash


def create_session(user_data: dict) -> str:
    token = hashlib.sha256(f"{user_data}{time.time()}".encode()).hexdigest()
    SESSION_STORE[token] = user_data
    return token


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if token and token in SESSION_STORE:
        return SESSION_STORE[token]
    return None


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def verify_test_secret(secret: str) -> bool:
    if not TEST_SECRET:
        return False
    return secret == TEST_SECRET


def check_test_secret_param(request: Request) -> bool:
    secret = request.query_params.get("secret", "")
    return verify_test_secret(secret)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    secret: Optional[str] = None
    url: str
    mode: str = "full"


class GenerateBatchRequest(BaseModel):
    secret: Optional[str] = None
    scene_numbers: list[int]


class GenerateAllRequest(BaseModel):
    secret: Optional[str] = None


class RegenerateSceneRequest(BaseModel):
    secret: Optional[str] = None
    scene_number: int


class EditPromptRequest(BaseModel):
    secret: Optional[str] = None
    scene_number: int
    visual_prompt: str


class AuthGenerateRequest(BaseModel):
    url: str
    mode: str = "full"


class AuthGenerateBatchRequest(BaseModel):
    scene_numbers: list[int]


class AuthRegenerateSceneRequest(BaseModel):
    scene_number: int


class AuthEditPromptRequest(BaseModel):
    scene_number: int
    visual_prompt: str


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Documentary Agent — Login</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0f;color:#e4e4e7;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
       display:flex;align-items:center;justify-content:center;min-height:100vh}
  .card{background:#13131a;border:1px solid #2a2a3a;border-radius:16px;padding:48px;
        text-align:center;max-width:420px;width:90%}
  h1{font-size:28px;margin-bottom:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  p{color:#888;margin-bottom:32px;font-size:14px}
  .tg-btn{margin-top:16px}
  .test-link{display:inline-block;margin-top:24px;color:#6366f1;text-decoration:none;
             font-size:13px;opacity:0.7;transition:opacity 0.2s}
  .test-link:hover{opacity:1}
</style>
</head>
<body>
<div class="card">
  <h1>AI Documentary Agent</h1>
  <p>Sign in with Telegram to access the dashboard</p>
  <div class="tg-btn">
    <script async src="https://telegram.org/js/telegram-widget.js?22"
            data-telegram-login="{{bot_username}}"
            data-size="large"
            data-radius="8"
            data-auth-url="/auth/telegram"
            data-request-access="write"></script>
  </div>
  <a class="test-link" href="/dashboard?secret={{test_secret}}">Test mode &rarr;</a>
</div>
</body>
</html>"""

DASHBOARD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Documentary Agent — Dashboard</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0f;color:#e4e4e7;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
       min-height:100vh}
  a{color:#8b5cf6;text-decoration:none}
  a:hover{text-decoration:underline}

  .header{display:flex;align-items:center;justify-content:space-between;padding:20px 32px;
          border-bottom:1px solid #2a2a3a;background:#13131a}
  .header h1{font-size:20px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header .links{display:flex;gap:16px;font-size:14px}

  .container{max-width:960px;margin:0 auto;padding:32px 16px}

  .card{background:#13131a;border:1px solid #2a2a3a;border-radius:12px;padding:24px;margin-bottom:24px}
  .card h2{font-size:18px;margin-bottom:16px;color:#c4b5fd}

  .input-row{display:flex;gap:12px}
  .input-row input{flex:1;background:#1a1a24;border:1px solid #2a2a3a;border-radius:8px;
                    padding:12px 16px;color:#e4e4e7;font-size:14px;outline:none}
  .input-row input:focus{border-color:#6366f1}
  .input-row input::placeholder{color:#555}

  .btn{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;
       border-radius:8px;padding:12px 24px;font-size:14px;cursor:pointer;font-weight:600;
       transition:opacity 0.2s}
  .btn:hover{opacity:0.85}
  .btn:disabled{opacity:0.4;cursor:not-allowed}
  .btn-sm{padding:8px 16px;font-size:13px}
  .btn-outline{background:transparent;border:1px solid #6366f1;color:#8b5cf6}
  .btn-outline:hover{background:rgba(99,102,241,0.1)}

  .project-list{display:flex;flex-direction:column;gap:12px}
  .project-item{display:flex;align-items:center;justify-content:space-between;
                 background:#1a1a24;border:1px solid #2a2a3a;border-radius:8px;padding:16px;
                 cursor:pointer;transition:border-color 0.2s}
  .project-item:hover{border-color:#6366f1}
  .project-item .info{flex:1;min-width:0}
  .project-item .title{font-size:14px;font-weight:600;white-space:nowrap;
                         overflow:hidden;text-overflow:ellipsis;margin-bottom:4px}
  .project-item .meta{font-size:12px;color:#888}
  .status-badge{display:inline-block;padding:4px 10px;border-radius:12px;font-size:11px;
                 font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
  .status-idle{background:rgba(136,136,136,0.15);color:#888}
  .status-running{background:rgba(56,189,248,0.15);color:#38bdf8}
  .status-completed{background:rgba(74,222,128,0.15);color:#4ade80}
  .status-error{background:rgba(248,113,113,0.15);color:#f87171}

  .empty{text-align:center;color:#555;padding:32px;font-size:14px}

  .mode-select{display:flex;gap:8px;margin-bottom:16px}
  .mode-btn{padding:8px 16px;border-radius:6px;font-size:13px;cursor:pointer;
             background:#1a1a24;border:1px solid #2a2a3a;color:#888;transition:all 0.2s}
  .mode-btn.active{border-color:#6366f1;color:#c4b5fd;background:rgba(99,102,241,0.1)}
</style>
</head>
<body>
<div class="header">
  <h1>AI Documentary Agent</h1>
  <div class="links">
    <span style="color:#888;font-size:13px" id="userLabel"></span>
    <a href="/logout" id="logoutLink" style="display:none">Logout</a>
  </div>
</div>
<div class="container">
  <div class="card">
    <h2>New Project</h2>
    <div class="mode-select">
      <div class="mode-btn active" data-mode="full" onclick="selectMode(this)">Full Pipeline</div>
      <div class="mode-btn" data-mode="analysis" onclick="selectMode(this)">Analysis Only</div>
    </div>
    <div class="input-row">
      <input type="text" id="urlInput" placeholder="Paste YouTube URL..." />
      <button class="btn" id="generateBtn" onclick="startGenerate()">Generate</button>
    </div>
    <p id="genError" style="color:#f87171;font-size:13px;margin-top:8px;display:none"></p>
  </div>

  <div class="card">
    <h2>Active Projects</h2>
    <div id="projectList" class="project-list">
      <div class="empty">No projects yet. Paste a YouTube URL above to get started.</div>
    </div>
  </div>
</div>

<script>
const AUTH_MODE = "{{auth_mode}}";
const SECRET = "{{secret}}";
let selectedMode = "full";

function selectMode(el){
  document.querySelectorAll('.mode-btn').forEach(function(b){b.classList.remove('active')});
  el.classList.add('active');
  selectedMode = el.dataset.mode;
}

function apiUrl(path){
  if(AUTH_MODE === "test") return "/api/test" + path + (path.indexOf("?") >= 0 ? "&" : "?") + "secret=" + SECRET;
  return "/api" + path;
}

function apiBody(data){
  if(AUTH_MODE === "test") return JSON.stringify(Object.assign({}, data, {secret: SECRET}));
  return JSON.stringify(data);
}

function projectUrl(vid){
  if(AUTH_MODE === "test") return "/project/" + vid + "?secret=" + SECRET;
  return "/project/" + vid;
}

function escHtml(s){
  if(!s) return "";
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function startGenerate(){
  var url = document.getElementById("urlInput").value.trim();
  if(!url){document.getElementById("genError").textContent="Please enter a URL";
           document.getElementById("genError").style.display="block";return}
  document.getElementById("genError").style.display="none";
  var btn = document.getElementById("generateBtn");
  btn.disabled = true; btn.textContent = "Starting...";
  try{
    var resp = await fetch(apiUrl("/generate"), {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: apiBody({url: url, mode: selectedMode})});
    var data = await resp.json();
    if(!resp.ok) throw new Error(data.detail || "Failed");
    if(data.video_id) window.location.href = projectUrl(data.video_id);
    else { btn.disabled=false; btn.textContent="Generate"; loadProjects(); }
  }catch(e){
    document.getElementById("genError").textContent=e.message;
    document.getElementById("genError").style.display="block";
    btn.disabled=false; btn.textContent="Generate";
  }
}

async function loadProjects(){
  try{
    var resp = await fetch(apiUrl("/status"));
    if(!resp.ok) return;
    var data = await resp.json();
    var list = document.getElementById("projectList");
    var projects = [];
    if(Array.isArray(data)) projects = data;
    else if(data && data.video_id) projects = [data];
    else if(data && data.projects) projects = data.projects;
    else if(data && typeof data === "object") projects = [data];

    if(projects.length === 0){
      list.innerHTML = '<div class="empty">No projects yet. Paste a YouTube URL above to get started.</div>';
      return;
    }
    list.innerHTML = projects.map(function(p){
      var vid = p.video_id || p.id || "unknown";
      var title = p.title || p.video_title || vid;
      var state = p.state || p.status || "idle";
      var step = p.current_step || p.step || "";
      return '<div class="project-item" onclick="window.location.href=projectUrl(\'' + vid + '\')">' +
        '<div class="info">' +
          '<div class="title">' + escHtml(title) + '</div>' +
          '<div class="meta">' + escHtml(vid) + (step ? " — " + escHtml(step) : "") + '</div>' +
        '</div>' +
        '<span class="status-badge status-' + state + '">' + state + '</span>' +
      '</div>';
    }).join("");
  }catch(e){console.error("loadProjects",e)}
}

// init
if(AUTH_MODE !== "test"){
  document.getElementById("logoutLink").style.display="inline";
  document.getElementById("userLabel").textContent="{{username}}";
}
loadProjects();
setInterval(loadProjects, 8000);
</script>
</body>
</html>"""

PROJECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Project — AI Documentary Agent</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0f;color:#e4e4e7;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
       min-height:100vh;padding-bottom:64px}
  a{color:#8b5cf6;text-decoration:none}
  a:hover{text-decoration:underline}

  .header{display:flex;align-items:center;justify-content:space-between;padding:16px 32px;
          border-bottom:1px solid #2a2a3a;background:#13131a;position:sticky;top:0;z-index:100}
  .header h1{font-size:18px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header .links{display:flex;gap:16px;font-size:14px;align-items:center}

  .container{max-width:1100px;margin:0 auto;padding:24px 16px}

  .section{background:#13131a;border:1px solid #2a2a3a;border-radius:12px;padding:24px;
           margin-bottom:20px;display:none}
  .section.visible{display:block}
  .section h2{font-size:16px;margin-bottom:16px;color:#c4b5fd;display:flex;align-items:center;gap:8px}
  .section h2 .num{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;
                    width:24px;height:24px;border-radius:6px;display:flex;align-items:center;
                    justify-content:center;font-size:12px;font-weight:700}

  /* Status Bar */
  .status-bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  .status-badge{display:inline-block;padding:6px 14px;border-radius:12px;font-size:12px;
                 font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
  .status-idle{background:rgba(136,136,136,0.15);color:#888}
  .status-running{background:rgba(56,189,248,0.15);color:#38bdf8;animation:pulse 2s infinite}
  .status-completed{background:rgba(74,222,128,0.15);color:#4ade80}
  .status-error{background:rgba(248,113,113,0.15);color:#f87171}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.6}}

  .progress-wrap{flex:1;min-width:200px}
  .progress-label{font-size:12px;color:#888;margin-bottom:4px}
  .progress-bar{height:6px;background:#1a1a24;border-radius:3px;overflow:hidden}
  .progress-fill{height:100%;background:linear-gradient(90deg,#6366f1,#8b5cf6);border-radius:3px;
                  transition:width 0.5s ease}

  .btn{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;
       border-radius:8px;padding:10px 20px;font-size:13px;cursor:pointer;font-weight:600;
       transition:opacity 0.2s;white-space:nowrap}
  .btn:hover{opacity:0.85}
  .btn:disabled{opacity:0.35;cursor:not-allowed}
  .btn-sm{padding:6px 14px;font-size:12px}
  .btn-outline{background:transparent;border:1px solid #6366f1;color:#8b5cf6}
  .btn-outline:hover{background:rgba(99,102,241,0.1)}
  .btn-danger{background:linear-gradient(135deg,#ef4444,#dc2626)}
  .btn-danger:hover{opacity:0.85}
  .btn-green{background:linear-gradient(135deg,#22c55e,#16a34a)}

  /* Source info */
  .source-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  @media(max-width:600px){.source-grid{grid-template-columns:1fr}}
  .source-grid .label{font-size:12px;color:#888;margin-bottom:2px}
  .source-grid .value{font-size:14px}
  .transcript-box{margin-top:12px;background:#1a1a24;border-radius:8px;padding:16px;
                   font-size:13px;line-height:1.6;color:#aaa;max-height:120px;overflow:hidden;
                   transition:max-height 0.3s ease;cursor:pointer;position:relative}
  .transcript-box.expanded{max-height:2000px}
  .transcript-box::after{content:"Click to expand";position:absolute;bottom:0;left:0;right:0;
                          padding:16px 0 8px;text-align:center;font-size:11px;color:#6366f1;
                          background:linear-gradient(transparent,#1a1a24 60%)}
  .transcript-box.expanded::after{display:none}

  /* Analysis & Script */
  .text-box{background:#1a1a24;border-radius:8px;padding:16px;font-size:13px;line-height:1.7;
             color:#ccc;overflow-y:auto;white-space:pre-wrap}
  .text-box.analysis{max-height:300px}
  .text-box.script{max-height:400px}
  .meta-row{display:flex;gap:16px;margin-top:12px;font-size:12px;color:#888}

  /* Scene Grid */
  .scene-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:768px){.scene-grid{grid-template-columns:1fr}}

  .scene-card{background:#1a1a24;border:1px solid #2a2a3a;border-radius:10px;padding:16px;
              transition:border-color 0.2s}
  .scene-card:hover{border-color:#3a3a4a}
  .scene-header{display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}
  .scene-num{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;
              padding:3px 10px;border-radius:6px;font-size:12px;font-weight:700}
  .scene-duration{background:rgba(136,136,136,0.15);color:#aaa;padding:3px 8px;
                   border-radius:6px;font-size:11px}
  .scene-mood{background:rgba(139,92,246,0.15);color:#c4b5fd;padding:3px 8px;
               border-radius:6px;font-size:11px}
  .scene-status{margin-left:auto;font-size:13px}
  .scene-status-pending{color:#888}
  .scene-status-generating{color:#38bdf8;animation:pulse 2s infinite}
  .scene-status-done{color:#4ade80}
  .scene-status-failed{color:#f87171}

  .scene-field{margin-bottom:10px}
  .scene-field-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;
                      margin-bottom:3px}
  .scene-text{font-size:13px;line-height:1.5;color:#bbb;max-height:60px;overflow:hidden;
               transition:max-height 0.3s ease;cursor:pointer}
  .scene-text.expanded{max-height:800px}
  .scene-text-toggle{font-size:11px;color:#6366f1;cursor:pointer;margin-top:2px}

  .scene-info-row{display:flex;gap:12px;font-size:12px;color:#888;margin-bottom:8px;flex-wrap:wrap}
  .scene-thumb{width:100%;border-radius:6px;margin-top:8px;max-height:180px;object-fit:cover;
               border:1px solid #2a2a3a}
  .scene-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}

  /* Generation Controls */
  .cost-panel{background:#1a1a24;border-radius:8px;padding:16px;margin-bottom:16px}
  .cost-row{display:flex;justify-content:space-between;font-size:13px;padding:4px 0}
  .cost-row .label{color:#888}
  .cost-row .value{color:#e4e4e7;font-weight:600}
  .cost-total{border-top:1px solid #2a2a3a;margin-top:8px;padding-top:8px;
              font-size:14px;font-weight:600}

  .gen-buttons{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
  .gen-progress{font-size:13px;color:#38bdf8;margin-top:12px}

  /* Output */
  .video-player{width:100%;border-radius:10px;background:#000;max-height:480px}
  .output-meta{display:flex;gap:16px;margin-top:12px;font-size:13px;color:#888;flex-wrap:wrap}
  .download-row{margin-top:16px}

  /* Loading */
  .loading{text-align:center;padding:48px;color:#555}
  .spinner{display:inline-block;width:32px;height:32px;border:3px solid #2a2a3a;
           border-top-color:#6366f1;border-radius:50%;animation:spin 0.8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* Edit prompt modal */
  .modal-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);
                  z-index:200;display:none;align-items:center;justify-content:center}
  .modal-overlay.show{display:flex}
  .modal{background:#13131a;border:1px solid #2a2a3a;border-radius:12px;padding:24px;
         width:90%;max-width:600px}
  .modal h3{margin-bottom:16px;color:#c4b5fd}
  .modal textarea{width:100%;background:#1a1a24;border:1px solid #2a2a3a;border-radius:8px;
                   padding:12px;color:#e4e4e7;font-size:13px;line-height:1.6;resize:vertical;
                   min-height:120px;outline:none;font-family:inherit}
  .modal textarea:focus{border-color:#6366f1}
  .modal-actions{display:flex;gap:12px;justify-content:flex-end;margin-top:16px}
</style>
</head>
<body>

<div class="header">
  <h1><a href="{{dashboard_url}}" style="-webkit-text-fill-color:inherit">AI Documentary Agent</a></h1>
  <div class="links">
    <a href="{{dashboard_url}}">Dashboard</a>
  </div>
</div>

<div class="container" id="main">
  <div class="loading" id="loadingState">
    <div class="spinner"></div>
    <p style="margin-top:16px">Loading project...</p>
  </div>

  <!-- Status Bar -->
  <div class="section" id="sectionStatus">
    <div class="status-bar">
      <span class="status-badge" id="stateBadge">idle</span>
      <div class="progress-wrap">
        <div class="progress-label" id="stepLabel">Waiting...</div>
        <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
      </div>
      <button class="btn btn-danger btn-sm" id="cancelBtn" onclick="cancelPipeline()" style="display:none">Cancel</button>
    </div>
  </div>

  <!-- Source Info -->
  <div class="section" id="sectionSource">
    <h2><span class="num">1</span> Source Info</h2>
    <div class="source-grid">
      <div><div class="label">Title</div><div class="value" id="srcTitle">—</div></div>
      <div><div class="label">Channel</div><div class="value" id="srcChannel">—</div></div>
    </div>
    <div class="transcript-box" id="srcTranscript" onclick="this.classList.toggle('expanded')">
      No transcript loaded.
    </div>
  </div>

  <!-- Analysis -->
  <div class="section" id="sectionAnalysis">
    <h2><span class="num">2</span> Virality Analysis</h2>
    <div class="text-box analysis" id="analysisText">—</div>
  </div>

  <!-- Script -->
  <div class="section" id="sectionScript">
    <h2><span class="num">3</span> Rewritten Script</h2>
    <div class="text-box script" id="scriptText">—</div>
    <div class="meta-row">
      <span id="scriptWordCount">—</span>
      <span id="scriptDuration">—</span>
    </div>
  </div>

  <!-- Scene Architecture -->
  <div class="section" id="sectionScenes">
    <h2><span class="num">4</span> Scene Architecture</h2>
    <div class="scene-grid" id="sceneGrid"></div>
  </div>

  <!-- Generation Controls -->
  <div class="section" id="sectionGenControls">
    <h2><span class="num">5</span> Generation Controls</h2>
    <div class="cost-panel" id="costPanel">
      <div class="cost-row"><span class="label">Loading cost estimate...</span></div>
    </div>
    <div class="gen-buttons" id="genButtons"></div>
    <div class="gen-progress" id="genProgressText" style="display:none"></div>
  </div>

  <!-- Output -->
  <div class="section" id="sectionOutput">
    <h2><span class="num">6</span> Final Output</h2>
    <video class="video-player" id="videoPlayer" controls></video>
    <div class="output-meta" id="outputMeta"></div>
    <div class="download-row" id="downloadRow"></div>
  </div>
</div>

<!-- Edit Prompt Modal -->
<div class="modal-overlay" id="editModal">
  <div class="modal">
    <h3>Edit Visual Prompt — Scene <span id="editSceneNum"></span></h3>
    <textarea id="editPromptText"></textarea>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closeEditModal()">Cancel</button>
      <button class="btn" id="editSaveBtn" onclick="saveEditPrompt()">Save &amp; Regenerate</button>
    </div>
  </div>
</div>

<script>
var VIDEO_ID = "{{video_id}}";
var AUTH_MODE = "{{auth_mode}}";
var SECRET = "{{secret}}";
var projectData = null;
var pollTimer = null;
var editingScene = null;

// --- API helpers ---
function apiBase(){return AUTH_MODE === "test" ? "/api/test" : "/api"}

function apiFetch(path, opts){
  var url = apiBase() + path;
  if(AUTH_MODE === "test"){
    url += (url.indexOf("?") >= 0 ? "&" : "?") + "secret=" + SECRET;
  }
  return fetch(url, opts);
}

function apiPost(path, body){
  var data = AUTH_MODE === "test" ? Object.assign({}, body, {secret: SECRET}) : body;
  var url = apiBase() + path;
  if(AUTH_MODE === "test"){
    url += (url.indexOf("?") >= 0 ? "&" : "?") + "secret=" + SECRET;
  }
  return fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });
}

function fileUrl(path){
  if(AUTH_MODE === "test") return "/api/test/file?secret=" + SECRET + "&path=" + encodeURIComponent(path);
  return "/download/" + encodeURIComponent(path);
}

// --- Rendering helpers ---
function escHtml(s){
  if(!s) return "";
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function show(id){document.getElementById(id).classList.add("visible")}
function hide(id){document.getElementById(id).classList.remove("visible")}

var STEP_PROGRESS = {
  "idle": 0,
  "downloading": 10,
  "transcribing": 20,
  "analyzing": 40,
  "scripting": 55,
  "scene_planning": 70,
  "generating_scenes": 80,
  "generating_audio": 85,
  "assembling": 92,
  "completed": 100,
  "error": 0,
  "cancelled": 0
};

function renderStatus(data){
  show("sectionStatus");
  var state = data.state || data.status || "idle";
  var step = data.current_step || data.step || state;
  var badge = document.getElementById("stateBadge");
  badge.textContent = state;
  badge.className = "status-badge status-" + state;
  document.getElementById("stepLabel").textContent = step ? ("Step: " + step) : "Waiting...";
  var pct = STEP_PROGRESS[step] || STEP_PROGRESS[state] || 0;
  document.getElementById("progressFill").style.width = pct + "%";
  var cancelBtn = document.getElementById("cancelBtn");
  cancelBtn.style.display = (state === "running") ? "inline-block" : "none";
}

function renderSource(data){
  var src = data.source || data;
  var title = src.video_title || src.title || data.title;
  var channel = src.channel || src.channel_name || "";
  var transcript = src.transcript || data.transcript || "";
  if(!title && !transcript) return;
  show("sectionSource");
  document.getElementById("srcTitle").textContent = title || "\u2014";
  document.getElementById("srcChannel").textContent = channel || "\u2014";
  if(transcript){
    var preview = transcript.length > 500 ? transcript.substring(0, 500) + "..." : transcript;
    document.getElementById("srcTranscript").textContent = preview;
    document.getElementById("srcTranscript").setAttribute("data-full", transcript);
  }
}

function renderAnalysis(data){
  var text = data.analysis || data.virality_analysis || "";
  if(!text) return;
  show("sectionAnalysis");
  document.getElementById("analysisText").textContent = text;
}

function renderScript(data){
  var text = data.script || data.rewritten_script || "";
  if(!text) return;
  show("sectionScript");
  document.getElementById("scriptText").textContent = text;
  var words = text.split(/\s+/).length;
  document.getElementById("scriptWordCount").textContent = words + " words";
  var mins = Math.round(words / 150);
  document.getElementById("scriptDuration").textContent = "~" + (mins || 1) + " min estimated";
}

function sceneStatusIcon(status){
  switch(status){
    case "done": case "completed": return '<span class="scene-status scene-status-done" title="Done">&#x2705;</span>';
    case "generating": return '<span class="scene-status scene-status-generating" title="Generating">&#x1f504;</span>';
    case "failed": case "error": return '<span class="scene-status scene-status-failed" title="Failed">&#x274c;</span>';
    default: return '<span class="scene-status scene-status-pending" title="Pending">&#x23f3;</span>';
  }
}

function toggleSceneText(el){el.classList.toggle("expanded")}

function renderScenes(data){
  var scenes = data.scenes || [];
  if(scenes.length === 0) return;
  show("sectionScenes");
  var grid = document.getElementById("sceneGrid");
  grid.innerHTML = scenes.map(function(sc, i){
    var num = sc.scene_number || sc.number || (i + 1);
    var duration = sc.duration || "";
    var mood = sc.mood || "";
    var narration = sc.narration || sc.narration_text || "";
    var prompt = sc.visual_prompt || sc.prompt || "";
    var camera = sc.camera || sc.camera_movement || "";
    var lighting = sc.lighting || "";
    var status = sc.status || "pending";
    var thumbPath = sc.thumbnail || sc.image_path || sc.output_image || "";
    var narrationShort = narration.length > 150 ? narration.substring(0,150) + "..." : narration;
    var promptShort = prompt.length > 120 ? prompt.substring(0,120) + "..." : prompt;

    var thumbHtml = "";
    if(thumbPath && (status === "done" || status === "completed")){
      thumbHtml = '<img class="scene-thumb" src="' + fileUrl(thumbPath) + '" alt="Scene ' + num + '" loading="lazy" onerror="this.style.display=\'none\'"/>';
    }

    var html = '<div class="scene-card" id="scene-card-' + num + '">';
    html += '<div class="scene-header">';
    html += '<span class="scene-num">Scene ' + num + '</span>';
    if(duration) html += '<span class="scene-duration">' + escHtml(duration) + '</span>';
    if(mood) html += '<span class="scene-mood">' + escHtml(mood) + '</span>';
    html += sceneStatusIcon(status);
    html += '</div>';
    if(narration){
      html += '<div class="scene-field">';
      html += '<div class="scene-field-label">Narration</div>';
      html += '<div class="scene-text" onclick="toggleSceneText(this)" title="Click to expand">' + escHtml(narrationShort) + '</div>';
      html += '</div>';
    }
    if(prompt){
      html += '<div class="scene-field">';
      html += '<div class="scene-field-label">Visual Prompt</div>';
      html += '<div class="scene-text" onclick="toggleSceneText(this)" title="Click to expand">' + escHtml(promptShort) + '</div>';
      html += '</div>';
    }
    if(camera || lighting){
      html += '<div class="scene-info-row">';
      if(camera) html += '<span>Camera: ' + escHtml(camera) + '</span>';
      if(lighting) html += '<span>Lighting: ' + escHtml(lighting) + '</span>';
      html += '</div>';
    }
    html += thumbHtml;
    html += '<div class="scene-actions">';
    html += '<button class="btn btn-sm btn-outline" onclick="regenerateSceneAction(' + num + ')">Regenerate</button>';
    html += '<button class="btn btn-sm btn-outline" onclick="openEditModal(' + num + ')">Edit Prompt</button>';
    html += '</div>';
    html += '</div>';
    return html;
  }).join("");
}

function renderGenControls(data){
  var scenes = data.scenes || [];
  if(scenes.length === 0) return;
  show("sectionGenControls");
  loadCostEstimate();
  renderGenButtons(data);
}

var costLoaded = false;
async function loadCostEstimate(){
  if(costLoaded) return;
  try{
    var resp = await apiFetch("/project/" + VIDEO_ID + "/cost");
    if(!resp.ok) return;
    var cost = await resp.json();
    costLoaded = true;
    var panel = document.getElementById("costPanel");
    var rows = "";
    if(cost.video_credits !== undefined) rows += '<div class="cost-row"><span class="label">Video generation</span><span class="value">' + cost.video_credits + ' credits</span></div>';
    if(cost.tts_credits !== undefined) rows += '<div class="cost-row"><span class="label">TTS / Audio</span><span class="value">' + cost.tts_credits + ' credits</span></div>';
    if(cost.music_credits !== undefined) rows += '<div class="cost-row"><span class="label">Music</span><span class="value">' + cost.music_credits + ' credits</span></div>';
    if(cost.total !== undefined) rows += '<div class="cost-row cost-total"><span class="label">Total estimated</span><span class="value">' + cost.total + ' credits</span></div>';
    if(cost.estimate) rows += '<div class="cost-row"><span class="label">Estimate</span><span class="value">' + escHtml(String(cost.estimate)) + '</span></div>';
    if(!rows) rows = '<div class="cost-row"><span class="label">Cost estimate not available</span></div>';
    panel.innerHTML = rows;
  }catch(e){console.error("cost",e)}
}

function renderGenButtons(data){
  var scenes = data.scenes || [];
  var state = data.state || "idle";
  var container = document.getElementById("genButtons");
  var allDone = scenes.every(function(s){return s.status === "done" || s.status === "completed"});
  var anyDone = scenes.some(function(s){return s.status === "done" || s.status === "completed"});
  var anyGenerating = scenes.some(function(s){return s.status === "generating"});
  var pendingScenes = scenes.filter(function(s){return s.status !== "done" && s.status !== "completed" && s.status !== "generating"});
  var isRunning = state === "running" || anyGenerating;
  var disabledAttr = isRunning ? " disabled" : "";

  var html = "";
  if(!allDone && scenes.length >= 3){
    var firstBatchNums = scenes.slice(0,3).map(function(s){return s.scene_number || s.number});
    var firstBatchDone = scenes.slice(0,3).every(function(s){return s.status === "done" || s.status === "completed"});
    if(!firstBatchDone){
      html += '<button class="btn" onclick="generateBatch([' + firstBatchNums.join(",") + '])"' + disabledAttr + '>Generate First 30s</button>';
    }
  }
  if(!allDone){
    html += '<button class="btn btn-green" onclick="generateAll()"' + disabledAttr + '>Generate All</button>';
  }
  if(anyDone && !allDone && pendingScenes.length > 0){
    var remainingNums = pendingScenes.map(function(s){return s.scene_number || s.number});
    html += '<button class="btn btn-outline" onclick="generateBatch([' + remainingNums.join(",") + '])"' + disabledAttr + '>Generate All Remaining (' + remainingNums.length + ')</button>';
    var nextBatchNums = remainingNums.slice(0, 3);
    if(nextBatchNums.length > 0 && nextBatchNums.length < remainingNums.length){
      html += '<button class="btn btn-outline" onclick="generateBatch([' + nextBatchNums.join(",") + '])"' + disabledAttr + '>Continue Next Batch</button>';
    }
  }
  if(allDone){
    html += '<span style="color:#4ade80;font-weight:600">All scenes generated!</span>';
  }
  container.innerHTML = html;

  var genProg = document.getElementById("genProgressText");
  if(anyGenerating){
    var genScene = scenes.find(function(s){return s.status === "generating"});
    var genNum = genScene ? (genScene.scene_number || genScene.number) : "?";
    var doneCount = scenes.filter(function(s){return s.status === "done" || s.status === "completed"}).length;
    genProg.textContent = "Generating scene " + genNum + "... (" + doneCount + "/" + scenes.length + " done)";
    genProg.style.display = "block";
  } else {
    genProg.style.display = "none";
  }
}

function renderOutput(data){
  var output = data.output || data.final_video || data.video_path || "";
  var outputFile = typeof output === "string" ? output : (output.path || output.file || "");
  if(!outputFile) return;
  show("sectionOutput");
  var player = document.getElementById("videoPlayer");
  var src = fileUrl(outputFile);
  if(player.getAttribute("src") !== src){
    player.src = src;
  }
  var meta = document.getElementById("outputMeta");
  var metaHtml = "";
  if(data.output_duration) metaHtml += "<span>Duration: " + escHtml(String(data.output_duration)) + "</span>";
  if(data.output_resolution) metaHtml += "<span>Resolution: " + escHtml(String(data.output_resolution)) + "</span>";
  if(data.output_size) metaHtml += "<span>Size: " + escHtml(String(data.output_size)) + "</span>";
  if(typeof output === "object"){
    if(output.duration) metaHtml += "<span>Duration: " + escHtml(String(output.duration)) + "</span>";
    if(output.resolution) metaHtml += "<span>Resolution: " + escHtml(String(output.resolution)) + "</span>";
    if(output.size) metaHtml += "<span>Size: " + escHtml(String(output.size)) + "</span>";
  }
  meta.innerHTML = metaHtml;
  var dlRow = document.getElementById("downloadRow");
  dlRow.innerHTML = '<a class="btn" href="' + src + '" download>Download Video</a>';
}

function renderProject(data){
  projectData = data;
  document.getElementById("loadingState").style.display = "none";
  renderStatus(data);
  renderSource(data);
  renderAnalysis(data);
  renderScript(data);
  renderScenes(data);
  renderGenControls(data);
  renderOutput(data);
}

// --- Actions ---
async function cancelPipeline(){
  if(!confirm("Cancel the pipeline?")) return;
  try{
    await apiPost("/project/" + VIDEO_ID + "/cancel", {});
  }catch(e){console.error("cancel error",e)}
  pollProject();
}

async function generateBatch(sceneNumbers){
  try{
    var resp = await apiPost("/project/" + VIDEO_ID + "/generate-batch", {scene_numbers: sceneNumbers});
    if(!resp.ok){var d = await resp.json(); alert(d.detail || "Failed");}
    else pollProject();
  }catch(e){alert("Error: " + e.message)}
}

async function generateAll(){
  try{
    var resp = await apiPost("/project/" + VIDEO_ID + "/generate-all", {});
    if(!resp.ok){var d = await resp.json(); alert(d.detail || "Failed");}
    else pollProject();
  }catch(e){alert("Error: " + e.message)}
}

async function regenerateSceneAction(num){
  try{
    var resp = await apiPost("/project/" + VIDEO_ID + "/regenerate-scene", {scene_number: num});
    if(!resp.ok){var d = await resp.json(); alert(d.detail || "Failed");}
    else pollProject();
  }catch(e){alert("Error: " + e.message)}
}

function openEditModal(num){
  editingScene = num;
  document.getElementById("editSceneNum").textContent = num;
  var scenes = (projectData && projectData.scenes) || [];
  var sc = scenes.find(function(s){return (s.scene_number || s.number) === num});
  document.getElementById("editPromptText").value = sc ? (sc.visual_prompt || sc.prompt || "") : "";
  document.getElementById("editModal").classList.add("show");
}

function closeEditModal(){
  document.getElementById("editModal").classList.remove("show");
  editingScene = null;
}

async function saveEditPrompt(){
  if(!editingScene) return;
  var prompt = document.getElementById("editPromptText").value;
  var btn = document.getElementById("editSaveBtn");
  btn.disabled = true; btn.textContent = "Saving...";
  try{
    var resp = await apiPost("/project/" + VIDEO_ID + "/edit-prompt", {
      scene_number: editingScene, visual_prompt: prompt
    });
    if(!resp.ok){var d = await resp.json(); alert(d.detail || "Failed");}
    closeEditModal();
    pollProject();
  }catch(e){alert("Error: " + e.message)}
  finally{btn.disabled = false; btn.textContent = "Save & Regenerate"}
}

// --- Polling ---
async function pollProject(){
  try{
    var resp = await apiFetch("/project/" + VIDEO_ID);
    if(!resp.ok) return;
    var data = await resp.json();
    renderProject(data);
  }catch(e){console.error("poll",e)}
}

// init
pollProject();
pollTimer = setInterval(pollProject, 3000);

// cleanup on leave
window.addEventListener("beforeunload", function(){ if(pollTimer) clearInterval(pollTimer); });

// close modal on overlay click
document.getElementById("editModal").addEventListener("click", function(e){
  if(e.target === this) closeEditModal();
});

// expand transcript with full text
document.getElementById("srcTranscript").addEventListener("click", function(){
  var full = this.getAttribute("data-full");
  if(full && this.classList.contains("expanded")){
    this.textContent = full;
  }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helper to render templates
# ---------------------------------------------------------------------------

def render_login_page() -> str:
    bot_username = getattr(config, "TELEGRAM_BOT_USERNAME", os.getenv("TELEGRAM_BOT_USERNAME", "your_bot"))
    html = LOGIN_PAGE
    html = html.replace("{{bot_username}}", bot_username)
    html = html.replace("{{test_secret}}", TEST_SECRET or "")
    return html


def render_dashboard(auth_mode: str = "telegram", secret: str = "", username: str = "") -> str:
    html = DASHBOARD_PAGE
    html = html.replace("{{auth_mode}}", auth_mode)
    html = html.replace("{{secret}}", secret)
    html = html.replace("{{username}}", username)
    return html


def render_project_page(video_id: str, auth_mode: str = "telegram", secret: str = "") -> str:
    html = PROJECT_PAGE
    html = html.replace("{{video_id}}", video_id)
    html = html.replace("{{auth_mode}}", auth_mode)
    html = html.replace("{{secret}}", secret)
    if auth_mode == "test":
        dashboard_url = "/dashboard?secret=" + secret
    else:
        dashboard_url = "/"
    html = html.replace("{{dashboard_url}}", dashboard_url)
    return html


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = get_current_user(request)
    if user:
        return HTMLResponse(render_dashboard(
            auth_mode="telegram",
            username=user.get("first_name", "User"),
        ))
    return HTMLResponse(render_login_page())


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, secret: str = ""):
    user = get_current_user(request)
    if user:
        return HTMLResponse(render_dashboard(
            auth_mode="telegram",
            username=user.get("first_name", "User"),
        ))
    if secret and verify_test_secret(secret):
        return HTMLResponse(render_dashboard(auth_mode="test", secret=secret))
    return RedirectResponse("/")


@app.get("/project/{video_id}", response_class=HTMLResponse)
async def project_page(request: Request, video_id: str, secret: str = ""):
    user = get_current_user(request)
    if user:
        return HTMLResponse(render_project_page(video_id, auth_mode="telegram"))
    if secret and verify_test_secret(secret):
        return HTMLResponse(render_project_page(video_id, auth_mode="test", secret=secret))
    return RedirectResponse("/")


# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------

@app.get("/auth/telegram")
async def auth_telegram(request: Request):
    params = dict(request.query_params)
    if not verify_telegram_auth(dict(params)):
        raise HTTPException(status_code=403, detail="Invalid Telegram auth")
    user_data = {
        "user_id": int(params.get("id", 0)),
        "first_name": params.get("first_name", ""),
        "last_name": params.get("last_name", ""),
        "username": params.get("username", ""),
        "photo_url": params.get("photo_url", ""),
    }
    token = create_session(user_data)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, max_age=86400 * 7)
    return response


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token and token in SESSION_STORE:
        del SESSION_STORE[token]
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Routes — Test API endpoints (secret-based auth)
# ---------------------------------------------------------------------------

@app.post("/api/test/generate")
async def test_generate(body: GenerateRequest):
    if not verify_test_secret(body.secret or ""):
        raise HTTPException(status_code=403, detail="Invalid secret")

    url = body.url
    mode = body.mode

    def run_bg():
        try:
            if mode == "analysis":
                run_analysis(url)
            else:
                run_pipeline(url)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()

    video_id = _extract_video_id(url)
    return {"status": "started", "mode": mode, "video_id": video_id}


@app.get("/api/test/status")
async def test_status(request: Request, secret: str = ""):
    if not verify_test_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    status = read_status()
    return status


@app.get("/api/test/project/{video_id}")
async def test_project_data(video_id: str, secret: str = ""):
    if not verify_test_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    data = get_project_data(video_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    return data


@app.get("/api/test/project/{video_id}/scenes")
async def test_project_scenes(video_id: str, secret: str = ""):
    if not verify_test_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    scenes = get_scene_status(video_id)
    return {"video_id": video_id, "scenes": scenes}


@app.post("/api/test/project/{video_id}/generate-batch")
async def test_generate_batch(video_id: str, body: GenerateBatchRequest):
    if not verify_test_secret(body.secret or ""):
        raise HTTPException(status_code=403, detail="Invalid secret")

    def run_bg():
        try:
            generate_scene_batch(video_id, body.scene_numbers)
        except Exception as e:
            logger.error(f"Batch generation error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "started", "video_id": video_id, "scene_numbers": body.scene_numbers}


@app.post("/api/test/project/{video_id}/generate-all")
async def test_generate_all(video_id: str, body: GenerateAllRequest):
    if not verify_test_secret(body.secret or ""):
        raise HTTPException(status_code=403, detail="Invalid secret")

    def run_bg():
        try:
            scenes = get_scene_status(video_id)
            pending = [
                s.get("scene_number", s.get("number", i + 1))
                for i, s in enumerate(scenes)
                if s.get("status") not in ("done", "completed")
            ]
            if pending:
                generate_scene_batch(video_id, pending)
            generate_project_audio(video_id)
            assemble_project(video_id)
        except Exception as e:
            logger.error(f"Generate all error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "started", "video_id": video_id}


@app.post("/api/test/project/{video_id}/regenerate-scene")
async def test_regenerate_scene(video_id: str, body: RegenerateSceneRequest):
    if not verify_test_secret(body.secret or ""):
        raise HTTPException(status_code=403, detail="Invalid secret")

    def run_bg():
        try:
            regenerate_scene(video_id, body.scene_number)
        except Exception as e:
            logger.error(f"Regenerate scene error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "started", "video_id": video_id, "scene_number": body.scene_number}


@app.post("/api/test/project/{video_id}/edit-prompt")
async def test_edit_prompt(video_id: str, body: EditPromptRequest):
    if not verify_test_secret(body.secret or ""):
        raise HTTPException(status_code=403, detail="Invalid secret")

    def run_bg():
        try:
            edit_scene_prompt(video_id, body.scene_number, body.visual_prompt)
            regenerate_scene(video_id, body.scene_number)
        except Exception as e:
            logger.error(f"Edit prompt error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "updated", "video_id": video_id, "scene_number": body.scene_number}


@app.get("/api/test/project/{video_id}/cost")
async def test_cost_estimate(video_id: str, secret: str = ""):
    if not verify_test_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    cost = estimate_cost(video_id)
    return cost


@app.get("/api/test/file")
async def test_file_download(secret: str = "", path: str = ""):
    if not verify_test_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    if not path:
        raise HTTPException(status_code=400, detail="Missing path parameter")
    # Sanitize path to prevent directory traversal
    safe_path = Path(OUTPUT_DIR) / path
    try:
        safe_path = safe_path.resolve()
        base_path = Path(OUTPUT_DIR).resolve()
        if not str(safe_path).startswith(str(base_path)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(safe_path))


@app.post("/api/test/project/{video_id}/cancel")
async def test_cancel(video_id: str, body: GenerateAllRequest):
    if not verify_test_secret(body.secret or ""):
        raise HTTPException(status_code=403, detail="Invalid secret")
    cancel_pipeline(video_id)
    return {"status": "cancelled", "video_id": video_id}


# ---------------------------------------------------------------------------
# Routes — Telegram-auth API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def auth_generate(request: Request, body: AuthGenerateRequest):
    user = require_user(request)
    url = body.url
    mode = body.mode

    def run_bg():
        try:
            if mode == "analysis":
                run_analysis(url)
            else:
                run_pipeline(url)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    video_id = _extract_video_id(url)
    return {"status": "started", "mode": mode, "video_id": video_id}


@app.get("/api/status")
async def auth_status(request: Request):
    user = require_user(request)
    status = read_status()
    return status


@app.get("/api/project/{video_id}")
async def auth_project_data(request: Request, video_id: str):
    user = require_user(request)
    data = get_project_data(video_id)
    if not data:
        raise HTTPException(status_code=404, detail="Project not found")
    return data


@app.get("/api/project/{video_id}/scenes")
async def auth_project_scenes(request: Request, video_id: str):
    user = require_user(request)
    scenes = get_scene_status(video_id)
    return {"video_id": video_id, "scenes": scenes}


@app.post("/api/project/{video_id}/generate-batch")
async def auth_generate_batch(request: Request, video_id: str, body: AuthGenerateBatchRequest):
    user = require_user(request)

    def run_bg():
        try:
            generate_scene_batch(video_id, body.scene_numbers)
        except Exception as e:
            logger.error(f"Batch generation error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "started", "video_id": video_id, "scene_numbers": body.scene_numbers}


@app.post("/api/project/{video_id}/generate-all")
async def auth_generate_all(request: Request, video_id: str):
    user = require_user(request)

    def run_bg():
        try:
            scenes = get_scene_status(video_id)
            pending = [
                s.get("scene_number", s.get("number", i + 1))
                for i, s in enumerate(scenes)
                if s.get("status") not in ("done", "completed")
            ]
            if pending:
                generate_scene_batch(video_id, pending)
            generate_project_audio(video_id)
            assemble_project(video_id)
        except Exception as e:
            logger.error(f"Generate all error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "started", "video_id": video_id}


@app.post("/api/project/{video_id}/regenerate-scene")
async def auth_regenerate_scene(request: Request, video_id: str, body: AuthRegenerateSceneRequest):
    user = require_user(request)

    def run_bg():
        try:
            regenerate_scene(video_id, body.scene_number)
        except Exception as e:
            logger.error(f"Regenerate scene error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "started", "video_id": video_id, "scene_number": body.scene_number}


@app.post("/api/project/{video_id}/edit-prompt")
async def auth_edit_prompt(request: Request, video_id: str, body: AuthEditPromptRequest):
    user = require_user(request)

    def run_bg():
        try:
            edit_scene_prompt(video_id, body.scene_number, body.visual_prompt)
            regenerate_scene(video_id, body.scene_number)
        except Exception as e:
            logger.error(f"Edit prompt error: {e}")

    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    return {"status": "updated", "video_id": video_id, "scene_number": body.scene_number}


@app.get("/api/project/{video_id}/cost")
async def auth_cost_estimate(request: Request, video_id: str):
    user = require_user(request)
    cost = estimate_cost(video_id)
    return cost


@app.post("/api/project/{video_id}/cancel")
async def auth_cancel(request: Request, video_id: str):
    user = require_user(request)
    cancel_pipeline(video_id)
    return {"status": "cancelled", "video_id": video_id}


@app.get("/api/outputs")
async def auth_outputs(request: Request):
    user = require_user(request)
    outputs = _list_outputs()
    return {"outputs": outputs}


@app.get("/download/{video_id}/{filename}")
async def download_file(request: Request, video_id: str, filename: str):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    safe_path = Path(OUTPUT_DIR) / video_id / filename
    try:
        safe_path = safe_path.resolve()
        base_path = Path(OUTPUT_DIR).resolve()
        if not str(safe_path).startswith(str(base_path)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(safe_path))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    if not url:
        return ""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _list_outputs() -> list:
    """List output directories."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return []
    results = []
    for d in sorted(output_path.iterdir(), reverse=True):
        if d.is_dir():
            files = [f.name for f in d.iterdir() if f.is_file()]
            results.append({
                "video_id": d.name,
                "files": files,
            })
    return results


# ---------------------------------------------------------------------------
# Run with uvicorn
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = getattr(config, "WEB_HOST", "0.0.0.0")
    port = getattr(config, "WEB_PORT", 8000)
    uvicorn.run(app, host=host, port=int(port))
