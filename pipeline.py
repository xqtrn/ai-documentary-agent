"""
AI Documentary Agent - Pipeline Orchestrator

Supports both full sequential runs and incremental/batch generation.
Pipeline steps:
  1. source    - YouTube transcript download
  2. virality  - Claude analysis
  3. script    - Claude rewrite
  4. scenes    - Scene splitting (engine-adaptive prompts)
  5. video     - Multi-engine video generation
  6. audio     - Runway TTS voiceover
  7. music     - Runway sound effects
  8. assembly  - FFmpeg final assembly
  9. metadata  - Claude metadata generation
"""

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scripts.source_analyzer import analyze_source
from scripts.virality_analyzer import analyze_virality
from scripts.scriptwriter import rewrite_script
from scripts.scene_splitter import split_into_scenes
from scripts.video_gen import generate_videos, generate_single_scene
from scripts.voiceover import generate_audio
from scripts.music_gen import generate_music
from scripts.assembler import assemble_video
from scripts.metadata import generate_all_metadata

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_pipeline_status = {
    "state": "idle",
    "step": None,
    "step_number": 0,
    "total_steps": 9,
    "message": "",
    "progress": 0.0,
    "video_id": None,
    "error": None,
    "engine": None,
}
_status_lock = threading.Lock()

_pipeline_cancel = threading.Event()

_log_buffer: list[str] = []
_log_buffer_lock = threading.Lock()

MAX_LOG_LINES = 2000

STEP_FILES = {
    "source":   "step1_source.json",
    "virality": "step2_virality.json",
    "script":   "step3_script.json",
    "scenes":   "step4_scenes.json",
    "video":    "step5_videos.json",
    "audio":    "step6_audio.json",
    "music":    "step7_music.json",
    "assembly": "step8_assembly.json",
    "metadata": "step9_metadata.json",
}

STEP_ORDER = [
    "source", "virality", "script", "scenes",
    "video", "audio", "music", "assembly", "metadata",
]


# ---------------------------------------------------------------------------
# Logging handler
# ---------------------------------------------------------------------------

class StatusLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _log_buffer_lock:
                _log_buffer.append(msg)
                while len(_log_buffer) > MAX_LOG_LINES:
                    _log_buffer.pop(0)
        except Exception:
            self.handleError(record)


def get_log_buffer(since: int = 0) -> list[str]:
    with _log_buffer_lock:
        return list(_log_buffer[since:])


def clear_log_buffer() -> None:
    with _log_buffer_lock:
        _log_buffer.clear()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def write_status(*, state: str | None = None, step: str | None = None,
                 step_number: int | None = None, message: str | None = None,
                 progress: float | None = None, video_id: str | None = None,
                 error: str | None = None, engine: str | None = None) -> None:
    with _status_lock:
        if state is not None:
            _pipeline_status["state"] = state
        if step is not None:
            _pipeline_status["step"] = step
        if step_number is not None:
            _pipeline_status["step_number"] = step_number
        if message is not None:
            _pipeline_status["message"] = message
        if progress is not None:
            _pipeline_status["progress"] = progress
        if video_id is not None:
            _pipeline_status["video_id"] = video_id
        if error is not None:
            _pipeline_status["error"] = error
        if engine is not None:
            _pipeline_status["engine"] = engine


def read_status() -> dict:
    with _status_lock:
        return dict(_pipeline_status)


def _reset_status() -> None:
    write_status(state="idle", step=None, step_number=0, message="",
                 progress=0.0, video_id=None, error=None, engine=None)


# ---------------------------------------------------------------------------
# Cancel helpers
# ---------------------------------------------------------------------------

def request_cancel() -> None:
    _pipeline_cancel.set()
    write_status(state="cancelled", message="Cancel requested — stopping after current step.")
    logger.info("Pipeline cancellation requested.")


cancel_pipeline = request_cancel


def _check_cancel() -> None:
    if _pipeline_cancel.is_set():
        raise PipelineCancelled("Pipeline cancelled by user.")


class PipelineCancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# Path / data helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
        if parsed.path.startswith("/embed/") or parsed.path.startswith("/v/"):
            return parsed.path.split("/")[2]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
    if re.match(r'^[\w-]{11}$', url):
        return url
    raise ValueError(f"Cannot extract video ID from: {url}")


def get_output_dir(url_or_video_id: str) -> Path:
    try:
        video_id = extract_video_id(url_or_video_id)
    except ValueError:
        video_id = url_or_video_id
    output_dir = Path(config.OUTPUT_DIR) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_step_data(output_dir: Path, step_file: str) -> dict | None:
    path = output_dir / step_file
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def save_checkpoint(output_dir: Path, step_file: str, data: dict) -> Path:
    path = output_dir / step_file
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Checkpoint saved: %s", path)
    return path


def get_checkpoint(video_id: str, step_name: str) -> dict | None:
    output_dir = get_output_dir(video_id)
    step_file = STEP_FILES.get(step_name)
    if step_file is None:
        return None
    return load_step_data(output_dir, step_file)


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

def _load_history() -> list:
    """Load history from persistent JSON file."""
    history_path = Path(config.HISTORY_FILE)
    if not history_path.exists():
        return []
    try:
        with open(history_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(history: list) -> None:
    """Save history to persistent JSON file."""
    history_path = Path(config.HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)


def add_history_record(video_id: str, engine: str, results: dict) -> dict:
    """Add a completed project to history."""
    assembly = results.get("assembly", {})
    source = results.get("source", {})
    scenes = results.get("scenes", {})

    record = {
        "video_id": video_id,
        "engine": engine,
        "title": source.get("video_title", source.get("title", video_id)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": assembly.get("duration_sec", 0),
        "file_size_mb": assembly.get("file_size_mb", 0),
        "resolution": assembly.get("resolution", ""),
        "scene_count": scenes.get("scene_count", 0),
        "clips_used": assembly.get("clips_used", 0),
        "final_video": assembly.get("final_video", ""),
        "thumbnail": assembly.get("thumbnail", ""),
    }

    history = _load_history()
    # Remove existing entry for this video_id if any
    history = [h for h in history if h.get("video_id") != video_id]
    history.insert(0, record)
    _save_history(history)
    logger.info("History record added for %s (engine=%s)", video_id, engine)
    return record


def get_history() -> list:
    """Get all history records."""
    return _load_history()


def get_history_record(video_id: str) -> dict | None:
    """Get a single history record."""
    for record in _load_history():
        if record.get("video_id") == video_id:
            return record
    return None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(url: str, *, resume: bool = True, engine: str = None,
                  video_id: str = None, voice_key: str = None) -> dict:
    """Run the complete 9-step pipeline.

    Parameters
    ----------
    url : str
        YouTube video URL.
    resume : bool
        If True, skip steps with existing checkpoints.
    engine : str
        Video engine key. Defaults to config.DEFAULT_ENGINE.
    video_id : str
        Override video_id (for unique runs per engine). If None, extracted from URL.
    voice_key : str
        Voice key for TTS. Defaults to config.DEFAULT_VOICE.
    """
    engine = engine or config.DEFAULT_ENGINE
    _pipeline_cancel.clear()
    clear_log_buffer()

    if video_id is None:
        video_id = extract_video_id(url)
    output_dir = Path(config.OUTPUT_DIR) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    write_status(state="running", step="source", step_number=1, progress=0.0,
                 message="Starting pipeline…", video_id=video_id, error=None, engine=engine)

    results: dict = {"video_id": video_id, "output_dir": str(output_dir), "engine": engine}

    try:
        # Step 1 — Source
        _check_cancel()
        write_status(step="source", step_number=1, message="Downloading transcript…", progress=0.0)
        source_data = load_step_data(output_dir, STEP_FILES["source"]) if resume else None
        if source_data is None:
            logger.info("Step 1/9: Analyzing source — %s", url)
            source_data = analyze_source(url, output_dir)
            save_checkpoint(output_dir, STEP_FILES["source"], source_data)
        else:
            logger.info("Step 1/9: Source — loaded from checkpoint.")
        results["source"] = source_data

        # Step 2 — Virality
        _check_cancel()
        write_status(step="virality", step_number=2, message="Running virality analysis…", progress=11.0)
        virality_data = load_step_data(output_dir, STEP_FILES["virality"]) if resume else None
        if virality_data is None:
            logger.info("Step 2/9: Virality analysis.")
            virality_data = analyze_virality(source_data, output_dir)
            save_checkpoint(output_dir, STEP_FILES["virality"], virality_data)
        else:
            logger.info("Step 2/9: Virality — loaded from checkpoint.")
        results["virality"] = virality_data

        # Step 3 — Script
        _check_cancel()
        write_status(step="script", step_number=3, message="Rewriting script…", progress=22.0)
        script_data = load_step_data(output_dir, STEP_FILES["script"]) if resume else None
        if script_data is None:
            logger.info("Step 3/9: Scriptwriting.")
            script_data = rewrite_script(source_data, virality_data, output_dir)
            save_checkpoint(output_dir, STEP_FILES["script"], script_data)
        else:
            logger.info("Step 3/9: Script — loaded from checkpoint.")
        results["script"] = script_data

        # Step 4 — Scenes (engine-adaptive)
        _check_cancel()
        write_status(step="scenes", step_number=4, message="Splitting into scenes…", progress=33.0)
        scenes_data = load_step_data(output_dir, STEP_FILES["scenes"]) if resume else None
        if scenes_data is None:
            logger.info("Step 4/9: Scene splitting (engine=%s).", engine)
            scenes_data = split_into_scenes(script_data, output_dir, engine=engine)
            save_checkpoint(output_dir, STEP_FILES["scenes"], scenes_data)
        else:
            logger.info("Step 4/9: Scenes — loaded from checkpoint.")
        results["scenes"] = scenes_data

        # Step 5 — Video (multi-engine)
        _check_cancel()
        write_status(step="video", step_number=5, message=f"Generating videos ({engine})…", progress=44.0)
        video_data = load_step_data(output_dir, STEP_FILES["video"]) if resume else None
        if video_data is None:
            logger.info("Step 5/9: Video generation (engine=%s).", engine)
            video_data = generate_videos(scenes_data, output_dir, engine=engine)
            save_checkpoint(output_dir, STEP_FILES["video"], video_data)
        else:
            logger.info("Step 5/9: Videos — loaded from checkpoint.")
        results["video"] = video_data

        # Step 6 — Audio
        _check_cancel()
        write_status(step="audio", step_number=6, message="Generating voiceover…", progress=55.0)
        audio_data = load_step_data(output_dir, STEP_FILES["audio"]) if resume else None
        if audio_data is None:
            logger.info("Step 6/9: Voiceover generation (voice=%s).", voice_key)
            audio_data = generate_audio(script_data, scenes_data, output_dir, voice_key=voice_key)
            save_checkpoint(output_dir, STEP_FILES["audio"], audio_data)
        else:
            logger.info("Step 6/9: Audio — loaded from checkpoint.")
        results["audio"] = audio_data

        # Step 7 — Music (optional: continues to assembly if 429/rate-limited)
        _check_cancel()
        write_status(step="music", step_number=7, message="Generating background music…", progress=66.0)
        music_data = load_step_data(output_dir, STEP_FILES["music"]) if resume else None
        if music_data is None:
            logger.info("Step 7/9: Music generation.")
            total_duration = sum(
                s.get("duration_sec", s.get("duration", 10)) for s in scenes_data.get("scenes", [])
            )
            try:
                music_data = generate_music(total_duration, output_dir, scenes=scenes_data.get("scenes", []))
                save_checkpoint(output_dir, STEP_FILES["music"], music_data)
            except Exception as music_exc:
                err_str = str(music_exc).lower()
                if any(kw in err_str for kw in ("429", "rate", "limit", "daily", "quota")):
                    logger.warning("Step 7/9: Music skipped (rate-limited): %s", music_exc)
                    music_data = {"music_path": None, "skipped": True, "reason": str(music_exc)}
                    save_checkpoint(output_dir, STEP_FILES["music"], music_data)
                else:
                    raise
        else:
            logger.info("Step 7/9: Music — loaded from checkpoint.")
        results["music"] = music_data

        # Step 8 — Assembly
        _check_cancel()
        write_status(step="assembly", step_number=8, message="Assembling final video…", progress=77.0)
        assembly_data = load_step_data(output_dir, STEP_FILES["assembly"]) if resume else None
        if assembly_data is None:
            logger.info("Step 8/9: Video assembly.")
            assembly_data = assemble_video(
                scenes_data, video_data, audio_data, music_data, output_dir
            )
            save_checkpoint(output_dir, STEP_FILES["assembly"], assembly_data)
        else:
            logger.info("Step 8/9: Assembly — loaded from checkpoint.")
        results["assembly"] = assembly_data

        # Step 9 — Metadata
        _check_cancel()
        write_status(step="metadata", step_number=9, message="Generating metadata…", progress=88.0)
        metadata_data = load_step_data(output_dir, STEP_FILES["metadata"]) if resume else None
        if metadata_data is None:
            logger.info("Step 9/9: Metadata generation.")
            metadata_data = generate_all_metadata(
                source_data, script_data, scenes_data, output_dir
            )
            save_checkpoint(output_dir, STEP_FILES["metadata"], metadata_data)
        else:
            logger.info("Step 9/9: Metadata — loaded from checkpoint.")
        results["metadata"] = metadata_data

        # Save to history
        add_history_record(video_id, engine, results)

        write_status(state="completed", step="done", step_number=9,
                     message="Pipeline complete!", progress=100.0)
        logger.info("Pipeline completed for %s (engine=%s)", video_id, engine)
        return results

    except PipelineCancelled:
        write_status(state="cancelled", message="Pipeline was cancelled.")
        logger.warning("Pipeline cancelled for %s", video_id)
        return results

    except Exception as exc:
        write_status(state="failed", message=str(exc), error=str(exc))
        logger.exception("Pipeline failed for %s: %s", video_id, exc)
        raise


# ---------------------------------------------------------------------------
# Incremental functions
# ---------------------------------------------------------------------------

def run_analysis(url: str, engine: str = None) -> dict:
    """Run steps 1-4 (source, virality, script, scenes)."""
    engine = engine or config.DEFAULT_ENGINE
    _pipeline_cancel.clear()

    video_id = extract_video_id(url)
    output_dir = get_output_dir(url)

    write_status(state="running", step="source", step_number=1, progress=0.0,
                 message="Starting analysis…", video_id=video_id, error=None, engine=engine)

    results: dict = {"video_id": video_id, "output_dir": str(output_dir), "engine": engine}

    try:
        _check_cancel()
        write_status(step="source", step_number=1, message="Downloading transcript…", progress=0.0)
        logger.info("Analysis step 1/4: Analyzing source — %s", url)
        source_data = analyze_source(url, output_dir)
        save_checkpoint(output_dir, STEP_FILES["source"], source_data)
        results["source"] = source_data

        _check_cancel()
        write_status(step="virality", step_number=2, message="Running virality analysis…", progress=25.0)
        logger.info("Analysis step 2/4: Virality analysis.")
        virality_data = analyze_virality(source_data, output_dir)
        save_checkpoint(output_dir, STEP_FILES["virality"], virality_data)
        results["virality"] = virality_data

        _check_cancel()
        write_status(step="script", step_number=3, message="Rewriting script…", progress=50.0)
        logger.info("Analysis step 3/4: Scriptwriting.")
        script_data = rewrite_script(source_data, virality_data, output_dir)
        save_checkpoint(output_dir, STEP_FILES["script"], script_data)
        results["script"] = script_data

        _check_cancel()
        write_status(step="scenes", step_number=4, message="Splitting into scenes…", progress=75.0)
        logger.info("Analysis step 4/4: Scene splitting (engine=%s).", engine)
        scenes_data = split_into_scenes(script_data, output_dir, engine=engine)
        save_checkpoint(output_dir, STEP_FILES["scenes"], scenes_data)
        results["scenes"] = scenes_data

        write_status(state="completed", step="scenes", step_number=4,
                     message="Analysis complete!", progress=100.0)
        logger.info("Analysis completed for %s", video_id)
        return results

    except PipelineCancelled:
        write_status(state="cancelled", message="Analysis was cancelled.")
        logger.warning("Analysis cancelled for %s", video_id)
        return results

    except Exception as exc:
        write_status(state="failed", message=str(exc), error=str(exc))
        logger.exception("Analysis failed for %s: %s", video_id, exc)
        raise


def generate_scene_batch(video_id: str, scene_numbers: list, engine: str = None) -> dict:
    """Generate video for specific scenes."""
    engine = engine or config.DEFAULT_ENGINE
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(f"No scenes data found for {video_id}. Run analysis first.")

    all_scenes = scenes_data.get("scenes", [])
    filtered_scenes = [s for s in all_scenes if s.get("scene_number") in scene_numbers]
    if not filtered_scenes:
        raise ValueError(
            f"None of the requested scene numbers {scene_numbers} exist. "
            f"Available: {[s.get('scene_number') for s in all_scenes]}"
        )

    filtered_scenes_data = {**scenes_data, "scenes": filtered_scenes}

    write_status(state="running", step="video", step_number=5,
                 message=f"Generating scenes {scene_numbers} ({engine})…",
                 video_id=video_id, error=None, engine=engine)
    logger.info("Generating scenes %s for %s (engine=%s)", scene_numbers, video_id, engine)

    new_video_data = generate_videos(filtered_scenes_data, output_dir, engine=engine)

    # Merge with existing
    existing_video_data = load_step_data(output_dir, STEP_FILES["video"]) or {}
    existing_map: dict = {}
    for sv in existing_video_data.get("generated_scenes", []):
        existing_map[sv.get("scene_number")] = sv
    for sv in new_video_data.get("generated_scenes", []):
        existing_map[sv.get("scene_number")] = sv

    merged = {
        **existing_video_data,
        **new_video_data,
        "generated_scenes": sorted(existing_map.values(), key=lambda s: s.get("scene_number", 0)),
    }

    save_checkpoint(output_dir, STEP_FILES["video"], merged)
    write_status(state="completed", step="video",
                 message=f"Scenes {scene_numbers} generated.", progress=100.0)
    logger.info("Scene batch complete for %s", video_id)
    return merged


def generate_project_audio(video_id: str, voice_key: str = None) -> dict:
    """Generate voiceover and music (steps 6-7)."""
    output_dir = get_output_dir(video_id)

    script_data = load_step_data(output_dir, STEP_FILES["script"])
    if script_data is None:
        raise FileNotFoundError(f"No script data for {video_id}.")

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(f"No scenes data for {video_id}.")

    write_status(state="running", step="audio", step_number=6,
                 message="Generating voiceover…", video_id=video_id, error=None)
    logger.info("Generating voiceover for %s (voice=%s)", video_id, voice_key)
    audio_data = generate_audio(script_data, scenes_data, output_dir, voice_key=voice_key)
    save_checkpoint(output_dir, STEP_FILES["audio"], audio_data)

    write_status(step="music", step_number=7, message="Generating background music…", progress=50.0)
    logger.info("Generating music for %s", video_id)
    total_duration = sum(
        s.get("duration_sec", s.get("duration", 10)) for s in scenes_data.get("scenes", [])
    )
    music_data = generate_music(total_duration, output_dir, scenes=scenes_data.get("scenes", []))
    save_checkpoint(output_dir, STEP_FILES["music"], music_data)

    write_status(state="completed", step="music", message="Audio generation complete!", progress=100.0)
    logger.info("Audio generation complete for %s", video_id)
    return {"audio": audio_data, "music": music_data}


def assemble_project(video_id: str) -> dict:
    """Assemble the final video (step 8)."""
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    video_data = load_step_data(output_dir, STEP_FILES["video"])
    audio_data = load_step_data(output_dir, STEP_FILES["audio"])
    music_data = load_step_data(output_dir, STEP_FILES["music"])

    missing = []
    if scenes_data is None: missing.append("scenes (step 4)")
    if video_data is None: missing.append("video (step 5)")
    if audio_data is None: missing.append("audio (step 6)")
    # Music is optional — assembly can proceed with voiceover only
    if missing:
        raise FileNotFoundError(f"Missing data for assembly: {', '.join(missing)}")

    write_status(state="running", step="assembly", step_number=8,
                 message="Assembling final video…", video_id=video_id, error=None)
    logger.info("Assembling video for %s", video_id)

    assembly_data = assemble_video(scenes_data, video_data, audio_data, music_data, output_dir)
    save_checkpoint(output_dir, STEP_FILES["assembly"], assembly_data)

    write_status(state="completed", step="assembly", message="Assembly complete!", progress=100.0)
    logger.info("Assembly complete for %s", video_id)
    return assembly_data


# ---------------------------------------------------------------------------
# Project introspection
# ---------------------------------------------------------------------------

def get_project_data(video_id: str) -> dict:
    """Get all available project data."""
    output_dir = get_output_dir(video_id)

    project: dict = {
        "video_id": video_id,
        "output_dir": str(output_dir),
        "steps_completed": [],
    }

    for step_name in STEP_ORDER:
        data = load_step_data(output_dir, STEP_FILES[step_name])
        if data is not None:
            project[step_name] = data
            project["steps_completed"].append(step_name)

    project["source_info"] = project.get("source")
    project["virality_analysis"] = project.get("virality")
    project["script"] = project.get("script")
    project["scene_status"] = get_scene_status(video_id)

    try:
        project["cost_estimate"] = estimate_cost(video_id)
    except Exception:
        project["cost_estimate"] = None

    output_files = []
    for fpath in output_dir.iterdir():
        if fpath.is_file():
            output_files.append(str(fpath))
    project["output_files"] = sorted(output_files)

    # Add history info
    hist = get_history_record(video_id)
    if hist:
        project["history"] = hist

    return project


def get_scene_status(video_id: str) -> list:
    """Get generation status of each scene."""
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        return []

    video_data = load_step_data(output_dir, STEP_FILES["video"]) or {}
    video_map: dict = {}
    for vs in video_data.get("generated_scenes", []):
        video_map[vs.get("scene_number")] = vs

    statuses = []
    for scene in scenes_data.get("scenes", []):
        sn = scene.get("scene_number")
        vs = video_map.get(sn)
        if vs is not None:
            status = vs.get("status", "completed")
            video_path = vs.get("video_path")
        else:
            status = "pending"
            video_path = None

        statuses.append({
            "scene_number": sn,
            "status": status,
            "video_path": video_path,
        })

    return statuses


def regenerate_scene(video_id: str, scene_number: int, engine: str = None) -> dict:
    """Regenerate video for a single scene."""
    engine = engine or config.DEFAULT_ENGINE
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(f"No scenes data for {video_id}.")

    target_scene = None
    for s in scenes_data.get("scenes", []):
        if s.get("scene_number") == scene_number:
            target_scene = s
            break
    if target_scene is None:
        raise ValueError(f"Scene {scene_number} not found in {video_id}.")

    write_status(state="running", step="video",
                 message=f"Regenerating scene {scene_number} ({engine})…",
                 video_id=video_id, error=None, engine=engine)
    logger.info("Regenerating scene %d for %s (engine=%s)", scene_number, video_id, engine)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    new_scene_video = generate_single_scene(target_scene, videos_dir, engine=engine)

    existing_video_data = load_step_data(output_dir, STEP_FILES["video"]) or {"generated_scenes": []}
    scenes_map: dict = {}
    for sv in existing_video_data.get("generated_scenes", []):
        scenes_map[sv.get("scene_number")] = sv
    scenes_map[scene_number] = new_scene_video

    merged = {
        **existing_video_data,
        "generated_scenes": sorted(scenes_map.values(), key=lambda s: s.get("scene_number", 0)),
    }
    save_checkpoint(output_dir, STEP_FILES["video"], merged)

    write_status(state="completed", step="video",
                 message=f"Scene {scene_number} regenerated.", progress=100.0)
    logger.info("Scene %d regenerated for %s", scene_number, video_id)
    return new_scene_video


def edit_scene_prompt(video_id: str, scene_number: int, new_prompt: str) -> dict:
    """Edit a scene's visual prompt."""
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(f"No scenes data for {video_id}.")

    updated_scene = None
    for scene in scenes_data.get("scenes", []):
        if scene.get("scene_number") == scene_number:
            scene["visual_prompt"] = new_prompt
            updated_scene = scene
            break

    if updated_scene is None:
        raise ValueError(f"Scene {scene_number} not found in {video_id}.")

    save_checkpoint(output_dir, STEP_FILES["scenes"], scenes_data)
    logger.info("Updated prompt for scene %d of %s", scene_number, video_id)
    return updated_scene


def estimate_cost(video_id: str, engine: str = None) -> dict:
    """Estimate credits needed for remaining work."""
    engine = engine or config.DEFAULT_ENGINE
    engine_cfg = config.ENGINE_CONFIG.get(engine, {})
    cost_per_sec = engine_cfg.get("cost_per_sec", 5)

    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(f"No scenes data for {video_id}.")

    scenes = scenes_data.get("scenes", [])
    scene_count = len(scenes)

    video_data = load_step_data(output_dir, STEP_FILES["video"]) or {}
    completed_nums = set()
    for sv in video_data.get("generated_scenes", []):
        if sv.get("status") in ("completed", "success", None):
            completed_nums.add(sv.get("scene_number"))

    pending_scenes = [s for s in scenes if s.get("scene_number") not in completed_nums]
    pending_video_duration = sum(s.get("duration_sec", 10) for s in pending_scenes)
    video_credits = pending_video_duration * cost_per_sec

    audio_data = load_step_data(output_dir, STEP_FILES["audio"])
    if audio_data is None:
        total_chars = sum(len(s.get("narration", "")) for s in scenes)
        tts_credits = total_chars / 50
    else:
        tts_credits = 0

    music_data = load_step_data(output_dir, STEP_FILES["music"])
    total_duration = sum(s.get("duration_sec", 10) for s in scenes)
    music_credits = (total_duration / 6) if music_data is None else 0

    total_credits = video_credits + tts_credits + music_credits

    return {
        "engine": engine,
        "cost_per_sec": cost_per_sec,
        "scene_count": scene_count,
        "pending_scenes": len(pending_scenes),
        "total_duration_sec": total_duration,
        "pending_video_duration_sec": pending_video_duration,
        "video_credits": round(video_credits, 1),
        "tts_credits": round(tts_credits, 1),
        "music_credits": round(music_credits, 1),
        "total_credits": round(total_credits, 1),
    }


