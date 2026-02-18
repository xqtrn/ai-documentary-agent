"""
AI Documentary Agent - Pipeline Orchestrator

Supports both full sequential runs and incremental/batch generation.
Pipeline steps:
  1. source    - YouTube transcript download
  2. virality  - Claude analysis
  3. script    - Claude rewrite
  4. scenes    - Scene splitting
  5. video     - Runway video generation
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
    "state": "idle",       # idle | running | completed | failed | cancelled
    "step": None,
    "step_number": 0,
    "total_steps": 9,
    "message": "",
    "progress": 0.0,
    "video_id": None,
    "error": None,
}
_status_lock = threading.Lock()

_pipeline_cancel = threading.Event()

# Log buffer consumed by the web UI
_log_buffer: list[str] = []
_log_buffer_lock = threading.Lock()

MAX_LOG_LINES = 2000

# Step file names used for checkpointing
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
# Logging handler that feeds the web UI
# ---------------------------------------------------------------------------

class StatusLogHandler(logging.Handler):
    """Captures log records into the shared _log_buffer list."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _log_buffer_lock:
                _log_buffer.append(msg)
                # Trim if the buffer grows too large
                while len(_log_buffer) > MAX_LOG_LINES:
                    _log_buffer.pop(0)
        except Exception:
            self.handleError(record)


def get_log_buffer(since: int = 0) -> list[str]:
    """Return log lines starting from index *since*."""
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
                 error: str | None = None) -> None:
    """Thread-safe update of the global pipeline status dict."""
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


def read_status() -> dict:
    """Return a snapshot of the current pipeline status."""
    with _status_lock:
        return dict(_pipeline_status)


def _reset_status() -> None:
    """Reset status back to idle defaults."""
    write_status(state="idle", step=None, step_number=0, message="",
                 progress=0.0, video_id=None, error=None)


# ---------------------------------------------------------------------------
# Cancel helpers
# ---------------------------------------------------------------------------

def request_cancel() -> None:
    """Signal the running pipeline to cancel at the next checkpoint."""
    _pipeline_cancel.set()
    write_status(state="cancelled", message="Cancel requested — stopping after current step.")
    logger.info("Pipeline cancellation requested.")


# Alias for web.py compatibility
cancel_pipeline = request_cancel


def _check_cancel() -> None:
    """Raise if cancellation was requested."""
    if _pipeline_cancel.is_set():
        raise PipelineCancelled("Pipeline cancelled by user.")


class PipelineCancelled(Exception):
    """Raised when the pipeline is cancelled mid-run."""


# ---------------------------------------------------------------------------
# Path / data helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
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
    # Fallback: treat the whole string as a video ID already
    if re.match(r'^[\w-]{11}$', url):
        return url
    raise ValueError(f"Cannot extract video ID from: {url}")


def get_output_dir(url_or_video_id: str) -> Path:
    """Get (and create) the output directory for a project.

    Accepts either a full YouTube URL or a bare video ID.
    """
    try:
        video_id = extract_video_id(url_or_video_id)
    except ValueError:
        # Assume it is already a video_id-style string
        video_id = url_or_video_id

    output_dir = Path(config.OUTPUT_DIR) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_step_data(output_dir: Path, step_file: str) -> dict | None:
    """Load checkpoint data from a JSON file, or return None."""
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
    """Persist step data as JSON and return the file path."""
    path = output_dir / step_file
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Checkpoint saved: %s", path)
    return path


def get_checkpoint(video_id: str, step_name: str) -> dict | None:
    """Public helper: load checkpoint for a given step by video ID."""
    output_dir = get_output_dir(video_id)
    step_file = STEP_FILES.get(step_name)
    if step_file is None:
        return None
    return load_step_data(output_dir, step_file)


# ---------------------------------------------------------------------------
# Full pipeline (sequential)
# ---------------------------------------------------------------------------

def run_pipeline(url: str, *, resume: bool = True) -> dict:
    """Run the complete 9-step pipeline sequentially.

    Parameters
    ----------
    url : str
        YouTube video URL.
    resume : bool
        If True, skip steps that already have checkpoint files.

    Returns
    -------
    dict
        Final project data including all step outputs.
    """
    _pipeline_cancel.clear()
    clear_log_buffer()

    video_id = extract_video_id(url)
    output_dir = get_output_dir(url)

    write_status(state="running", step="source", step_number=1, progress=0.0,
                 message="Starting pipeline…", video_id=video_id, error=None)

    results: dict = {"video_id": video_id, "output_dir": str(output_dir)}

    try:
        # ------------------------------------------------------------------
        # Step 1 — Source
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="source", step_number=1, message="Downloading transcript…",
                     progress=0.0)
        source_data = load_step_data(output_dir, STEP_FILES["source"]) if resume else None
        if source_data is None:
            logger.info("Step 1/9: Analyzing source — %s", url)
            source_data = analyze_source(url, output_dir)
            save_checkpoint(output_dir, STEP_FILES["source"], source_data)
        else:
            logger.info("Step 1/9: Source — loaded from checkpoint.")
        results["source"] = source_data

        # ------------------------------------------------------------------
        # Step 2 — Virality
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="virality", step_number=2,
                     message="Running virality analysis…", progress=11.0)
        virality_data = load_step_data(output_dir, STEP_FILES["virality"]) if resume else None
        if virality_data is None:
            logger.info("Step 2/9: Virality analysis.")
            virality_data = analyze_virality(source_data, output_dir)
            save_checkpoint(output_dir, STEP_FILES["virality"], virality_data)
        else:
            logger.info("Step 2/9: Virality — loaded from checkpoint.")
        results["virality"] = virality_data

        # ------------------------------------------------------------------
        # Step 3 — Script
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="script", step_number=3,
                     message="Rewriting script…", progress=22.0)
        script_data = load_step_data(output_dir, STEP_FILES["script"]) if resume else None
        if script_data is None:
            logger.info("Step 3/9: Scriptwriting.")
            script_data = rewrite_script(source_data, virality_data, output_dir)
            save_checkpoint(output_dir, STEP_FILES["script"], script_data)
        else:
            logger.info("Step 3/9: Script — loaded from checkpoint.")
        results["script"] = script_data

        # ------------------------------------------------------------------
        # Step 4 — Scenes
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="scenes", step_number=4,
                     message="Splitting into scenes…", progress=33.0)
        scenes_data = load_step_data(output_dir, STEP_FILES["scenes"]) if resume else None
        if scenes_data is None:
            logger.info("Step 4/9: Scene splitting.")
            scenes_data = split_into_scenes(script_data, output_dir)
            save_checkpoint(output_dir, STEP_FILES["scenes"], scenes_data)
        else:
            logger.info("Step 4/9: Scenes — loaded from checkpoint.")
        results["scenes"] = scenes_data

        # ------------------------------------------------------------------
        # Step 5 — Video generation
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="video", step_number=5,
                     message="Generating scene videos…", progress=44.0)
        video_data = load_step_data(output_dir, STEP_FILES["video"]) if resume else None
        if video_data is None:
            logger.info("Step 5/9: Video generation.")
            video_data = generate_videos(scenes_data, output_dir)
            save_checkpoint(output_dir, STEP_FILES["video"], video_data)
        else:
            logger.info("Step 5/9: Videos — loaded from checkpoint.")
        results["video"] = video_data

        # ------------------------------------------------------------------
        # Step 6 — Audio / voiceover
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="audio", step_number=6,
                     message="Generating voiceover…", progress=55.0)
        audio_data = load_step_data(output_dir, STEP_FILES["audio"]) if resume else None
        if audio_data is None:
            logger.info("Step 6/9: Voiceover generation.")
            audio_data = generate_audio(script_data, scenes_data, output_dir)
            save_checkpoint(output_dir, STEP_FILES["audio"], audio_data)
        else:
            logger.info("Step 6/9: Audio — loaded from checkpoint.")
        results["audio"] = audio_data

        # ------------------------------------------------------------------
        # Step 7 — Music
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="music", step_number=7,
                     message="Generating background music…", progress=66.0)
        music_data = load_step_data(output_dir, STEP_FILES["music"]) if resume else None
        if music_data is None:
            logger.info("Step 7/9: Music generation.")
            total_duration = sum(
                s.get("duration_sec", s.get("duration", 10)) for s in scenes_data.get("scenes", [])
            )
            music_data = generate_music(total_duration, output_dir, scenes=scenes_data.get("scenes", []))
            save_checkpoint(output_dir, STEP_FILES["music"], music_data)
        else:
            logger.info("Step 7/9: Music — loaded from checkpoint.")
        results["music"] = music_data

        # ------------------------------------------------------------------
        # Step 8 — Assembly
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="assembly", step_number=8,
                     message="Assembling final video…", progress=77.0)
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

        # ------------------------------------------------------------------
        # Step 9 — Metadata
        # ------------------------------------------------------------------
        _check_cancel()
        write_status(step="metadata", step_number=9,
                     message="Generating metadata…", progress=88.0)
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

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        write_status(state="completed", step="done", step_number=9,
                     message="Pipeline complete!", progress=100.0)
        logger.info("Pipeline completed successfully for %s", video_id)
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
# Incremental / batch functions
# ---------------------------------------------------------------------------

def run_analysis(url: str) -> dict:
    """Run steps 1-4 only (source, virality, script, scenes).

    Creates the output directory from the video ID, executes the four
    analysis steps, and returns a dict containing all step data plus the
    output_dir path.
    """
    _pipeline_cancel.clear()

    video_id = extract_video_id(url)
    output_dir = get_output_dir(url)

    write_status(state="running", step="source", step_number=1, progress=0.0,
                 message="Starting analysis…", video_id=video_id, error=None)

    results: dict = {"video_id": video_id, "output_dir": str(output_dir)}

    try:
        # Step 1 — Source
        _check_cancel()
        write_status(step="source", step_number=1,
                     message="Downloading transcript…", progress=0.0)
        logger.info("Analysis step 1/4: Analyzing source — %s", url)
        source_data = analyze_source(url, output_dir)
        save_checkpoint(output_dir, STEP_FILES["source"], source_data)
        results["source"] = source_data

        # Step 2 — Virality
        _check_cancel()
        write_status(step="virality", step_number=2,
                     message="Running virality analysis…", progress=25.0)
        logger.info("Analysis step 2/4: Virality analysis.")
        virality_data = analyze_virality(source_data, output_dir)
        save_checkpoint(output_dir, STEP_FILES["virality"], virality_data)
        results["virality"] = virality_data

        # Step 3 — Script
        _check_cancel()
        write_status(step="script", step_number=3,
                     message="Rewriting script…", progress=50.0)
        logger.info("Analysis step 3/4: Scriptwriting.")
        script_data = rewrite_script(source_data, virality_data, output_dir)
        save_checkpoint(output_dir, STEP_FILES["script"], script_data)
        results["script"] = script_data

        # Step 4 — Scenes
        _check_cancel()
        write_status(step="scenes", step_number=4,
                     message="Splitting into scenes…", progress=75.0)
        logger.info("Analysis step 4/4: Scene splitting.")
        scenes_data = split_into_scenes(script_data, output_dir)
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


def generate_scene_batch(video_id: str, scene_numbers: list) -> dict:
    """Generate video for specific scenes (step 5, partial).

    Loads the scenes data from step4_scenes.json, filters to only the
    requested scene numbers, generates video for those scenes, then merges
    the results into step5_videos.json (preserving any previously generated
    scenes).

    Parameters
    ----------
    video_id : str
        The YouTube video ID (project identifier).
    scene_numbers : list
        List of 1-based scene numbers to generate.

    Returns
    -------
    dict
        The updated (merged) video data.
    """
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(
            f"No scenes data found for {video_id}. Run analysis first."
        )

    # Build a filtered scenes_data containing only the requested scenes
    all_scenes = scenes_data.get("scenes", [])
    filtered_scenes = [
        s for s in all_scenes if s.get("scene_number") in scene_numbers
    ]
    if not filtered_scenes:
        raise ValueError(
            f"None of the requested scene numbers {scene_numbers} exist. "
            f"Available: {[s.get('scene_number') for s in all_scenes]}"
        )

    filtered_scenes_data = {**scenes_data, "scenes": filtered_scenes}

    write_status(state="running", step="video", step_number=5,
                 message=f"Generating video for scenes {scene_numbers}…",
                 video_id=video_id, error=None)
    logger.info("Generating video for scenes %s of %s", scene_numbers, video_id)

    new_video_data = generate_videos(filtered_scenes_data, output_dir)

    # Merge with existing video data (if any)
    existing_video_data = load_step_data(output_dir, STEP_FILES["video"]) or {}
    existing_scenes_map: dict = {}
    for sv in existing_video_data.get("scenes", []):
        existing_scenes_map[sv.get("scene_number")] = sv
    for sv in new_video_data.get("scenes", []):
        existing_scenes_map[sv.get("scene_number")] = sv

    merged_video_data = {
        **existing_video_data,
        **new_video_data,
        "scenes": sorted(existing_scenes_map.values(),
                         key=lambda s: s.get("scene_number", 0)),
    }

    save_checkpoint(output_dir, STEP_FILES["video"], merged_video_data)
    write_status(state="completed", step="video",
                 message=f"Scenes {scene_numbers} generated.", progress=100.0)
    logger.info("Scene batch generation complete for %s", video_id)
    return merged_video_data


def generate_project_audio(video_id: str) -> dict:
    """Generate voiceover and music (steps 6-7).

    Loads script_data and scenes_data from their checkpoint files, runs TTS
    voiceover generation and music generation, then saves checkpoints.

    Returns
    -------
    dict
        ``{"audio": audio_data, "music": music_data}``
    """
    output_dir = get_output_dir(video_id)

    script_data = load_step_data(output_dir, STEP_FILES["script"])
    if script_data is None:
        raise FileNotFoundError(
            f"No script data found for {video_id}. Run analysis first."
        )

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(
            f"No scenes data found for {video_id}. Run analysis first."
        )

    write_status(state="running", step="audio", step_number=6,
                 message="Generating voiceover…", video_id=video_id, error=None)
    logger.info("Generating voiceover for %s", video_id)
    audio_data = generate_audio(script_data, scenes_data, output_dir)
    save_checkpoint(output_dir, STEP_FILES["audio"], audio_data)

    write_status(step="music", step_number=7,
                 message="Generating background music…", progress=50.0)
    logger.info("Generating music for %s", video_id)
    total_duration = sum(
        s.get("duration_sec", s.get("duration", 10)) for s in scenes_data.get("scenes", [])
    )
    music_data = generate_music(total_duration, output_dir, scenes=scenes_data.get("scenes", []))
    save_checkpoint(output_dir, STEP_FILES["music"], music_data)

    write_status(state="completed", step="music",
                 message="Audio generation complete!", progress=100.0)
    logger.info("Audio generation complete for %s", video_id)

    return {"audio": audio_data, "music": music_data}


def assemble_project(video_id: str) -> dict:
    """Assemble the final video (step 8).

    Loads all required step data and runs the FFmpeg assembly step.
    """
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    video_data = load_step_data(output_dir, STEP_FILES["video"])
    audio_data = load_step_data(output_dir, STEP_FILES["audio"])
    music_data = load_step_data(output_dir, STEP_FILES["music"])

    missing = []
    if scenes_data is None:
        missing.append("scenes (step 4)")
    if video_data is None:
        missing.append("video (step 5)")
    if audio_data is None:
        missing.append("audio (step 6)")
    if music_data is None:
        missing.append("music (step 7)")
    if missing:
        raise FileNotFoundError(
            f"Missing required data for assembly: {', '.join(missing)}"
        )

    write_status(state="running", step="assembly", step_number=8,
                 message="Assembling final video…", video_id=video_id, error=None)
    logger.info("Assembling video for %s", video_id)

    assembly_data = assemble_video(
        scenes_data, video_data, audio_data, music_data, output_dir
    )
    save_checkpoint(output_dir, STEP_FILES["assembly"], assembly_data)

    write_status(state="completed", step="assembly",
                 message="Assembly complete!", progress=100.0)
    logger.info("Assembly complete for %s", video_id)
    return assembly_data


# ---------------------------------------------------------------------------
# Project introspection
# ---------------------------------------------------------------------------

def get_project_data(video_id: str) -> dict:
    """Get all available project data for the dashboard.

    Returns a comprehensive dict with every step's data (if available),
    per-scene video status, cost estimates, and output file paths.
    """
    output_dir = get_output_dir(video_id)

    project: dict = {
        "video_id": video_id,
        "output_dir": str(output_dir),
        "steps_completed": [],
    }

    # Load every step that has a checkpoint
    for step_name in STEP_ORDER:
        data = load_step_data(output_dir, STEP_FILES[step_name])
        if data is not None:
            project[step_name] = data
            project["steps_completed"].append(step_name)

    # Convenience aliases
    project["source_info"] = project.get("source")
    project["virality_analysis"] = project.get("virality")
    project["script"] = project.get("script")

    # Per-scene video status
    project["scene_status"] = get_scene_status(video_id)

    # Cost estimate
    try:
        project["cost_estimate"] = estimate_cost(video_id)
    except Exception:
        project["cost_estimate"] = None

    # Collect output files
    output_files = []
    for fpath in output_dir.iterdir():
        if fpath.is_file():
            output_files.append(str(fpath))
    project["output_files"] = sorted(output_files)

    return project


def get_scene_status(video_id: str) -> list:
    """Get the generation status of each scene's video.

    Returns a list of dicts::

        [
            {
                "scene_number": 1,
                "status": "completed" | "pending" | "failed",
                "video_path": "/path/to/video.mp4" or None,
                "image_path": "/path/to/image.png" or None,
            },
            ...
        ]
    """
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        return []

    video_data = load_step_data(output_dir, STEP_FILES["video"]) or {}
    video_scenes_map: dict = {}
    for vs in video_data.get("scenes", []):
        video_scenes_map[vs.get("scene_number")] = vs

    statuses = []
    for scene in scenes_data.get("scenes", []):
        sn = scene.get("scene_number")
        vs = video_scenes_map.get(sn)
        if vs is not None:
            status = vs.get("status", "completed")
            video_path = vs.get("video_path")
            image_path = vs.get("image_path")
        else:
            status = "pending"
            video_path = None
            image_path = None

        statuses.append({
            "scene_number": sn,
            "status": status,
            "video_path": video_path,
            "image_path": image_path,
        })

    return statuses


def regenerate_scene(video_id: str, scene_number: int) -> dict:
    """Regenerate the video for a single scene.

    Uses ``video_gen.generate_single_scene`` and updates the
    step5_videos.json checkpoint with the new result.

    Returns the updated scene entry dict.
    """
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(
            f"No scenes data found for {video_id}. Run analysis first."
        )

    # Find the target scene
    target_scene = None
    for s in scenes_data.get("scenes", []):
        if s.get("scene_number") == scene_number:
            target_scene = s
            break
    if target_scene is None:
        raise ValueError(
            f"Scene {scene_number} not found in project {video_id}."
        )

    write_status(state="running", step="video",
                 message=f"Regenerating scene {scene_number}…",
                 video_id=video_id, error=None)
    logger.info("Regenerating scene %d for %s", scene_number, video_id)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    new_scene_video = generate_single_scene(target_scene, videos_dir)

    # Merge into existing video checkpoint
    existing_video_data = load_step_data(output_dir, STEP_FILES["video"]) or {"scenes": []}
    scenes_map: dict = {}
    for sv in existing_video_data.get("scenes", []):
        scenes_map[sv.get("scene_number")] = sv
    scenes_map[scene_number] = new_scene_video

    merged = {
        **existing_video_data,
        "scenes": sorted(scenes_map.values(),
                         key=lambda s: s.get("scene_number", 0)),
    }
    save_checkpoint(output_dir, STEP_FILES["video"], merged)

    write_status(state="completed", step="video",
                 message=f"Scene {scene_number} regenerated.", progress=100.0)
    logger.info("Scene %d regenerated for %s", scene_number, video_id)
    return new_scene_video


def edit_scene_prompt(video_id: str, scene_number: int,
                      new_prompt: str) -> dict:
    """Edit a scene's visual prompt in step4_scenes.json.

    Updates the prompt text for the specified scene and saves the
    checkpoint. Does NOT automatically regenerate — call
    ``regenerate_scene`` afterwards if desired.

    Returns the updated scene dict.
    """
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(
            f"No scenes data found for {video_id}. Run analysis first."
        )

    updated_scene = None
    for scene in scenes_data.get("scenes", []):
        if scene.get("scene_number") == scene_number:
            scene["visual_prompt"] = new_prompt
            updated_scene = scene
            break

    if updated_scene is None:
        raise ValueError(
            f"Scene {scene_number} not found in project {video_id}."
        )

    save_checkpoint(output_dir, STEP_FILES["scenes"], scenes_data)
    logger.info("Updated visual prompt for scene %d of %s", scene_number, video_id)
    return updated_scene


def estimate_cost(video_id: str) -> dict:
    """Estimate the credits needed for remaining generation work.

    Calculation basis:
      - Video: ``scene_count * avg_duration * 5`` credits per second
      - TTS:   ``total_characters / 50`` credits
      - Music: ``total_duration / 6`` credits

    Returns a breakdown dict with per-category and total estimates.
    """
    output_dir = get_output_dir(video_id)

    scenes_data = load_step_data(output_dir, STEP_FILES["scenes"])
    if scenes_data is None:
        raise FileNotFoundError(
            f"No scenes data found for {video_id}. Run analysis first."
        )

    scenes = scenes_data.get("scenes", [])
    scene_count = len(scenes)

    # Determine which scenes still need video generation
    video_data = load_step_data(output_dir, STEP_FILES["video"]) or {}
    completed_scene_numbers = set()
    for sv in video_data.get("scenes", []):
        if sv.get("status") in ("completed", None):
            completed_scene_numbers.add(sv.get("scene_number"))

    pending_scenes = [
        s for s in scenes if s.get("scene_number") not in completed_scene_numbers
    ]

    # Video cost
    pending_video_duration = sum(s.get("duration", 5) for s in pending_scenes)
    video_credits = pending_video_duration * 5

    # TTS cost — based on total voiceover character count
    audio_data = load_step_data(output_dir, STEP_FILES["audio"])
    if audio_data is None:
        total_chars = sum(len(s.get("voiceover", "")) for s in scenes)
        tts_credits = total_chars / 50
    else:
        tts_credits = 0  # Already generated

    # Music cost
    music_data = load_step_data(output_dir, STEP_FILES["music"])
    total_duration = sum(s.get("duration", 5) for s in scenes)
    if music_data is None:
        music_credits = total_duration / 6
    else:
        music_credits = 0  # Already generated

    total_credits = video_credits + tts_credits + music_credits

    return {
        "scene_count": scene_count,
        "pending_scenes": len(pending_scenes),
        "total_duration_sec": total_duration,
        "pending_video_duration_sec": pending_video_duration,
        "video_credits": round(video_credits, 1),
        "tts_credits": round(tts_credits, 1),
        "music_credits": round(music_credits, 1),
        "total_credits": round(total_credits, 1),
    }
