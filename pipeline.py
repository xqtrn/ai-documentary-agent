#!/usr/bin/env python3
"""AI Documentary Agent - Main Pipeline CLI."""

import argparse
import json
import logging
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import config
from scripts.source_analyzer import analyze_source
from scripts.virality_analyzer import analyze_virality
from scripts.scriptwriter import rewrite_script
from scripts.scene_splitter import split_into_scenes
from scripts.video_gen import generate_videos
from scripts.voiceover import generate_audio
from scripts.music_gen import generate_music
from scripts.assembler import assemble_video
from scripts.metadata import generate_all_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

STEPS = [
    "source", "virality", "script", "scenes",
    "video", "audio", "music", "assembly", "metadata",
]

STEP_NAMES = {
    "source": "Source Analysis",
    "virality": "Virality Analysis",
    "script": "Script Rewrite",
    "scenes": "Scene Splitting",
    "video": "Video Generation",
    "audio": "Voiceover + SFX",
    "music": "Background Music",
    "assembly": "Final Assembly",
    "metadata": "Metadata Generation",
}

# Global pipeline state for web UI
_pipeline_lock = threading.Lock()
_pipeline_cancel = threading.Event()
_log_buffer = deque(maxlen=100)


class StatusLogHandler(logging.Handler):
    """Capture log messages into a shared buffer for the web UI."""
    def emit(self, record):
        msg = self.format(record)
        _log_buffer.append(msg)


# Install the status log handler on root logger
_status_handler = StatusLogHandler()
_status_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_status_handler)


def _status_path() -> Path:
    p = Path(config.OUTPUT_DIR) / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_status(state: str, step_index: int = 0, step_name: str = "", url: str = "",
                 error: str = "", output_dir: str = ""):
    """Write pipeline status to status.json for the web UI."""
    total = len(STEPS)
    pct = int((step_index / total) * 100) if total else 0
    data = {
        "state": state,  # idle / running / completed / error
        "step_index": step_index,
        "step_total": total,
        "step_name": step_name,
        "progress_pct": pct,
        "url": url,
        "error": error,
        "output_dir": output_dir,
        "logs": list(_log_buffer),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _status_path().write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass


def read_status() -> dict:
    """Read current pipeline status."""
    p = _status_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"state": "idle", "step_index": 0, "step_total": len(STEPS),
            "step_name": "", "progress_pct": 0, "url": "", "error": "",
            "output_dir": "", "logs": [], "updated_at": ""}


def get_output_dir(url: str) -> Path:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    slug = match.group(1) if match else "unknown"
    output_dir = Path(config.OUTPUT_DIR) / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_checkpoint(output_dir: Path):
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        data = json.loads(checkpoint_file.read_text())
        return data.get("last_step")
    return None


def save_checkpoint(output_dir: Path, step: str):
    checkpoint_file = output_dir / "checkpoint.json"
    data = {"last_step": step}
    checkpoint_file.write_text(json.dumps(data))


def load_step_data(output_dir: Path, step_file: str) -> dict:
    path = output_dir / step_file
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return json.loads(path.read_text())


def cancel_pipeline():
    """Signal the pipeline to cancel."""
    _pipeline_cancel.set()


def _check_cancel():
    if _pipeline_cancel.is_set():
        raise InterruptedError("Pipeline cancelled by user")


def run_pipeline(url: str, start_from: str = None):
    """Run the full pipeline."""
    _pipeline_cancel.clear()
    _log_buffer.clear()

    output_dir = get_output_dir(url)
    logger.info("Output directory: %s", output_dir)

    last_step = get_checkpoint(output_dir)
    if start_from:
        start_idx = STEPS.index(start_from)
    elif last_step:
        start_idx = STEPS.index(last_step) + 1
        logger.info("Resuming from after step: %s", last_step)
    else:
        start_idx = 0

    def run_step(idx, name, fn, *args):
        _check_cancel()
        logger.info("=" * 60)
        logger.info("STEP %d: %s", idx + 1, STEP_NAMES[name])
        write_status("running", idx + 1, STEP_NAMES[name], url, output_dir=str(output_dir))
        result = fn(*args)
        save_checkpoint(output_dir, name)
        return result

    try:
        write_status("running", 0, "Starting...", url, output_dir=str(output_dir))

        # Step 1
        if start_idx <= 0:
            source_data = run_step(0, "source", analyze_source, url, output_dir)
        else:
            source_data = load_step_data(output_dir, "step1_source.json")

        # Step 2
        if start_idx <= 1:
            virality_data = run_step(1, "virality", analyze_virality, source_data, output_dir)
        else:
            virality_data = load_step_data(output_dir, "step2_virality.json")

        # Step 3
        if start_idx <= 2:
            script_data = run_step(2, "script", rewrite_script, source_data, virality_data, output_dir)
        else:
            script_data = load_step_data(output_dir, "step3_script.json")

        # Step 4
        if start_idx <= 3:
            scenes_data = run_step(3, "scenes", split_into_scenes, script_data, output_dir)
        else:
            scenes_data = load_step_data(output_dir, "step4_scenes.json")

        # Step 5
        if start_idx <= 4:
            video_data = run_step(4, "video", generate_videos, scenes_data, output_dir)
            if video_data.get("successful", 0) == 0:
                raise RuntimeError(f"Video generation failed: 0/{video_data.get('total_scenes', 0)} scenes succeeded")
        else:
            video_data = load_step_data(output_dir, "step5_videos.json")

        # Step 6
        if start_idx <= 5:
            audio_data = run_step(5, "audio", generate_audio, script_data, scenes_data, output_dir)
        else:
            audio_data = load_step_data(output_dir, "step6_audio.json")

        # Step 7
        if start_idx <= 6:
            total_duration = scenes_data.get("total_duration_sec", 1200)
            music_data = run_step(6, "music", generate_music, total_duration, output_dir)
        else:
            music_data = load_step_data(output_dir, "step7_music.json")

        # Step 8
        if start_idx <= 7:
            run_step(7, "assembly", assemble_video, scenes_data, video_data, audio_data, music_data, output_dir)

        # Metadata
        if start_idx <= 8:
            _check_cancel()
            logger.info("=" * 60)
            logger.info("STEP 9: Metadata Generation")
            write_status("running", 9, "Metadata Generation", url, output_dir=str(output_dir))
            generate_all_metadata(source_data, script_data, scenes_data, output_dir)
            save_checkpoint(output_dir, "metadata")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        write_status("completed", len(STEPS), "Done", url, output_dir=str(output_dir))
        return str(output_dir)

    except InterruptedError:
        logger.warning("Pipeline cancelled")
        write_status("idle", 0, "Cancelled", url, output_dir=str(output_dir))
        raise
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        write_status("error", 0, "", url, error=str(e), output_dir=str(output_dir))
        raise


def main():
    parser = argparse.ArgumentParser(description="AI Documentary Agent Pipeline")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--start-from", choices=STEPS, help="Resume from a specific step")
    args = parser.parse_args()

    try:
        output = run_pipeline(args.url, args.start_from)
        print(f"\nDone! Output: {output}")
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
