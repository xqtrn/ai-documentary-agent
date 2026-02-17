"""Step 7: Generate background music using Beatoven.ai API.

Falls back to generating a silent audio track if Beatoven is unavailable,
so the pipeline can continue without background music.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

import httpx

import config

logger = logging.getLogger(__name__)

BASE_URL = config.BEATOVEN_BASE_URL


def get_headers():
    return {
        "Authorization": f"Bearer {config.BEATOVEN_API_KEY}",
        "Content-Type": "application/json",
    }


def _generate_silent_audio(duration_sec: int, output_path: Path) -> Path:
    """Generate a silent MP3 file as fallback when music API is unavailable."""
    logger.info("Generating silent audio fallback (%ds)...", duration_sec)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", str(duration_sec),
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg silent audio failed: {result.stderr}")
    logger.info("Silent audio fallback saved: %s", output_path)
    return output_path


def _try_beatoven(duration_sec: int, music_path: Path) -> bool:
    """Try to generate music via Beatoven API. Returns True on success."""
    try:
        # Step 1: Create a composition
        resp = httpx.post(
            f"{BASE_URL}/tracks",
            headers=get_headers(),
            json={
                "title": "Documentary Background",
                "genre": "cinematic",
                "mood": "dramatic",
                "tempo": "medium",
                "duration": duration_sec,
                "instruments": ["orchestra", "piano", "strings"],
            },
            timeout=15,
        )
        resp.raise_for_status()
        track = resp.json()
        track_id = track.get("id") or track.get("track_id")
        logger.info("Beatoven track created: %s", track_id)

        # Step 2: Start composition/render
        resp = httpx.post(
            f"{BASE_URL}/tracks/{track_id}/compose",
            headers=get_headers(),
            json={"format": "mp3"},
            timeout=15,
        )
        resp.raise_for_status()
        task = resp.json()
        task_id = task.get("id") or task.get("task_id") or track_id
        logger.info("Beatoven composition started: %s", task_id)

        # Step 3: Poll until done
        for i in range(120):
            resp = httpx.get(
                f"{BASE_URL}/tracks/{track_id}/status",
                headers=get_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            status = resp.json()

            state = status.get("status", "").lower()
            if state in ("completed", "done", "ready"):
                download_url = status.get("download_url") or status.get("url") or status.get("output_url")
                if download_url:
                    dl_resp = httpx.get(download_url, timeout=120)
                    dl_resp.raise_for_status()
                    music_path.write_bytes(dl_resp.content)
                    logger.info("Background music saved: %s", music_path)
                    return True
            elif state in ("failed", "error"):
                logger.error("Beatoven composition failed: %s", status)
                return False

            time.sleep(5)

        logger.error("Beatoven composition timed out")
        return False

    except Exception as e:
        logger.warning("Beatoven API failed: %s", e)
        return False


def generate_music(duration_sec: int, output_dir: Path) -> dict:
    """Generate cinematic documentary background music with fallback."""
    logger.info("Generating background music (%ds)...", duration_sec)

    music_dir = output_dir / "audio"
    music_dir.mkdir(parents=True, exist_ok=True)
    music_path = music_dir / "background_music.mp3"

    # Try Beatoven first
    success = _try_beatoven(duration_sec, music_path)

    if not success:
        logger.warning("Beatoven unavailable, using silent audio fallback")
        _generate_silent_audio(duration_sec, music_path)

    result = {
        "music_path": str(music_path),
        "duration_sec": duration_sec,
        "source": "beatoven" if success else "silent_fallback",
    }

    with open(output_dir / "step7_music.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
