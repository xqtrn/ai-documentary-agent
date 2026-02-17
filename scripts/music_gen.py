"""Step 7: Generate background music using Beatoven.ai API."""

import json
import logging
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


def generate_music(duration_sec: int, output_dir: Path) -> dict:
    """Generate cinematic documentary background music."""
    logger.info("Generating background music (%ds)...", duration_sec)

    music_dir = output_dir / "audio"
    music_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create a composition
    for attempt in range(config.MAX_RETRIES):
        try:
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
                timeout=60,
            )
            resp.raise_for_status()
            track = resp.json()
            track_id = track.get("id") or track.get("track_id")
            logger.info("Beatoven track created: %s", track_id)
            break
        except Exception as e:
            logger.warning("Beatoven create attempt %d/%d failed: %s", attempt + 1, config.MAX_RETRIES, e)
            if attempt == config.MAX_RETRIES - 1:
                raise
            time.sleep(5 * (attempt + 1))

    # Step 2: Start composition/render
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{BASE_URL}/tracks/{track_id}/compose",
                headers=get_headers(),
                json={"format": "mp3"},
                timeout=60,
            )
            resp.raise_for_status()
            task = resp.json()
            task_id = task.get("id") or task.get("task_id") or track_id
            logger.info("Beatoven composition started: %s", task_id)
            break
        except Exception as e:
            logger.warning("Beatoven compose attempt %d/%d failed: %s", attempt + 1, config.MAX_RETRIES, e)
            if attempt == config.MAX_RETRIES - 1:
                raise
            time.sleep(5 * (attempt + 1))

    # Step 3: Poll until done
    music_path = music_dir / "background_music.mp3"
    max_polls = 120
    for i in range(max_polls):
        try:
            resp = httpx.get(
                f"{BASE_URL}/tracks/{track_id}/status",
                headers=get_headers(),
                timeout=30,
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
                    break
            elif state in ("failed", "error"):
                raise RuntimeError(f"Beatoven composition failed: {status}")

        except httpx.HTTPStatusError:
            pass  # May not be ready yet

        time.sleep(5)
    else:
        raise TimeoutError("Beatoven composition timed out")

    result = {
        "music_path": str(music_path),
        "duration_sec": duration_sec,
        "track_id": track_id,
    }

    with open(output_dir / "step7_music.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
