"""Step 7: Generate background music.

Tries Beatoven API first, then ElevenLabs sound generation,
then falls back to FFmpeg-generated ambient music pad (NOT silence).
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


def _try_beatoven(duration_sec: int, music_path: Path) -> bool:
    """Try to generate music via Beatoven API. Returns True on success."""
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
            timeout=15,
        )
        resp.raise_for_status()
        track = resp.json()
        track_id = track.get("id") or track.get("track_id")
        logger.info("Beatoven track created: %s", track_id)

        resp = httpx.post(
            f"{BASE_URL}/tracks/{track_id}/compose",
            headers=get_headers(),
            json={"format": "mp3"},
            timeout=15,
        )
        resp.raise_for_status()

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
                    logger.info("Beatoven music saved: %s", music_path)
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


def _try_elevenlabs_music(duration_sec: int, music_path: Path) -> bool:
    """Try generating background music using ElevenLabs sound generation API."""
    try:
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }

        # ElevenLabs sound gen supports up to ~22 seconds per request
        chunk_duration = min(duration_sec, 22)
        chunks_needed = max(1, -(-duration_sec // chunk_duration))  # ceil division

        chunk_paths = []
        for i in range(chunks_needed):
            this_duration = min(chunk_duration, duration_sec - i * chunk_duration)
            if this_duration <= 0:
                break

            prompt = (
                "cinematic documentary background music, orchestral strings and piano, "
                "dramatic ambient atmosphere, slow tempo, continuous instrumental soundtrack, "
                "dark and emotional tone, film score style"
            )

            resp = httpx.post(
                "https://api.elevenlabs.io/v1/sound-generation",
                headers=headers,
                json={
                    "text": prompt,
                    "duration_seconds": this_duration,
                },
                timeout=120,
            )
            resp.raise_for_status()

            chunk_path = music_path.parent / f"music_chunk_{i:03d}.mp3"
            chunk_path.write_bytes(resp.content)
            chunk_paths.append(chunk_path)
            logger.info("Music chunk %d/%d generated (%d bytes)", i + 1, chunks_needed, len(resp.content))

        if not chunk_paths:
            return False

        if len(chunk_paths) == 1:
            chunk_paths[0].rename(music_path)
        else:
            concat_list = music_path.parent / "music_concat.txt"
            with open(concat_list, "w") as f:
                for p in chunk_paths:
                    f.write(f"file '{p}'\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-c", "copy", str(music_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
            for p in chunk_paths:
                p.unlink(missing_ok=True)
            concat_list.unlink(missing_ok=True)

        logger.info("ElevenLabs music saved: %s", music_path)
        return True
    except Exception as e:
        logger.warning("ElevenLabs music generation failed: %s", e)
        for f in music_path.parent.glob("music_chunk_*.mp3"):
            f.unlink(missing_ok=True)
        return False


def _generate_ambient_pad(duration_sec: int, output_path: Path) -> Path:
    """Generate a cinematic ambient pad using FFmpeg audio synthesis.

    Creates a dark ambient drone by layering sine waves in a minor chord
    with low-pass filtering and fade effects. This is REAL audio (NOT silence).
    """
    logger.info("Generating ambient pad fallback (%ds)...", duration_sec)

    fade_out_start = max(0, duration_sec - 3)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=110:duration={duration_sec}",
        "-f", "lavfi", "-i", f"sine=frequency=164.81:duration={duration_sec}",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration_sec}",
        "-f", "lavfi", "-i", f"sine=frequency=261.63:duration={duration_sec}",
        "-f", "lavfi", "-i", f"sine=frequency=196:duration={duration_sec}",
        "-filter_complex",
        f"[0]volume=0.12[a];"
        f"[1]volume=0.08[b];"
        f"[2]volume=0.06[c];"
        f"[3]volume=0.04[d];"
        f"[4]volume=0.07[e];"
        f"[a][b][c][d][e]amix=inputs=5:normalize=0,"
        f"lowpass=f=600,"
        f"afade=t=in:d=3,"
        f"afade=t=out:st={fade_out_start}:d=3",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg ambient pad failed: {result.stderr}")

    logger.info("Ambient pad saved: %s", output_path)
    return output_path


def generate_music(duration_sec: int, output_dir: Path) -> dict:
    """Generate cinematic documentary background music with fallback chain."""
    logger.info("Generating background music (%ds)...", duration_sec)

    music_dir = output_dir / "audio"
    music_dir.mkdir(parents=True, exist_ok=True)
    music_path = music_dir / "background_music.mp3"

    source = "unknown"

    # Try Beatoven first
    if _try_beatoven(duration_sec, music_path):
        source = "beatoven"
    # Try ElevenLabs sound generation
    elif _try_elevenlabs_music(duration_sec, music_path):
        source = "elevenlabs"
    # Fall back to FFmpeg ambient pad (NOT silence)
    else:
        logger.warning("Beatoven and ElevenLabs music unavailable, generating ambient pad")
        _generate_ambient_pad(duration_sec, music_path)
        source = "ffmpeg_ambient"

    result = {
        "music_path": str(music_path),
        "duration_sec": duration_sec,
        "source": source,
    }

    with open(output_dir / "step7_music.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
