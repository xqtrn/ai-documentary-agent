"""Dynamic background music generation using Runway Sound Effect API.

Generates multiple mood-matched music segments for different sections of the
documentary, then concatenates them with FFmpeg crossfades.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx
from runwayml import RunwayML

import config

logger = logging.getLogger(__name__)

MAX_CHUNK_SEC = 30  # Runway sound effect max per request

# Music prompts for different documentary moods
MOOD_MUSIC = {
    "opening": "epic cinematic orchestral music, dramatic tension-building strings and brass, deep war drums, film score introduction, slow build",
    "educational": "light curious documentary background music, gentle piano and strings, educational atmosphere, moderate tempo",
    "conflict": "intense dramatic percussion, war drums, urgent strings, battle tension, cinematic action score",
    "emotional": "emotional cinematic piano and strings, reflective melancholic atmosphere, gentle crescendo, film score",
    "resolution": "resolving orchestral swell, hopeful strings and piano, documentary conclusion, peaceful yet powerful",
    "default": "cinematic orchestral documentary background music, dramatic strings and piano, emotional ambient atmosphere, slow tempo, film score",
}


def _download_audio(url: str, dest: Path, timeout: float = 120.0) -> Path:
    logger.info("Downloading audio -> %s", dest)
    with httpx.Client(timeout=timeout, follow_redirects=True) as http:
        resp = http.get(url)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    logger.info("Downloaded %d bytes to %s", dest.stat().st_size, dest)
    return dest


def _concatenate_with_crossfade(file_paths: list[Path], output_path: Path, crossfade_sec: float = 2.0) -> Path:
    """Concatenate audio files with crossfade transitions."""
    if len(file_paths) == 1:
        import shutil
        shutil.copy2(file_paths[0], output_path)
        return output_path

    if len(file_paths) == 2:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(file_paths[0]),
            "-i", str(file_paths[1]),
            "-filter_complex",
            f"[0:a][1:a]acrossfade=d={crossfade_sec}:c1=tri:c2=tri[out]",
            "-map", "[out]",
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("Crossfade failed, falling back to concat: %s", result.stderr[:200])
            return _simple_concat(file_paths, output_path)
        return output_path

    # For 3+ files, use simple concat (complex crossfade filter gets unwieldy)
    return _simple_concat(file_paths, output_path)


def _simple_concat(file_paths: list[Path], output_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=output_path.parent
    ) as f:
        for p in file_paths:
            f.write(f"file '{p.resolve()}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

    Path(list_path).unlink(missing_ok=True)
    return output_path


def _generate_segment(client: RunwayML, prompt: str, duration_sec: int, output_path: Path) -> Path:
    """Generate a single music segment via Runway Sound Effect."""
    clamped = min(duration_sec, MAX_CHUNK_SEC)
    logger.info("Generating music segment (%d sec): %.60s...", clamped, prompt)

    try:
        task = client.sound_effect.create(
            model=config.RUNWAY_MUSIC_MODEL,
            prompt_text=prompt,
            duration=clamped,
        )
    except Exception as exc:
        if "credit" in str(exc).lower() or "insufficient" in str(exc).lower():
            raise RuntimeError(f"Runway music credit exhaustion: {exc}") from exc
        raise

    logger.info("Music task %s created, waiting...", task.id)
    result = task.wait_for_task_output()

    if not result or not result.output or len(result.output) == 0:
        raise RuntimeError(f"Music task {task.id} returned empty output.")

    return _download_audio(result.output[0], output_path)


def _determine_mood_segments(scenes: list, total_duration: int) -> list[dict]:
    """Determine music mood segments based on scene moods."""
    if not scenes:
        return [{"mood": "default", "duration": total_duration, "prompt": MOOD_MUSIC["default"]}]

    segments = []
    scene_count = len(scenes)

    for i, scene in enumerate(scenes):
        mood = scene.get("mood", "").lower()
        duration = scene.get("duration_sec", 10)

        # Map scene mood to music mood
        if i == 0:
            music_mood = "opening"
        elif "battle" in mood or "war" in mood or "conflict" in mood or "action" in mood or "revolution" in mood:
            music_mood = "conflict"
        elif "tense" in mood or "dramatic" in mood or "intense" in mood:
            music_mood = "conflict"
        elif "somber" in mood or "sad" in mood or "reflective" in mood or "emotional" in mood:
            music_mood = "emotional"
        elif "triumph" in mood or "victory" in mood or "resolution" in mood or "hope" in mood:
            music_mood = "resolution"
        elif i == scene_count - 1:
            music_mood = "resolution"
        else:
            music_mood = "educational"

        prompt = MOOD_MUSIC.get(music_mood, MOOD_MUSIC["default"])
        segments.append({"mood": music_mood, "duration": duration, "prompt": prompt})

    # Merge consecutive segments with the same mood
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg["mood"] == merged[-1]["mood"]:
            merged[-1]["duration"] += seg["duration"]
        else:
            merged.append(seg)

    return merged


def generate_music(duration_sec: int, output_dir, scenes: list = None) -> dict:
    """Generate dynamic background music with mood-matched segments.

    Args:
        duration_sec: Total desired duration.
        output_dir: Base output directory.
        scenes: Optional list of scene dicts for mood matching.

    Returns:
        Dict with music_path, duration_sec, source.
    """
    config.check_api_key("RUNWAY_API_KEY")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    music_dir = output_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)

    client = RunwayML(api_key=config.RUNWAY_API_KEY)

    # Determine mood segments
    segments = _determine_mood_segments(scenes or [], duration_sec)
    logger.info("Music plan: %d mood segment(s) for %d sec total", len(segments), duration_sec)

    segment_paths: list[Path] = []
    for i, seg in enumerate(segments):
        seg_duration = seg["duration"]
        seg_prompt = seg["prompt"]

        # Split long segments into chunks
        remaining = seg_duration
        chunk_idx = 0
        while remaining > 0:
            chunk_dur = min(remaining, MAX_CHUNK_SEC)
            chunk_path = music_dir / f"music_seg{i:02d}_chunk{chunk_idx:02d}.mp3"
            _generate_segment(client, seg_prompt, chunk_dur, chunk_path)
            segment_paths.append(chunk_path)
            remaining -= chunk_dur
            chunk_idx += 1

    # Concatenate all segments
    final_path = music_dir / "background_music.mp3"
    _concatenate_with_crossfade(segment_paths, final_path)

    # Clean up chunks
    for sp in segment_paths:
        if sp != final_path:
            sp.unlink(missing_ok=True)

    logger.info("Music generation complete: %s (%d sec)", final_path, duration_sec)

    result = {
        "music_path": str(final_path),
        "duration_sec": duration_sec,
        "source": "runway",
        "segments": len(segments),
    }

    step_file = output_dir / "step7_music.json"
    step_file.write_text(json.dumps(result, indent=2))
    logger.info("Music step result saved to %s", step_file)

    return result
