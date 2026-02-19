"""Voiceover + SFX generation using Runway API.

TTS: eleven_multilingual_v2 — natural documentary narration (Runway preset voices)
SFX: eleven_text_to_sound_v2 — per-scene ambient sound effects
"""

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import httpx
from runwayml import RunwayML

import config

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 1000  # Runway TTS limit per request

# Mapping of voice keys to Runway preset voice names
# See: https://docs.dev.runwayml.com/api
VOICE_KEY_TO_PRESET = {
    "george": "Mark",       # Deep, authoritative male
    "daniel": "James",      # Calm, clear male
    "brian": "Bernard",     # Deep American male
    "james": "James",       # Mature British male
    "liam": "Noah",         # Young, energetic male
}
DEFAULT_PRESET = "Mark"


def _split_script_into_chunks(text: str, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    """Split script text into chunks at sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            sub_parts = re.split(r',\s*', sentence)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                if current and len(current) + len(part) + 2 > max_chars:
                    chunks.append(current.strip())
                    current = part
                elif current:
                    current += ", " + part
                else:
                    current = part
            continue

        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def _download_audio(url: str, dest: Path, timeout: float = 120.0) -> Path:
    """Download audio file from URL."""
    logger.info("Downloading audio -> %s", dest)
    with httpx.Client(timeout=timeout, follow_redirects=True) as http:
        resp = http.get(url)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    logger.info("Downloaded %d bytes to %s", dest.stat().st_size, dest)
    return dest


def _concatenate_audio(file_paths: list[Path], output_path: Path) -> Path:
    """Concatenate multiple audio files with FFmpeg."""
    if len(file_paths) == 1:
        import shutil
        shutil.copy2(file_paths[0], output_path)
        return output_path

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


def _get_preset_voice(voice_key: str | None = None) -> str:
    """Get Runway preset voice name from voice key."""
    voice_key = (voice_key or config.DEFAULT_VOICE).lower()
    preset = VOICE_KEY_TO_PRESET.get(voice_key, DEFAULT_PRESET)
    logger.info("Voice key '%s' -> Runway preset '%s'", voice_key, preset)
    return preset


def generate_voiceover(script_text: str, output_dir, voice_key: str | None = None) -> Path:
    """Generate voiceover via Runway TTS API.

    Parameters
    ----------
    script_text : str
        The narration script text.
    output_dir : path-like
        Directory to save output audio.
    voice_key : str, optional
        Key for voice selection (mapped to Runway preset).
        Defaults to config.DEFAULT_VOICE.
    """
    config.check_api_key("RUNWAY_API_KEY")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = RunwayML(api_key=config.RUNWAY_API_KEY)

    # Resolve voice to Runway preset
    preset_name = _get_preset_voice(voice_key)
    logger.info("Using Runway preset voice: %s", preset_name)

    # Clean script markers
    clean_text = script_text
    for marker in ["[HOOK]", "[CLIMAX]", "[FINALE]"]:
        clean_text = clean_text.replace(marker, "")

    chunks = _split_script_into_chunks(clean_text)
    if not chunks:
        raise ValueError("Script text is empty.")

    logger.info("Script split into %d TTS chunk(s) for Runway TTS.", len(chunks))

    chunk_paths: list[Path] = []
    for i, chunk_text in enumerate(chunks):
        chunk_path = output_dir / f"voiceover_chunk_{i:03d}.mp3"
        logger.info("TTS chunk %d/%d (%d chars)...", i + 1, len(chunks), len(chunk_text))

        try:
            task = client.text_to_speech.create(
                model=config.RUNWAY_TTS_MODEL,
                prompt_text=chunk_text,
                voice={
                    "type": "runway-preset",
                    "presetId": preset_name,
                },
            )
        except Exception as exc:
            if "credit" in str(exc).lower() or "insufficient" in str(exc).lower():
                raise RuntimeError(f"Runway TTS credit exhaustion: {exc}") from exc
            raise

        logger.info("TTS task %s created, waiting...", task.id)
        result = task.wait_for_task_output()

        if not result or not result.output or len(result.output) == 0:
            raise RuntimeError(f"TTS task {task.id} returned empty output.")

        audio_url = result.output[0]
        _download_audio(audio_url, chunk_path)
        chunk_paths.append(chunk_path)

    final_path = output_dir / "voiceover.mp3"
    _concatenate_audio(chunk_paths, final_path)

    if len(chunk_paths) > 1:
        for cp in chunk_paths:
            cp.unlink(missing_ok=True)

    logger.info("Voiceover complete: %s (voice=%s, preset=%s)", final_path, voice_key, preset_name)
    return final_path


def generate_sfx(scenes: list, output_dir) -> list:
    """Generate per-scene sound effects via Runway Sound Effect API."""
    config.check_api_key("RUNWAY_API_KEY")
    output_dir = Path(output_dir)
    sfx_dir = output_dir / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)

    client = RunwayML(api_key=config.RUNWAY_API_KEY)
    sfx_results = []

    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        sfx_prompt = scene.get("sfx_prompt", "")
        duration = min(scene.get("duration_sec", 10), 30)

        if not sfx_prompt:
            # Generate a default SFX prompt from mood
            mood = scene.get("mood", "").lower()
            if not mood:
                continue
            sfx_prompt = _mood_to_sfx_prompt(mood, scene.get("visual_prompt", ""))
            if not sfx_prompt:
                continue

        logger.info("Generating SFX for scene %d: %.60s...", scene_num, sfx_prompt)

        try:
            task = client.sound_effect.create(
                model=config.RUNWAY_MUSIC_MODEL,
                prompt_text=sfx_prompt,
                duration=duration,
            )

            logger.info("SFX task %s created, waiting...", task.id)
            result = task.wait_for_task_output()

            if not result or not result.output or len(result.output) == 0:
                logger.warning("SFX task %s returned empty output, skipping.", task.id)
                continue

            sfx_path = sfx_dir / f"sfx_{scene_num:03d}.mp3"
            _download_audio(result.output[0], sfx_path)

            sfx_results.append({
                "scene_number": scene_num,
                "path": str(sfx_path),
                "prompt": sfx_prompt,
            })
            logger.info("SFX generated for scene %d", scene_num)

        except Exception as e:
            if "credit" in str(e).lower():
                logger.warning("SFX credit issue, stopping SFX generation: %s", str(e)[:100])
                break
            logger.warning("SFX failed for scene %d: %s, skipping", scene_num, str(e)[:100])

    logger.info("Generated %d SFX clips", len(sfx_results))
    return sfx_results


def _mood_to_sfx_prompt(mood: str, visual_prompt: str) -> str:
    mood_map = {
        "battle": "sounds of distant battle, swords clashing, war drums, soldiers shouting",
        "war": "distant artillery fire, marching soldiers on cobblestone, wind howling",
        "tense": "deep suspenseful drone, subtle heartbeat, wind, distant murmuring crowd",
        "dramatic": "dramatic orchestral hit, deep rumbling bass, tension rising",
        "storm": "thunder rolling, heavy rain on rooftops, wind howling through streets",
        "explosion": "distant explosion with rumble, debris falling, crowd screaming",
        "triumphant": "crowd cheering loudly, church bells ringing, celebratory atmosphere",
        "epic": "epic deep drums, brass fanfare swell, crowd roaring",
        "action": "fast percussion, impacts, whooshing movements, crowd panic",
        "somber": "gentle wind, distant church bell, quiet crowd murmuring",
        "awe": "wide ambient reverb, crowd gasping, wind",
        "revolutionary": "angry crowd chanting, drums beating, fists pounding on wood",
    }
    for key, sfx in mood_map.items():
        if key in mood:
            return sfx
    return "ambient atmospheric sounds, subtle cinematic environmental texture"


def generate_audio(script_data: dict, scenes_data: dict, output_dir, voice_key: str | None = None) -> dict:
    """Main entry: generate voiceover + per-scene SFX.

    Parameters
    ----------
    voice_key : str, optional
        Key for voice selection (mapped to Runway preset).
    """
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    script_text = script_data.get("script", "")
    if not script_text:
        raise ValueError("script_data must contain a non-empty 'script' key.")

    logger.info("Generating voiceover for %d-char script (voice_key=%s)...", len(script_text), voice_key)
    voiceover_path = generate_voiceover(script_text, audio_dir, voice_key=voice_key)

    logger.info("Generating per-scene SFX...")
    sfx_results = generate_sfx(scenes_data.get("scenes", []), audio_dir)

    result = {
        "voiceover_path": str(voiceover_path),
        "voice_key": voice_key or config.DEFAULT_VOICE,
        "sfx": sfx_results,
        "sfx_count": len(sfx_results),
    }

    step_file = output_dir / "step6_audio.json"
    step_file.write_text(json.dumps(result, indent=2))
    logger.info("Audio step result saved to %s", step_file)

    return result
