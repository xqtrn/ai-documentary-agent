"""Step 6: Generate voiceover and SFX using ElevenLabs."""

import json
import logging
import subprocess
import time
from pathlib import Path

import httpx

import config

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elevenlabs.io/v1"


def get_headers(for_tts=False):
    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
    }
    if for_tts:
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "audio/mpeg"
    else:
        headers["Content-Type"] = "application/json"
    return headers


def get_voice_id(voice_name: str) -> str:
    """Look up voice ID by name."""
    resp = httpx.get(f"{BASE_URL}/voices", headers=get_headers(), timeout=30)
    resp.raise_for_status()
    voices = resp.json()["voices"]
    
    # Log available voices
    voice_names = [v["name"] for v in voices]
    logger.info("Available voices: %s", ", ".join(voice_names))
    
    for v in voices:
        if v["name"].lower() == voice_name.lower():
            logger.info("Found voice '%s': %s", voice_name, v["voice_id"])
            return v["voice_id"]
    
    # Fallback to first available
    logger.warning("Voice '%s' not found, using '%s'", voice_name, voices[0]["name"])
    return voices[0]["voice_id"]


def generate_voiceover(script_text: str, output_dir: Path) -> Path:
    """Generate full narration audio."""
    logger.info("Generating voiceover (%d chars)...", len(script_text))

    voice_id = get_voice_id(config.ELEVENLABS_VOICE)

    # Clean script text
    clean_text = script_text
    for marker in ["[HOOK]", "[CLIMAX]", "[FINALE]"]:
        clean_text = clean_text.replace(marker, "")

    # Split into chunks
    max_chars = 5000
    chunks = []
    paragraphs = clean_text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    logger.info("Split into %d chunks for TTS", len(chunks))

    audio_parts = []
    for i, chunk in enumerate(chunks):
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{BASE_URL}/text-to-speech/{voice_id}",
                    headers=get_headers(for_tts=True),
                    json={
                        "text": chunk,
                        "model_id": config.ELEVENLABS_MODEL,
                        "voice_settings": {
                            "stability": 0.6,
                            "similarity_boost": 0.8,
                            "style": 0.4,
                            "use_speaker_boost": True,
                        },
                    },
                    timeout=120,
                )
                
                if resp.status_code == 401:
                    # Log the actual error body for debugging
                    logger.error("ElevenLabs 401 response body: %s", resp.text[:500])
                    logger.error("API key prefix: %s...", config.ELEVENLABS_API_KEY[:10])
                
                resp.raise_for_status()

                chunk_path = output_dir / f"voiceover_part_{i:03d}.mp3"
                chunk_path.write_bytes(resp.content)
                audio_parts.append(chunk_path)
                logger.info("Voiceover chunk %d/%d generated (%d bytes)", i + 1, len(chunks), len(resp.content))
                break

            except Exception as e:
                logger.warning("TTS attempt %d/%d failed for chunk %d: %s", attempt + 1, config.MAX_RETRIES, i, e)
                if attempt == config.MAX_RETRIES - 1:
                    raise
                time.sleep(5 * (attempt + 1))

    # Concatenate audio parts
    if len(audio_parts) == 1:
        final_path = output_dir / "voiceover.mp3"
        audio_parts[0].rename(final_path)
    else:
        concat_list = output_dir / "concat_list.txt"
        with open(concat_list, "w") as f:
            for p in audio_parts:
                f.write(f"file '{p}'\n")

        final_path = output_dir / "voiceover.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(final_path)],
            check=True, capture_output=True, timeout=120,
        )
        for p in audio_parts:
            p.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)

    logger.info("Voiceover saved: %s", final_path)
    return final_path


def generate_sfx(scenes: list, output_dir: Path) -> list:
    """Generate SFX for key scenes using ElevenLabs Sound Effects."""
    logger.info("Generating SFX for key scenes...")

    sfx_dir = output_dir / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    sfx_results = []

    sfx_moods = {"tense", "dramatic", "action", "battle", "war", "explosion", "storm", "triumphant", "epic"}

    for scene in scenes:
        mood = scene.get("mood", "").lower()
        if not any(m in mood for m in sfx_moods):
            continue

        scene_num = scene["scene_number"]
        sfx_prompt = _mood_to_sfx_prompt(mood, scene.get("visual_prompt", ""))

        for attempt in range(config.MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{BASE_URL}/sound-generation",
                    headers=get_headers(),
                    json={
                        "text": sfx_prompt,
                        "duration_seconds": scene.get("duration_sec", 10),
                    },
                    timeout=120,
                )
                resp.raise_for_status()

                sfx_path = sfx_dir / f"sfx_{scene_num:03d}.mp3"
                sfx_path.write_bytes(resp.content)
                sfx_results.append({
                    "scene_number": scene_num,
                    "path": str(sfx_path),
                    "prompt": sfx_prompt,
                })
                logger.info("SFX generated for scene %d", scene_num)
                break
            except Exception as e:
                logger.warning("SFX attempt %d/%d failed for scene %d: %s", attempt + 1, config.MAX_RETRIES, scene_num, e)
                if attempt == config.MAX_RETRIES - 1:
                    logger.error("SFX generation failed for scene %d, skipping", scene_num)
                time.sleep(3 * (attempt + 1))

    logger.info("Generated %d SFX clips", len(sfx_results))
    return sfx_results


def _mood_to_sfx_prompt(mood: str, visual_prompt: str) -> str:
    mood_map = {
        "battle": "sounds of distant battle, swords clashing, war drums",
        "war": "distant artillery, marching soldiers, wind",
        "tense": "deep suspenseful drone, heartbeat, subtle tension",
        "dramatic": "dramatic orchestral hit, deep bass, tension",
        "storm": "thunder, heavy rain, wind howling",
        "explosion": "distant explosion, rumble, debris",
        "triumphant": "triumphant fanfare, crowd cheering, bells",
        "epic": "epic orchestral swell, deep drums, brass",
        "action": "fast-paced percussion, impacts, whooshes",
    }
    for key, sfx in mood_map.items():
        if key in mood:
            return sfx
    return "ambient atmospheric sound, subtle cinematic texture"


def generate_audio(script_data: dict, scenes_data: dict, output_dir: Path) -> dict:
    """Main entry: generate voiceover + SFX."""
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    voiceover_path = generate_voiceover(script_data["script"], audio_dir)
    sfx_results = generate_sfx(scenes_data["scenes"], audio_dir)

    result = {
        "voiceover_path": str(voiceover_path),
        "sfx": sfx_results,
        "sfx_count": len(sfx_results),
    }

    with open(output_dir / "step6_audio.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
