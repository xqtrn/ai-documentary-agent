"""Step 6: Generate voiceover and SFX.

Primary: ElevenLabs TTS
Fallback: edge-tts (Microsoft Edge's TTS API, works from cloud IPs)
"""

import asyncio
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

    voice_names = [v["name"] for v in voices]
    logger.info("Available voices: %s", ", ".join(voice_names))

    for v in voices:
        name = v["name"].lower()
        target = voice_name.lower()
        if name == target or name.startswith(target + " ") or name.startswith(target + " -"):
            logger.info("Found voice '%s': %s (ID: %s)", voice_name, v["name"], v["voice_id"])
            return v["voice_id"]

    logger.warning("Voice '%s' not found, using '%s'", voice_name, voices[0]["name"])
    return voices[0]["voice_id"]


def _is_cloud_blocked(resp) -> bool:
    """Check if ElevenLabs is blocking free tier from cloud IP."""
    if resp.status_code == 401:
        body = resp.text.lower()
        return "unusual activity" in body or "free tier" in body or "proxy" in body or "vpn" in body
    return False


def generate_voiceover_elevenlabs(script_text: str, output_dir: Path) -> Path:
    """Generate voiceover using ElevenLabs TTS."""
    logger.info("Trying ElevenLabs TTS (%d chars)...", len(script_text))

    voice_id = get_voice_id(config.ELEVENLABS_VOICE)
    clean_text = script_text
    for marker in ["[HOOK]", "[CLIMAX]", "[FINALE]"]:
        clean_text = clean_text.replace(marker, "")

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

    logger.info("Split into %d chunks for ElevenLabs TTS", len(chunks))
    audio_parts = []
    for i, chunk in enumerate(chunks):
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

        if _is_cloud_blocked(resp):
            raise RuntimeError("ElevenLabs free tier blocked from cloud IP")

        resp.raise_for_status()
        chunk_path = output_dir / f"voiceover_part_{i:03d}.mp3"
        chunk_path.write_bytes(resp.content)
        audio_parts.append(chunk_path)
        logger.info("ElevenLabs chunk %d/%d generated (%d bytes)", i + 1, len(chunks), len(resp.content))

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

    logger.info("ElevenLabs voiceover saved: %s", final_path)
    return final_path


def generate_voiceover_edge_tts(script_text: str, output_dir: Path) -> Path:
    """Generate voiceover using edge-tts (Microsoft Edge's free TTS API)."""
    import edge_tts

    logger.info("Using edge-tts fallback (%d chars)...", len(script_text))

    clean_text = script_text
    for marker in ["[HOOK]", "[CLIMAX]", "[FINALE]"]:
        clean_text = clean_text.replace(marker, "")

    # Good documentary narration voices
    voice = "en-US-GuyNeural"  # Deep, authoritative male voice
    final_path = output_dir / "voiceover.mp3"

    async def _generate():
        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(str(final_path))

    # Run async edge-tts
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_generate())).result(timeout=300)
        else:
            loop.run_until_complete(_generate())
    except RuntimeError:
        asyncio.run(_generate())

    logger.info("edge-tts voiceover saved: %s", final_path)
    return final_path


def generate_voiceover(script_text: str, output_dir: Path) -> Path:
    """Generate voiceover with ElevenLabs, falling back to edge-tts."""
    # Try ElevenLabs first
    try:
        return generate_voiceover_elevenlabs(script_text, output_dir)
    except Exception as e:
        logger.warning("ElevenLabs TTS failed: %s", str(e)[:200])
        logger.info("Falling back to edge-tts (Microsoft Edge TTS)...")

    # Fallback to edge-tts
    return generate_voiceover_edge_tts(script_text, output_dir)


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
            if _is_cloud_blocked(resp):
                logger.warning("ElevenLabs SFX blocked from cloud IP, skipping SFX")
                break
            resp.raise_for_status()

            sfx_path = sfx_dir / f"sfx_{scene_num:03d}.mp3"
            sfx_path.write_bytes(resp.content)
            sfx_results.append({
                "scene_number": scene_num,
                "path": str(sfx_path),
                "prompt": sfx_prompt,
            })
            logger.info("SFX generated for scene %d", scene_num)
        except Exception as e:
            logger.warning("SFX generation failed for scene %d: %s, skipping", scene_num, str(e)[:100])

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
