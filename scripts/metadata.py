"""Generate YouTube metadata, SRT subtitles, and thumbnail prompt."""

import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

METADATA_PROMPT = """You are a YouTube SEO expert. Generate optimized metadata for a documentary video.

SCRIPT SUMMARY (first 500 words):
{script_preview}

ORIGINAL VIDEO TOPIC: {original_title}

Generate:
1. **title**: Compelling YouTube title (under 70 chars, includes power words, creates curiosity)
2. **description**: YouTube description (1500-2000 chars). Include:
   - Compelling first 2 lines (shown in search)
   - Topic overview
   - Timestamps placeholder
   - Call to action (subscribe, like)
   - Relevant hashtags (5-8)
3. **tags**: List of 15-20 relevant YouTube tags for SEO
4. **thumbnail_prompt**: A detailed prompt for generating a cinematic thumbnail image (16:9). Must be visually striking, NO TEXT in the image. The thumbnail should convey the topic through imagery alone.

Return as JSON:
{{
  "title": "...",
  "description": "...",
  "tags": ["...", "..."],
  "thumbnail_prompt": "..."
}}

Output ONLY valid JSON."""


def generate_metadata(source_data: dict, script_data: dict, output_dir: Path) -> dict:
    """Generate YouTube metadata."""
    logger.info("Generating YouTube metadata...")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    script_preview = script_data["script"][:2000]

    prompt = METADATA_PROMPT.format(
        script_preview=script_preview,
        original_title=source_data.get("title", "Documentary"),
    )

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    metadata = json.loads(raw)

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Metadata generated: %s", metadata.get("title"))
    return metadata


def generate_srt(scenes_data: dict, output_dir: Path) -> Path:
    """Generate SRT subtitles from scene narrations."""
    logger.info("Generating SRT subtitles...")

    srt_path = output_dir / "subtitles.srt"
    lines = []
    current_time = 0.0

    for i, scene in enumerate(scenes_data["scenes"], 1):
        duration = scene.get("duration_sec", 10)
        narration = scene.get("narration", "").strip()
        if not narration:
            current_time += duration
            continue

        start = _format_srt_time(current_time)
        end = _format_srt_time(current_time + duration)

        # Split long narrations into subtitle chunks (~15 words each)
        words = narration.split()
        chunk_size = 15
        chunks = [" ".join(words[j:j+chunk_size]) for j in range(0, len(words), chunk_size)]

        chunk_duration = duration / max(len(chunks), 1)
        for k, chunk in enumerate(chunks):
            sub_start = _format_srt_time(current_time + k * chunk_duration)
            sub_end = _format_srt_time(current_time + (k + 1) * chunk_duration)
            lines.append(f"{len(lines)//4 + 1}")
            lines.append(f"{sub_start} --> {sub_end}")
            lines.append(chunk)
            lines.append("")

        current_time += duration

    srt_path.write_text("\n".join(lines))
    logger.info("SRT saved: %s", srt_path)
    return srt_path


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_all_metadata(source_data: dict, script_data: dict, scenes_data: dict, output_dir: Path) -> dict:
    """Generate all metadata files."""
    metadata = generate_metadata(source_data, script_data, output_dir)
    srt_path = generate_srt(scenes_data, output_dir)

    result = {
        "metadata": metadata,
        "srt_path": str(srt_path),
    }

    with open(output_dir / "step_metadata.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
