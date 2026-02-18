"""Step 4: Split script into scenes with hyperrealistic cinematic visual prompts."""

import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

SCENE_SPLIT_PROMPT = """You are a world-class cinematic director creating prompts for Grok Imagine, xAI's video generation model. Your goal: every scene must look like a real Hollywood film.

SCRIPT:
{script}

TASK: Break this script into {scene_min}-{scene_max} scenes, each 8 seconds long.

For each scene provide:
1. **scene_number**: Sequential number
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: A cinematic video prompt (MAX 500 characters, see rules below)
4. **camera**: Specific camera movement
5. **lighting**: Specific lighting
6. **mood**: Emotional mood for music/SFX
7. **sfx_prompt**: Sound effect description (MUST avoid violent/graphic language — use atmospheric sounds only: crowd murmur, wind, rain, bells, footsteps, crackling fire, distant drums)
8. **duration_sec**: 8

VISUAL PROMPT RULES (MAX 500 CHARACTERS EACH):
- Style: "Hyperrealistic cinematic 4K, shot on ARRI Alexa 65, anamorphic lens"
- NEVER: "painting", "illustration", "animated", "cartoon", "artistic"
- Every scene MUST have motion: crowds moving, flags waving, smoke rising, flames flickering
- Include camera movement: drone, tracking, crane, steadicam
- Include period-accurate costumes and architecture
- Scene 1 MUST be an epic aerial establishing shot
- End every prompt with: "no text, no letters, no words, no subtitles, no watermark"
- Keep prompts CONCISE — under 500 characters. Grok works better with focused prompts.

SFX PROMPT RULES:
- Describe atmospheric ambient sounds only
- Be specific: "large crowd murmuring, distant church bells, horse hooves on cobblestone"
- NEVER include graphic/violent descriptions — content moderation will block them
- Focus on: wind, rain, fire crackling, crowd murmur, bells, footsteps, drums, nature sounds

OUTPUT FORMAT: Return ONLY a JSON array, no markdown code blocks.

[
  {{
    "scene_number": 1,
    "narration": "exact narration text",
    "visual_prompt": "hyperrealistic cinematic description under 500 chars",
    "camera": "specific camera movement",
    "lighting": "specific lighting",
    "mood": "emotional mood",
    "sfx_prompt": "atmospheric sound description",
    "duration_sec": 8
  }}
]"""


def _fix_json(text: str) -> str:
    """Fix common JSON issues from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end])
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text.strip()


def _parse_json_robust(raw_text: str) -> list:
    """Try multiple strategies to parse JSON from LLM output."""
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fixed = _fix_json(text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*\]', fixed, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            arr_text = re.sub(r',\s*([}\]])', r'\1', match.group())
            try:
                return json.loads(arr_text)
            except json.JSONDecodeError:
                pass

    objects = []
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', fixed):
        try:
            obj = json.loads(match.group())
            if "scene_number" in obj or "narration" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    if objects:
        return objects

    raise json.JSONDecodeError(
        f"Could not parse JSON. First 200 chars: {text[:200]}", text, 0
    )


def split_into_scenes(script_data: dict, output_dir) -> dict:
    """Split script into scenes with visual prompts."""
    output_dir = Path(output_dir)
    logger.info("Splitting script into scenes...")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = SCENE_SPLIT_PROMPT.format(
        script=script_data["script"],
        scene_min=config.SCENES_COUNT_MIN,
        scene_max=config.SCENES_COUNT_MAX,
    )

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text
    logger.info("Scene splitter raw response: %d chars", len(raw_text))

    scenes = _parse_json_robust(raw_text)

    if not isinstance(scenes, list) or len(scenes) == 0:
        raise RuntimeError(f"Scene splitter returned invalid data: {type(scenes)}")

    total_duration = sum(s.get("duration_sec", 10) for s in scenes)

    result = {
        "scenes": scenes,
        "scene_count": len(scenes),
        "total_duration_sec": total_duration,
        "model": config.CLAUDE_MODEL,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    with open(output_dir / "step4_scenes.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Scene split complete: %d scenes, ~%ds total", len(scenes), total_duration)
    return result
