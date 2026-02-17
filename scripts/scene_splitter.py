"""Step 4: Split script into scenes with visual prompts using Claude."""

import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

SCENE_SPLIT_PROMPT = """You are a cinematic AI video director. Your job is to break a documentary script into individual scenes and create detailed visual prompts for AI video generation.

SCRIPT:
{script}

TASK: Break this script into {scene_min}-{scene_max} scenes, each 8-12 seconds long.

For each scene, provide:
1. **scene_number**: Sequential number
2. **narration**: The exact narration text for this scene (from the script)
3. **visual_prompt**: A detailed prompt for AI video generation (Runway Gen-4)
4. **camera**: Camera movement/angle (e.g., "slow dolly forward", "aerial pan left", "close-up static")
5. **lighting**: Lighting description (e.g., "golden hour warm light", "dramatic side lighting")
6. **mood**: Emotional mood (e.g., "tense", "awe-inspiring", "somber")
7. **duration_sec**: Target duration (8-12 seconds)

CRITICAL RULES FOR VISUAL PROMPTS:
- NEVER include ANY text, letters, words, numbers, signs, labels, titles, subtitles, captions, banners, inscriptions, writing in the visual prompt
- NO newspapers, documents, books with visible text, maps with labels, screens with text
- Instead of "soldier reads a letter" → "soldier holds a folded paper, camera on his emotional face"
- Instead of "newspaper headline about war" → "stack of newspapers on a table, shot with shallow depth of field, focus on hands of the reader"
- Instead of "map showing territory" → "abstract aerial view of varied terrain, rivers and mountains from above"
- Every prompt MUST end with: "no text, no letters, no words, no subtitles, no signs, no writing, no numbers, no captions"
- Style: photorealistic cinematic, NOT animation, NOT stock footage
- Include specific details: ethnicity of people, clothing period, architecture style, weather, time of day

OUTPUT FORMAT: Return a JSON array of scene objects. Output ONLY valid JSON, no markdown code blocks, no trailing commas.

[
  {{
    "scene_number": 1,
    "narration": "exact narration text",
    "visual_prompt": "detailed visual description ending with no-text instruction",
    "camera": "camera movement",
    "lighting": "lighting description",
    "mood": "emotional mood",
    "duration_sec": 10
  }}
]"""


def _fix_json(text: str) -> str:
    """Fix common JSON issues from LLM output."""
    # Remove markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end])

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix single quotes to double quotes (careful with apostrophes in text)
    # Only do this if the JSON doesn't parse as-is
    return text.strip()


def _parse_json_robust(raw_text: str) -> list:
    """Try multiple strategies to parse JSON from LLM output."""
    text = raw_text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Fix common issues
    fixed = _fix_json(text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find JSON array in text
    match = re.search(r'\[.*\]', fixed, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try fixing the extracted array
            arr_text = re.sub(r',\s*([}\]])', r'\1', match.group())
            try:
                return json.loads(arr_text)
            except json.JSONDecodeError:
                pass

    # Strategy 4: Try to parse as individual objects and collect
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

    # Last resort: raise with context
    raise json.JSONDecodeError(
        f"Could not parse JSON after all strategies. First 200 chars: {text[:200]}",
        text, 0
    )


def split_into_scenes(script_data: dict, output_dir: Path) -> dict:
    """Split script into scenes with visual prompts."""
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
        raise RuntimeError(f"Scene splitter returned invalid data: expected list of scenes, got {type(scenes)}")

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
