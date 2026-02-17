"""Step 4: Split script into scenes with visual prompts using Claude."""

import json
import logging
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

OUTPUT FORMAT: Return a JSON array of scene objects. Output ONLY valid JSON, no markdown code blocks.

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

    # Parse JSON from response (handle potential markdown wrapping)
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        json_text = "\n".join(lines[1:-1])

    scenes = json.loads(json_text)

    total_duration = sum(s.get("duration_sec", 10) for s in scenes)

    result = {
        "scenes": scenes,
        "scene_count": len(scenes),
        "total_duration_sec": total_duration,
        "model": config.CLAUDE_MODEL,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    # Save checkpoint
    with open(output_dir / "step4_scenes.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Scene split complete: %d scenes, ~%ds total", len(scenes), total_duration)
    return result
