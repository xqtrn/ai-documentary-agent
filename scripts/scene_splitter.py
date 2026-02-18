"""Step 4: Split script into scenes with hyperrealistic cinematic visual prompts."""

import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

SCENE_SPLIT_PROMPT = """You are a world-class cinematic director and DP creating prompts for Grok Imagine, xAI's text-to-video AI model. Every scene must look indistinguishable from a real Hollywood film shot on ARRI Alexa 65.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each 8 seconds long. Output exactly {scene_count} scenes — no more, no fewer. Distribute ALL narration evenly.

For each scene provide:
1. **scene_number**: Sequential number (1 through {scene_count})
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: A richly detailed cinematic video prompt (TARGET: 2500-3800 characters — see rules below)
4. **camera**: Specific camera movement
5. **lighting**: Specific lighting setup
6. **mood**: Emotional mood
7. **sfx_prompt**: Atmospheric sound effects description (see SFX rules)
8. **duration_sec**: 8

═══════════════════════════════════════════════════════
VISUAL PROMPT RULES — TARGET 2500-3800 CHARACTERS EACH
═══════════════════════════════════════════════════════

Write each prompt like a professional director's shot list covering EVERY visual department. The maximum is 4000 characters — aim for 2500-3800.

Include ALL of the following in each prompt:

**SHOT TYPE & CAMERA:**
Start with the shot type. Camera and lens: "Shot on ARRI Alexa 65, 40mm anamorphic lens, T2.0". Camera movement: steadicam tracking, dolly push-in, crane descending, handheld, drone aerial. Camera height, depth of field, start and end positions.

**SETTING & ARCHITECTURE:**
Detailed 18th century French location. Materials, surfaces, scale. Period-accurate buildings, streets, interiors. Props, furniture, scattered objects. Condition: intact, weathered, destroyed.

**PEOPLE & COSTUMES:**
Precise clothing for every visible person. Materials: "rough undyed linen", "worn velvet", "soot-stained leather". Historical details: tricorn hats, wigs, stockings, clogs, revolutionary cockades. Body language, facial expressions, specific actions. Crowd size and behavior.

**LIGHTING & ATMOSPHERE:**
Time of day, light source and direction. Color temperature. Shadow quality. God rays, flame flicker, reflections on wet surfaces. Weather: fog, rain, heat haze. Particles: dust, smoke, ash, sparks, embers.

**MOTION (CRITICAL — every scene needs visible movement):**
Crowds surging, flags waving, smoke drifting, flames flickering, fabric fluttering, horses galloping, dust swirling, rain falling, torches guttering, papers blowing.

**STYLE & CONSTRAINTS:**
- ALWAYS begin with: "Hyperrealistic cinematic 4K footage"
- NEVER: "painting", "illustration", "animated", "cartoon", "artistic", "stylized"
- Scene 1 MUST be an epic aerial establishing shot
- ALWAYS end with: "no text, no letters, no words, no subtitles, no watermarks, no UI elements, no logos"

═══════════════════════════════════════════════════════
SFX PROMPT RULES
═══════════════════════════════════════════════════════
- Atmospheric ambient sounds ONLY
- Specific: "large crowd murmuring, distant church bells, horse hooves on cobblestone"
- NEVER include graphic/violent language — use: wind, rain, fire crackling, crowd murmur, bells, footsteps, drums, fabric rustling, wood creaking, thunder

═══════════════════════════════════════════════════════

CRITICAL: Output EXACTLY {scene_count} scenes. Keep each visual_prompt between 2500-3800 characters.

OUTPUT FORMAT: Return ONLY a valid JSON array. No markdown code blocks.

[
  {{
    "scene_number": 1,
    "narration": "narration text",
    "visual_prompt": "detailed cinematic description 2500-3800 chars",
    "camera": "camera movement",
    "lighting": "lighting setup",
    "mood": "emotional mood",
    "sfx_prompt": "atmospheric sounds",
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

    target_count = config.SCENES_COUNT_MAX

    prompt = SCENE_SPLIT_PROMPT.format(
        script=script_data["script"],
        scene_count=target_count,
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

    # Enforce max scene count
    if len(scenes) > config.SCENES_COUNT_MAX:
        logger.warning(
            "Scene splitter returned %d scenes, truncating to %d",
            len(scenes), config.SCENES_COUNT_MAX,
        )
        scenes = scenes[:config.SCENES_COUNT_MAX]
        for i, scene in enumerate(scenes):
            scene["scene_number"] = i + 1

    # Enforce max prompt length (xAI limit is 4096 chars)
    XAI_MAX_PROMPT = 4000
    for scene in scenes:
        vp = scene.get("visual_prompt", "")
        if len(vp) > XAI_MAX_PROMPT:
            logger.warning(
                "Scene %d visual_prompt is %d chars, truncating to %d",
                scene.get("scene_number", 0), len(vp), XAI_MAX_PROMPT,
            )
            # Truncate at last sentence boundary before limit
            truncated = vp[:XAI_MAX_PROMPT]
            last_period = truncated.rfind(".")
            if last_period > XAI_MAX_PROMPT - 200:
                truncated = truncated[:last_period + 1]
            scene["visual_prompt"] = truncated

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
