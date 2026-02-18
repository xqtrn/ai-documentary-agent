"""Step 4: Split script into scenes with adaptive cinematic visual prompts.

Prompt length adapts to the selected video engine's capabilities.
"""

import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

# Prompt templates by prompt length category
SCENE_PROMPT_SHORT = """You are a world-class cinematic director creating prompts for an AI video model. Every scene must look like a real Hollywood film.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each {scene_duration} seconds long.

For each scene provide:
1. **scene_number**: Sequential number
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: A hyperrealistic cinematic prompt (TARGET: 400-800 characters, MAX {max_prompt_chars})
4. **camera**: Specific camera movement (tracking, dolly, crane, steadicam, drone, handheld)
5. **lighting**: Specific lighting (golden hour, candlelight, overcast with god rays, etc.)
6. **mood**: Emotional mood for music/SFX selection
7. **sfx_prompt**: Sound effect description for this scene
8. **duration_sec**: {scene_duration}

VISUAL PROMPT RULES:
- Style: "hyperrealistic cinematic 4K film, shot on ARRI Alexa 65, anamorphic lens"
- NEVER use: "painting", "illustration", "animated", "cartoon", "stylized", "digital art"
- Every scene MUST have MOTION: crowds moving, flags waving, flames flickering, smoke rising
- Include camera movement in the prompt
- Include period-accurate details
- NEVER include text, letters, words, numbers, signs
- End every prompt with: "no text, no letters, no words, no subtitles"
- Scene 1 MUST be an epic aerial establishing shot

SFX RULES:
- Specific atmospheric sounds: "large crowd murmuring, distant church bells"
- Include environmental sounds: wind, rain, fire crackling

OUTPUT: Return ONLY a valid JSON array, no markdown.

[
  {{
    "scene_number": 1,
    "narration": "text",
    "visual_prompt": "cinematic description 400-800 chars ending with no-text instruction",
    "camera": "camera movement",
    "lighting": "lighting",
    "mood": "mood",
    "sfx_prompt": "sound effects",
    "duration_sec": {scene_duration}
  }}
]"""

SCENE_PROMPT_LONG = """You are a world-class cinematic director and DP creating prompts for an AI video model with a large prompt window. Every scene must look indistinguishable from a real Hollywood film shot on ARRI Alexa 65.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each {scene_duration} seconds long.

For each scene provide:
1. **scene_number**: Sequential number
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: A richly detailed cinematic video prompt (TARGET: 2500-3800 characters, MAX {max_prompt_chars})
4. **camera**: Specific camera movement
5. **lighting**: Specific lighting setup
6. **mood**: Emotional mood
7. **sfx_prompt**: Atmospheric sound effects description
8. **duration_sec**: {scene_duration}

VISUAL PROMPT RULES — TARGET 2500-3800 CHARACTERS EACH:

Write each prompt like a professional director's shot list covering EVERY visual department.

Include ALL of the following:

**SHOT TYPE & CAMERA:**
Camera and lens: "Shot on ARRI Alexa 65, 40mm anamorphic lens, T2.0". Camera movement: steadicam tracking, dolly push-in, crane descending, handheld, drone aerial.

**SETTING & ARCHITECTURE:**
Detailed period-accurate location. Materials, surfaces, scale. Period buildings, streets, interiors.

**PEOPLE & COSTUMES:**
Precise clothing. Materials: "rough undyed linen", "worn velvet". Historical details. Body language, facial expressions.

**LIGHTING & ATMOSPHERE:**
Time of day, light source, color temperature, shadow quality. God rays, flame flicker. Weather: fog, rain, heat haze. Particles: dust, smoke, ash, sparks.

**MOTION (CRITICAL):**
Crowds surging, flags waving, smoke drifting, flames flickering, fabric fluttering, dust swirling, rain falling.

**CONSTRAINTS:**
- Begin with: "Hyperrealistic cinematic 4K footage"
- NEVER: "painting", "illustration", "animated", "cartoon"
- Scene 1 MUST be epic aerial establishing shot
- End with: "no text, no letters, no words, no subtitles, no watermarks, no UI elements, no logos"

SFX RULES:
- Atmospheric ambient sounds ONLY
- Specific: "large crowd murmuring, distant church bells, horse hooves on cobblestone"

OUTPUT: Return ONLY a valid JSON array, no markdown.

[
  {{
    "scene_number": 1,
    "narration": "text",
    "visual_prompt": "detailed cinematic description 2500-3800 chars",
    "camera": "camera movement",
    "lighting": "lighting",
    "mood": "mood",
    "sfx_prompt": "atmospheric sounds",
    "duration_sec": {scene_duration}
  }}
]"""

SCENE_PROMPT_MEDIUM = """You are a world-class cinematic director creating prompts for an AI video model. Every scene must look like a real Hollywood film.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each {scene_duration} seconds long.

For each scene provide:
1. **scene_number**: Sequential number
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: A detailed hyperrealistic cinematic prompt (TARGET: 500-1500 characters)
4. **camera**: Specific camera movement
5. **lighting**: Specific lighting
6. **mood**: Emotional mood
7. **sfx_prompt**: Sound effect description
8. **duration_sec**: {scene_duration}

VISUAL PROMPT RULES:
- Style: "hyperrealistic cinematic 4K film, shot on ARRI Alexa 65, anamorphic lens"
- NEVER: "painting", "illustration", "animated", "cartoon", "stylized"
- Every scene MUST have visible MOTION: crowds, flags, flames, smoke, wind
- Include camera movement, lighting, period-accurate details
- Include specific setting details, costumes, atmosphere
- NEVER include text/letters/words
- End every prompt with: "no text, no letters, no words, no subtitles"
- Scene 1 MUST be an epic aerial establishing shot

SFX RULES:
- Specific atmospheric sounds: "large crowd murmuring, distant church bells"

OUTPUT: Return ONLY a valid JSON array, no markdown.

[
  {{
    "scene_number": 1,
    "narration": "text",
    "visual_prompt": "cinematic description",
    "camera": "camera movement",
    "lighting": "lighting",
    "mood": "mood",
    "sfx_prompt": "sounds",
    "duration_sec": {scene_duration}
  }}
]"""


def _get_prompt_template(engine: str) -> str:
    """Select prompt template based on engine's prompt capacity."""
    engine_cfg = config.ENGINE_CONFIG.get(engine, {})
    max_chars = engine_cfg.get("max_prompt_chars")

    if max_chars is None:
        # Unlimited (Sora) — use medium length
        return SCENE_PROMPT_MEDIUM
    elif max_chars >= 3000:
        # Long prompts (Grok Imagine)
        return SCENE_PROMPT_LONG
    else:
        # Short prompts (Runway models)
        return SCENE_PROMPT_SHORT


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


def split_into_scenes(script_data: dict, output_dir, engine: str = None) -> dict:
    """Split script into scenes with visual prompts adapted to the video engine.

    Args:
        script_data: Dict with "script" key.
        output_dir: Output directory.
        engine: Video engine key. Defaults to config.DEFAULT_ENGINE.

    Returns:
        Dict with scenes list and metadata.
    """
    engine = engine or config.DEFAULT_ENGINE
    engine_cfg = config.ENGINE_CONFIG.get(engine, {})
    max_prompt_chars = engine_cfg.get("max_prompt_chars", 1000)
    scene_duration = engine_cfg.get("max_duration_sec", 10)

    output_dir = Path(output_dir)
    logger.info("Splitting script into scenes for engine=%s (max_prompt=%s)...", engine, max_prompt_chars)

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    target_count = config.SCENES_COUNT_MAX
    template = _get_prompt_template(engine)

    prompt = template.format(
        script=script_data["script"],
        scene_count=target_count,
        scene_duration=scene_duration,
        max_prompt_chars=max_prompt_chars or "unlimited",
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

    # Enforce max prompt length per engine
    if max_prompt_chars:
        safety_limit = max_prompt_chars - 50
        for scene in scenes:
            vp = scene.get("visual_prompt", "")
            if len(vp) > safety_limit:
                logger.warning(
                    "Scene %d visual_prompt is %d chars, truncating to %d",
                    scene.get("scene_number", 0), len(vp), safety_limit,
                )
                truncated = vp[:safety_limit]
                last_period = truncated.rfind(".")
                if last_period > safety_limit - 200:
                    truncated = truncated[:last_period + 1]
                scene["visual_prompt"] = truncated

    total_duration = sum(s.get("duration_sec", 10) for s in scenes)

    result = {
        "scenes": scenes,
        "scene_count": len(scenes),
        "total_duration_sec": total_duration,
        "model": config.CLAUDE_MODEL,
        "engine": engine,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    with open(output_dir / "step4_scenes.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Scene split complete: %d scenes, ~%ds total (engine=%s)", len(scenes), total_duration, engine)
    return result
