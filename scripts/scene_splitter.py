"""Step 4: Split script into scenes with hyperrealistic cinematic visual prompts."""

import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

SCENE_SPLIT_PROMPT = """You are a world-class cinematic director creating prompts for Runway Gen-4.5, the most advanced AI video model. Your goal: every scene must look like a real Hollywood film, NOT like AI art.

SCRIPT:
{script}

TASK: Break this script into {scene_min}-{scene_max} scenes, each 8-10 seconds long.

For each scene provide:
1. **scene_number**: Sequential number
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: A HYPERREALISTIC cinematic prompt (see rules below, MAX 500 characters)
4. **camera**: Specific camera movement (tracking, dolly, crane, steadicam, drone, handheld)
5. **lighting**: Specific lighting (golden hour, candlelight, overcast with god rays, etc.)
6. **mood**: Emotional mood for music/SFX selection
7. **sfx_prompt**: Sound effect description for this scene (crowd noise, battle sounds, rain, etc.)
8. **duration_sec**: 8-10 seconds

HYPERREALISTIC VISUAL PROMPT RULES (CRITICAL):
- Style MUST be: "hyperrealistic cinematic 4K film, shot on ARRI Alexa 65, anamorphic lens"
- NEVER use words: "painting", "illustration", "artistic", "animated", "cartoon", "stylized", "digital art", "render"
- Every scene MUST have significant MOTION: crowds moving, flags waving, horses galloping, flames flickering, smoke rising, rain falling, soldiers marching, people running, wind in hair/clothes
- Think "what would be expensive to film in real life?" — massive crowds (thousands), military battles, cavalry charges, drone shots over battlefields, fire, explosions, weather effects
- Include SPECIFIC camera movement in the prompt: "slow tracking shot", "aerial drone descending", "steadicam following through crowd", "crane shot rising above"
- Include SPECIFIC environmental details: exact weather, time of day, dust particles in air, fog, smoke, reflections
- Include period-accurate details: costumes, architecture, weapons, vehicles
- NEVER include ANY text, letters, words, numbers, signs, labels, titles, subtitles
- Every prompt MUST end with: "no text, no letters, no words, no subtitles"

SCENE 1 RULE (OPENING SHOT):
- Scene 1 MUST be an epic wide aerial establishing shot — the most dynamic, cinematic shot in the entire video
- Use aerial drone or sweeping crane movement showing the FULL SCALE of the topic
- Show thousands of people, massive locations, weather effects, smoke
- Camera must be in CONSTANT motion — never static
- Example: "Hyperrealistic cinematic 4K aerial drone shot slowly descending over 18th century Paris, thousands of citizens flooding cobblestone streets, smoke rising from distant barricades, overcast sky with dramatic god rays, shot on ARRI Alexa 65, anamorphic lens, no text, no letters, no words, no subtitles"

SFX PROMPT RULES:
- Describe realistic ambient sounds for the scene
- Be specific: "large crowd murmuring and shouting in a city square, distant church bells" NOT just "crowd noise"
- Include environmental sounds: wind, rain, fire crackling, horse hooves on cobblestone
- For battle scenes: specify weapons (cannons, muskets, swords), crowd reactions, explosions

OUTPUT FORMAT: Return ONLY a JSON array, no markdown code blocks.

[
  {{
    "scene_number": 1,
    "narration": "exact narration text",
    "visual_prompt": "hyperrealistic cinematic description with camera movement, ending with no-text instruction",
    "camera": "specific camera movement",
    "lighting": "specific lighting",
    "mood": "emotional mood",
    "sfx_prompt": "detailed sound effect description",
    "duration_sec": 10
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
