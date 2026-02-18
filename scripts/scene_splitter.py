"""Step 4: Split script into scenes with hyperrealistic cinematic visual prompts."""

import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

SCENE_SPLIT_PROMPT = """You are a world-class cinematic director and DP (Director of Photography) creating prompts for Grok Imagine, xAI's text-to-video AI model. Your goal: every scene must look indistinguishable from a real Hollywood film shot on ARRI Alexa 65.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each 8 seconds long. You MUST output exactly {scene_count} scenes — no more, no fewer. Distribute ALL narration evenly across the scenes.

For each scene provide:
1. **scene_number**: Sequential number (1 through {scene_count})
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: An EXTREMELY DETAILED cinematic video prompt (see rules below — NO character limit, write as much as needed)
4. **camera**: Specific camera movement description
5. **lighting**: Specific lighting setup
6. **mood**: Emotional mood for music/SFX
7. **sfx_prompt**: Sound effect description (atmospheric sounds ONLY — see SFX rules below)
8. **duration_sec**: 8

═══════════════════════════════════════════════════════════════
VISUAL PROMPT RULES — WRITE LIKE A PROFESSIONAL FILM DIRECTOR
═══════════════════════════════════════════════════════════════

There is NO character limit on visual prompts. Write each prompt like a professional film screenplay / director's shot list. More detail = more control = better result. Each prompt should read like instructions given to EVERY department on a film set.

For EACH visual prompt, include ALL of the following:

**DIRECTOR'S VISION:**
- Overall mood and atmosphere of the scene
- Emotional tone: tension, fear, triumph, despair, hope, awe
- Pacing: frantic chaos or slow building tension
- What the viewer should FEEL in this moment

**CINEMATOGRAPHER (Camera & Lens):**
- Camera and lens: "ARRI Alexa 65, 40mm anamorphic lens, T2.0"
- Camera movement: steadicam tracking, dolly push-in, crane descending, handheld shaky, drone aerial
- Camera height: eye level, low angle looking up, bird's eye overhead
- Depth of field: shallow DOF with blurred background, or deep focus
- Start and end position of camera movement

**LIGHTING / GAFFER:**
- Time of day: dawn, midday, sunset, night, twilight
- Light source: natural sun, candles, torches, fireplace, overcast sky
- Light direction: backlit, side-lit, front-lit, from below
- Color temperature: warm golden, cool bluish, neutral
- Shadows: hard contrasty or soft diffused
- Special lighting: god rays through clouds, flame flicker, reflections on wet cobblestones

**COSTUME DESIGN:**
- Precise clothing description for EVERY visible character
- Materials and textures: "rough undyed linen", "worn velvet", "soot-stained leather apron"
- Clothing condition: new/worn, clean/dirty, torn/intact
- Historical details: tricorn hats, wigs, stockings, clogs, revolutionary cockades
- Accessories: belts, buckles, hats, scarves, weapons

**MAKEUP & HAIR:**
- Faces: clean/dirty, sweaty, sooty, scarred
- Hair: neat/disheveled, wigs, loose
- Age and condition: wrinkles, hollow cheeks, bags under eyes from exhaustion
- Skin texture and detail

**ACTORS / PERFORMANCE:**
- Facial expressions: "clenched jaws, narrowed eyes, nostrils flared with rage"
- Body language: "hunched shoulders, arms clutched to chest" or "chest out, chin raised"
- Specific actions: what each visible person is doing in frame
- Gaze direction: looking at camera, to the side, upward, at another character
- Crowd behavior: unified movement, scattered panic, silent stillness

**PRODUCTION DESIGN (Location & Set):**
- Detailed location description: architecture, wall materials, floor, ceiling
- Props: furniture, items on tables, food, weapons, tools
- Space condition: clean/cluttered, intact/destroyed
- Scale: cramped room or massive square with thousands of people
- Period-accurate 18th century French details

**ATMOSPHERE & VFX:**
- Weather: rain, fog, snow, wind, heat haze
- Particles in air: dust, smoke, ash, sparks, raindrops
- Fire, explosions, destruction if present
- Crowd size: "dozens", "hundreds", "thousands filling the square wall to wall"

**NEGATIVE CONSTRAINTS (MANDATORY at end of EVERY prompt):**
"no text, no letters, no words, no subtitles, no watermarks, no UI elements, no logos, no title cards, no credits, no captions"

**STYLE CONSTRAINTS (MANDATORY in EVERY prompt):**
- ALWAYS: "Hyperrealistic cinematic 4K footage"
- NEVER use: "painting", "illustration", "animated", "cartoon", "artistic", "stylized"
- Every scene MUST have significant motion
- Scene 1 MUST be an epic aerial establishing shot

═══════════════════════════════════════════════════════════════
SFX PROMPT RULES
═══════════════════════════════════════════════════════════════
- Describe atmospheric ambient sounds ONLY
- Be specific: "large crowd murmuring, distant church bells, horse hooves on cobblestone"
- NEVER include graphic/violent descriptions — content moderation will block them
- Focus on: wind, rain, fire crackling, crowd murmur, bells, footsteps, drums, nature sounds, fabric rustling, wood creaking

═══════════════════════════════════════════════════════════════

CRITICAL: Output EXACTLY {scene_count} scene objects. No more, no fewer.

OUTPUT FORMAT: Return ONLY a valid JSON array. No markdown code blocks, no explanation text.

[
  {{
    "scene_number": 1,
    "narration": "exact narration text for this scene",
    "visual_prompt": "EXTREMELY DETAILED cinematic description — multiple paragraphs covering every visual department",
    "camera": "specific camera movement",
    "lighting": "specific lighting setup",
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

    # Enforce max scene count — truncate if Claude returned too many
    if len(scenes) > config.SCENES_COUNT_MAX:
        logger.warning(
            "Scene splitter returned %d scenes, truncating to %d",
            len(scenes), config.SCENES_COUNT_MAX,
        )
        scenes = scenes[:config.SCENES_COUNT_MAX]
        for i, scene in enumerate(scenes):
            scene["scene_number"] = i + 1

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
