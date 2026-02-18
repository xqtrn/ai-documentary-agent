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

SCENE_PROMPT_LONG = """You are an Oscar-winning film director, cinematographer, and production designer creating ULTRA-DETAILED visual prompts for Grok Imagine, xAI's text-to-video AI. Each prompt must be so detailed that a blind person could reconstruct the exact frame.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each {scene_duration} seconds long.

For each scene provide:
1. **scene_number**: Sequential number (1 through {scene_count})
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: ULTRA-DETAILED cinematic prompt — MUST be EXACTLY 4080-4090 characters. THIS IS NON-NEGOTIABLE. Count precisely. If your prompt is under 4000 characters, ADD MORE DETAIL until you reach 4080. Under 4000 chars = FAILURE.
4. **camera**: Specific camera movement
5. **lighting**: Specific lighting setup
6. **mood**: Emotional mood
7. **sfx_prompt**: Atmospheric sound effects
8. **duration_sec**: {scene_duration}

================================================================
VISUAL PROMPT LENGTH: EXACTLY 4080-4090 CHARACTERS PER SCENE
================================================================

This is the MOST IMPORTANT rule. Each visual_prompt string must contain EXACTLY 4080 to 4090 characters (not words — CHARACTERS including spaces). You MUST count. A prompt under 4000 characters is UNACCEPTABLE and means you skipped required departments below. Fill EVERY department with exhaustive detail until you hit exactly 4080-4090 chars.

Every visual_prompt MUST begin with: "Hyperrealistic cinematic 4K footage"
Every visual_prompt MUST end with: "no text, no letters, no words, no subtitles, no watermarks, no UI elements, no logos, no close-up faces, no modern elements, aesthetically beautiful cinematography"

Cover ALL 10 departments below IN ORDER. The character counts are MINIMUMS per department:

**DEPT 1 — CAMERA & LENS (250+ chars):**
Exact camera: "Shot on ARRI Alexa 65 with Panavision C-Series 40mm anamorphic lens at T2.0, 6.5K open-gate." Camera starting position (height, distance, angle). Movement path: direction, speed ("tracking left to right at 0.3 meters per second"), arc, acceleration. End position. Depth of field: what is in focus vs bokeh. Stabilization method (Steadicam, dolly, crane, drone gimbal, handheld).

**DEPT 2 — SETTING & ARCHITECTURE (500+ chars):**
Exact 18th-century French location type: narrow medieval street, grand boulevard, palace courtyard, market square. Building materials: rough-hewn limestone blocks, exposed timber framing, crumbling plaster over brick, slate rooftops with copper gutters turned green with verdigris. Street surface: uneven cobblestones with gaps filled with mud, central drainage channel. Specific structures: a four-story townhouse with iron juliet balconies, a stone church with a weathered bell tower, a wooden market stall with canvas awning. Distances: street is 6 meters wide, buildings rise 12-15 meters. Props scattered: overturned wooden handcart, broken barrel spilling grain, discarded broadsheets on wet ground, iron street lantern (unlit). Condition: walls stained with soot, moss growing between stones, cracked windowpanes, hanging laundry on lines between buildings.

**DEPT 3 — CROWD COMPOSITION (350+ chars):**
EXACT number: "approximately 2500 people fill the street stretching 200 meters." Breakdown: 55% working men aged 20-50, 20% women of all ages, 15% elderly men and women, 10% adolescents and children. Every person has a SPECIFIC ACTION VERB — men SURGE forward, women CLUTCH children to their chests, boys SCRAMBLE atop walls for a view, elderly men LEAN on walking sticks and SHAKE fists. Crowd density: packed shoulder-to-shoulder in center, thinning at edges. ALL moving in ONE unified direction — specify which direction. Small clusters: a group of three women PASSING a water jug, two men ARGUING while WALKING, a father HOISTING a child onto his shoulders.

**DEPT 4 — COSTUMES BY CLASS (350+ chars):**
Working class men: rough undyed linen shirts with rolled sleeves, patched brown woolen breeches held up by leather belts, wooden sabots caked with mud, some wearing grimy leather aprons over shirts, red cotton liberty caps (bonnet rouge) on many heads. Working women: faded linen chemises under heavy grey wool skirts, cotton fichus crossed over chests, hair pinned under plain cotton bonnets, wooden sabots. Children: oversized hand-me-down shirts, barefoot or in rope-soled shoes. Middle class (few): worn but clean wool frock coats in dark blue or brown, cotton stockings, buckled leather shoes, tricorn hats. Fabric conditions: threadbare, sweat-stained, mud-splattered hems, patched elbows, frayed cuffs. Colors: muted earth tones — off-white, grey-brown, faded indigo, dirty cream, with occasional vivid red of liberty caps.

**DEPT 5 — FACES & BODY LANGUAGE (250+ chars):**
ALL people shown at MEDIUM or WIDE shot distance — waist-up minimum, NO close-ups. Emotional energy conveyed through posture: clenched fists, raised arms, hunched shoulders, wide stances. Men's faces show grim determination, jaw muscles tight, brows furrowed. Women's expressions convey fierce protectiveness, eyes scanning surroundings. General descriptors only: weathered skin, gaunt cheekbones from hunger, sun-darkened complexions. Natural attractive human faces with historically appropriate features, no grotesque or distorted features. Hair: men with natural shoulder-length hair tied with ribbon or tucked under liberty caps — NO modern haircuts.

**DEPT 6 — LIGHTING (350+ chars):**
Primary source: [specify per scene — dawn sun at 15 degrees above horizon / overcast diffused daylight / golden hour / torchlight]. Direction from camera-left at 45 degrees. Color temperature: [specify 2700K-5600K]. Hard or soft shadows — hard from direct sun, soft from overcast. Fill light from [reflected light off limestone walls / bonfire glow / ambient sky]. Rim light catching edges of smoke and dust particles. Specific effects: god rays piercing through gaps between buildings, torchlight casting warm orange flicker on faces (2700K), wet cobblestones reflecting sky in puddles as silver-blue mirror surfaces. Practical lights: iron lanterns, wooden torches with pitch-soaked rags, candles in windows. Contrast ratio: 6:1 between highlights and shadows.

**DEPT 7 — ATMOSPHERE & PARTICLES (300+ chars):**
Weather: overcast with occasional breaks in clouds. Smoke: grey-white woodsmoke rising from multiple sources, density medium — obscuring background buildings at 50+ meters, drifting left to right with prevailing wind. Fine dust particles visible in any light beam, golden motes floating. Scattered ash from distant fires, tiny white flecks descending slowly. Ground-level mist in shadows and alleys. Wind from the west at 8 km/h — causing flags to ripple, loose hair to stream, torch flames to bend, canvas awnings to flap. Humidity high: visible perspiration on foreheads, condensation on iron surfaces.

**DEPT 8 — MOTION IN EVERY PART OF FRAME (350+ chars):**
Camera: [specific movement per scene]. Crowd: entire mass SURGES in unified direction at walking pace. Individual motions: a man WAVES a tricolor flag overhead in wide arcs, a woman PUSHES through the crowd CARRYING a basket, three soldiers MARCH in lockstep, a horse STAMPS and TOSSES its head, a dog WEAVES between legs. Environmental: smoke BILLOWS and CURLS upward, flames from torches LICK and GUTTER in wind, loose papers TUMBLE across cobblestones, a wooden shutter BANGS repeatedly against a wall, laundry on lines SNAPS in gusts, water in gutters RIPPLES from vibration of thousands of feet. Foreground motion differs from background.

**DEPT 9 — COLOR PALETTE (250+ chars):**
Overall grade: warm amber-golden in highlights, cool blue-grey in shadows, slightly desaturated mids. Dominant shadow color: deep blue-grey with hints of warm brown near firelight. Highlight color: pale golden cream. Key accent colors: vivid red of liberty caps (scarlet-vermillion), deep blue of rare frock coats (navy-indigo), warm orange of torch flames, dirty white of linen shirts. Contrast: high contrast dramatic chiaroscuro with deep blacks and bright highlights. Saturation: 70% — naturalistic but with slight warmth push. Film grain: subtle organic grain as if shot on 35mm celluloid.

**DEPT 10 — FOREGROUND / MIDGROUND / BACKGROUND LAYERS (350+ chars):**
FOREGROUND (0-3m from camera): slightly soft focus — a shoulder in rough linen entering frame left, the iron tip of a pike cutting across upper right, cobblestones with mud and scattered straw at bottom, a torch held high casting lens flare. MIDGROUND (3-20m): sharpest focus — the main crowd mass, primary character actions, building facades at street level, market stalls, a fountain or monument. BACKGROUND (20m+): progressively softer — rooftop silhouettes against the sky, church spire or palace dome, columns of smoke rising from multiple points, clouds or clear sky with appropriate color gradient. NO empty areas anywhere in the composition — every zone must be filled with detail.

================================================================
GROK IMAGINE ANTI-HALLUCINATION RULES (include in every prompt):
================================================================
- NO close-up faces — Grok distorts them. Waist-up minimum distance
- NO text, signs with writing, banners with words, shop names — Grok makes gibberish
- NO modern elements: no glasses, wristwatches, modern clothing, synthetic fabrics, zippers
- NO symmetrical compositions — use asymmetric, organic, off-center framing
- NO static poses or frozen tableaux — every person must have a specific ACTION verb
- NO empty spaces — fill foreground, midground, background completely
- NO anachronistic hairstyles — no modern cuts, no buzz cuts, no gel
- CROWD DIRECTION: all people MUST move in ONE unified direction
- Specify EXACT crowd count: "approximately 2500 people" not just "a crowd"
- Always include: "aesthetically beautiful cinematography, natural attractive human faces"

================================================================
SCENE STRUCTURE (French Revolution):
================================================================
Scene 1: EPIC AERIAL — Drone/crane descending over revolutionary Paris, massive crowd, smoke, scale
Scene 2: THE CONTRAST — Tracking through desperate poverty vs aristocratic excess
Scene 3: THE STORMING — Crowd charging toward the Bastille, peak revolutionary energy
Scene 4: THE AFTERMATH — Slow pullback, dust settling, emotional weight of transformation

================================================================
SFX PROMPT RULES:
================================================================
- Atmospheric ambient sounds ONLY
- Be specific: "2000-person crowd murmuring and chanting, distant church bells tolling slowly, horse hooves clattering on cobblestone, wooden cart wheels creaking"
- NEVER include graphic/violent sounds — use: wind, rain, fire crackling, crowd murmur, bells, footsteps, drums, fabric rustling, wood creaking, thunder

================================================================
FINAL REMINDER — READ THIS CAREFULLY:
================================================================
1. EXACTLY {scene_count} scenes
2. Each visual_prompt = EXACTLY 4080-4090 characters. Count carefully. This is the MOST CRITICAL requirement.
3. Begin each: "Hyperrealistic cinematic 4K footage"
4. End each: "no text, no letters, no words, no subtitles, no watermarks, no UI elements, no logos, no close-up faces, no modern elements, aesthetically beautiful cinematography"
5. Cover ALL 10 departments in order with the minimum character counts specified
6. If a prompt is under 4000 characters, you MUST expand it — add more architectural detail, more crowd actions, more atmospheric particles, more color description

OUTPUT: Return ONLY a valid JSON array. No markdown code blocks, no commentary.

[
  {{
    "scene_number": 1,
    "narration": "narration text",
    "visual_prompt": "EXACTLY 4080-4090 characters of ultra-detailed cinematic description covering all 10 departments",
    "camera": "camera movement",
    "lighting": "lighting setup",
    "mood": "emotional mood",
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


EXPAND_PROMPT = """The visual_prompt for scene {scene_number} is only {current_len} characters. It MUST be EXACTLY 4080-4090 characters.

Current prompt:
{current_prompt}

EXPAND this prompt to EXACTLY 4080-4090 characters by adding MORE detail to these departments:
- More SETTING detail: additional buildings, street features, architectural ornaments, distances
- More CROWD detail: additional individual actions, subgroups, specific numbers per area
- More COSTUME detail: additional fabric descriptions, wear patterns, accessories, specific garments
- More ATMOSPHERE: additional particle types, wind effects, temperature visual cues, humidity signs
- More COLOR: additional color descriptions for specific objects, light reflections, shadow hues
- More FOREGROUND/BACKGROUND: additional layered depth elements, objects at different distances

RULES:
- Keep the same opening "Hyperrealistic cinematic 4K footage" and closing "no text, no letters..." etc
- Keep all existing content — only ADD more detail between existing sentences
- The final prompt MUST be 4080-4090 characters. Count precisely.
- Return ONLY the expanded visual_prompt text, nothing else. No JSON, no quotes, no explanation."""


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


def _expand_short_prompts(client, scenes: list, min_chars: int = 3500, max_prompt_chars: int = 4096) -> list:
    """Re-prompt Claude to expand any visual_prompt that is under min_chars."""
    target_max = max_prompt_chars - 6  # 4090 for Grok
    for scene in scenes:
        vp = scene.get("visual_prompt", "")
        sn = scene.get("scene_number", 0)
        if len(vp) < min_chars:
            logger.warning(
                "Scene %d prompt is only %d chars (min %d). Expanding...",
                sn, len(vp), min_chars,
            )
            expand_msg = EXPAND_PROMPT.format(
                scene_number=sn,
                current_len=len(vp),
                current_prompt=vp,
            )
            try:
                resp = client.messages.create(
                    model=config.CLAUDE_MODEL,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": expand_msg}],
                )
                expanded = resp.content[0].text.strip()
                # Remove any wrapping quotes
                if expanded.startswith('"') and expanded.endswith('"'):
                    expanded = expanded[1:-1]
                if len(expanded) > len(vp):
                    # Truncate if over limit
                    if len(expanded) > target_max:
                        truncated = expanded[:target_max]
                        last_period = truncated.rfind(".")
                        if last_period > target_max - 200:
                            truncated = truncated[:last_period + 1]
                        expanded = truncated
                    scene["visual_prompt"] = expanded
                    logger.info(
                        "Scene %d expanded: %d -> %d chars",
                        sn, len(vp), len(expanded),
                    )
                else:
                    logger.warning("Scene %d expansion did not increase length.", sn)
            except Exception as exc:
                logger.warning("Scene %d expansion failed: %s", sn, exc)
    return scenes


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

    # Log prompt lengths
    for scene in scenes:
        vp = scene.get("visual_prompt", "")
        logger.info("Scene %d visual_prompt: %d chars", scene.get("scene_number", 0), len(vp))

    # For long-prompt engines: expand any prompts that are too short
    if max_prompt_chars and max_prompt_chars >= 3000:
        scenes = _expand_short_prompts(client, scenes, min_chars=3500, max_prompt_chars=max_prompt_chars)

    # Enforce max prompt length per engine (safety truncation)
    if max_prompt_chars:
        safety_limit = max_prompt_chars - 6  # tight limit: 4090 for Grok (4096 API max)
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
