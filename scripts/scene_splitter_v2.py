"""Step 4 V2: Split script into scenes with year-anchored anti-hallucination prompts.

Enhanced version of scene_splitter.py with build_grok_prompt() that anchors
every visual element (flags, clothing, architecture, objects) to the exact
historical year. Designed for Grok Imagine V2 (4096 char max).

Original scene_splitter.py is preserved unchanged.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

# ===================================================================
# All original templates from scene_splitter.py (unchanged)
# ===================================================================

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

SCENE_PROMPT_LONG = """You are a world-class cinematic director creating prompts for an AI video model. Every scene must look indistinguishable from a real Hollywood film. You MUST follow ALL 15 anti-hallucination rules below — violations produce unwatchable AI artifacts.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each {scene_duration} seconds long.

================================================================
15 MANDATORY ANTI-HALLUCINATION RULES — NOW 18 RULES
================================================================

RULE 1 — STATE YEAR AND LOCATION: Begin every visual_prompt with "France, 1789."
RULE 2 — NO FAMOUS BUILDINGS OR CITIES: NEVER write "Bastille", "Notre Dame", "Versailles", "Paris", "Tuileries". Use ONLY generic descriptions: "narrow European cobblestone street lined with 4-story limestone buildings with slate rooftops"
RULE 3 — NO MODERN EQUIPMENT: Explicitly ban: "NO film cameras, cranes, tripods, microphones, electric lights, modern vehicles, glasses, wristwatches, modern hairstyles, rubber shoes, zippers, synthetic fabrics"
RULE 4 — EVERY PERSON UNIQUELY DRESSED: Describe 5+ individuals with "varied colors (brown, grey, tan, off-white, faded blue), different states of wear, different hat types, NO TWO PEOPLE identical in clothing or pose"
RULE 5 — NO CLOSE-UP FACES: "waist-up minimum, NEVER extreme close-up of any face"
RULE 6 — CORRECT FLAG COLORS: "THREE VERTICAL STRIPES: dark navy blue on left, pure white center, bright crimson red on right. NO yellow, NO orange, NO green, NO horizontal stripes"
RULE 7 — CORRECT MILITARY UNIFORMS: "white wool coats with blue lapels and red collar trim, white breeches, black boots, tall black bicorn hats, Charleville muskets with bayonets. NO green uniforms, NO red coats, NO camouflage, NO modern helmets"
RULE 8 — NO SYMMETRY: "Asymmetric organic composition — crowd denser on left, buildings vary in height, people at different distances from camera"
RULE 9 — EVERY PERSON UNIQUE ACTION: Describe at least 5 individuals doing different things: "woman carrying ceramic jug stumbling, old man leaning on walking stick shouting, young boy running between legs, man pushing handcart with sacks, woman holding infant wrapped in shawl"
RULE 10 — NO TEXT/SIGNS/SYMBOLS: "absolutely NO letters, NO words, NO numbers, NO shop names, NO banners with writing, NO icons, NO symbols floating in air, NO digital artifacts, NO geometric shapes, NO watermarks, NO X marks"
RULE 11 — FILL EVERY FRAME: "FOREGROUND (0-5m): 3-4 specific individuals with unique clothing and actions. MIDGROUND (5-30m): dense crowd of 200+ people filling street wall-to-wall. BACKGROUND (30m+): building facades with open shuttered windows, rooftops with chimney smoke, sky with clouds"
RULE 12 — SPECIFY EXACT MOTION: "entire crowd moves RIGHT TO LEFT in unified surge. Camera tracks alongside at walking pace. Smoke drifts left to right from west wind. Flags on pikes wave eastward. Torch flames bend in wind direction"
RULE 13 — NATURAL LIGHTING ONLY: "overcast grey daylight OR warm golden-hour amber sunlight. Orange flickering torchlight from hand-held torches. Candlelight glow from windows. NO electric lights, NO studio lighting, NO fluorescent, NO neon"
RULE 14 — GROUND-LEVEL REALISM: "uneven wet cobblestones with puddles reflecting light, scattered straw and hay, mud patches, fallen leaves, knocked-over wooden barrel, horse manure, worn stone curbs, iron drain grates"
RULE 15 — ATMOSPHERIC DEPTH: "light grey haze increases with distance, buildings 100m+ away desaturated and softer, dust particles visible in light shafts from gaps between buildings, smoke from fires adds layered depth to background"
RULE 16 — CONSTANT CAMERA MOTION: Camera NEVER static. Every shot uses dolly, crane, steadicam, or drone movement. Specify exact direction and speed: "steadicam pushing forward at walking pace", "crane descending 2 meters over 8 seconds", "drone tracking left-to-right". Static tripod shots are FORBIDDEN.
RULE 17 — ZERO FROZEN PEOPLE: Every single person in frame MUST be in motion. Foreground people: running, stumbling, shouting with open mouths, swinging arms, pushing. Midground crowd: surging as a mass, arms raised, bobbing heads. Background: silhouettes moving across rooftops, leaning from windows. If a person is standing still, they are WRONG — give them an action.
RULE 18 — VISIBLE EMOTIONS IN FOREGROUND: The 3-4 foreground characters (0-5m from camera) must show CLEAR facial emotions visible to viewer: "determined gaze, furrowed brow with concentration, lips pressed tightly in resolve, eyes wide absorbing the chaos, weathered face showing years of hardship, quiet dignity mixed with rising anger". Emotions should be REAL and HUMAN — not exaggerated theatrical screaming. Think Kubrick's Barry Lyndon or Ridley Scott's Napoleon — restrained intensity, not comic book.

================================================================

For each scene provide:
1. **scene_number**: Sequential number (1 through {scene_count})
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: A richly detailed cinematic video prompt (TARGET: 3500-3900 characters, MAX {max_prompt_chars}). Must follow ALL 18 rules above.
4. **camera**: Specific camera movement
5. **lighting**: Specific lighting setup
6. **mood**: Emotional mood
7. **sfx_prompt**: Atmospheric sound effects for Runway audio sync
8. **duration_sec**: {scene_duration}

VISUAL PROMPT STRUCTURE (in this exact order):
1. "Hyperrealistic cinematic 4K footage, shot on ARRI Alexa 65" + lens + camera movement
2. "France, 1789." + generic location description (RULE 2)
3. Crowd composition with 5+ unique individuals (RULES 4, 9)
4. Costume details with varied colors and materials (RULE 4, 7)
5. Three-layer composition: foreground, midground, background (RULE 11)
6. Lighting and atmosphere (RULES 13, 15)
7. Ground details (RULE 14)
8. Motion directions — EVERY person moving, camera moving, smoke/flags/fire moving (RULES 12, 16, 17)
9. Foreground character emotions — restrained historical: determined gaze, furrowed brow, quiet resolve, weathered faces (RULE 18)
10. End with: "absolutely no text, no signs, no writing, no letters, no words, no subtitles, no watermarks, no UI elements, no logos, no modern elements, no film equipment"

SFX PROMPT RULES (for Runway audio generation):
- Specific atmospheric sounds synced to visuals
- Example: "massive crowd roaring and chanting in unison, cobblestones under thousands of feet creating rhythmic thunder, distant sporadic musket shots echoing between stone buildings, crackling of large bonfires, iron church bells clanging frantically in the distance, horses neighing, wooden carts creaking"
- NEVER: music, narration, speech, dialogue

SCENE GUIDELINES:
- Scene 1: Wide aerial/crane establishing shot descending over generic European street, reveal crowd scale
- Scene 2: Ground-level dolly/steadicam through contrast — peasant hovel → palace hallway (generic interiors, NO named buildings)
- Scene 3: Medium-wide tracking shot with crowd surging toward generic stone fortress (NOT "Bastille"), smoke, torches, correct tricolor flags
- Scene 4: Slow crane pulling back from quiet aftermath — abandoned items, single tricolor flag, empty cobblestone street

OUTPUT: Return ONLY a valid JSON array, no markdown.

[
  {{
    "scene_number": 1,
    "narration": "narration text",
    "visual_prompt": "3500-3900 chars following all 18 rules — constant motion, restrained historical emotions, no famous buildings",
    "camera": "camera movement",
    "lighting": "lighting setup",
    "mood": "emotional mood",
    "sfx_prompt": "detailed atmospheric sounds for Runway",
    "duration_sec": {scene_duration}
  }}
]"""

# ===================================================================
# NEW: Enhanced V2 template — Claude generates 4080-4090 char prompts
# with year-anchored rules embedded directly in the system prompt.
# ===================================================================

SCENE_PROMPT_ENHANCED = """You are a cinematic director writing ULTRA-DETAILED visual prompts for Grok Imagine V2 video generation.

YEAR: {year}. LOCATION: {location}.

ABSOLUTE REQUIREMENT: Each visual_prompt MUST be EXACTLY 4080-4090 characters. Count precisely after writing. If under 4000 characters — ADD MORE DETAIL until you reach 4080+. This is NON-NEGOTIABLE. Short prompts produce generic video. Every extra character of description = better visual output.

You are generating 4 scenes for a {duration}-second documentary about: {topic}

SCRIPT CONTEXT:
{script_summary}

HISTORICAL ACCURACY RULES FOR {year} {location}:
- YEAR ANCHOR: Every visual element must be labeled with the year. Example: "{year}-era limestone buildings" not just "buildings"
- FLAGS: {flag_rule}
- CLOTHING: {clothing_rule}
- OBJECTS: {objects_rule}
- ARCHITECTURE: {architecture_rule}

GROK IMAGINE V2 — KNOWN ISSUES TO PREVENT:
- NO close-up faces — Grok distorts facial features at close range. All people at MEDIUM or WIDE shot (waist-up minimum, full body preferred)
- NO text, signs, banners with writing, shop names, newspapers — Grok generates illegible gibberish text
- NO modern elements — Grok inserts modern clothing, glasses, wristwatches, sneakers if not explicitly forbidden
- NO symmetrical compositions — Grok defaults to artificial symmetry. Always specify: asymmetric, organic, offset framing
- NO static poses — Grok defaults to frozen tableaux. Every single person must have a specific ACTION VERB (running, pushing, pointing, carrying, stumbling)
- NO empty spaces — fill every layer: foreground crowd + midground action + background architecture/sky/smoke
- NO anachronistic hairstyles — specify: men have natural hair tied with ribbon, cotton liberty caps, or tricorn hats. NOT modern undercuts or fades
- CROWD DIRECTION: All people must move in ONE unified direction. Say: "entire crowd surges LEFT TO RIGHT toward the fortress" — Grok defaults to random wandering
- EXACT NUMBERS: "approximately 3000 people fill the cobblestone street" not "a large crowd"
- BEAUTY RULE: always append "aesthetically beautiful cinematography, natural attractive human faces, no grotesque or distorted features, photorealistic skin textures"

FOR EACH SCENE — DESCRIBE ALL OF THESE DEPARTMENTS (this is why prompts must be 4080+ chars):

**CAMERA & LENS**: Exact movement (steadicam/crane/drone/dolly/handheld), speed (slow/medium/fast), direction (left-to-right/descending/pushing in), focal length (24mm/35mm/50mm), aperture (f/1.8/f/2.8/f/5.6), depth of field (shallow/deep), any special effects (lens flare, dust on lens)

**SETTING & ARCHITECTURE ({year})**: Building materials (limestone/timber/brick), street width in meters, building height in stories, specific architectural details (mansard roofs/Gothic arches/wrought iron), what year these structures were built, what is NOT present (no Eiffel Tower, no Haussmann boulevards, no gas lamps if pre-1820), ground surface (wet cobblestones/mud/packed earth), debris and environmental detail

**CROWD COMPOSITION**: Exact percentage breakdown (e.g., "60% working-class men, 25% women, 10% children, 5% bourgeoisie"), what each group is carrying (pitchforks/torches/bread/muskets/flags), their movement direction and speed

**COSTUMES ({year} — historically accurate)**: Fabric type for each social class (rough linen/wool/silk), specific garments (culottes/chemise/sabots/tricorn), colors (muted ochres/dark browns/dirty whites), condition (worn/torn/dirty), footwear, headwear, accessories. FORBIDDEN: zippers, elastic, synthetic fabrics, rubber soles, modern cuts

**FACES & EXPRESSIONS**: Shot distance rule (no closer than waist-up), emotion descriptors (determined/desperate/furious/fearful), age distribution, natural beauty standard, no grotesque or distorted features

**LIGHTING**: Primary source (late afternoon sun/torchlight/overcast sky), direction (raking from left/backlit/overhead), color temperature in Kelvin (2700K torchlight/5500K daylight/7000K overcast), shadow hardness (hard-edged/soft/diffuse), any secondary sources (reflected light from water/windows)

**ATMOSPHERE & WEATHER**: Season, temperature suggestion, wind direction and strength, particles in air (smoke density/dust/fog/rain), visibility distance in meters, smell suggestion (smoke/gunpowder/sweat/mud) for mood reference

**MOTION LAYERS**: Camera movement speed and direction, crowd movement direction and speed, individual hero actions (woman in foreground pushes cart LEFT, man in midground raises torch RIGHT, child in background runs AWAY), environmental motion (flags rippling, smoke drifting, birds startling)

**COLOR PALETTE**: Primary colors (desaturated ochre/dark charcoal/blood red), secondary colors (dirty white/rust/deep shadow), specific hex-level description if helpful, overall grade (warm golden hour/cold grey overcast/high-contrast torchlight)

**DEPTH LAYERS**:
- FOREGROUND (0-3m): Specific individuals, their exact actions, costumes, expressions
- MIDGROUND (3-15m): Main crowd action, key visual element, architecture fragment
- BACKGROUND (15m+): Full architectural context, sky, smoke, scale reference

Generate exactly 4 scenes following this structure:

SCENE 1 (0:00-0:08) — EPIC AERIAL ESTABLISHING SHOT
Drone descending from 200m altitude down to 15m above the crowd. We see revolutionary {location} from above — the density of {year}-era streets, smoke rising from multiple points. As drone descends, we begin to see individual faces and torches. Camera movement: slow vertical descent combined with slight northward drift.

SCENE 2 (0:08-0:16) — THE HUMAN COST
Ground level steadicam moving through the crowd. We see the faces of the revolution — exhausted mothers, young men with hollow cheeks, elderly citizens who have waited decades. Contrast of desperate poverty against ornate {year} architecture. The camera weaves between individuals, each with their own story.

SCENE 3 (0:16-0:24) — THE STORMING ACTION
The crowd surges toward its target. Maximum kinetic energy. Tracking shot moving LEFT TO RIGHT with the crowd. Torches, screaming faces (at distance), weapons raised. This is the moment of historical rupture.

SCENE 4 (0:24-0:32) — THE AFTERMATH / RESOLUTION
Slow crane pulling back and upward. The immediate aftermath — some jubilation, some shock, some grief. The camera rises to show the scale of what has happened — thousands of people, changed forever. Emotional, quiet, monumental.

OUTPUT FORMAT — return valid JSON:
{{
  "scenes": [
    {{
      "scene_number": 1,
      "time_start": "0:00",
      "time_end": "0:08",
      "visual_prompt": "EXACTLY 4080-4090 CHARS HERE — count after writing",
      "narration": "What the narrator says during this scene (2-3 sentences)",
      "mood": "epic/tense/dramatic/somber",
      "camera": "specific camera movement description"
    }}
  ]
}}

CRITICAL: After writing each visual_prompt — count the characters. If under 4080 — add more detail to COSTUMES, ATMOSPHERE, or CROWD COMPOSITION sections until you reach 4080-4090. Do not truncate. Do not summarize. More detail = better video.
"""


SCENE_PROMPT_UNLIMITED = """You are an Oscar-winning film director, cinematographer, and production designer creating MAXIMUM-DETAIL visual prompts for Sora 2 Pro, OpenAI's premier text-to-video AI. Sora 2 Pro has NO character limit — use this to write the most detailed, comprehensive prompts possible. Each prompt should read like a complete professional film shooting script.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each {scene_duration} seconds long.

For each scene provide:
1. **scene_number**: Sequential number (1 through {scene_count})
2. **narration**: The exact narration text for this scene
3. **visual_prompt**: Detailed cinematic prompt — MUST be 3000-5000 characters. Target exactly 4000 characters. Under 3000 = too short, over 5000 = too long. Be precise and dense, not repetitive.
4. **camera**: Specific camera movement
5. **lighting**: Specific lighting setup
6. **mood**: Emotional mood
7. **sfx_prompt**: Atmospheric sound effects
8. **duration_sec**: {scene_duration}

================================================================
VISUAL PROMPT LENGTH: MINIMUM 3000, TARGET 4000, MAXIMUM 5000
================================================================

Each visual_prompt MUST be 3000-5000 characters. Target exactly 4000 characters. Write like a film director giving dense, precise instructions. Be specific but concise — do NOT pad with repetition. Under 3000 = too sparse. Over 5000 = too verbose and wastes tokens. Aim for 4000 characters of rich, non-repetitive detail.

Every visual_prompt MUST begin with: "Hyperrealistic cinematic 4K footage, shot on ARRI Alexa 65"
Every visual_prompt MUST end with: "absolutely no text, no signs, no writing, no letters, no words, no subtitles, no watermarks, no UI elements, no logos, no modern elements"

Cover ALL 12 departments below IN ORDER. The character counts are MINIMUMS per department:

**DEPT 1 — CAMERA & LENS (200+ chars):**
Exact camera model, lens focal length (e.g., Panavision C-Series 40mm anamorphic at T2.0). Movement type: steadicam, dolly, crane, drone, handheld. Movement path described second by second. Movement speed. Depth of field. Stabilization method. Start and end framing.

**DEPT 2 — SETTING & ARCHITECTURE (400+ chars):**
Exact period location with period-accurate details. Building materials, street details, scale markers, specific structures, props, condition.

**DEPT 3 — CROWD COMPOSITION (400+ chars):**
Exact numbers, demographics, specific action verbs, crowd direction, density, individual clusters.

**DEPT 4 — COSTUMES BY SOCIAL CLASS (400+ chars):**
Poor men, poor women, children, bourgeois, soldiers. Fabric textures, condition.

**DEPT 5 — FACES & EXPRESSIONS (200+ chars):**
Distance rule, specific emotions through body language, age details, hair.

**DEPT 6 — LIGHTING (300+ chars):**
Time of day, light source, color temperature, shadows, special effects, practical lights, contrast ratio.

**DEPT 7 — ATMOSPHERE & PARTICLES (300+ chars):**
Weather, smoke, particles, wind, temperature cues.

**DEPT 8 — MOTION (300+ chars):**
Camera motion, crowd motion, environmental motion, foreground motion.

**DEPT 9 — COLOR PALETTE (250+ chars):**
Overall grade, dominant colors, key accents, contrast, saturation, film grain.

**DEPT 10 — FOREGROUND / MIDGROUND / BACKGROUND (200+ chars):**
Three distinct layers described separately.

**DEPT 11 — SOUND DESIGN CUES (150+ chars):**
Implied soundscape to guide atmosphere.

**DEPT 12 — CINEMATIC REFERENCES (150+ chars):**
Reference specific films or cinematographers for visual style.

================================================================
ANTI-HALLUCINATION RULES (include in every prompt):
================================================================
- NO close-up faces — keep all people at medium or wide shot distance, waist-up minimum
- NO text, NO signs, NO banners with writing, NO shop names, NO numbers
- NO modern elements: no modern clothing, no glasses, no wristwatches, no modern hairstyles
- NO symmetrical compositions — use asymmetric organic framing
- NO static frozen poses — every person has a specific action verb
- NO empty spaces — fill every area of the frame
- CROWD DIRECTION: all people move in ONE unified direction
- Always include: "aesthetically beautiful cinematography, natural attractive human faces"

OUTPUT: Return ONLY a valid JSON array. No markdown code blocks, no commentary.

[
  {{
    "scene_number": 1,
    "narration": "narration text",
    "visual_prompt": "3000-5000 characters of dense cinematic description covering all 12 departments",
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


EXPAND_PROMPT_UNLIMITED = """The visual_prompt for scene {scene_number} is only {current_len} characters. For Sora 2 Pro, each prompt should be at least 3000 characters (target 4000-5000).

Current prompt:
{current_prompt}

EXPAND this prompt to at least 3500 characters by adding MORE detail:
- More CAMERA detail: second-by-second movement path, lens specifics, focus pull descriptions
- More SETTING detail: additional buildings, architectural ornaments, specific distances, material textures
- More CROWD detail: additional individual character actions, specific numbers per area, more subgroups
- More COSTUME detail: additional fabric descriptions, specific garments per social class, wear patterns
- More LIGHTING detail: multiple light sources, color temperatures, shadow directions, reflections
- More ATMOSPHERE: additional particle types, smoke density variations, wind effects, humidity signs
- More MOTION: additional environmental movement, individual character actions, foreground/background
- More COLOR: color grading details, accent colors, shadow hues, highlight tones
- CINEMATIC REFERENCES: reference specific films or cinematographers for the visual style

RULES:
- Keep the same opening "Hyperrealistic cinematic 4K footage" and closing "no text, no signs..." etc
- Keep all existing content — only ADD more detail between existing sentences
- Target 3500-5000 characters. More detail is always better. There is NO upper limit.
- Return ONLY the expanded visual_prompt text, nothing else. No JSON, no quotes, no explanation."""


# ===================================================================
# Original helper functions (unchanged from scene_splitter.py)
# ===================================================================

def _get_prompt_template(engine: str) -> str:
    """Select prompt template based on engine's prompt capacity."""
    engine_cfg = config.ENGINE_CONFIG.get(engine, {})
    max_chars = engine_cfg.get("max_prompt_chars")

    if max_chars is None:
        return SCENE_PROMPT_UNLIMITED
    elif max_chars >= 3000:
        return SCENE_PROMPT_LONG
    else:
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
    """Try multiple strategies to parse JSON from LLM output.

    Handles both plain arrays [...] and {"scenes": [...]} wrapper format.
    """
    text = raw_text.strip()

    def _unwrap(parsed):
        """If parsed is a dict with 'scenes' key, extract the array."""
        if isinstance(parsed, dict) and "scenes" in parsed:
            return parsed["scenes"]
        return parsed

    try:
        return _unwrap(json.loads(text))
    except json.JSONDecodeError:
        pass

    fixed = _fix_json(text)
    try:
        return _unwrap(json.loads(fixed))
    except json.JSONDecodeError:
        pass

    # Try to find {"scenes": [...]} wrapper first
    scenes_match = re.search(r'\{\s*"scenes"\s*:\s*\[.*?\]\s*\}', fixed, re.DOTALL)
    if scenes_match:
        try:
            parsed = json.loads(scenes_match.group())
            return _unwrap(parsed)
        except json.JSONDecodeError:
            pass

    match = re.search(r'\[.*\]', fixed, re.DOTALL)
    if match:
        try:
            return _unwrap(json.loads(match.group()))
        except json.JSONDecodeError:
            arr_text = re.sub(r',\s*([}\]])', r'\1', match.group())
            try:
                return _unwrap(json.loads(arr_text))
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
    target_max = max_prompt_chars - 6
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
                if expanded.startswith('"') and expanded.endswith('"'):
                    expanded = expanded[1:-1]
                if len(expanded) > len(vp):
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


def _expand_short_prompts_unlimited(client, scenes: list, min_chars: int = 2500) -> list:
    """Re-prompt Claude to expand short prompts for unlimited engines (Sora)."""
    for scene in scenes:
        vp = scene.get("visual_prompt", "")
        sn = scene.get("scene_number", 0)
        if len(vp) < min_chars:
            logger.warning(
                "Scene %d prompt is only %d chars (min %d for unlimited engine). Expanding...",
                sn, len(vp), min_chars,
            )
            expand_msg = EXPAND_PROMPT_UNLIMITED.format(
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
                if expanded.startswith('"') and expanded.endswith('"'):
                    expanded = expanded[1:-1]
                if len(expanded) > len(vp):
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


# ===================================================================
# NEW: Year-anchored anti-hallucination prompt functions
# ===================================================================

def _extract_year_from_title(title: str) -> int:
    """Extract 4-digit year from video title. Returns 1789 as default."""
    match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', title)
    return int(match.group(1)) if match else 1789


def _extract_location_from_title(title: str) -> str:
    """Extract location from video title using simple keyword matching."""
    title_lower = title.lower()
    location_map = {
        "french revolution": "Paris, France",
        "france": "France",
        "paris": "Paris, France",
        "russia": "Russia",
        "moscow": "Moscow, Russia",
        "england": "England",
        "london": "London, England",
        "america": "United States",
        "rome": "Rome, Italy",
        "italy": "Italy",
        "germany": "Germany",
        "china": "China",
        "japan": "Japan",
        "egypt": "Egypt",
    }
    for keyword, location in location_map.items():
        if keyword in title_lower:
            return location
    return "Historical location"


def _get_flag_rule(year: int, location: str) -> str:
    """Return accurate flag description for the given year and location."""
    loc = location.lower()

    if "france" in loc or "paris" in loc:
        if year < 1790:
            return (f"FRANCE {year}: The French royal flag is plain WHITE (white Bourbon flag). "
                    f"The tricolor (blue/white/red vertical stripes) was NOT YET ADOPTED — "
                    f"it was introduced September 1790. In {year}, ONLY white royal flags or "
                    f"no flags at all. Zero tricolor flags permitted.")
        elif 1790 <= year <= 1794:
            return (f"FRANCE {year}: The French tricolor IS in use — three VERTICAL stripes: "
                    f"BLUE (left), WHITE (center), RED (right). Stripes are vertical, not horizontal. "
                    f"Carried on wooden poles or pikes. No eagles, no Napoleon symbols (pre-Empire). "
                    f"Proportions roughly 2:3.")
        elif year > 1804:
            return (f"FRANCE {year}: French tricolor — vertical blue/white/red. "
                    f"Napoleon-era flags may include eagle standards for military. "
                    f"Correct proportions for {year}.")
        else:
            return (f"FRANCE {year}: French tricolor — three VERTICAL stripes blue/white/red. "
                    f"Only flags historically present in {year} France.")

    elif "russia" in loc or "moscow" in loc or "saint pete" in loc or "petersburg" in loc:
        if year < 1700:
            return f"RUSSIA {year}: No standardized national flag yet. Military banners only — Orthodox crosses on dark backgrounds."
        elif 1700 <= year < 1858:
            return f"RUSSIA {year}: Russian Imperial flag — black double-headed eagle on yellow/gold background, or naval white-blue-red tricolor for ships."
        else:
            return f"RUSSIA {year}: Russian Imperial tricolor — white/blue/red HORIZONTAL stripes. Imperial eagle emblem if official context."

    elif "england" in loc or "britain" in loc or "london" in loc:
        if year < 1606:
            return f"ENGLAND {year}: St George's Cross — red cross on white background. Union Jack not yet created."
        elif 1606 <= year < 1801:
            return f"BRITAIN {year}: Union Jack of 1606 — combined St George's Cross and St Andrew's Cross. No Irish diagonal stripe yet (added 1801)."
        else:
            return f"BRITAIN {year}: Full Union Jack — St George (England), St Andrew (Scotland), St Patrick (Ireland) combined."

    elif "usa" in loc or "america" in loc or "washington" in loc or "united states" in loc:
        stars = min(50, max(13, year - 1776 + 13)) if year > 1776 else 0
        if year < 1776:
            return f"AMERICA {year}: No US flag exists. British colonial flags or no flags."
        elif year == 1776:
            return f"USA 1776: The Betsy Ross flag — 13 stars in a circle on blue canton, 13 alternating red/white stripes."
        else:
            return f"USA {year}: American flag with approximately {stars} stars (exact count for {year}). Red/white stripes, blue canton with stars."

    else:
        return (f"LOCATION {location}, {year}: ONLY show flags and national symbols that "
                f"historically existed in {year} in this exact location. "
                f"Research the correct flag for this region in {year} before rendering. "
                f"If uncertain — show NO flags rather than incorrect ones.")


def _get_period_clothing(year: int, location: str) -> str:
    """Return period-accurate clothing description."""
    loc = location.lower()

    if "france" in loc or "paris" in loc:
        if 1780 <= year <= 1800:
            return (f"MEN {year} France: Knee breeches (culottes) or long trousers for working class, "
                    f"linen shirts, wool coats or jackets, tricorn or round hats, leather shoes with buckles. "
                    f"Working class: rough linen smocks, clogs (sabots), red Phrygian caps. "
                    f"WOMEN {year} France: Long skirts to ankles, linen blouses, wool shawls, "
                    f"mob caps (white linen bonnets), aprons. No corsets visible externally. "
                    f"ALL HANDMADE — no factory-produced garments.")
        elif year < 1780:
            return (f"MEN {year} France: Powdered wigs for nobility, tricorn hats, "
                    f"knee breeches, silk or wool coats with large cuffs and buttons. "
                    f"WOMEN {year} France: Wide panniers under skirts, powdered hair for nobility, "
                    f"simpler linen for common women. Long sleeves always.")

    if year < 1500:
        return (f"Medieval clothing {year}: Wool tunics, linen undergarments, leather belts, "
                f"hood or coif head coverings. Hand-stitched. Earthtones: brown, grey, dark blue, ochre. "
                f"No bright synthetic colors. Shoes: soft leather turnshoes or wooden clogs.")
    elif 1500 <= year < 1700:
        return (f"Renaissance/Early Modern {year}: Doublets and hose for men, "
                f"ruff collars for wealthy, simple linen for poor. "
                f"Women: bodices, full skirts, linen coifs. All hand-woven, hand-stitched fabrics.")
    elif 1700 <= year < 1800:
        return (f"18th century {year}: Men — frock coats, waistcoats, breeches, stockings, buckled shoes. "
                f"Women — stays/corsets under petticoats, long skirts, linen caps. "
                f"Wealthy: silk and fine wool. Poor: rough linen and homespun wool.")
    elif 1800 <= year < 1900:
        return (f"19th century {year}: Men — frock coats or sack coats, top hats or bowlers, "
                f"waistcoats, cravats. Women — full skirts with crinolines/bustles depending on decade, "
                f"bonnets, gloves. No synthetic dyes before 1856.")
    else:
        return f"Historically accurate clothing for {year}. No anachronistic materials or styles."


def _get_period_architecture(year: int, location: str) -> str:
    """Return period-accurate architecture description."""
    loc = location.lower()

    if ("france" in loc or "paris" in loc) and 1700 <= year <= 1800:
        return (f"Paris {year}: Narrow cobblestone streets 4-6 meters wide. "
                f"3-5 story limestone buildings, mansard roofs, wooden shutters. "
                f"Ground floors: shops with wooden signs (no neon, no printed signs). "
                f"Wrought iron balconies. Gas lighting NOT YET EXIST — oil lanterns on iron brackets. "
                f"NO Eiffel Tower (built 1889). NO Haussmann boulevards (1853-1870). "
                f"Medieval street pattern — winding, irregular, organic.")

    elif year < 1600:
        return (f"Medieval/Renaissance architecture {year}: Stone or timber-frame buildings, "
                f"thatched or tile roofs, small windows with wooden shutters or oiled parchment. "
                f"No glass windows in common buildings. Churches with Gothic arches if European.")
    elif 1600 <= year < 1800:
        return (f"Early modern architecture {year}: Stone or brick buildings, "
                f"multi-pane sash windows, chimneys, slate or tile roofs. "
                f"Baroque or classical detailing for wealthy buildings. "
                f"Narrow irregular streets, no urban planning.")
    elif 1800 <= year < 1900:
        return (f"19th century architecture {year}: Neoclassical or Victorian depending on location. "
                f"Larger windows, iron railings, gas street lamps (post-1820s). "
                f"Industrial buildings have brick and cast iron. No electric lights (pre-1880s).")
    else:
        return f"Period-accurate architecture for {year}. No buildings constructed after {year}."


def _get_period_objects(year: int, location: str) -> str:
    """Return period-accurate objects and tools."""
    if year < 1700:
        return (f"Objects {year}: Hand-forged iron tools, wooden barrels and crates, "
                f"clay pots, wicker baskets, leather bags, tallow candles, oil lamps. "
                f"Horse-drawn carts with wooden wheels and iron rims. "
                f"Weapons if military: muskets/pikes/swords appropriate for {year}.")
    elif 1700 <= year < 1800:
        return (f"Objects {year}: Flintlock muskets (if military), wooden carts, "
                f"clay jugs and earthenware, wicker baskets, hemp rope, "
                f"oil lanterns, tallow candles, printing press (large wooden manual). "
                f"NO steam engines visible (Watt's engine 1769 but rare). "
                f"Coins: no paper money in common use. Hand pumps for water.")
    elif 1800 <= year < 1850:
        return (f"Objects {year}: Early steam engines possible in industrial areas, "
                f"iron tools, gas lamps on streets (post-1820), "
                f"early newspapers (hand-set type). Horse-drawn carriages. "
                f"NO photography (Daguerre 1839). NO telegraph (1837). "
                f"NO railways except in Britain post-1825.")
    else:
        return (f"Period-accurate objects for {year}. "
                f"No electricity (unless post-1880s context). "
                f"No internal combustion engines (pre-1885). "
                f"No photography unless post-1839.")


def _get_lighting(mood: str) -> str:
    """Return lighting description based on mood."""
    mood_lower = mood.lower()
    if any(w in mood_lower for w in ["tense", "dramatic", "dangerous", "violent"]):
        return "late afternoon golden hour casting long dramatic shadows, or torchlight at night — deep contrasts"
    elif any(w in mood_lower for w in ["triumphant", "celebrat", "joyful"]):
        return "bright midday sun, clear sky, warm golden tones"
    elif any(w in mood_lower for w in ["somber", "sad", "mourning", "defeat"]):
        return "overcast grey sky diffusing light evenly, flat cold light, desaturated tones"
    else:
        return "natural daylight — direction consistent with time of day, soft shadows"


def _get_ground_detail(location: str, year: int) -> str:
    """Return ground-level detail for the location and year."""
    loc = location.lower()
    if "paris" in loc or "france" in loc:
        return "Wet cobblestones (Paris limestone setts), puddles in ruts, mud at edges, scattered straw and horse manure, fallen cabbage leaves from market carts"
    elif "london" in loc or "england" in loc or "britain" in loc:
        return "Cobblestone or packed earth street, horse dung, scattered straw, puddles, coal dust near buildings"
    elif "russia" in loc:
        return "Packed earth or timber-plank streets, mud in spring/autumn, snow in winter, birch bark scraps"
    else:
        return f"Historically accurate ground surface for {location} in {year}: period-appropriate paving, dirt, or stone"


def flags_description_from_narration(narration: str, year: int, location: str) -> str:
    """Detect if narration mentions flags and add specific rules."""
    narration_lower = narration.lower()
    if any(w in narration_lower for w in ["flag", "banner", "standard", "tricolor", "colours", "colors"]):
        return f"FLAGS DETECTED IN SCENE — apply year {year} flag rules above with ABSOLUTE precision."
    return f"If any flags appear spontaneously — they must match {year} {location} historical accuracy exactly."


def build_grok_prompt(scene: dict, year: int, location: str) -> str:
    """Build a 4096-char anti-hallucination prompt for Grok Imagine V2.

    Every visual element is anchored to the exact year to prevent anachronisms.
    Flags, clothing, architecture, and objects are described with year-specific accuracy.

    Args:
        scene: dict with keys: narration, action, mood, scene_number
        year: exact year of the historical event (e.g. 1789)
        location: city/country (e.g. "Paris, France")

    Returns:
        str: prompt up to 4096 chars (truncated if needed)
    """
    year_anchor = f"{location}, {year}"

    flag_rule = _get_flag_rule(year, location)
    clothing = _get_period_clothing(year, location)
    architecture = _get_period_architecture(year, location)
    objects = _get_period_objects(year, location)

    narration = scene.get("narration", "")
    action = scene.get("action", scene.get("visual_description", narration[:200]))
    mood = scene.get("mood", "tense and dramatic")

    # Camera motion — always dynamic, never static
    camera_options = [
        f"Steadicam tracking shot through the scene, moving LEFT TO RIGHT at ground level, {year}",
        f"Slow crane shot descending from rooftop to street level, revealing crowd below, {year}",
        f"Handheld documentary-style follow shot — urgent, shaky, immersive, {year}",
        f"Dolly shot pushing INTO the scene, depth increasing, figures growing larger, {year}",
        f"Drone aerial descending to crowd level in {location}, {year}",
    ]
    cam_idx = int(hashlib.md5(narration.encode()).hexdigest(), 16) % len(camera_options)
    camera = camera_options[cam_idx]

    flags_desc = flags_description_from_narration(narration, year, location)

    prompt = f"""{year_anchor}. {location}. {mood.capitalize()} atmosphere.

ACTION: {action}

PEOPLE IN SCENE (all dressed as {year} {location} residents):
At least 7 individuals visible — each uniquely dressed, unique action, unique expression.
No two people identical. Ages mixed: elderly, middle-aged, young adults, one child.

HISTORICALLY ACCURATE CLOTHING FOR {year} {location}:
{clothing}
ZERO modern materials: no synthetic fabrics, no elastic, no zippers, no rubber soles.

ARCHITECTURE ({year} {location} — period accurate):
{architecture}

FLAGS AND NATIONAL SYMBOLS — CRITICAL RULE FOR {year}:
{flag_rule}
{flags_desc}

PERIOD-ACCURATE OBJECTS ({year} — no anachronisms):
{objects}
FORBIDDEN: cameras, glasses with modern frames, printed newspapers with modern fonts,
factory-made uniform clothing, synthetic ropes, rubber, plastic of any kind.

CAMERA: {camera}

LIGHTING: Natural light only — {_get_lighting(mood)}.
No electric lighting. No artificial illumination except torches/candles/oil lamps if nighttime.

GROUND DETAIL: {_get_ground_detail(location, year)}

ATMOSPHERE: {mood}. Dust particles, smoke, or mist in air.
Depth: sharp foreground figures, softer midground, hazy background architecture.

COMPOSITION: Asymmetric organic crowd composition. Rule of thirds.
Strong diagonal lines from lower-left to upper-right.
Every single person IS IN MOTION — no one standing completely still.

ABSOLUTE PROHIBITIONS:
— Nothing manufactured after {year}
— No text, signs, labels, inscriptions visible anywhere in frame
— No theatrical screaming — restrained historical emotion only
— No static camera — camera MUST be moving throughout shot
— No symmetrical/artificial composition"""

    # Truncate to 4096 chars max (Grok API limit)
    if len(prompt) > 4096:
        prompt = prompt[:4093] + "..."

    return prompt


# ===================================================================
# Main function: split_into_scenes (enhanced V2)
# ===================================================================

def split_into_scenes(script_data: dict, output_dir, engine: str = None,
                      source_data: dict = None) -> dict:
    """Split script into scenes with visual prompts adapted to the video engine.

    V2 Enhancement: When source_data is provided, uses SCENE_PROMPT_ENHANCED
    template so Claude generates 4080-4090 char year-anchored prompts directly.

    Args:
        script_data: Dict with "script" key.
        output_dir: Output directory.
        engine: Video engine key. Defaults to config.DEFAULT_ENGINE.
        source_data: Optional source data dict with video_title for year/location extraction.

    Returns:
        Dict with scenes list and metadata.
    """
    engine = engine or config.DEFAULT_ENGINE
    engine_cfg = config.ENGINE_CONFIG.get(engine, {})
    max_prompt_chars = engine_cfg.get("max_prompt_chars", 1000)
    scene_duration = config.SCENE_DURATION_SEC

    output_dir = Path(output_dir)
    logger.info("V2: Splitting script into scenes for engine=%s (max_prompt=%s)...", engine, max_prompt_chars)

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    target_count = config.SCENES_COUNT_MAX

    # ===============================================================
    # V2 ENHANCED PATH: Use SCENE_PROMPT_ENHANCED with year-anchored rules
    # Claude generates 4080-4090 char prompts directly — no post-processing
    # ===============================================================
    if source_data:
        title = source_data.get("video_title", source_data.get("title", ""))
        year = _extract_year_from_title(title)
        location = _extract_location_from_title(title)
        logger.info("V2 ENHANCED: year=%d, location=%s, title=%s", year, location, title)

        # Fill year-anchored rules from helper functions
        flag_rule = _get_flag_rule(year, location)
        clothing_rule = _get_period_clothing(year, location)
        objects_rule = _get_period_objects(year, location)
        architecture_rule = _get_period_architecture(year, location)

        # Build topic and script summary
        topic = title or "historical event"
        script_text = script_data.get("script", "")
        script_summary = script_text[:3000]  # First 3000 chars for context
        duration = target_count * scene_duration

        prompt = SCENE_PROMPT_ENHANCED.format(
            year=year,
            location=location,
            duration=duration,
            topic=topic,
            script_summary=script_summary,
            flag_rule=flag_rule,
            clothing_rule=clothing_rule,
            objects_rule=objects_rule,
            architecture_rule=architecture_rule,
        )

        logger.info("V2 ENHANCED: Prompt template filled — %d chars, sending to Claude...", len(prompt))

        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=32768,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text
        logger.info("V2 ENHANCED: Claude response — %d chars", len(raw_text))

        scenes = _parse_json_robust(raw_text)

        if not isinstance(scenes, list) or len(scenes) == 0:
            raise RuntimeError(f"Scene splitter returned invalid data: {type(scenes)}")

        # Enforce max scene count
        if len(scenes) > config.SCENES_COUNT_MAX:
            logger.warning("Truncating %d scenes to %d", len(scenes), config.SCENES_COUNT_MAX)
            scenes = scenes[:config.SCENES_COUNT_MAX]
            for i, scene in enumerate(scenes):
                scene["scene_number"] = i + 1

        # Ensure duration_sec is set
        for scene in scenes:
            if "duration_sec" not in scene:
                scene["duration_sec"] = scene_duration

        # Log prompt lengths — these should be 4080-4090 chars from Claude
        all_long = True
        for scene in scenes:
            vp = scene.get("visual_prompt", "")
            sn = scene.get("scene_number", 0)
            logger.info("V2 ENHANCED: Scene %d visual_prompt = %d chars", sn, len(vp))
            if len(vp) < 3500:
                all_long = False

        # If Claude didn't generate long enough prompts, expand them
        if not all_long:
            logger.warning("V2 ENHANCED: Some prompts under 3500 chars — expanding...")
            scenes = _expand_short_prompts(client, scenes, min_chars=3500, max_prompt_chars=max_prompt_chars or 4096)

    # ===============================================================
    # STANDARD PATH: Use original templates
    # ===============================================================
    else:
        logger.info("V2: No source_data — using standard templates.")
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
        logger.info("V2: Scene splitter raw response: %d chars", len(raw_text))

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
            logger.info("V2: Scene %d visual_prompt (Claude): %d chars", scene.get("scene_number", 0), len(vp))

        # For long-prompt engines (Grok Imagine): expand any prompts that are too short
        if max_prompt_chars and max_prompt_chars >= 3000:
            scenes = _expand_short_prompts(client, scenes, min_chars=3500, max_prompt_chars=max_prompt_chars)

        # For unlimited engines (Sora): expand any prompts under 2500 chars
        if max_prompt_chars is None:
            scenes = _expand_short_prompts_unlimited(client, scenes, min_chars=2500)

    # Enforce max prompt length per engine (safety truncation)
    if max_prompt_chars:
        safety_limit = max_prompt_chars - 6
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
        "enhanced_prompts": source_data is not None,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    with open(output_dir / "step4_scenes.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("V2: Scene split complete: %d scenes, ~%ds total (engine=%s, enhanced=%s)",
                len(scenes), total_duration, engine, source_data is not None)
    return result
