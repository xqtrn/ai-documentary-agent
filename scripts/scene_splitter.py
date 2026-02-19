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

SCENE_PROMPT_LONG = """You are a world-class cinematic director creating prompts for an AI video model. Every scene must look indistinguishable from a real Hollywood film. You MUST follow ALL 15 anti-hallucination rules below — violations produce unwatchable AI artifacts.

SCRIPT:
{script}

TASK: Break this script into EXACTLY {scene_count} scenes, each {scene_duration} seconds long.

================================================================
15 MANDATORY ANTI-HALLUCINATION RULES → NOW 18 RULES
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
Exact camera model, lens focal length (e.g., Panavision C-Series 40mm anamorphic at T2.0). Movement type: steadicam, dolly, crane, drone, handheld. Movement path described second by second: "camera starts at eye level, slowly rises to 30 feet, then pushes forward through the crowd." Movement speed: "extremely slow dolly push-in over 10 seconds." Depth of field: "shallow DOF with f/1.4, subject sharp, background rendered in soft circular bokeh." Stabilization method. Start and end framing.

**DEPT 2 — SETTING & ARCHITECTURE (400+ chars):**
Exact 18th-century French location with period-accurate details. Building materials: "rough limestone walls with crumbling mortar, exposed timber frame, slate rooftops with green verdigris copper gutters." Street details: "uneven cobblestone street approximately 4 meters wide, open sewage gutters along both sides, puddles reflecting the sky." Scale markers: "the street stretches 200 meters into the distance, flanked by 4-story buildings with iron juliet balconies." Specific structures: church steeple, market stalls, stone fountain. Props: overturned handcart, broken barrel spilling grain, scattered broadsheets, iron street lantern. Condition: soot-stained walls, moss between stones, cracked windowpanes, laundry lines between buildings.

**DEPT 3 — CROWD COMPOSITION (400+ chars):**
Exact numbers: "approximately 3000 people fill the boulevard from wall to wall." Demographics: "60% men aged 25-50, 25% women aged 20-45, 10% elderly, 5% children." What each group is doing with SPECIFIC ACTION VERBS: men SURGE forward with fists raised, women CLUTCH children protectively, boys SCRAMBLE atop walls, elderly men SHAKE walking sticks overhead. Crowd direction: "entire mass moves LEFT TO RIGHT toward the fortress gates." Crowd density: "shoulder to shoulder in the center, thinning near building walls." Individual clusters: three women passing a water jug, two men arguing while marching, a father hoisting a child onto his shoulders to see over the crowd.

**DEPT 4 — COSTUMES BY SOCIAL CLASS (400+ chars):**
Poor men: "rough undyed linen shirts, open at the collar, threadbare brown wool waistcoats, patched knee-length breeches, no stockings, wooden clogs or bare feet, cotton liberty caps (red Phrygian bonnets) on many heads, some wearing grimy leather aprons." Poor women: "faded grey cotton dresses, white linen aprons stained with soot, linen bonnets or mob caps, wooden clogs, woolen shawls wrapped around shoulders." Children: oversized hand-me-down shirts, barefoot or rope-soled shoes. Bourgeois (few): "dark broadcloth coats with brass buttons, clean white cravats, tricorn hats, leather shoes with buckles." Soldiers (if present): "blue and white uniforms with red facings, white crossbelts, tall bicorn hats, bayoneted Charleville muskets." Fabric textures: "rough homespun wool, fraying linen, sun-bleached cotton." Condition: "clothes are dirty, sweat-stained, torn at elbows and knees, mud-splattered hems."

**DEPT 5 — FACES & EXPRESSIONS (200+ chars):**
DISTANCE RULE: "all faces shown from medium or wide shot distance, waist-up minimum, NEVER extreme close-up." Specific emotions conveyed through body language: "clenched jaws, narrowed eyes, mouths open shouting, raised fists." Beauty rule: "aesthetically beautiful cinematography, natural attractive human faces, no grotesque or distorted features." Age details: "weathered skin, hollow cheeks from hunger, calloused hands, sun-darkened complexions." Hair: natural period-appropriate styles, men with shoulder-length hair tied back or under liberty caps.

**DEPT 6 — LIGHTING (300+ chars):**
Time of day with specifics: "late golden hour, sun 15 degrees above horizon." Light source: "warm directional sunlight from screen-left, supplemented by flickering orange torchlight from within the crowd." Color temperature: "natural daylight 5600K mixed with warm torchlight 2800K." Shadows: "long dramatic shadows stretching across the cobblestones from right to left, deep black in doorways and alleys." Special effects: "god rays breaking through gaps between rooftops, cutting through smoke haze, catching dust motes." Practical lights: iron lanterns on walls, wooden torches with pitch-soaked rags. Contrast ratio: "6:1 between highlights and deepest shadows."

**DEPT 7 — ATMOSPHERE & PARTICLES (300+ chars):**
Weather: "overcast sky with breaks of golden light, humidity visible in the air." Smoke: "thick grey-white smoke drifts from left to right across the mid-ground, density increasing toward the horizon, rising from multiple sources." Particles: "fine ash and embers float in the air, dust kicked up by thousands of feet creates a low ground haze, golden motes visible in light beams." Wind: "gentle wind from the west moving fabric, hair, flags, and smoke, canvas awnings flapping, torch flames bending." Temperature cues: visible perspiration, condensation on cold metal.

**DEPT 8 — MOTION (300+ chars):**
Camera motion: exact path and speed described second by second. Crowd motion: unified direction with specific individual actions described for 3-5 distinct people. Environmental motion: "smoke drifting, flames flickering 3-4 feet high, torn fabric fluttering, French tricolor flags snapping in the wind, torches casting dancing shadows, loose papers tumbling across cobblestones, wooden shutters banging in the wind, water in gutters rippling from ground vibration." Foreground motion: specific actions of 2-3 individual figures closest to camera.

**DEPT 9 — COLOR PALETTE (250+ chars):**
Overall grade: warm amber-golden highlights, cool blue-grey shadows, slightly desaturated midtones. Dominant colors: muted earth tones (off-white, grey-brown, faded indigo, dirty cream). Key accents: vivid scarlet-vermillion of liberty caps, warm orange of torch flames, deep navy of rare frock coats, silver-blue reflections in puddles. Contrast: "high contrast dramatic chiaroscuro with deep blacks and bright highlights." Saturation: 70% naturalistic with slight warmth push. Film grain: "subtle organic grain as if shot on 35mm celluloid, gentle vignetting at frame edges."

**DEPT 10 — FOREGROUND / MIDGROUND / BACKGROUND (200+ chars):**
Three distinct layers described separately. FOREGROUND (0-3m): slightly soft, a shoulder entering frame, iron pike tip crossing upper frame, cobblestones with mud and straw, torch casting lens flare. MIDGROUND (3-20m): sharpest focus, main crowd mass, primary actions, building facades, market stalls. BACKGROUND (20m+): progressively softer, rooftop silhouettes, church spire or palace dome, columns of smoke, clouds with color gradient. No empty areas anywhere.

**DEPT 11 — SOUND DESIGN CUES (150+ chars):**
Even though this is a visual prompt, describe the IMPLIED soundscape to guide the AI's sense of atmosphere: "the deafening roar of thousands of voices echoing between stone walls, the rhythmic thud of marching feet on cobblestone, distant cannon fire rumbling like thunder, the crackle of flames, church bells tolling a warning."

**DEPT 12 — CINEMATIC REFERENCES (150+ chars):**
Reference specific films or cinematographers for the visual style: "Shot in the style of Roger Deakins' work in 1917 — long unbroken takes with fluid camera movement through chaotic environments. Color grading inspired by Barry Lyndon — natural candlelight warmth with deep rich shadows. Composition echoing the battle sequences of Ridley Scott's Napoleon."

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

================================================================
SCENE STRUCTURE (French Revolution):
================================================================
Scene 1: EPIC AERIAL — Drone descending over revolutionary Paris at dusk, massive crowd flooding streets, scale of 3000+ people, burning buildings on horizon
Scene 2: THE CONTRAST — Steadicam through starving peasant hovel, match cut to opulent Versailles, Louis XVI alone on golden throne
Scene 3: THE STORMING — Medium-wide tracking shot of crowd surging toward Bastille fortress, smoke and fire, determined faces, raised weapons
Scene 4: THE AFTERMATH — Slow crane pulling back from quiet aftermath, abandoned royal symbols, empty cobblestone street, single French tricolor flag waving

================================================================
SFX PROMPT RULES:
================================================================
- Atmospheric ambient sounds ONLY
- Be specific: "3000-person crowd roaring and chanting revolutionary slogans, distant church bells tolling urgently, horse hooves clattering on cobblestone, wooden cart wheels creaking, wind howling through narrow streets"
- NEVER include graphic/violent sounds

================================================================
FINAL REMINDER:
================================================================
1. EXACTLY {scene_count} scenes
2. Each visual_prompt = MINIMUM 3000 characters, TARGET 4000, MAXIMUM 5000. Do NOT exceed 5000 characters.
3. Begin each: "Hyperrealistic cinematic 4K footage, shot on ARRI Alexa 65"
4. End each: "absolutely no text, no signs, no writing, no letters, no words, no subtitles, no watermarks, no UI elements, no logos, no modern elements"
5. Cover ALL 12 departments in order with the minimum character counts specified
6. If a prompt is under 3000 characters, you MUST expand it — add more architectural detail, more costume descriptions, more atmospheric particles, more lighting specifics, more motion descriptions

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


def _get_prompt_template(engine: str) -> str:
    """Select prompt template based on engine's prompt capacity."""
    engine_cfg = config.ENGINE_CONFIG.get(engine, {})
    max_chars = engine_cfg.get("max_prompt_chars")

    if max_chars is None:
        # Unlimited (Sora 2 Pro, Sora 2) — use maximum detail
        return SCENE_PROMPT_UNLIMITED
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
    scene_duration = config.SCENE_DURATION_SEC

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

    # For long-prompt engines (Grok Imagine): expand any prompts that are too short
    if max_prompt_chars and max_prompt_chars >= 3000:
        scenes = _expand_short_prompts(client, scenes, min_chars=3500, max_prompt_chars=max_prompt_chars)

    # For unlimited engines (Sora): expand any prompts under 2500 chars
    if max_prompt_chars is None:
        scenes = _expand_short_prompts_unlimited(client, scenes, min_chars=2500)

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
