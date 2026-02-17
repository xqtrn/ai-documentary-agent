"""Step 3: Rewrite and improve the script using Claude."""

import json
import logging
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are an elite documentary scriptwriter known for creating viral, binge-worthy YouTube documentaries.

ORIGINAL VIDEO ANALYSIS:
{analysis}

ORIGINAL TRANSCRIPT (for reference only - DO NOT copy):
{transcript}

TASK: Write a COMPLETELY NEW documentary script on the same topic. This must be original text - not a rewrite or paraphrase of the original.

REQUIREMENTS:
1. **Length**: Target {target_min}-{target_max} minutes when read aloud (~150 words/minute = {word_min}-{word_max} words total)
2. **Style**: Cinematic documentary narration. Dramatic, authoritative, engaging.
3. **First 30 seconds**: MAXIMUM hook power. The opening must be so compelling that 70%+ of viewers stay. Use a shocking fact, a provocative question, or drop the viewer into the most intense moment.
4. **Re-hooks every 2-3 minutes**: Add attention-grabbing moments throughout. Use:
   - Open loops ("But what happened next would change everything...")
   - Cliffhangers before transitions
   - Surprising revelations
   - Rhetorical questions
   - Pattern interrupts
5. **Narrative structure**: Build tension progressively. Use mystery/reveal structure.
6. **Emotional arc**: Take the viewer on an emotional journey. Mix tension, wonder, fear, hope.
7. **Finale**: Strong payoff that resolves all open loops. Leave the viewer with a powerful closing thought.
8. **NO FLUFF**: Every sentence must earn its place. Cut anything that doesn't serve the story.

OUTPUT FORMAT:
Write the complete script as continuous narration text. Use paragraph breaks for natural pauses.
Mark re-hook points with [HOOK] at the start of the paragraph.
Mark the climax with [CLIMAX].
Mark the finale with [FINALE].

Write the complete script now. Remember: ORIGINAL content only, cinematic style, {word_min}-{word_max} words."""


def rewrite_script(source_data: dict, virality_data: dict, output_dir: Path) -> dict:
    """Rewrite the script with improvements."""
    logger.info("Rewriting script...")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    target_min = config.TARGET_DURATION_MIN
    target_max = config.TARGET_DURATION_MAX
    word_min = target_min * 150
    word_max = target_max * 150

    transcript = source_data["transcript_text"]
    if len(transcript) > 60000:
        transcript = transcript[:60000] + "\n\n[TRUNCATED]"

    prompt = REWRITE_PROMPT.format(
        analysis=virality_data["analysis"],
        transcript=transcript,
        target_min=target_min,
        target_max=target_max,
        word_min=word_min,
        word_max=word_max,
    )

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )

    script_text = response.content[0].text
    word_count = len(script_text.split())
    est_duration = word_count / 150

    result = {
        "script": script_text,
        "word_count": word_count,
        "estimated_duration_min": round(est_duration, 1),
        "model": config.CLAUDE_MODEL,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    # Save checkpoint
    with open(output_dir / "step3_script.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(output_dir / "script.txt", "w") as f:
        f.write(script_text)

    logger.info("Script rewrite complete: %d words, ~%.1f min", word_count, est_duration)
    return result
