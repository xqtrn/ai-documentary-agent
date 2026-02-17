"""Step 2: Analyze why the video went viral using Claude."""

import json
import logging
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are an expert YouTube content strategist and viral video analyst.

Analyze this YouTube video transcript and metadata to understand WHY it became popular.

VIDEO METADATA:
- Title: {title}
- Channel: {channel}
- Language: {language}

TRANSCRIPT:
{transcript}

Provide a detailed analysis in the following structure:

## 1. Hook Analysis (First 30 seconds)
- What hooks are used in the opening?
- How does it grab attention immediately?
- Rate the hook effectiveness (1-10)

## 2. Narrative Structure
- What storytelling framework is used? (Hero's journey, mystery reveal, chronological, etc.)
- How does it maintain viewer attention throughout?
- Where are the key tension/interest peaks?

## 3. Retention Techniques
- What open loops are used?
- Where are the "pattern interrupts"?
- How often does it re-hook the viewer?

## 4. Emotional Triggers
- What emotions does it target?
- What are the most powerful emotional moments?

## 5. Pacing & Rhythm
- How does the pacing change throughout?
- Are there slow/fast sections? Where and why?

## 6. Weaknesses & Improvement Opportunities
- What could be improved?
- Where does attention likely drop?
- What hooks or techniques are missing?

## 7. Key Takeaways for Rewrite
- Top 5 elements to keep/amplify
- Top 5 elements to fix/add
- Recommended narrative approach for the rewrite

Be specific with quotes from the transcript."""


def analyze_virality(source_data: dict, output_dir: Path) -> dict:
    """Analyze why the video went viral."""
    logger.info("Analyzing virality for: %s", source_data["title"])

    # Check API key early with clear error
    api_key = config.check_api_key("ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)

    # Truncate transcript if too long
    transcript = source_data["transcript_text"]
    if len(transcript) > 80000:
        transcript = transcript[:80000] + "\n\n[TRANSCRIPT TRUNCATED]"

    prompt = ANALYSIS_PROMPT.format(
        title=source_data["title"],
        channel=source_data.get("channel", "Unknown"),
        language=source_data.get("language", "unknown"),
        transcript=transcript,
    )

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is invalid or has no credits! "
            "Please check your key at https://console.anthropic.com/settings/keys "
            "and add billing at https://console.anthropic.com/settings/billing"
        )
    except anthropic.RateLimitError:
        raise RuntimeError(
            "Anthropic API rate limit reached. Please wait a moment and try again, "
            "or check your billing at https://console.anthropic.com/settings/billing"
        )

    analysis_text = response.content[0].text

    result = {
        "analysis": analysis_text,
        "model": config.CLAUDE_MODEL,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    # Save checkpoint
    with open(output_dir / "step2_virality.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(output_dir / "analysis.md", "w") as f:
        f.write(f"# Virality Analysis: {source_data['title']}\n\n")
        f.write(f"**Original URL:** {source_data['url']}\n\n")
        f.write(analysis_text)

    logger.info(
        "Virality analysis complete (%d input + %d output = %d tokens)",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.input_tokens + response.usage.output_tokens,
    )
    return result
