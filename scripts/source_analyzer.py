"""Step 1: Download YouTube transcript and metadata."""

import json
import logging
import re
import urllib.request
import urllib.error
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract video ID from URL: {url}")


def download_metadata(url: str, video_id: str) -> dict:
    """Get video metadata via noembed/oembed APIs (no yt-dlp, no auth needed)."""
    logger.info("Downloading metadata for %s", url)

    metadata = {}

    # Try noembed first (no rate limits, no auth)
    try:
        noembed_url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
        req = urllib.request.Request(noembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            metadata["title"] = data.get("title", "")
            metadata["channel"] = data.get("author_name", "")
            metadata["thumbnail_url"] = data.get("thumbnail_url", "")
            logger.info("Got metadata from noembed: %s", metadata["title"])
    except Exception as e:
        logger.warning("noembed failed: %s, trying oembed", e)

    # Fallback to YouTube oembed
    if not metadata.get("title"):
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                metadata["title"] = data.get("title", "")
                metadata["channel"] = data.get("author_name", "")
                metadata["thumbnail_url"] = data.get("thumbnail_url", "")
                logger.info("Got metadata from oembed: %s", metadata["title"])
        except Exception as e:
            logger.warning("oembed also failed: %s", e)

    # Set defaults for fields that oembed/noembed don't provide
    metadata.setdefault("title", f"Video {video_id}")
    metadata.setdefault("channel", "Unknown")
    metadata.setdefault("thumbnail_url", "")
    metadata["description"] = ""  # Not available via oembed
    metadata["tags"] = []
    metadata["view_count"] = 0
    metadata["like_count"] = 0
    metadata["duration"] = 0
    metadata["upload_date"] = ""

    return metadata


def download_transcript(video_id: str) -> tuple[list[dict], str]:
    """Download transcript, return (segments, language)."""
    logger.info("Downloading transcript for %s", video_id)
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try manually created first, then auto-generated
    try:
        transcript = transcript_list.find_manually_created_transcript(["en", "ru", "es", "de", "fr"])
    except Exception:
        transcript = transcript_list.find_generated_transcript(["en", "ru", "es", "de", "fr"])

    segments = transcript.fetch()
    # Convert to plain dicts
    segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in segments]
    return segments, transcript.language_code


def get_full_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments)


def analyze_source(url: str, output_dir: Path) -> dict:
    """Main entry point for Step 1."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = extract_video_id(url)
    metadata = download_metadata(url, video_id)
    segments, language = download_transcript(video_id)
    full_text = get_full_text(segments)

    result = {
        "video_id": video_id,
        "url": url,
        "title": metadata.get("title", ""),
        "description": metadata.get("description", ""),
        "tags": metadata.get("tags", []),
        "view_count": metadata.get("view_count", 0),
        "like_count": metadata.get("like_count", 0),
        "duration": metadata.get("duration", 0),
        "channel": metadata.get("channel", ""),
        "upload_date": metadata.get("upload_date", ""),
        "language": language,
        "transcript_segments": segments,
        "transcript_text": full_text,
    }

    # Save checkpoint
    with open(output_dir / "step1_source.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Source analysis complete: %s (%d segments, lang=%s)", metadata.get("title"), len(segments), language)
    return result
