"""Step 1: Download YouTube transcript and metadata."""

import json
import logging
import re
import urllib.request
import urllib.error
from pathlib import Path

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


def download_metadata(video_id: str) -> dict:
    """Get video metadata via noembed/oembed APIs (no yt-dlp, no auth needed)."""
    logger.info("Downloading metadata via noembed/oembed for %s", video_id)
    metadata = {}

    # Try noembed first
    for api_name, api_url in [
        ("noembed", f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"),
        ("oembed", f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"),
    ]:
        if metadata.get("title"):
            break
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                metadata["title"] = data.get("title", "")
                metadata["channel"] = data.get("author_name", "")
                metadata["thumbnail_url"] = data.get("thumbnail_url", "")
                logger.info("Got metadata from %s: %s", api_name, metadata["title"])
        except Exception as e:
            logger.warning("%s failed: %s", api_name, e)

    # Defaults
    metadata.setdefault("title", f"Video {video_id}")
    metadata.setdefault("channel", "Unknown")
    metadata.setdefault("thumbnail_url", f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg")
    metadata.setdefault("description", "")
    metadata.setdefault("tags", [])
    metadata.setdefault("view_count", 0)
    metadata.setdefault("like_count", 0)
    metadata.setdefault("duration", 0)
    metadata.setdefault("upload_date", "")
    return metadata


def download_transcript(video_id: str) -> tuple[list[dict], str]:
    """Download transcript using youtube-transcript-api v1.x API."""
    logger.info("Downloading transcript for %s", video_id)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise RuntimeError("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")

    # v1.x API: instantiate, then call .fetch() or .list()
    ytt = YouTubeTranscriptApi()

    # Try fetching transcript
    try:
        # First try to list available transcripts and pick best one
        transcript_list = ytt.list(video_id)
        
        # Try manually created English first
        best = None
        for t in transcript_list:
            if t.language_code == "en" and not t.is_generated:
                best = t
                break
        # Then any manual transcript
        if not best:
            for t in transcript_list:
                if not t.is_generated:
                    best = t
                    break
        # Then auto-generated English
        if not best:
            for t in transcript_list:
                if t.language_code == "en":
                    best = t
                    break
        # Then any transcript
        if not best and transcript_list:
            best = transcript_list[0]

        if best:
            snippets = best.fetch()
            segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in snippets]
            return segments, best.language_code

    except Exception as e:
        logger.warning("list/fetch approach failed: %s, trying direct fetch", e)

    # Fallback: direct fetch
    try:
        snippets = ytt.fetch(video_id)
        segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in snippets]
        return segments, "en"
    except Exception as e2:
        raise RuntimeError(
            f"Failed to download transcript for {video_id}: {e2}\n"
            "This usually means YouTube is blocking requests from this server's IP. "
            "The video might not have subtitles, or you may need to use a proxy."
        )


def get_full_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments)


def analyze_source(url: str, output_dir: Path) -> dict:
    """Main entry point for Step 1."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = extract_video_id(url)
    logger.info("Video ID: %s", video_id)

    metadata = download_metadata(video_id)
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

    logger.info(
        "Source analysis complete: %s (%d segments, %d chars, lang=%s)",
        metadata.get("title"), len(segments), len(full_text), language,
    )
    return result
