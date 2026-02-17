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
    """Get video metadata via noembed/oembed APIs."""
    logger.info("Downloading metadata via noembed/oembed for %s", video_id)
    metadata = {}

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


def _try_fetch_transcript(video_id: str, use_proxy: bool = False):
    """Attempt to fetch transcript, optionally via Tor SOCKS5 proxy."""
    from youtube_transcript_api import YouTubeTranscriptApi
    import requests

    if use_proxy:
        logger.info("Trying transcript download via Tor proxy...")
        session = requests.Session()
        session.proxies = {
            "http": "socks5h://127.0.0.1:9050",
            "https": "socks5h://127.0.0.1:9050",
        }
        ytt = YouTubeTranscriptApi(session=session)
    else:
        logger.info("Trying transcript download (direct)...")
        ytt = YouTubeTranscriptApi()

    snippets = ytt.fetch(video_id)
    segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in snippets]
    return segments


def download_transcript(video_id: str) -> tuple[list[dict], str]:
    """Download transcript with Tor proxy fallback."""
    logger.info("Downloading transcript for %s", video_id)

    # Try direct first
    try:
        segments = _try_fetch_transcript(video_id, use_proxy=False)
        logger.info("Direct fetch succeeded: %d segments", len(segments))
        return segments, "en"
    except Exception as e:
        logger.warning("Direct fetch failed: %s", str(e)[:200])

    # Try via Tor proxy
    try:
        segments = _try_fetch_transcript(video_id, use_proxy=True)
        logger.info("Tor proxy fetch succeeded: %d segments", len(segments))
        return segments, "en"
    except Exception as e:
        logger.warning("Tor proxy fetch also failed: %s", str(e)[:200])

    raise RuntimeError(
        f"Failed to download transcript for {video_id}. "
        "Both direct and Tor proxy attempts were blocked by YouTube. "
        "Please try: (1) Wait a few minutes and retry, "
        "(2) Use a different video, or "
        "(3) Add a residential proxy via PROXY_URL environment variable."
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

    with open(output_dir / "step1_source.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        "Source analysis complete: %s (%d segments, %d chars, lang=%s)",
        metadata.get("title"), len(segments), len(full_text), language,
    )
    return result
