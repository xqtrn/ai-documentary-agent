"""Step 1: Download YouTube transcript and metadata.

Strategy for transcript download (cloud IPs are blocked by YouTube):
1. Try youtube-transcript-api directly (works on residential IPs)
2. Try Invidious public instances (open source YouTube frontends)
3. Try scraping YouTube page for timedtext caption URLs
4. Check if user provided transcript via the web UI (paste/upload)
5. Fail with clear instructions
"""

import json
import logging
import re
import urllib.request
import urllib.error
from pathlib import Path
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

# Invidious instances that tend to work
INVIDIOUS_INSTANCES = [
    "https://inv.nadeko.net",
    "https://iv.datura.network",
    "https://invidious.protokolla.fi",
    "https://yewtu.be",
    "https://inv.tux.pizza",
    "https://vid.puffyan.us",
    "https://invidious.lunar.icu",
]


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


def _try_youtube_transcript_api(video_id: str) -> list[dict] | None:
    """Method 1: youtube-transcript-api (only works on non-cloud IPs)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        logger.info("Trying youtube-transcript-api (direct)...")
        ytt = YouTubeTranscriptApi()
        snippets = ytt.fetch(video_id)
        segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in snippets]
        logger.info("youtube-transcript-api succeeded: %d segments", len(segments))
        return segments
    except Exception as e:
        logger.warning("youtube-transcript-api failed: %s", str(e)[:150])
        return None


def _try_youtube_transcript_api_tor(video_id: str) -> list[dict] | None:
    """Method 1b: youtube-transcript-api via Tor SOCKS5 proxy."""
    try:
        import requests
        from youtube_transcript_api import YouTubeTranscriptApi
        logger.info("Trying youtube-transcript-api via Tor proxy...")
        session = requests.Session()
        session.proxies = {
            "http": "socks5h://127.0.0.1:9050",
            "https": "socks5h://127.0.0.1:9050",
        }
        ytt = YouTubeTranscriptApi(session=session)
        snippets = ytt.fetch(video_id)
        segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in snippets]
        logger.info("Tor proxy succeeded: %d segments", len(segments))
        return segments
    except Exception as e:
        logger.warning("Tor proxy failed: %s", str(e)[:150])
        return None


def _try_invidious(video_id: str) -> list[dict] | None:
    """Method 2: Try Invidious public instances."""
    for instance in INVIDIOUS_INSTANCES:
        try:
            logger.info("Trying Invidious: %s", instance)
            # Get caption list
            caps_url = f"{instance}/api/v1/captions/{video_id}"
            req = urllib.request.Request(caps_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            captions = data.get("captions", [])
            if not captions:
                continue

            # Find English caption
            en_cap = None
            for c in captions:
                lc = c.get("languageCode", c.get("language_code", ""))
                label = c.get("label", "")
                if lc == "en" and "auto" not in label.lower():
                    en_cap = c
                    break
            if not en_cap:
                for c in captions:
                    lc = c.get("languageCode", c.get("language_code", ""))
                    if lc == "en":
                        en_cap = c
                        break
            if not en_cap:
                en_cap = captions[0]

            # Download VTT
            vtt_path = en_cap.get("url", "")
            if not vtt_path:
                continue
            vtt_url = f"{instance}{vtt_path}"
            req2 = urllib.request.Request(vtt_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                vtt_data = resp2.read().decode("utf-8", errors="replace")

            if len(vtt_data) < 50:
                logger.warning("Invidious %s returned empty VTT", instance)
                continue

            # Parse VTT
            segments = _parse_vtt(vtt_data)
            if segments:
                logger.info("Invidious %s succeeded: %d segments", instance, len(segments))
                return segments
        except Exception as e:
            logger.warning("Invidious %s failed: %s", instance, str(e)[:100])
            continue
    return None


def _try_scrape_youtube_page(video_id: str) -> list[dict] | None:
    """Method 3: Scrape YouTube page for timedtext URLs and fetch XML captions."""
    try:
        import http.cookiejar
        logger.info("Trying YouTube page scraping...")

        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

        req = urllib.request.Request(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
        page = opener.open(req, timeout=20).read().decode("utf-8", errors="replace")

        m = re.search(r'"captionTracks":(\[.*?\])', page)
        if not m:
            logger.warning("No captionTracks found on YouTube page")
            return None

        tracks = json.loads(m.group(1))
        if not tracks:
            return None

        # Find English manual > English auto > any
        en_track = None
        for t in tracks:
            if t.get("languageCode") == "en" and t.get("kind") != "asr":
                en_track = t
                break
        if not en_track:
            for t in tracks:
                if t.get("languageCode") == "en":
                    en_track = t
                    break
        if not en_track:
            en_track = tracks[0]

        caption_url = en_track.get("baseUrl", "")
        if not caption_url:
            return None

        # Try fetching caption XML with cookies
        req2 = urllib.request.Request(caption_url, headers=headers)
        xml_data = opener.open(req2, timeout=15).read().decode("utf-8", errors="replace")

        if len(xml_data) < 50:
            # YouTube returns empty for cloud IPs even with valid URLs
            logger.warning("YouTube timedtext returned empty (cloud IP blocked)")
            return None

        # Parse XML captions
        segments = _parse_caption_xml(xml_data)
        if segments:
            logger.info("YouTube page scraping succeeded: %d segments", len(segments))
            return segments
        return None
    except Exception as e:
        logger.warning("YouTube page scraping failed: %s", str(e)[:150])
        return None


def _check_user_transcript(video_id: str, output_dir: Path) -> list[dict] | None:
    """Method 4: Check if user pasted/uploaded transcript via web UI."""
    # Check for user-provided transcript file
    user_file = output_dir / "user_transcript.txt"
    if not user_file.exists():
        user_file = Path(config.OUTPUT_DIR) / "user_transcript.txt"
    if not user_file.exists():
        return None

    logger.info("Found user-provided transcript file: %s", user_file)
    text = user_file.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None

    # Split into fake segments (one per sentence/paragraph)
    segments = []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    t = 0.0
    for p in paragraphs:
        dur = max(2.0, len(p) / 15)  # ~15 chars per second speech
        segments.append({"text": p, "start": round(t, 2), "duration": round(dur, 2)})
        t += dur

    logger.info("User transcript loaded: %d segments, %d chars", len(segments), len(text))
    return segments


def _check_bundled_transcript(video_id: str) -> list[dict] | None:
    """Method 5: Check for transcripts bundled in the repo's transcripts/ directory."""
    try:
        # Check local bundled transcripts directory
        bundled_json = Path(__file__).parent.parent / "transcripts" / f"{video_id}.json"
        if bundled_json.exists():
            logger.info("Found bundled transcript: %s", bundled_json)
            segments = json.loads(bundled_json.read_text())
            logger.info("Bundled transcript loaded: %d segments", len(segments))
            return segments

        bundled_txt = Path(__file__).parent.parent / "transcripts" / f"{video_id}.txt"
        if bundled_txt.exists():
            logger.info("Found bundled transcript text: %s", bundled_txt)
            text = bundled_txt.read_text(encoding="utf-8", errors="replace").strip()
            segments = []
            t = 0.0
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    dur = max(2.0, len(line) / 15)
                    segments.append({"text": line, "start": round(t, 2), "duration": round(dur, 2)})
                    t += dur
            logger.info("Bundled text transcript loaded: %d segments", len(segments))
            return segments
    except Exception as e:
        logger.warning("Bundled transcript check failed: %s", e)
    return None


def _parse_vtt(vtt_text: str) -> list[dict]:
    """Parse WebVTT subtitle format."""
    segments = []
    lines = vtt_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            parts = line.split("-->")
            start = _parse_timestamp(parts[0].strip())
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            # Remove VTT tags
            text = re.sub(r"<[^>]+>", "", text)
            if text:
                segments.append({"text": text, "start": round(start, 2), "duration": 0})
        i += 1
    return segments


def _parse_caption_xml(xml_text: str) -> list[dict]:
    """Parse YouTube caption XML format."""
    try:
        root = ElementTree.fromstring(xml_text)
        segments = []
        for elem in root.findall(".//text"):
            text = (elem.text or "").strip()
            start = float(elem.get("start", 0))
            dur = float(elem.get("dur", 0))
            if text:
                segments.append({"text": text, "start": round(start, 2), "duration": round(dur, 2)})
        return segments
    except Exception:
        return []


def _parse_timestamp(ts: str) -> float:
    """Parse VTT timestamp like '00:01:23.456' or '01:23.456'."""
    m = re.match(r"(\d+):(\d+):(\d+)[.,](\d+)", ts)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000
    m2 = re.match(r"(\d+):(\d+)[.,](\d+)", ts)
    if m2:
        return int(m2.group(1)) * 60 + int(m2.group(2)) + int(m2.group(3)) / 1000
    return 0.0


def get_full_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments)


# Need config for OUTPUT_DIR in _check_user_transcript
import config


def download_transcript(video_id: str, output_dir: Path) -> tuple[list[dict], str]:
    """Download transcript using multiple fallback methods."""
    logger.info("Downloading transcript for %s (trying multiple methods)", video_id)

    errors = []

    # Method 1: youtube-transcript-api (direct)
    result = _try_youtube_transcript_api(video_id)
    if result:
        return result, "en"
    errors.append("Direct API: blocked by YouTube")

    # Method 1b: youtube-transcript-api via Tor
    result = _try_youtube_transcript_api_tor(video_id)
    if result:
        return result, "en"
    errors.append("Tor proxy: blocked or unavailable")

    # Method 2: Invidious instances
    result = _try_invidious(video_id)
    if result:
        return result, "en"
    errors.append("Invidious: all instances failed")

    # Method 3: Scrape YouTube page directly
    result = _try_scrape_youtube_page(video_id)
    if result:
        return result, "en"
    errors.append("YouTube scraping: timedtext blocked for cloud IP")

    # Method 4: User-provided transcript
    result = _check_user_transcript(video_id, output_dir)
    if result:
        return result, "en"

    # Method 5: Bundled transcript in repo
    result = _check_bundled_transcript(video_id)
    if result:
        return result, "en"
    errors.append("Bundled transcript: not found for this video")

    # All methods failed
    error_details = "\n".join(f"  - {e}" for e in errors)
    raise RuntimeError(
        f"❌ Cannot download transcript for video {video_id}.\n"
        f"All automatic methods failed:\n{error_details}\n\n"
        f"🔧 SOLUTION: Paste the transcript manually!\n"
        f"In the web dashboard, paste the video's transcript text in the "
        f"'Manual Transcript' field, then click Generate again.\n\n"
        f"You can get the transcript from:\n"
        f"  1. YouTube → click '...' below video → 'Show transcript'\n"
        f"  2. Google 'French Revolution OverSimplified transcript'\n"
        f"  3. Websites like downsub.com or kome.ai"
    )


def analyze_source(url: str, output_dir: Path) -> dict:
    """Main entry point for Step 1."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = extract_video_id(url)
    logger.info("Video ID: %s", video_id)

    metadata = download_metadata(video_id)
    segments, language = download_transcript(video_id, output_dir)
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
