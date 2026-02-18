"""Video generation using Grok Imagine (xAI) text-to-video.

Uses the xAI REST API directly. No SDK needed.
On credit exhaustion the module raises immediately — no placeholders.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

import config

logger = logging.getLogger(__name__)


def _download_file(url: str, dest: Path, timeout: float = 180.0) -> Path:
    """Download a file from a URL to a local path."""
    logger.info("Downloading %s -> %s", url, dest)
    with httpx.Client(timeout=timeout, follow_redirects=True) as http:
        resp = http.get(url)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    logger.info("Downloaded %d bytes to %s", dest.stat().st_size, dest)
    return dest


def _is_credit_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    return any(kw in err_str for kw in ("credit", "insufficient", "quota", "billing", "rate_limit"))


def _submit_video(prompt: str, duration: int) -> str:
    """Submit a video generation request to xAI Grok Imagine. Returns request_id."""
    config.check_api_key("XAI_API_KEY")

    url = f"{config.XAI_VIDEO_BASE_URL}/videos/generations"
    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.XAI_VIDEO_MODEL,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": config.XAI_VIDEO_ASPECT_RATIO,
        "resolution": config.XAI_VIDEO_RESOLUTION,
    }

    with httpx.Client(timeout=60.0) as http:
        resp = http.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    request_id = data.get("request_id") or data.get("id")
    if not request_id:
        raise RuntimeError(f"xAI returned no request_id: {data}")
    return request_id


def _poll_video(request_id: str, max_wait: int = 300, interval: int = 5) -> dict:
    """Poll xAI until video is done. Returns the response dict with video URL.

    xAI response formats:
      - Pending: {"status": "pending"}
      - Done:    {"video": {"url": "..."}, "model": "..."}  (no status field)
      - Failed:  {"status": "failed", ...}
    """
    config.check_api_key("XAI_API_KEY")

    url = f"{config.XAI_VIDEO_BASE_URL}/videos/{request_id}"
    headers = {"Authorization": f"Bearer {config.XAI_API_KEY}"}

    deadline = time.time() + max_wait
    while time.time() < deadline:
        with httpx.Client(timeout=30.0) as http:
            resp = http.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # xAI returns video.url directly when done (no "status" field)
        video_obj = data.get("video")
        if isinstance(video_obj, dict) and video_obj.get("url"):
            logger.info("xAI video %s completed.", request_id)
            return data

        status = data.get("status", "").lower()
        if status in ("failed", "error"):
            raise RuntimeError(f"xAI video generation failed: {data}")

        logger.debug("xAI video %s status: %s", request_id, status or "waiting")
        time.sleep(interval)

    raise TimeoutError(f"xAI video {request_id} did not complete within {max_wait}s")


def generate_single_scene(scene: dict, output_dir) -> dict:
    """Generate video for a single scene using Grok Imagine.

    Returns:
        Dict with scene_number, video_path, status, duration.

    Raises:
        RuntimeError: On credit exhaustion (fatal).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_num = scene.get("scene_number", 0)
    visual_prompt = scene.get("visual_prompt", "")
    duration = scene.get("duration_sec", config.XAI_VIDEO_DURATION)

    if not visual_prompt:
        raise ValueError(f"Scene {scene_num} has no visual_prompt.")

    video_path = output_dir / f"scene_{scene_num:03d}.mp4"

    last_error: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            logger.info(
                "Scene %d: generating video (attempt %d/%d, %d sec, model=%s)...",
                scene_num, attempt, config.MAX_RETRIES, duration, config.XAI_VIDEO_MODEL,
            )

            request_id = _submit_video(visual_prompt, duration)
            logger.info("Scene %d: request_id=%s, polling...", scene_num, request_id)

            result = _poll_video(request_id)

            # Extract video URL from response
            video_url = None
            if "video" in result and isinstance(result["video"], dict):
                video_url = result["video"].get("url")
            if not video_url:
                video_url = result.get("url")
            if not video_url and isinstance(result.get("output"), list) and result["output"]:
                video_url = result["output"][0]
            if not video_url:
                raise RuntimeError(f"No video URL in xAI response: {result}")

            _download_file(video_url, video_path)
            logger.info("Scene %d: video saved to %s", scene_num, video_path)

            return {
                "scene_number": scene_num,
                "video_path": str(video_path),
                "status": "success",
                "duration": duration,
            }

        except Exception as exc:
            if _is_credit_error(exc):
                raise RuntimeError(
                    f"xAI credit exhaustion at scene {scene_num}: {exc}. "
                    "Top up at https://console.x.ai"
                ) from exc

            last_error = exc
            logger.warning(
                "Scene %d attempt %d/%d failed: %s",
                scene_num, attempt, config.MAX_RETRIES, exc,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(2 ** attempt)

    logger.error("Scene %d failed after %d retries: %s", scene_num, config.MAX_RETRIES, last_error)
    return {
        "scene_number": scene_num,
        "video_path": None,
        "status": "failed",
        "error": str(last_error),
    }


def generate_videos(scenes_data: dict, output_dir) -> dict:
    """Generate videos for all scenes.

    Returns:
        Dict with generated_scenes list, counts, status.

    Raises:
        RuntimeError: On credit exhaustion (stops immediately).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_data.get("scenes", [])
    if not scenes:
        raise ValueError("scenes_data must contain a non-empty 'scenes' list.")

    logger.info("Starting video generation for %d scene(s) with %s...", len(scenes), config.XAI_VIDEO_MODEL)

    scene_results: list[dict] = []
    successful = 0
    failed = 0

    for i, scene in enumerate(scenes):
        scene_num = scene.get("scene_number", i + 1)
        logger.info("Processing scene %d/%d (scene_number=%d)...", i + 1, len(scenes), scene_num)

        try:
            result = generate_single_scene(scene, videos_dir)
        except RuntimeError as exc:
            if "credit" in str(exc).lower():
                logger.error("Credit exhaustion at scene %d — stopping.", scene_num)
                scene_results.append({
                    "scene_number": scene_num,
                    "video_path": None,
                    "error": str(exc),
                })
                failed += 1

                output = {
                    "generated_scenes": scene_results,
                    "total_scenes": len(scenes),
                    "successful": successful,
                    "failed": failed,
                    "status": "credit_exhausted",
                }
                step_file = output_dir / "step5_videos.json"
                step_file.write_text(json.dumps(output, indent=2))
                raise
            raise

        scene_results.append(result)
        if result.get("status") == "success":
            successful += 1
        else:
            failed += 1

    status = "success" if failed == 0 else ("partial" if successful > 0 else "failed")
    logger.info(
        "Video generation complete: %d/%d successful. Status: %s",
        successful, len(scenes), status,
    )

    output = {
        "generated_scenes": scene_results,
        "total_scenes": len(scenes),
        "successful": successful,
        "failed": failed,
        "status": status,
    }

    step_file = output_dir / "step5_videos.json"
    step_file.write_text(json.dumps(output, indent=2))
    logger.info("Video step result saved to %s", step_file)

    return output
