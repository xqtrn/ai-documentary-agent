"""Video generation using Runway Gen-4.5 text-to-video.

Single-step pipeline: text prompt → video. No intermediate image generation.
Gen-4.5 produces the highest quality hyperrealistic cinematic video.
On credit exhaustion the module raises immediately — no placeholders.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
from runwayml import RunwayML

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
    return any(kw in err_str for kw in ("credit", "insufficient", "quota", "billing"))


def generate_single_scene(scene: dict, output_dir) -> dict:
    """Generate video for a single scene using gen4.5 text-to-video.

    Args:
        scene: Scene dict with scene_number, visual_prompt, duration_sec.
        output_dir: Directory to store the video file.

    Returns:
        Dict with scene_number, video_path, status, duration.
        On failure: includes "error" key.

    Raises:
        RuntimeError: On credit exhaustion (fatal, never retry).
    """
    config.check_api_key("RUNWAY_API_KEY")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = RunwayML(api_key=config.RUNWAY_API_KEY)

    scene_num = scene.get("scene_number", 0)
    visual_prompt = scene.get("visual_prompt", "")
    duration = min(scene.get("duration_sec", config.SCENE_DURATION_SEC), 10)

    if not visual_prompt:
        raise ValueError(f"Scene {scene_num} has no visual_prompt.")

    video_path = output_dir / f"scene_{scene_num:03d}.mp4"

    last_error: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            logger.info(
                "Scene %d: generating video (attempt %d/%d, %d sec, model=%s)...",
                scene_num, attempt, config.MAX_RETRIES, duration, config.RUNWAY_VIDEO_MODEL,
            )

            task = client.text_to_video.create(
                model=config.RUNWAY_VIDEO_MODEL,
                prompt_text=visual_prompt,
                ratio="1280:720",
                duration=duration,
            )

            logger.info("Scene %d: task %s created, waiting...", scene_num, task.id)
            result = task.wait_for_task_output()

            if not result or not result.output or len(result.output) == 0:
                raise RuntimeError(f"Video task {task.id} returned empty output.")

            video_url = result.output[0]
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
                    f"Runway credit exhaustion at scene {scene_num}: {exc}. "
                    "Top up at https://app.runwayml.com."
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

    Args:
        scenes_data: Dict with "scenes" list.
        output_dir: Base project output directory.

    Returns:
        Dict with generated_scenes list (assembler-compatible), counts, status.

    Raises:
        RuntimeError: On credit exhaustion (stops immediately).
    """
    config.check_api_key("RUNWAY_API_KEY")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_data.get("scenes", [])
    if not scenes:
        raise ValueError("scenes_data must contain a non-empty 'scenes' list.")

    logger.info("Starting video generation for %d scene(s) with %s...", len(scenes), config.RUNWAY_VIDEO_MODEL)

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
