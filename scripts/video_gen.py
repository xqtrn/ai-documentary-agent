"""Step 5: Generate video clips using Runway SDK.

No placeholders — if Runway credits run out, the pipeline stops with a clear error.
"""

import json
import logging
import time
from pathlib import Path

import httpx

import config

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 950  # Runway limit is 1000, leave buffer


def _truncate_prompt(prompt: str, max_len: int = MAX_PROMPT_LENGTH) -> str:
    """Truncate prompt to max length, keeping the no-text suffix."""
    suffix = " no text, no letters, no words, no subtitles, no signs, no writing, no numbers, no captions"
    if len(prompt) <= max_len:
        return prompt
    available = max_len - len(suffix) - 2
    truncated = prompt[:available].rsplit(".", 1)[0]
    if not truncated:
        truncated = prompt[:available]
    return truncated + "." + suffix


def _is_credits_error(error_str: str) -> bool:
    """Check if an error is due to exhausted Runway credits."""
    lower = error_str.lower()
    return any(phrase in lower for phrase in [
        "not have enough credits",
        "insufficient credits",
        "credits exhausted",
        "quota exceeded",
    ])


def generate_image_runway(client, prompt: str, scene_num: int, output_dir: Path) -> tuple:
    """Generate a reference image using Runway text_to_image."""
    import runwayml

    full_prompt = _truncate_prompt(
        f"{prompt}. Photorealistic, cinematic, high detail. "
        "no text, no letters, no words, no subtitles, no signs, no writing, no numbers, no captions"
    )
    logger.info("Scene %d image prompt (%d chars): %s", scene_num, len(full_prompt), full_prompt[:100] + "...")

    task = client.text_to_image.create(
        model="gen4_image",
        prompt_text=full_prompt,
        ratio="1280:720",
    )
    task_id = task.id
    logger.info("Scene %d image task created: %s", scene_num, task_id)

    try:
        result = task.wait_for_task_output()
    except runwayml.TaskFailedError as e:
        raise RuntimeError(f"Runway image task failed: {e}")
    except runwayml.TaskTimeoutError:
        raise TimeoutError(f"Runway image task {task_id} timed out")

    image_url = result.output[0]

    image_path = output_dir / f"scene_{scene_num:03d}.png"
    resp = httpx.get(image_url, timeout=120)
    resp.raise_for_status()
    image_path.write_bytes(resp.content)
    logger.info("Scene %d image saved: %s (%d bytes)", scene_num, image_path, len(resp.content))
    return image_path, image_url


def generate_video_runway(client, image_url: str, prompt: str, scene_num: int, duration: int, output_dir: Path) -> Path:
    """Generate video clip from reference image using Runway image_to_video."""
    import runwayml

    full_prompt = _truncate_prompt(
        f"{prompt}. Smooth cinematic motion, photorealistic. "
        "no text, no letters, no words, no subtitles, no signs"
    )
    logger.info("Scene %d video prompt (%d chars)", scene_num, len(full_prompt))

    task = client.image_to_video.create(
        model="gen4_turbo",
        prompt_image=image_url,
        prompt_text=full_prompt,
        duration=min(duration, 10),
        ratio="1280:720",
    )
    task_id = task.id
    logger.info("Scene %d video task created: %s", scene_num, task_id)

    try:
        result = task.wait_for_task_output()
    except runwayml.TaskFailedError as e:
        raise RuntimeError(f"Runway video task failed: {e}")
    except runwayml.TaskTimeoutError:
        raise TimeoutError(f"Runway video task {task_id} timed out")

    video_url = result.output[0]

    video_path = output_dir / f"scene_{scene_num:03d}.mp4"
    resp = httpx.get(video_url, timeout=120)
    resp.raise_for_status()
    video_path.write_bytes(resp.content)
    logger.info("Scene %d video saved: %s (%d bytes)", scene_num, video_path, len(resp.content))
    return video_path


def process_scene(client, scene: dict, images_dir: Path, videos_dir: Path) -> dict:
    """Process a single scene: generate image then video via Runway."""
    num = scene["scene_number"]
    prompt = scene["visual_prompt"]
    camera = scene.get("camera", "")
    lighting = scene.get("lighting", "")
    full_prompt = f"{prompt}. Camera: {camera}. Lighting: {lighting}"
    duration = scene.get("duration_sec", 10)

    logger.info("Processing scene %d...", num)

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            image_path, image_url = generate_image_runway(client, full_prompt, num, images_dir)
            video_path = generate_video_runway(client, image_url, full_prompt, num, duration, videos_dir)
            return {
                "scene_number": num,
                "image_path": str(image_path),
                "video_path": str(video_path),
                "duration_sec": duration,
                "source": "runway",
            }
        except Exception as e:
            last_error = str(e)
            if _is_credits_error(last_error):
                raise RuntimeError(
                    f"Runway credits exhausted! Cannot generate scene {num}. "
                    f"Please add more credits at https://dev.runwayml.com. Error: {last_error}"
                )
            logger.warning("Scene %d attempt %d/%d failed: %s", num, attempt + 1, config.MAX_RETRIES, last_error[:200])
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(10 * (attempt + 1))

    # All retries exhausted for this scene
    logger.error("Scene %d failed after %d attempts: %s", num, config.MAX_RETRIES, last_error)
    return {
        "scene_number": num,
        "duration_sec": duration,
        "error": last_error,
        "source": "failed",
    }


def generate_videos(scenes_data: dict, output_dir: Path) -> dict:
    """Generate all scene videos using Runway."""
    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_data["scenes"]
    results = []

    # Initialize Runway client — fail early if SDK missing or key invalid
    import runwayml
    client = runwayml.RunwayML(api_key=config.RUNWAY_API_KEY)
    logger.info("Runway client initialized, processing %d scenes...", len(scenes))

    for scene in scenes:
        result = process_scene(client, scene, images_dir, videos_dir)
        results.append(result)

    successful = [r for r in results if r.get("source") == "runway"]
    failed = [r for r in results if "error" in r]

    result = {
        "generated_scenes": results,
        "total_scenes": len(scenes),
        "successful": len(successful),
        "failed": len(failed),
        "runway_scenes": len(successful),
    }

    with open(output_dir / "step5_videos.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Video generation complete: %d/%d successful (%d failed)",
                len(successful), len(scenes), len(failed))
    return result
