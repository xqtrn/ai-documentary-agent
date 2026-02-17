"""Step 5: Generate video clips using Runway SDK."""

import json
import logging
import time
from pathlib import Path

import httpx
import runwayml

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


def generate_image(client: runwayml.RunwayML, prompt: str, scene_num: int, output_dir: Path) -> tuple:
    """Generate a reference image using Runway text_to_image."""
    full_prompt = _truncate_prompt(
        f"{prompt}. Photorealistic, cinematic, high detail. "
        "no text, no letters, no words, no subtitles, no signs, no writing, no numbers, no captions"
    )
    logger.info("Scene %d image prompt (%d chars): %s", scene_num, len(full_prompt), full_prompt[:100] + "...")

    for attempt in range(config.MAX_RETRIES):
        try:
            task = client.text_to_image.create(
                model="gen4_image",
                prompt_text=full_prompt,
                ratio="1280:720",
            )
            task_id = task.id
            logger.info("Scene %d image task created: %s", scene_num, task_id)

            # Use SDK's built-in polling with wait_for_task_output
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

        except Exception as e:
            logger.warning("Image gen attempt %d/%d failed for scene %d: %s", attempt + 1, config.MAX_RETRIES, scene_num, e)
            if attempt == config.MAX_RETRIES - 1:
                raise
            time.sleep(5 * (attempt + 1))


def generate_video(client: runwayml.RunwayML, image_url: str, prompt: str, scene_num: int, duration: int, output_dir: Path) -> Path:
    """Generate video clip from reference image using Runway image_to_video."""
    full_prompt = _truncate_prompt(
        f"{prompt}. Smooth cinematic motion, photorealistic. "
        "no text, no letters, no words, no subtitles, no signs"
    )
    logger.info("Scene %d video prompt (%d chars)", scene_num, len(full_prompt))

    for attempt in range(config.MAX_RETRIES):
        try:
            task = client.image_to_video.create(
                model="gen4_turbo",
                prompt_image=image_url,
                prompt_text=full_prompt,
                duration=min(duration, 10),
                ratio="1280:720",
            )
            task_id = task.id
            logger.info("Scene %d video task created: %s", scene_num, task_id)

            # Use SDK's built-in polling
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

        except Exception as e:
            logger.warning("Video gen attempt %d/%d failed for scene %d: %s", attempt + 1, config.MAX_RETRIES, scene_num, e)
            if attempt == config.MAX_RETRIES - 1:
                raise
            time.sleep(10 * (attempt + 1))


def process_scene(client: runwayml.RunwayML, scene: dict, images_dir: Path, videos_dir: Path) -> dict:
    """Process a single scene: generate image then video."""
    num = scene["scene_number"]
    prompt = scene["visual_prompt"]
    camera = scene.get("camera", "")
    lighting = scene.get("lighting", "")
    full_prompt = f"{prompt}. Camera: {camera}. Lighting: {lighting}"
    duration = scene.get("duration_sec", 10)

    logger.info("Processing scene %d...", num)

    image_path, image_url = generate_image(client, full_prompt, num, images_dir)
    video_path = generate_video(client, image_url, full_prompt, num, duration, videos_dir)

    return {
        "scene_number": num,
        "image_path": str(image_path),
        "video_path": str(video_path),
        "duration_sec": duration,
    }


def generate_videos(scenes_data: dict, output_dir: Path) -> dict:
    """Generate all scene videos sequentially."""
    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_data["scenes"]
    results = []

    client = runwayml.RunwayML(api_key=config.RUNWAY_API_KEY)

    for scene in scenes:
        try:
            result = process_scene(client, scene, images_dir, videos_dir)
            results.append(result)
        except Exception as e:
            logger.error("Scene %d failed: %s", scene["scene_number"], e)
            results.append({
                "scene_number": scene["scene_number"],
                "error": str(e),
            })

    result = {
        "generated_scenes": results,
        "total_scenes": len(scenes),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
    }

    with open(output_dir / "step5_videos.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Video generation complete: %d/%d successful", result["successful"], result["total_scenes"])
    return result
