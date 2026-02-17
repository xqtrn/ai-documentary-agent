"""Step 5: Generate video clips using Runway Gen-4 API."""

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

import config

logger = logging.getLogger(__name__)

HEADERS = {
    "Authorization": f"Bearer {config.RUNWAY_API_KEY}",
    "Content-Type": "application/json",
    "X-Runway-Version": "2024-11-06",
}


async def generate_image(client: httpx.AsyncClient, prompt: str, scene_num: int, output_dir: Path) -> Path:
    """Generate a reference image using Runway Gen-4 Image."""
    full_prompt = f"{prompt}. Photorealistic, cinematic, 16:9 aspect ratio, high detail. no text, no letters, no words, no subtitles, no signs, no writing, no numbers, no captions, no titles, no labels, no banners, no inscriptions"

    for attempt in range(config.MAX_RETRIES):
        try:
            resp = await client.post(
                f"{config.RUNWAY_BASE_URL}/image/generate",
                headers=HEADERS,
                json={
                    "model": config.RUNWAY_IMAGE_MODEL,
                    "prompt": full_prompt,
                    "num_images": 1,
                    "resolution": "1080p",
                    "aspect_ratio": "16:9",
                },
                timeout=120,
            )
            resp.raise_for_status()
            task = resp.json()
            task_id = task["id"]

            # Poll for completion
            image_path = await poll_runway_task(client, task_id, scene_num, output_dir, "image")
            return image_path

        except Exception as e:
            logger.warning("Image gen attempt %d/%d failed for scene %d: %s", attempt + 1, config.MAX_RETRIES, scene_num, e)
            if attempt == config.MAX_RETRIES - 1:
                raise
            await asyncio.sleep(5 * (attempt + 1))


async def generate_video_from_image(client: httpx.AsyncClient, image_url: str, prompt: str, scene_num: int, duration: int, output_dir: Path) -> Path:
    """Generate video clip from reference image using Runway Gen-4 Turbo."""
    full_prompt = f"{prompt}. Smooth cinematic motion, photorealistic. no text, no letters, no words, no subtitles, no signs"

    for attempt in range(config.MAX_RETRIES):
        try:
            resp = await client.post(
                f"{config.RUNWAY_BASE_URL}/video/generate",
                headers=HEADERS,
                json={
                    "model": config.RUNWAY_VIDEO_MODEL,
                    "prompt": full_prompt,
                    "image_url": image_url,
                    "duration": min(duration, 10),
                    "resolution": "1080p",
                    "aspect_ratio": "16:9",
                },
                timeout=120,
            )
            resp.raise_for_status()
            task = resp.json()
            task_id = task["id"]

            video_path = await poll_runway_task(client, task_id, scene_num, output_dir, "video")
            return video_path

        except Exception as e:
            logger.warning("Video gen attempt %d/%d failed for scene %d: %s", attempt + 1, config.MAX_RETRIES, scene_num, e)
            if attempt == config.MAX_RETRIES - 1:
                raise
            await asyncio.sleep(10 * (attempt + 1))


async def poll_runway_task(client: httpx.AsyncClient, task_id: str, scene_num: int, output_dir: Path, media_type: str) -> Path:
    """Poll Runway API until task completes, download result."""
    max_polls = 120  # 10 minutes max
    for _ in range(max_polls):
        resp = await client.get(
            f"{config.RUNWAY_BASE_URL}/tasks/{task_id}",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json()

        if status["status"] == "SUCCEEDED":
            output_url = status["output"][0]
            # Download the file
            ext = "png" if media_type == "image" else "mp4"
            file_path = output_dir / f"scene_{scene_num:03d}.{ext}"
            dl_resp = await client.get(output_url, timeout=120)
            dl_resp.raise_for_status()
            file_path.write_bytes(dl_resp.content)
            logger.info("Scene %d %s saved: %s", scene_num, media_type, file_path)
            return file_path

        elif status["status"] == "FAILED":
            raise RuntimeError(f"Runway task {task_id} failed: {status.get('error', 'unknown')}")

        await asyncio.sleep(5)

    raise TimeoutError(f"Runway task {task_id} timed out")


async def process_scene(client: httpx.AsyncClient, scene: dict, output_dir: Path, images_dir: Path, videos_dir: Path) -> dict:
    """Process a single scene: generate image then video."""
    num = scene["scene_number"]
    prompt = scene["visual_prompt"]
    camera = scene.get("camera", "")
    lighting = scene.get("lighting", "")
    full_prompt = f"{prompt}. Camera: {camera}. Lighting: {lighting}"
    duration = scene.get("duration_sec", 10)

    logger.info("Processing scene %d...", num)

    # Step 1: Generate reference image
    image_path = await generate_image(client, full_prompt, num, images_dir)

    # Get image URL from Runway (or use local path for upload)
    # For image-to-video, we need the image URL from the generation result
    # We'll read the task result which contains the URL
    image_url = str(image_path)  # Will be replaced with actual URL from generation

    # Step 2: Generate video from image
    video_path = await generate_video_from_image(client, image_url, full_prompt, num, duration, videos_dir)

    return {
        "scene_number": num,
        "image_path": str(image_path),
        "video_path": str(video_path),
        "duration_sec": duration,
    }


async def generate_all_videos(scenes_data: dict, output_dir: Path) -> dict:
    """Generate all scene videos in parallel batches."""
    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_data["scenes"]
    results = []

    async with httpx.AsyncClient() as client:
        # Process in batches
        for i in range(0, len(scenes), config.BATCH_SIZE):
            batch = scenes[i:i + config.BATCH_SIZE]
            logger.info("Processing batch %d-%d of %d scenes", i + 1, i + len(batch), len(scenes))

            tasks = [
                process_scene(client, scene, output_dir, images_dir, videos_dir)
                for scene in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error("Scene %d failed: %s", batch[j]["scene_number"], result)
                    results.append({
                        "scene_number": batch[j]["scene_number"],
                        "error": str(result),
                    })
                else:
                    results.append(result)

    result = {
        "generated_scenes": results,
        "total_scenes": len(scenes),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
    }

    # Save checkpoint
    with open(output_dir / "step5_videos.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Video generation complete: %d/%d successful", result["successful"], result["total_scenes"])
    return result


def generate_videos(scenes_data: dict, output_dir: Path) -> dict:
    """Sync wrapper for async video generation."""
    return asyncio.run(generate_all_videos(scenes_data, output_dir))
