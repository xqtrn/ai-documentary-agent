"""Step 5: Generate video clips using Runway SDK.

Falls back to generating placeholder videos using FFmpeg when Runway
credits are exhausted or the API is unavailable.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

import httpx

import config

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 950  # Runway limit is 1000, leave buffer

# Track if Runway credits are exhausted to skip remaining scenes
_credits_exhausted = False


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


def _generate_placeholder_video(scene_num: int, duration: int, output_dir: Path) -> Path:
    """Generate a placeholder dark gradient video clip using FFmpeg."""
    video_path = output_dir / f"scene_{scene_num:03d}.mp4"
    # Create a dark cinematic gradient - different hue per scene
    hue = (scene_num * 30) % 360
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"color=c=black:s=1280x720:d={duration}:r=24,format=yuv420p,"
        f"drawbox=x=0:y=0:w=1280:h=720:c=0x{_scene_color(scene_num)}@0.3:t=fill",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        # Simpler fallback if drawbox fails
        cmd2 = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x111122:s=1280x720:d={duration}:r=24,format=yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            str(video_path),
        ]
        subprocess.run(cmd2, capture_output=True, text=True, timeout=30, check=True)

    logger.info("Scene %d placeholder video generated: %s", scene_num, video_path)
    return video_path


def _scene_color(num: int) -> str:
    """Generate a dark color hex for each scene."""
    colors = ["1a1a2e", "16213e", "0f3460", "1a1a3e", "2d1b3d", "1b2d3d"]
    return colors[num % len(colors)]


def _generate_placeholder_image(scene_num: int, output_dir: Path) -> Path:
    """Generate a placeholder dark image using FFmpeg."""
    image_path = output_dir / f"scene_{scene_num:03d}.png"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=0x{_scene_color(scene_num)}:s=1280x720:d=1,format=rgb24",
        "-frames:v", "1",
        str(image_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
    return image_path


def _is_credits_error(error_str: str) -> bool:
    """Check if an error is due to exhausted Runway credits."""
    return "not have enough credits" in error_str.lower() or "insufficient credits" in error_str.lower()


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
    """Process a single scene: generate image then video, with fallback."""
    global _credits_exhausted
    num = scene["scene_number"]
    prompt = scene["visual_prompt"]
    camera = scene.get("camera", "")
    lighting = scene.get("lighting", "")
    full_prompt = f"{prompt}. Camera: {camera}. Lighting: {lighting}"
    duration = scene.get("duration_sec", 10)

    logger.info("Processing scene %d...", num)

    # If we already know credits are exhausted, skip Runway entirely
    if _credits_exhausted:
        logger.info("Scene %d: using placeholder (Runway credits exhausted)", num)
        image_path = _generate_placeholder_image(num, images_dir)
        video_path = _generate_placeholder_video(num, duration, videos_dir)
        return {
            "scene_number": num,
            "image_path": str(image_path),
            "video_path": str(video_path),
            "duration_sec": duration,
            "source": "placeholder",
        }

    # Try Runway with fallback
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
        error_str = str(e)
        if _is_credits_error(error_str):
            _credits_exhausted = True
            logger.warning("Runway credits exhausted, switching to placeholder videos for all remaining scenes")
        else:
            logger.warning("Runway failed for scene %d: %s, using placeholder", num, error_str[:150])

        image_path = _generate_placeholder_image(num, images_dir)
        video_path = _generate_placeholder_video(num, duration, videos_dir)
        return {
            "scene_number": num,
            "image_path": str(image_path),
            "video_path": str(video_path),
            "duration_sec": duration,
            "source": "placeholder",
        }


def generate_videos(scenes_data: dict, output_dir: Path) -> dict:
    """Generate all scene videos sequentially."""
    global _credits_exhausted
    _credits_exhausted = False

    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_data["scenes"]
    results = []

    # Try to initialize Runway client
    client = None
    try:
        import runwayml
        client = runwayml.RunwayML(api_key=config.RUNWAY_API_KEY)
    except Exception as e:
        logger.warning("Could not initialize Runway client: %s, using placeholders", e)
        _credits_exhausted = True

    for scene in scenes:
        result = process_scene(client, scene, images_dir, videos_dir)
        results.append(result)

    runway_count = sum(1 for r in results if r.get("source") == "runway")
    placeholder_count = sum(1 for r in results if r.get("source") == "placeholder")

    result = {
        "generated_scenes": results,
        "total_scenes": len(scenes),
        "successful": len(results),  # All scenes now succeed (with fallback)
        "failed": 0,
        "runway_scenes": runway_count,
        "placeholder_scenes": placeholder_count,
    }

    with open(output_dir / "step5_videos.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Video generation complete: %d total (%d Runway, %d placeholder)",
                len(results), runway_count, placeholder_count)
    return result
