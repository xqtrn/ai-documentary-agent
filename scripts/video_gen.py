"""Multi-engine video generation dispatcher.

Supports:
  - Runway models: veo3.1, veo3.1_fast, gen4.5, gen4_turbo, veo3
  - OpenAI models: sora-2-pro, sora-2
  - xAI models: grok-imagine (grok-imagine-video)
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
    return any(kw in err_str for kw in ("credit", "insufficient", "quota", "billing", "rate_limit"))


def _truncate_prompt(prompt: str, max_chars: int) -> str:
    """Truncate prompt to fit engine's character limit."""
    if max_chars is None or len(prompt) <= max_chars:
        return prompt
    logger.warning("Prompt is %d chars, truncating to %d", len(prompt), max_chars)
    truncated = prompt[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars - 200:
        truncated = truncated[:last_period + 1]
    return truncated


# ---------------------------------------------------------------------------
# Runway video generation (veo3.1, veo3.1_fast, gen4.5, gen4_turbo, veo3)
# ---------------------------------------------------------------------------

def _generate_runway_video(prompt: str, duration: int, video_path: Path, engine_cfg: dict) -> Path:
    """Generate video via Runway SDK."""
    config.check_api_key("RUNWAY_API_KEY")
    client = RunwayML(api_key=config.RUNWAY_API_KEY)

    prompt = _truncate_prompt(prompt, engine_cfg.get("max_prompt_chars", 1000))
    model = engine_cfg["model"]
    ratio = engine_cfg.get("ratio", "1280:720")
    max_dur = engine_cfg.get("max_duration_sec", 10)
    duration = min(duration, max_dur)

    logger.info("Runway: model=%s, ratio=%s, duration=%d", model, ratio, duration)

    task = client.text_to_video.create(
        model=model,
        prompt_text=prompt,
        ratio=ratio,
        duration=duration,
    )
    logger.info("Runway task %s created, waiting...", task.id)
    result = task.wait_for_task_output()

    if not result or not result.output or len(result.output) == 0:
        raise RuntimeError(f"Runway task {task.id} returned empty output.")

    return _download_file(result.output[0], video_path)


# ---------------------------------------------------------------------------
# OpenAI Sora video generation (sora-2-pro, sora-2)
# ---------------------------------------------------------------------------

def _generate_sora_video(prompt: str, duration: int, video_path: Path, engine_cfg: dict) -> Path:
    """Generate video via OpenAI Sora API."""
    config.check_api_key("OPENAI_API_KEY")

    model = engine_cfg["model"]
    resolution = engine_cfg.get("resolution", "1920x1080")
    max_dur = engine_cfg.get("max_duration_sec", 20)
    duration = min(duration, max_dur)

    # Sora requires seconds as STRING
    base_url = "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Step 1: Create video
    create_body = {
        "model": model,
        "prompt": prompt,
        "size": resolution,
        "seconds": str(duration),
    }
    logger.info("Sora: model=%s, size=%s, seconds=%s", model, resolution, str(duration))

    with httpx.Client(timeout=60.0) as http:
        resp = http.post(f"{base_url}/videos", headers=headers, json=create_body)
        resp.raise_for_status()
        data = resp.json()

    video_id = data.get("id")
    if not video_id:
        raise RuntimeError(f"Sora returned no video id: {data}")
    logger.info("Sora video %s created, polling...", video_id)

    # Step 2: Poll for completion
    deadline = time.time() + 600  # 10 min max
    while time.time() < deadline:
        with httpx.Client(timeout=30.0) as http:
            resp = http.get(f"{base_url}/videos/{video_id}", headers=headers)
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status", "").lower()
        if status == "completed":
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"Sora video failed: {data}")
        logger.debug("Sora video %s status: %s", video_id, status)
        time.sleep(5)
    else:
        raise TimeoutError(f"Sora video {video_id} did not complete within 600s")

    # Step 3: Download content
    with httpx.Client(timeout=30.0) as http:
        resp = http.get(f"{base_url}/videos/{video_id}/content", headers=headers)
        resp.raise_for_status()
        content_data = resp.json()

    # Get download URL from content response
    video_url = None
    if isinstance(content_data, dict):
        video_url = content_data.get("url") or content_data.get("download_url")
        if not video_url and "data" in content_data:
            items = content_data["data"]
            if isinstance(items, list) and items:
                video_url = items[0].get("url") or items[0].get("download_url")
    if not video_url:
        raise RuntimeError(f"No download URL in Sora content response: {content_data}")

    return _download_file(video_url, video_path)


# ---------------------------------------------------------------------------
# xAI Grok Imagine video generation
# ---------------------------------------------------------------------------

def _generate_grok_video(prompt: str, duration: int, video_path: Path, engine_cfg: dict) -> Path:
    """Generate video via xAI Grok Imagine API."""
    config.check_api_key("XAI_API_KEY")

    prompt = _truncate_prompt(prompt, engine_cfg.get("max_prompt_chars", 4096))
    model = engine_cfg["model"]
    max_dur = engine_cfg.get("max_duration_sec", 8)
    duration = min(duration, max_dur)
    aspect_ratio = engine_cfg.get("aspect_ratio", "16:9")
    resolution = engine_cfg.get("resolution", "720p")

    base_url = "https://api.x.ai/v1"
    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Submit
    body = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    logger.info("Grok: model=%s, duration=%d, prompt=%d chars", model, duration, len(prompt))

    with httpx.Client(timeout=60.0) as http:
        resp = http.post(f"{base_url}/videos/generations", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    request_id = data.get("request_id") or data.get("id")
    if not request_id:
        raise RuntimeError(f"xAI returned no request_id: {data}")
    logger.info("Grok video %s created, polling...", request_id)

    # Poll — xAI returns video.url when done, NOT a status field
    deadline = time.time() + 300
    while time.time() < deadline:
        with httpx.Client(timeout=30.0) as http:
            resp = http.get(f"{base_url}/videos/{request_id}", headers=headers)
            resp.raise_for_status()
            data = resp.json()

        video_obj = data.get("video")
        if isinstance(video_obj, dict) and video_obj.get("url"):
            logger.info("Grok video %s completed.", request_id)
            return _download_file(video_obj["url"], video_path)

        status = data.get("status", "").lower()
        if status in ("failed", "error"):
            raise RuntimeError(f"Grok video generation failed: {data}")
        time.sleep(5)

    raise TimeoutError(f"Grok video {request_id} did not complete within 300s")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PROVIDER_FN = {
    "runway": _generate_runway_video,
    "openai": _generate_sora_video,
    "xai": _generate_grok_video,
}


def generate_single_scene(scene: dict, output_dir, engine: str = None) -> dict:
    """Generate video for a single scene using the specified engine.

    Args:
        scene: Scene dict with scene_number, visual_prompt, duration_sec.
        output_dir: Directory to store the video file.
        engine: Engine key from ENGINE_CONFIG. Defaults to config.DEFAULT_ENGINE.

    Returns:
        Dict with scene_number, video_path, status, duration, engine.
    """
    engine = engine or config.DEFAULT_ENGINE
    engine_cfg = config.get_engine_config(engine)
    provider = engine_cfg["provider"]
    gen_fn = _PROVIDER_FN.get(provider)
    if gen_fn is None:
        raise ValueError(f"No generator for provider: {provider}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_num = scene.get("scene_number", 0)
    visual_prompt = scene.get("visual_prompt", "")
    duration = scene.get("duration_sec", config.SCENE_DURATION_SEC)

    if not visual_prompt:
        raise ValueError(f"Scene {scene_num} has no visual_prompt.")

    video_path = output_dir / f"scene_{scene_num:03d}.mp4"

    last_error: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            logger.info(
                "Scene %d: engine=%s (%s), attempt %d/%d, %d sec...",
                scene_num, engine, engine_cfg["display_name"],
                attempt, config.MAX_RETRIES, duration,
            )

            gen_fn(visual_prompt, duration, video_path, engine_cfg)
            logger.info("Scene %d: video saved to %s", scene_num, video_path)

            return {
                "scene_number": scene_num,
                "video_path": str(video_path),
                "status": "success",
                "duration": duration,
                "engine": engine,
            }

        except Exception as exc:
            if _is_credit_error(exc):
                raise RuntimeError(
                    f"Credit exhaustion ({engine}) at scene {scene_num}: {exc}"
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
        "engine": engine,
    }


def generate_videos(scenes_data: dict, output_dir, engine: str = None) -> dict:
    """Generate videos for all scenes using the specified engine.

    Returns:
        Dict with generated_scenes list (assembler-compatible), counts, status.
    """
    engine = engine or config.DEFAULT_ENGINE
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_data.get("scenes", [])
    if not scenes:
        raise ValueError("scenes_data must contain a non-empty 'scenes' list.")

    engine_cfg = config.get_engine_config(engine)
    logger.info(
        "Starting video generation for %d scene(s) with %s (%s)...",
        len(scenes), engine, engine_cfg["display_name"],
    )

    scene_results: list[dict] = []
    successful = 0
    failed = 0

    for i, scene in enumerate(scenes):
        scene_num = scene.get("scene_number", i + 1)
        logger.info("Processing scene %d/%d (scene_number=%d)...", i + 1, len(scenes), scene_num)

        try:
            result = generate_single_scene(scene, videos_dir, engine=engine)
        except RuntimeError as exc:
            if "credit" in str(exc).lower():
                logger.error("Credit exhaustion at scene %d — stopping.", scene_num)
                scene_results.append({
                    "scene_number": scene_num,
                    "video_path": None,
                    "error": str(exc),
                    "engine": engine,
                })
                failed += 1

                output = {
                    "generated_scenes": scene_results,
                    "total_scenes": len(scenes),
                    "successful": successful,
                    "failed": failed,
                    "status": "credit_exhausted",
                    "engine": engine,
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
        "Video generation complete: %d/%d successful. Engine: %s, Status: %s",
        successful, len(scenes), engine, status,
    )

    output = {
        "generated_scenes": scene_results,
        "total_scenes": len(scenes),
        "successful": successful,
        "failed": failed,
        "status": status,
        "engine": engine,
    }

    step_file = output_dir / "step5_videos.json"
    step_file.write_text(json.dumps(output, indent=2))
    logger.info("Video step result saved to %s", step_file)

    return output
