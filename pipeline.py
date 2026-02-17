#!/usr/bin/env python3
"""AI Documentary Agent - Main Pipeline CLI."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import config
from scripts.source_analyzer import analyze_source
from scripts.virality_analyzer import analyze_virality
from scripts.scriptwriter import rewrite_script
from scripts.scene_splitter import split_into_scenes
from scripts.video_gen import generate_videos
from scripts.voiceover import generate_audio
from scripts.music_gen import generate_music
from scripts.assembler import assemble_video
from scripts.metadata import generate_all_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

STEPS = [
    "source", "virality", "script", "scenes",
    "video", "audio", "music", "assembly", "metadata",
]


def get_output_dir(url: str) -> Path:
    """Create output directory based on video ID."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    slug = match.group(1) if match else "unknown"
    output_dir = Path(config.OUTPUT_DIR) / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_checkpoint(output_dir: Path) -> str | None:
    """Find the last completed step."""
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        data = json.loads(checkpoint_file.read_text())
        return data.get("last_step")
    return None


def save_checkpoint(output_dir: Path, step: str):
    """Save checkpoint after completing a step."""
    checkpoint_file = output_dir / "checkpoint.json"
    data = {"last_step": step}
    checkpoint_file.write_text(json.dumps(data))


def load_step_data(output_dir: Path, step_file: str) -> dict:
    """Load data from a previous step's checkpoint file."""
    path = output_dir / step_file
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return json.loads(path.read_text())


def run_pipeline(url: str, start_from: str | None = None):
    """Run the full pipeline."""
    output_dir = get_output_dir(url)
    logger.info("Output directory: %s", output_dir)

    # Determine starting step
    last_step = get_checkpoint(output_dir)
    if start_from:
        start_idx = STEPS.index(start_from)
    elif last_step:
        start_idx = STEPS.index(last_step) + 1
        logger.info("Resuming from after step: %s", last_step)
    else:
        start_idx = 0

    # Step 1: Source Analysis
    if start_idx <= 0:
        logger.info("=" * 60)
        logger.info("STEP 1: Source Analysis")
        source_data = analyze_source(url, output_dir)
        save_checkpoint(output_dir, "source")
    else:
        source_data = load_step_data(output_dir, "step1_source.json")

    # Step 2: Virality Analysis
    if start_idx <= 1:
        logger.info("=" * 60)
        logger.info("STEP 2: Virality Analysis")
        virality_data = analyze_virality(source_data, output_dir)
        save_checkpoint(output_dir, "virality")
    else:
        virality_data = load_step_data(output_dir, "step2_virality.json")

    # Step 3: Script Rewrite
    if start_idx <= 2:
        logger.info("=" * 60)
        logger.info("STEP 3: Script Rewrite")
        script_data = rewrite_script(source_data, virality_data, output_dir)
        save_checkpoint(output_dir, "script")
    else:
        script_data = load_step_data(output_dir, "step3_script.json")

    # Step 4: Scene Splitting
    if start_idx <= 3:
        logger.info("=" * 60)
        logger.info("STEP 4: Scene Splitting")
        scenes_data = split_into_scenes(script_data, output_dir)
        save_checkpoint(output_dir, "scenes")
    else:
        scenes_data = load_step_data(output_dir, "step4_scenes.json")

    # Step 5: Video Generation
    if start_idx <= 4:
        logger.info("=" * 60)
        logger.info("STEP 5: Video Generation")
        video_data = generate_videos(scenes_data, output_dir)
        save_checkpoint(output_dir, "video")
    else:
        video_data = load_step_data(output_dir, "step5_videos.json")

    # Step 6: Voiceover + SFX
    if start_idx <= 5:
        logger.info("=" * 60)
        logger.info("STEP 6: Voiceover + SFX")
        audio_data = generate_audio(script_data, scenes_data, output_dir)
        save_checkpoint(output_dir, "audio")
    else:
        audio_data = load_step_data(output_dir, "step6_audio.json")

    # Step 7: Background Music
    if start_idx <= 6:
        logger.info("=" * 60)
        logger.info("STEP 7: Background Music")
        total_duration = scenes_data.get("total_duration_sec", 1200)
        music_data = generate_music(total_duration, output_dir)
        save_checkpoint(output_dir, "music")
    else:
        music_data = load_step_data(output_dir, "step7_music.json")

    # Step 8: Assembly
    if start_idx <= 7:
        logger.info("=" * 60)
        logger.info("STEP 8: Final Assembly")
        assembly_data = assemble_video(scenes_data, video_data, audio_data, music_data, output_dir)
        save_checkpoint(output_dir, "assembly")

    # Metadata (runs in parallel conceptually, after script+scenes)
    if start_idx <= 8:
        logger.info("=" * 60)
        logger.info("STEP 9: Metadata Generation")
        meta_data = generate_all_metadata(source_data, script_data, scenes_data, output_dir)
        save_checkpoint(output_dir, "metadata")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("Output: %s", output_dir)
    logger.info("Final video: %s/final_video.mp4", output_dir)

    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="AI Documentary Agent Pipeline")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--start-from", choices=STEPS, help="Resume from a specific step")
    args = parser.parse_args()

    try:
        output = run_pipeline(args.url, args.start_from)
        print(f"\nDone! Output: {output}")
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
