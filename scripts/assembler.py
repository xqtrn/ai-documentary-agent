"""Step 8: Assemble final video using FFmpeg."""

import json
import logging
import subprocess
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def get_video_duration(path: str) -> float:
    """Get duration of a video/audio file using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def assemble_video(
    scenes_data: dict,
    video_data: dict,
    audio_data: dict,
    music_data: dict,
    output_dir: Path,
) -> dict:
    """Assemble final MP4 from all components."""
    logger.info("Assembling final video...")

    videos_dir = output_dir / "videos"
    final_path = output_dir / "final_video.mp4"

    # Get list of successfully generated video clips
    generated = [s for s in video_data["generated_scenes"] if "error" not in s]
    generated.sort(key=lambda x: x["scene_number"])

    if not generated:
        raise RuntimeError("No video clips were generated successfully")

    # Step 1: Create concat list for video clips with crossfade
    video_paths = [s["video_path"] for s in generated]

    # Build complex FFmpeg filter for crossfade transitions
    crossfade_duration = 0.5

    if len(video_paths) == 1:
        # Single clip, just copy
        concat_video = video_paths[0]
    else:
        # Create crossfade chain
        concat_video = str(output_dir / "concat_video.mp4")
        _concat_with_crossfade(video_paths, concat_video, crossfade_duration)

    # Step 2: Get paths
    voiceover_path = audio_data["voiceover_path"]
    music_path = music_data["music_path"]

    # Step 3: Mix audio - voiceover at full volume, music at -15dB
    mixed_audio = str(output_dir / "mixed_audio.mp3")
    _mix_audio(voiceover_path, music_path, audio_data.get("sfx", []), mixed_audio, output_dir)

    # Step 4: Combine video + mixed audio
    voiceover_duration = get_video_duration(voiceover_path)
    video_duration = get_video_duration(concat_video)

    # Use the shorter of voiceover or video as final duration
    target_duration = min(voiceover_duration, video_duration)

    cmd = [
        "ffmpeg", "-y",
        "-i", concat_video,
        "-i", mixed_audio,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-r", "24",
        "-s", "1920x1080",
        "-t", str(target_duration),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(final_path),
    ]

    logger.info("Running final FFmpeg assembly...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error("FFmpeg error: %s", result.stderr)
        raise RuntimeError(f"FFmpeg assembly failed: {result.stderr}")

    final_duration = get_video_duration(str(final_path))
    file_size_mb = final_path.stat().st_size / (1024 * 1024)

    assembly_result = {
        "final_video": str(final_path),
        "duration_sec": final_duration,
        "file_size_mb": round(file_size_mb, 1),
        "resolution": "1920x1080",
        "fps": 24,
        "clips_used": len(generated),
    }

    with open(output_dir / "step8_assembly.json", "w") as f:
        json.dump(assembly_result, f, ensure_ascii=False, indent=2)

    # Clean up intermediate files
    for f in [output_dir / "concat_video.mp4", output_dir / "mixed_audio.mp3"]:
        if f.exists():
            f.unlink()

    logger.info("Final video assembled: %s (%.1f MB, %.0fs)", final_path, file_size_mb, final_duration)
    return assembly_result


def _concat_with_crossfade(video_paths: list, output: str, crossfade_sec: float):
    """Concatenate videos with crossfade transitions."""
    # For many clips, use simple concat (crossfade filter graph gets too complex)
    if len(video_paths) > 20:
        _simple_concat(video_paths, output)
        return

    # Build FFmpeg filter graph for crossfade
    inputs = []
    for i, p in enumerate(video_paths):
        inputs.extend(["-i", p])

    # Build xfade filter chain
    filter_parts = []
    current = "[0:v]"
    for i in range(1, len(video_paths)):
        next_input = f"[{i}:v]"
        if i < len(video_paths) - 1:
            out = f"[v{i}]"
        else:
            out = "[vout]"

        offset = sum(
            _get_clip_duration(video_paths[j]) - crossfade_sec
            for j in range(i)
        )
        offset = max(0, offset)

        filter_parts.append(
            f"{current}{next_input}xfade=transition=fade:duration={crossfade_sec}:offset={offset}{out}"
        )
        current = out

    filter_graph = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_graph,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", "24",
        output,
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)


def _simple_concat(video_paths: list, output: str):
    """Simple concatenation without crossfade for large numbers of clips."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", "24",
        output,
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
    Path(list_path).unlink(missing_ok=True)


def _get_clip_duration(path: str) -> float:
    """Get clip duration, default 10s on error."""
    try:
        return get_video_duration(path)
    except Exception:
        return 10.0


def _mix_audio(voiceover: str, music: str, sfx_list: list, output: str, output_dir: Path):
    """Mix voiceover (full volume) + music (-15dB) + SFX."""
    # Start with voiceover + music
    inputs = ["-i", voiceover, "-i", music]
    filter_parts = [
        "[0:a]volume=1.0[voice]",
        "[1:a]volume=0.18[music]",  # -15dB ~ 0.18
    ]

    if sfx_list:
        # Add SFX inputs
        for i, sfx in enumerate(sfx_list):
            inputs.extend(["-i", sfx["path"]])
            sfx_idx = i + 2
            filter_parts.append(f"[{sfx_idx}:a]volume=0.5[sfx{i}]")

        # Mix all together
        mix_inputs = "[voice][music]" + "".join(f"[sfx{i}]" for i in range(len(sfx_list)))
        filter_parts.append(f"{mix_inputs}amix=inputs={2 + len(sfx_list)}:duration=first[out]")
    else:
        filter_parts.append("[voice][music]amix=inputs=2:duration=first[out]")

    filter_graph = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", "192k",
        output,
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
