import os

# API Keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RUNWAY_API_KEY = os.environ.get("RUNWAY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_USERNAME = os.environ.get("TELEGRAM_BOT_USERNAME", "ai_documentary_bot")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Claude model
CLAUDE_MODEL = "claude-opus-4-6"

# TTS (via Runway)
RUNWAY_TTS_MODEL = "eleven_multilingual_v2"
# RUNWAY_TTS_VOICE = "Mark"
RUNWAY_BASE_URL = "https://api.dev.runwayml.com/v1"

# Voice Configuration (ElevenLabs via Runway)
ELEVENLABS_VOICES = {
    "george": {
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "name": "George",
        "description": "Deep, authoritative British male. Perfect for historical documentaries.",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    },
    "daniel": {
        "voice_id": "onwK4e9ZLuTAKqWW03F9",
        "name": "Daniel",
        "description": "Calm, clear British male. Great for educational content.",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    },
    "brian": {
        "voice_id": "nPczCjzI2devNBz1zQrb",
        "name": "Brian",
        "description": "Deep American male. Warm and engaging narration style.",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    },
    "james": {
        "voice_id": "ZQe5CZNOzWyzPSCn5a3c",
        "name": "James",
        "description": "Mature British male. Authoritative documentary narrator.",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    },
    "liam": {
        "voice_id": "TX3LPaxmHKxFdv7VOQHJ",
        "name": "Liam",
        "description": "Young, energetic male. Dynamic storytelling style.",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    },
}
DEFAULT_VOICE = "george"

# Music & SFX (via Runway)
RUNWAY_MUSIC_MODEL = "eleven_text_to_sound_v2"

# Video Engine Configuration
ENGINE_CONFIG = {
    "veo3.1": {
        "provider": "runway",
        "model": "veo3.1",
        "display_name": "Veo 3.1",
        "description": "Google's latest video model via Runway. Best cinematic realism with native audio.",
        "max_prompt_chars": 1000,
        "max_duration_sec": 8,
        "has_native_audio": True,
        "cost_per_sec": 40,
        "ratio": "1280:720",
    },
    "veo3.1_fast": {
        "provider": "runway",
        "model": "veo3.1_fast",
        "display_name": "Veo 3.1 Fast",
        "description": "Faster variant of Veo 3.1. Good balance of speed and quality, with native audio.",
        "max_prompt_chars": 1000,
        "max_duration_sec": 8,
        "has_native_audio": True,
        "cost_per_sec": 20,
        "ratio": "1280:720",
    },
    "sora-2-pro": {
        "provider": "openai",
        "model": "sora-2-pro",
        "display_name": "Sora 2 Pro",
        "description": "OpenAI's best video model. Ultra-high quality cinematic output.",
        "max_prompt_chars": None,
        "max_duration_sec": 12,
        "has_native_audio": False,
        "cost_per_sec": 0,
        "resolution": "1792x1024",
    },
    "sora-2": {
        "provider": "openai",
        "model": "sora-2",
        "display_name": "Sora 2",
        "description": "OpenAI's standard video model. Great quality, faster generation.",
        "max_prompt_chars": None,
        "max_duration_sec": 12,
        "has_native_audio": False,
        "cost_per_sec": 0,
        "resolution": "1280x720",
    },
    "gen4.5": {
        "provider": "runway",
        "model": "gen4.5",
        "display_name": "Gen-4.5",
        "description": "Runway's premium model. Excellent cinematic quality, 12 credits/sec.",
        "max_prompt_chars": 1000,
        "max_duration_sec": 10,
        "has_native_audio": False,
        "cost_per_sec": 12,
        "ratio": "1280:720",
    },
    "gen4_turbo": {
        "provider": "runway",
        "model": "gen4_turbo",
        "display_name": "Gen-4 Turbo",
        "description": "Fast Runway model. Image-to-video only (needs gen4_image first).",
        "max_prompt_chars": 1000,
        "max_duration_sec": 10,
        "has_native_audio": False,
        "cost_per_sec": 5,
        "ratio": "1280:720",
        "image_to_video_only": True,
    },
    "veo3": {
        "provider": "runway",
        "model": "veo3",
        "display_name": "Veo 3",
        "description": "Google's Veo 3 via Runway. Good quality with native audio generation.",
        "max_prompt_chars": 1000,
        "max_duration_sec": 8,
        "has_native_audio": True,
        "cost_per_sec": 10,
        "ratio": "1280:720",
    },
    "grok-imagine": {
        "provider": "xai",
        "model": "grok-imagine-video",
        "display_name": "Grok Imagine",
        "description": "xAI's video model. Supports ultra-long prompts (4K chars). Free tier available.",
        "max_prompt_chars": 4096,
        "max_duration_sec": 8,
        "has_native_audio": False,
        "cost_per_sec": 0,
        "resolution": "720p",
        "aspect_ratio": "16:9",
    },
}

DEFAULT_ENGINE = os.environ.get("DEFAULT_ENGINE", "veo3.1")

# Scene config
TARGET_DURATION_MIN = 0.5
TARGET_DURATION_MAX = 1
SCENE_DURATION_SEC = 8
SCENES_COUNT_MIN = 3
SCENES_COUNT_MAX = 4

MAX_RETRIES = 1
BATCH_SIZE = 10
WEB_PORT = int(os.environ.get("PORT", "8080"))
TEST_SECRET = os.environ.get("RUN_SECRET", "test_pipeline_2026")

# Enhanced Prompts V2 — year-anchored anti-hallucination (scene_splitter_v2)
USE_ENHANCED_PROMPTS = os.environ.get("USE_ENHANCED_PROMPTS", "false").lower() == "true"

# History
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.json")


def check_api_key(key_name: str) -> str:
    val = globals().get(key_name, "")
    if not val:
        raise RuntimeError(
            f"{key_name} is not set! Add it to Railway environment variables."
        )
    return val


def get_engine_config(engine: str) -> dict:
    """Get engine configuration, falling back to default."""
    if engine not in ENGINE_CONFIG:
        raise ValueError(f"Unknown engine: {engine}. Available: {list(ENGINE_CONFIG.keys())}")
    return ENGINE_CONFIG[engine]
