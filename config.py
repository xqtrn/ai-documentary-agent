import os

# API Keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RUNWAY_API_KEY = os.environ.get("RUNWAY_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_USERNAME = os.environ.get("TELEGRAM_BOT_USERNAME", "ai_documentary_bot")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Models
CLAUDE_MODEL = "claude-opus-4-6"

# VIDEO — Grok Imagine (xAI)
XAI_VIDEO_MODEL = "grok-imagine-video"
XAI_VIDEO_BASE_URL = "https://api.x.ai/v1"
XAI_VIDEO_DURATION = 8
XAI_VIDEO_ASPECT_RATIO = "16:9"
XAI_VIDEO_RESOLUTION = "720p"

# VOICEOVER — ElevenLabs via Runway API
RUNWAY_TTS_MODEL = "eleven_multilingual_v2"
RUNWAY_TTS_VOICE = "James"
RUNWAY_BASE_URL = "https://api.dev.runwayml.com/v1"
RUNWAY_API_VERSION = "2024-11-06"

# MUSIC & SFX — Runway Sound Effects
RUNWAY_MUSIC_MODEL = "eleven_text_to_sound_v2"

# Scene config
TARGET_DURATION_MIN = 0.5
TARGET_DURATION_MAX = 1
SCENE_DURATION_SEC = 8
SCENES_COUNT_MIN = 4
SCENES_COUNT_MAX = 4

MAX_RETRIES = 3
BATCH_SIZE = 10
WEB_PORT = int(os.environ.get("PORT", "8080"))
TEST_SECRET = os.environ.get("RUN_SECRET", "test_pipeline_2026")


def check_api_key(key_name: str) -> str:
    val = globals().get(key_name, "")
    if not val:
        raise RuntimeError(
            f"{key_name} is not set! Add it to Railway environment variables."
        )
    return val
