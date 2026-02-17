import os

# API Keys - use .get() so missing keys don't crash on import
# They'll fail gracefully at the step that needs them
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RUNWAY_API_KEY = os.environ.get("RUNWAY_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
BEATOVEN_API_KEY = os.environ.get("BEATOVEN_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_USERNAME = os.environ.get("TELEGRAM_BOT_USERNAME", "ai_documentary_bot")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
CLAUDE_MODEL = "claude-opus-4-6"
ELEVENLABS_VOICE = "Adam"
ELEVENLABS_MODEL = "eleven_multilingual_v2"

TARGET_DURATION_MIN = 18
TARGET_DURATION_MAX = 22
SCENE_DURATION_SEC = 10
SCENES_COUNT_MIN = 120
SCENES_COUNT_MAX = 150

RUNWAY_IMAGE_MODEL = "gen4_image"
RUNWAY_VIDEO_MODEL = "gen4_turbo"
RUNWAY_BASE_URL = "https://api.dev.runwayml.com/v1"

BEATOVEN_BASE_URL = "https://api.beatoven.ai/api/v2"

MAX_RETRIES = 3
BATCH_SIZE = 10
WEB_PORT = int(os.environ.get("PORT", "8080"))


def check_api_key(key_name: str) -> str:
    """Check that an API key is set, raise clear error if not."""
    val = globals().get(key_name, "")
    if not val:
        raise RuntimeError(
            f"{key_name} is not set! "
            f"Please add it to your Railway environment variables. "
            f"Go to https://railway.app and set {key_name} in your project settings."
        )
    return val
