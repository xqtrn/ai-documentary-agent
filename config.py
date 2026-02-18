import os

# API Keys — Only Runway + Anthropic needed
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RUNWAY_API_KEY = os.environ.get("RUNWAY_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_USERNAME = os.environ.get("TELEGRAM_BOT_USERNAME", "ai_documentary_bot")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Models — ALWAYS use the best available
CLAUDE_MODEL = "claude-opus-4-6"
RUNWAY_VIDEO_MODEL = "gen4.5"                    # Best quality: 12 credits/sec
RUNWAY_TTS_MODEL = "eleven_multilingual_v2"      # via Runway API: 1 credit/50 chars
RUNWAY_MUSIC_MODEL = "eleven_text_to_sound_v2"   # via Runway API: 1 credit/6 sec
RUNWAY_BASE_URL = "https://api.dev.runwayml.com/v1"

# TTS voice — natural documentary narrator
RUNWAY_TTS_VOICE = "Mark"

# Scene config — quality over quantity
TARGET_DURATION_MIN = 0.5  # 30 sec
TARGET_DURATION_MAX = 1    # 60 sec
SCENE_DURATION_SEC = 10
SCENES_COUNT_MIN = 3
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
