#!/usr/bin/env python3
"""Telegram bot + web server launcher for AI Documentary Agent."""

import logging
import multiprocessing
import os
import re
import traceback
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import config
from pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

YOUTUBE_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]+"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI Documentary Agent\n\n"
        "Send me a YouTube URL and I'll create an AI-generated documentary based on it.\n\n"
        "The pipeline:\n"
        "1. Download & analyze the original video\n"
        "2. Analyze what makes it viral\n"
        "3. Rewrite a completely new script\n"
        "4. Generate AI video scenes\n"
        "5. Add voiceover, music & SFX\n"
        "6. Assemble final MP4\n\n"
        "This process takes a while. I'll send updates as each step completes."
    )


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    match = YOUTUBE_PATTERN.search(text)

    if not match:
        await update.message.reply_text(
            "Please send a valid YouTube URL.\n"
            "Example: https://youtube.com/watch?v=XXXXX"
        )
        return

    url = match.group(0)
    if not url.startswith("http"):
        url = "https://" + url

    await update.message.reply_text(f"Starting pipeline for:\n{url}\n\nThis will take a while...")

    try:
        await update.message.reply_text("Step 1/8: Analyzing source video...")
        output_dir = run_pipeline(url)
        output_path = Path(output_dir)

        await update.message.reply_text("Pipeline complete! Sending files...")

        video_path = output_path / "final_video.mp4"
        if video_path.exists():
            file_size = video_path.stat().st_size
            if file_size < 50 * 1024 * 1024:
                await update.message.reply_video(
                    video=open(video_path, "rb"),
                    caption="AI Documentary - Final Video",
                    supports_streaming=True,
                )
            else:
                await update.message.reply_text(
                    f"Video is too large for Telegram ({file_size / 1024 / 1024:.0f}MB). "
                    f"File saved at: {video_path}"
                )

        meta_path = output_path / "metadata.json"
        if meta_path.exists():
            await update.message.reply_document(
                document=open(meta_path, "rb"),
                caption="YouTube Metadata (title, description, tags)",
            )

        srt_path = output_path / "subtitles.srt"
        if srt_path.exists():
            await update.message.reply_document(
                document=open(srt_path, "rb"),
                caption="Subtitles (SRT)",
            )

        analysis_path = output_path / "analysis.md"
        if analysis_path.exists():
            await update.message.reply_document(
                document=open(analysis_path, "rb"),
                caption="Virality Analysis",
            )

        await update.message.reply_text("All done! Files sent above.")

    except Exception as e:
        error_msg = f"Pipeline failed:\n{str(e)}\n\n{traceback.format_exc()[-500:]}"
        logger.exception("Pipeline error")
        await update.message.reply_text(error_msg[:4000])


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    output_base = Path(config.OUTPUT_DIR)
    if not output_base.exists():
        await update.message.reply_text("No outputs yet.")
        return

    dirs = sorted(output_base.iterdir())
    if not dirs:
        await update.message.reply_text("No outputs yet.")
        return

    status_lines = []
    import json
    for d in dirs[-5:]:
        checkpoint = d / "checkpoint.json"
        if checkpoint.exists():
            data = json.loads(checkpoint.read_text())
            step = data.get("last_step", "unknown")
            status_lines.append(f"{d.name}: step={step}")
        else:
            status_lines.append(f"{d.name}: not started")

    await update.message.reply_text("Recent outputs:\n" + "\n".join(status_lines))


def run_bot():
    """Run the Telegram bot (blocking)."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        import time
        while True:
            logger.info("Waiting for TELEGRAM_BOT_TOKEN to be configured...")
            time.sleep(60)

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))

    logger.info("Telegram bot started. Waiting for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


def run_web():
    """Run the FastAPI web server."""
    import uvicorn
    from web import app as web_app
    logger.info("Starting web server on port %d...", config.WEB_PORT)
    uvicorn.run(web_app, host="0.0.0.0", port=config.WEB_PORT, log_level="info")


def main():
    # Run both bot and web server as separate processes
    bot_proc = multiprocessing.Process(target=run_bot, name="telegram-bot")
    web_proc = multiprocessing.Process(target=run_web, name="web-server")

    bot_proc.start()
    web_proc.start()

    logger.info("Both processes started: bot (pid=%d), web (pid=%d)", bot_proc.pid, web_proc.pid)

    # Wait for either to exit
    bot_proc.join()
    web_proc.join()


if __name__ == "__main__":
    main()
