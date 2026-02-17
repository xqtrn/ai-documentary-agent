FROM python:3.11-slim

# Install ffmpeg and yt-dlp dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp binary (latest)
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache bust: change this value to force a fresh COPY
ARG CACHEBUST=2
COPY . .

# Create output and data directories
RUN mkdir -p /app/output /app/data

EXPOSE 8080

CMD ["python", "bot.py"]
