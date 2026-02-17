FROM python:3.11-slim

# Install ffmpeg, tor proxy, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    tor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pysocks requests

COPY . .

# Create output and data directories
RUN mkdir -p /app/output /app/data

EXPOSE 8080

# Start Tor in background, then run the app
CMD tor --RunAsDaemon 1 --SocksPort 9050 && sleep 3 && python bot.py
