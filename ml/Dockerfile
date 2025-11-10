FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Copy application
COPY anomaly_detector.py .

# Set default environment variables
ENV ES_HOST=elasticsearch
ENV ES_PORT=9200
ENV DETECTION_INTERVAL=60
ENV LOG_LEVEL=INFO

# Run detector with unbuffered output
CMD ["python", "-u", "anomaly_detector.py"]
