# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=mlruns

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and configs
COPY configs/ ./configs/
COPY src/ ./src/
COPY tests/ ./tests/
COPY Makefile ./

# Expose MLflow tracking server default port
EXPOSE 5000

# Default command: run unit tests, then training
CMD ["python", "src/train.py", "--config", "configs/config.yaml"]

