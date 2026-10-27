# Production-serving image for the EfficientNetB3 brain-tumor classifier.
# CPU-only by default (works anywhere); swap the base image for an NVIDIA CUDA
# runtime to enable GPU. Python 3.11 is required by TensorFlow 2.15.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BTC_DATA_DIR=/data

# System libs needed by opencv / pillow / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source + package install.
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .

# The deployed model (weights + fitted temperature) is baked in; mount a
# different models/ directory to serve another one:
#   docker run -v $(pwd)/models:/app/models -p 8000:8000 neuroclass-api
COPY models ./models

# Run as a non-root user: if the service or one of its dependencies is ever
# compromised, this limits what the resulting process can touch on the host
# (no root inside the container). /app stays root-owned and read-only to it —
# the API never writes there — which also avoids a `chown -R` layer that would
# duplicate the ~95 MB of model files in the image.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "btc.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
