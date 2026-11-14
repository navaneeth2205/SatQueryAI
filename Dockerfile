# syntax=docker/dockerfile:1.4
# SatQuery AI Production Multi-Architecture Dockerfile
# Supports both CPU and CUDA GPU acceleration profiles

ARG TARGET_ENV=cpu

# ==========================================
# Base Environment
# ==========================================
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SATQUERY_HOME=/app

WORKDIR /app

# Install minimal OS dependencies for geospatial and scientific computation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY requirements.txt pyproject.toml /app/

# Create non-root application user for container security
RUN useradd -m -u 1000 -s /bin/bash satquery && \
    mkdir -p /app/artifacts /app/data && \
    chown -R satquery:satquery /app

# ==========================================
# CPU Build Stage
# ==========================================
FROM base AS build-cpu
USER satquery
RUN pip install --user --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --user --no-cache-dir -r requirements.txt

# ==========================================
# CUDA Build Stage
# ==========================================
FROM base AS build-cuda
USER satquery
RUN pip install --user --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --user --no-cache-dir -r requirements.txt

# ==========================================
# Final Production Runtime Stage
# ==========================================
FROM build-${TARGET_ENV} AS final

ENV PATH="/home/satquery/.local/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}"

WORKDIR /app

# Copy application source code and frozen production artifacts
COPY --chown=satquery:satquery satquery/ /app/satquery/
COPY --chown=satquery:satquery artifacts/ /app/artifacts/
COPY --chown=satquery:satquery data/ /app/data/
COPY --chown=satquery:satquery README.md /app/

# Install satquery in editable/production mode
RUN pip install --user --no-deps -e .

# Expose web service port
EXPOSE 5000

# Healthcheck to verify service availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/catalog || exit 1

# Default entrypoint runs satquery CLI; default CMD starts interactive web service
ENTRYPOINT ["python", "-m", "satquery.cli"]
CMD ["demo", "--host", "0.0.0.0", "--port", "5000"]
