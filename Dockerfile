# syntax=docker/dockerfile:1.4

# Stage 1: Development
FROM python:3.10-slim as dev

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    DEBIAN_FRONTEND=noninteractive

# System deps
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        git \
        libpq-dev \
        procps \
        htop \
        vim \
        less \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Copy only requirements to cache them in docker layer
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./
COPY requirements.txt ./

# Project initialization
RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir -r requirements.txt

# Development deps and tools
RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir \
        black \
        flake8 \
        mypy \
        pytest \
        pytest-cov \
        ipython \
        debugpy

# Copy project
COPY . .

# Create non-root user
RUN useradd -m -u 1000 developer \
    && chown -R developer:developer $PYSETUP_PATH

USER developer

# Stage 2: Builder
FROM python:3.10-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir -r requirements.txt \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 3: Production
FROM python:3.10-slim as prod

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /app/venv

# Copy wheels from builder
COPY --from=builder /app/wheels /app/wheels

# Install dependencies
RUN /app/venv/bin/pip install --no-cache /app/wheels/*

# Copy project
COPY . .

# Create non-root user
RUN useradd -m -u 1000 app \
    && chown -R app:app /app

USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Command to run the application
CMD ["python", "main.py"] 