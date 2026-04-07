# ── Stage 1: Base image ────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="telecom-analytics"
LABEL description="Telecom Analytics Platform – Streamlit Dashboard + ML Pipeline"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        default-libmysqlclient-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Dependencies ──────────────────────────────────────────────────────
FROM base AS deps

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application ───────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy source code
COPY . .

# Create required directories
RUN mkdir -p feature_store models_artifacts reports data

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Expose ports
EXPOSE 8501
# Streamlit
EXPOSE 5000
# MLflow UI

# ── Entrypoint: Streamlit dashboard by default ─────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
