# Production Dockerfile for EKM (Episodic Knowledge Mesh)
# Multi-stage build for smaller image size

# Stage 1: Builder stage for compiling dependencies
FROM python:3.11-slim AS builder

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install dependencies (including dev dependencies for now)
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production stage
FROM python:3.11-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 ekm && mkdir -p /app && chown -R ekm:ekm /app
USER ekm
WORKDIR /app

# Copy application source code
COPY --chown=ekm:ekm . .

# Create data directory for SQLite database
RUN mkdir -p /app/data && chown ekm:ekm /app/data

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV EKM_DB_URL=sqlite:///data/ekm.db
ENV GEMINI_API_KEY=""
ENV OPENAI_API_KEY=""
ENV EKM_LOG_LEVEL=INFO

# Expose API port (for research assistant server)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the research assistant server
CMD ["uvicorn", "examples.research_assistant.server:app", "--host", "0.0.0.0", "--port", "8000"]