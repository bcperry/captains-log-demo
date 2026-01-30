# Stage 1: Build React frontend
FROM node:22-slim AS frontend-builder

WORKDIR /frontend

# Build arguments for Vite environment variables (baked into JS bundle at build time)
ARG VITE_AZURE_CLIENT_ID=""
ARG VITE_AZURE_TENANT_ID=""
ARG VITE_AZURE_CLOUD="government"

# Convert build args to environment variables for npm run build
ENV VITE_AZURE_CLIENT_ID=${VITE_AZURE_CLIENT_ID}
ENV VITE_AZURE_TENANT_ID=${VITE_AZURE_TENANT_ID}
ENV VITE_AZURE_CLOUD=${VITE_AZURE_CLOUD}

# Copy package files for dependency installation
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source files
COPY frontend/ .

# Build production bundle with VITE_ env vars baked in
RUN npm run build

# Stage 2: Build Python application
FROM python:3.11-slim

# Install ffmpeg and other system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY app/pyproject.toml app/uv.lock ./

# Copy README.md referenced by pyproject.toml
COPY README.md ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev --no-editable

# Copy application code
COPY app/ .

# Copy React build artifacts from frontend-builder stage
COPY --from=frontend-builder /frontend/dist ./static

# Expose port 8001 for FastAPI
EXPOSE 8001

# Run FastAPI application using uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
