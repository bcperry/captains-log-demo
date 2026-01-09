# Stage 1: Build React frontend
FROM node:22-slim AS frontend-builder

WORKDIR /frontend

# Copy package files for dependency installation
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source files
COPY frontend/ .

# Build production bundle
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
