# Use Python 3.11 slim image
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

# Expose port 8000 for Streamlit
EXPOSE 8000

# Run Streamlit application using uv
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0", "--server.runOnSave=false"]
