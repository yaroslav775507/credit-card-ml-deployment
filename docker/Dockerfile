FROM python:3.12-slim

# Metadata
LABEL maintainer="ml-engineer@bank.example.com"
LABEL description="Credit card default prediction service"

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained models
COPY ../app ./app/
COPY ../models ./models/

# Non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Expose service port
EXPOSE 5000

# Health check (Docker will poll this)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Start Flask application
CMD ["python", "app/api.py"]
