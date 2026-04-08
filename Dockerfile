# Dockerfile for StartupOps AI Simulator - HF Spaces
# Uses FastAPI for OpenEnv API endpoints

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY env/ ./env/
COPY agents/ ./agents/
COPY configs/ ./configs/
COPY api.py .
COPY app.py .
COPY main.py .
COPY inference.py .
COPY openenv.yaml .

# Expose port 7860 for HF Spaces
EXPOSE 7860

# Run FastAPI app on port 7860 (HF Spaces requirement)
CMD ["python", "-c", "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=7860)"]
