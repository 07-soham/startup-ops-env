# Dockerfile for StartupOps AI Simulator - HF Spaces
# Runs the unified Gradio UI + FastAPI OpenEnv API on port 7860

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY env/ ./env/
COPY agents/ ./agents/
COPY configs/ ./configs/
COPY server/ ./server/
COPY api.py .
COPY app.py .
COPY main.py .
COPY inference.py .
COPY openenv.yaml .

# Expose port 7860 for HF Spaces
EXPOSE 7860

# Run the unified Gradio UI + FastAPI ASGI app on port 7860
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
