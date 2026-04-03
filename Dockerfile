FROM python:3.11-slim

# Metadata
LABEL maintainer="StartupOps Team"
LABEL description="StartupOpsEnv — Startup Operations RL Environment"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Hugging Face Spaces listens on 7860
EXPOSE 7860

# Gradio auto-detects HF environment and binds to 0.0.0.0:7860
CMD ["python", "app.py"]
