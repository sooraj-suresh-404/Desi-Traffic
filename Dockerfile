FROM python:3.10-slim

# Create a non-root user specifically for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install required packages including server dependencies
RUN pip install --no-cache-dir gymnasium pydantic openai stable-baselines3 pygame fastapi uvicorn "openenv[server]"

# Copy all files into the container
COPY --chown=user:user . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the OpenEnv server
CMD ["python", "server.py"]
