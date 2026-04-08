FROM python:3.10-slim

# Create a non-root user specifically for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

# A dummy web server on port 7860. This keeps the Hugging Face Space "Running" indefinitely,
# serves the 200 OK health check so you don't get a 504 Gateway Timeout,
# and allows the OpenEnv hackathon bot to evaluate the container!
CMD ["python", "-m", "http.server", "7860"]
