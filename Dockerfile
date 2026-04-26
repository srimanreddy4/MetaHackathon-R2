# Multi-stage build
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

ARG BUILD_MODE=in-repo
ARG ENV_NAME=app

COPY . /app/env

WORKDIR /app/env

# Install dependencies using pip (preserving original index + retry settings)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt \
    -i https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --retries 10 \
    --timeout 1000

# Final runtime stage
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the environment code
COPY --from=builder /app/env /app/env

WORKDIR /app/env

ENV PYTHONPATH="/app/env:/app/env/src:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:7860/health'); assert r.status_code==200"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]