FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env

RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir openenv-core>=0.2.0 fastapi>=0.104.0 uvicorn>=0.24.0 pydantic>=2.5.0 requests>=2.31.0 pyyaml>=6.0.0

ENV PYTHONPATH="/app/env:/app/env/src:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
