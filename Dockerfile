FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .

# 👇 Fix: explicit index + retries + timeout
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt \
    -i https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --retries 10 \
    --timeout 1000

COPY app.py .
COPY src/ src/
COPY openenv.yaml .
COPY README.md .
COPY pyproject.toml .
COPY .env.example .

ENV PYTHONPATH=/app:/app/src

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD python -c "import requests; r=requests.get('http://localhost:7860/health'); assert r.status_code==200"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]