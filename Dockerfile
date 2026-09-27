FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir '.[cpu,server]'

RUN mkdir -p /models/huggingface && chown -R app:app /models /app
USER app

EXPOSE 8000
VOLUME ["/models/huggingface"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"

CMD ["uvicorn", "moxhimt_runner.server_cpu:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--limit-concurrency", "32", "--timeout-keep-alive", "5"]
