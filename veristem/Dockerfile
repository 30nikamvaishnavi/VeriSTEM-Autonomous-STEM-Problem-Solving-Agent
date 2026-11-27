FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd -r veristem && useradd -r -g veristem veristem

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY datasets/ ./datasets/
COPY benchmarks/ ./benchmarks/
COPY scripts/ ./scripts/
COPY pyproject.toml .

RUN mkdir -p /app/data && chown -R veristem:veristem /app

USER veristem

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
