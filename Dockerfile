FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MODEL_DIR=/models HOME=/tmp
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-service.txt ./requirements-service.txt
RUN python -m pip install --no-cache-dir -r requirements-service.txt
COPY src/__init__.py src/poi_baseline.py src/frozen_encoders.py src/poi_inference.py ./src/
COPY service/ ./service/
RUN groupadd --gid 10001 poi && useradd --uid 10001 --gid 10001 --create-home poi
# Bundle внутри образа устраняет зависимость запуска от Windows bind mounts.
COPY models/poi_service_dinov2/ /models/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=3).read()"]
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5"]
