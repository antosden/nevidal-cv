FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN python -m pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-mobile-export.txt /tmp/requirements-mobile-export.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-mobile-export.txt && python -m pip check && python -m pip freeze > /opt/pip-freeze.txt
WORKDIR /app
CMD ["python", "scripts/export_mobile_litert.py", "--work", "/work/input", "--output", "/work/output"]
