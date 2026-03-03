FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation fonts-dejavu-core fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ src/
COPY samples/ samples/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e ".[formats,dev]"

CMD ["python", "scripts/test_pipeline.py"]
