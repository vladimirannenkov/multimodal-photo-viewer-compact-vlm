# syntax=docker/dockerfile:1
# Сборка под нужную видеокарту:  docker build --build-arg CUDA_VARIANT=cu128 -t imagehub .
#   cpu   - без видеокарты
#   cu126 - драйвер с поддержкой CUDA 12.6 (Pascal ... Ada)
#   cu128 - CUDA 12.8 (нужен для Blackwell, RTX 50xx)
# Колёса PyTorch уже содержат нужные библиотеки CUDA. На хосте нужны драйвер NVIDIA и NVIDIA Container Toolkit.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ARG CUDA_VARIANT=cu128
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    IMAGEHUB_DATA_DIR=/data IMAGEHUB_WEIGHTS_DIR=/weights IMAGEHUB_HOST=0.0.0.0 IMAGEHUB_PICTURES_DIR=/data/pictures

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libheif1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 1) torch нужного варианта (отдельным слоем: меняется редко)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VARIANT}
# 2) зависимости проекта и сам пакет
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

VOLUME ["/data", "/weights"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8765/api/queue')" || exit 1
ENTRYPOINT ["imagehub"]
CMD ["serve"]
