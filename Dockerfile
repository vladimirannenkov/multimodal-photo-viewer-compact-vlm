# syntax=docker/dockerfile:1
# Образ только для CPU: видеокарта и CUDA не нужны, PyTorch ставится в сборке для процессора.
# Сборка:  docker build -t imagehub:cpu .
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    IMAGEHUB_DATA_DIR=/data IMAGEHUB_WEIGHTS_DIR=/weights IMAGEHUB_HOST=0.0.0.0 IMAGEHUB_PICTURES_DIR=/data/pictures \
    IMAGEHUB_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libheif1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 1) torch для CPU (отдельным слоем: меняется редко)
RUN pip install --default-timeout=300 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 2) тяжёлые зависимости отдельными слоями с повторами: на нестабильном канале pip иногда получает повреждённый файл
#    (ошибка «hashes do not match»), тогда повторяется только этот слой, а не вся сборка
RUN for i in 1 2 3 4; do pip install --default-timeout=300 --retries 10 "transformers>=5.0" accelerate sentence-transformers timm einops sentencepiece huggingface_hub && break; done
RUN for i in 1 2 3 4; do pip install --default-timeout=300 --retries 10 "onnxruntime>=1.20" rapidocr-onnxruntime insightface opencv-python-headless && break; done
# 3) остальные зависимости проекта и сам пакет
COPY pyproject.toml README.md ./
COPY src ./src
RUN for i in 1 2 3 4; do pip install --default-timeout=300 --retries 10 . && break; done

VOLUME ["/data", "/weights"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8765/api/queue')" || exit 1
ENTRYPOINT ["imagehub"]
CMD ["serve"]
