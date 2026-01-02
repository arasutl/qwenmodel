FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CUDA-enabled PyTorch (PyPI default is usually CPU-only).
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 \
 && pip3 install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3", "handler.py"]