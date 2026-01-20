# DLM-DTI Refactored Docker Image
# Blackwell GPU (sm_120) support with PyTorch nightly + CUDA 12.8
# Build: docker build -t dlm-dti:latest .
# Run: docker run --gpus all -it dlm-dti:latest

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS base
SHELL ["/bin/bash", "-c"]

ARG USER_UID=1000
ARG USER_GID=1000
ENV USER_UID=${USER_UID}
ENV USER_GID=${USER_GID}

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HF_HOME=/workspace/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    build-essential \
    git \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch nightly + CUDA 12.8 (Blackwell GPU sm_120 support)
RUN uv pip install --python /opt/venv/bin/python \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ============================================================================
# Install PyTorch Geometric (optional, build from source for nightly compatibility)
# ============================================================================
RUN uv pip install --python /opt/venv/bin/python torch-geometric
RUN uv pip install --python /opt/venv/bin/python setuptools wheel
RUN uv pip install --python /opt/venv/bin/python \
    torch-scatter torch-sparse torch-cluster --no-binary torch-scatter,torch-sparse,torch-cluster --no-build-isolation

# Install model requirements (filter torch/nvidia wheels)
COPY requirements.txt /tmp/requirements.txt
RUN sed -E '/^(torch|torchvision|torchaudio|nvidia-)/d' /tmp/requirements.txt > /tmp/req_no_torch.txt \
    && uv pip install --python /opt/venv/bin/python -r /tmp/req_no_torch.txt

# Copy refactored code
COPY components /app/components
COPY src /app/src
COPY src /app/dlm-dti_models/src
COPY config.yaml /app/config.yaml

# Create directories with proper permissions
RUN mkdir -p /workspace/models /workspace/execution_logs /workspace/datasets \
    && mkdir -p /home/appuser \
    && groupadd -g ${USER_GID} appuser \
    && useradd -m -u ${USER_UID} -g ${USER_GID} appuser \
    && chown -R ${USER_UID}:${USER_GID} /app /workspace /home/appuser

ENV PYTHONPATH="/app:/app/src:/app/components:/app/dlm-dti_models"

USER appuser

CMD ["/bin/bash"]
