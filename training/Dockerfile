FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV MAX_JOBS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pulp
RUN pip install --upgrade --no-cache-dir pip setuptools wheel ninja uvtools prometheus_fastapi_instrumentator uvicorn watchfiles partial_json_parser python-multipart pybase64 hf_transfer
RUN pip install --no-cache-dir datasets transformers pydantic cachetools cloudpickle msgspec numba blake3 fastapi openai py-cpuinfo gguf compressed_tensors
RUN pip install --no-deps --no-cache-dir psutil regex rich bitsandbytes accelerate peft trl cut_cross_entropy unsloth_zoo


RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install --no-cache-dir flashinfer-python

RUN pip install --no-deps --no-cache-dir unsloth
RUN pip install --no-deps --no-cache-dir triton
RUN pip install --no-deps --no-cache xformers
RUN pip install --no-deps vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

WORKDIR /opt
RUN pip install --no-cache-dir jupyter

RUN git clone https://github.com/ERGO-Code/HiGHS.git
WORKDIR /opt/HiGHS
RUN cmake -S. -B build
RUN cmake --build build --parallel

WORKDIR /workspace

EXPOSE 8000

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8000", "--no-browser", "--allow-root", "--NotebookApp.token="]
