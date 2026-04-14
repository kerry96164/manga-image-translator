FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

WORKDIR /app

# --- 第一層：安裝系統工具 (這部分幾乎不會變，跑一次後永遠 CACHED) ---
RUN sed -i 's/archive.ubuntu.com/free.nchc.org.tw/g' /etc/apt/sources.list && \
    apt update --yes && \
    apt install --no-install-recommends -y g++ wget ffmpeg libsm6 libxext6 gimp libvulkan1 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt update --yes && \
    apt install -y libcudnn8=8*-1+cuda11.8 libcudnn8-dev=8*-1+cuda11.8 && \
    apt clean && rm -rf /var/lib/apt/lists/*

# --- 第二層：安裝 Python 套件 (只有修改 requirements.txt 時才會重跑) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- 第三層：模型預備 ---
COPY manga_translator/ /app/manga_translator/
COPY docker_prepare.py .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/app/models \
    python -u docker_prepare.py --continue-on-error

# --- 第四層：你的程式碼 (最常修改的部分放在最後) ---
COPY . .

ENV PYTHONPATH="/app"
ENTRYPOINT ["python", "-m", "manga_translator"]