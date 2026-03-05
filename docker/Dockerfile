FROM ghcr.io/prefix-dev/pixi:noble-cuda-13.0.0

ENV DEBIAN_FRONTEND="noninteractive"
ENV PIXI_CACHE_DIR=/workspace/.pixi

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    libgl1 \
    libegl1 \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxt6 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY source source
RUN --mount=type=cache,target=${PIXI_CACHE_DIR} \
    --mount=type=bind,source=pixi.lock,target=pixi.lock \
    --mount=type=bind,source=pixi.toml,target=pixi.toml \
    pixi install --locked -e rl
