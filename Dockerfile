FROM ghcr.io/prefix-dev/pixi:noble-cuda-13.0.0

ENV DEBIAN_FRONTEND="noninteractive"

ARG USERNAME=user
ARG UID=1000
ARG GID=1000

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

RUN groupadd --gid ${GID} ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} -m ${USERNAME}

USER ${USERNAME}

WORKDIR /workspace

ENV PIXI_CACHE_DIR=/home/user/.cache/rattler
RUN --mount=type=cache,target=${PIXI_CACHE_DIR},uid=${UID},gid=${GID} \
    --mount=type=bind,source=pixi.lock,target=pixi.lock \
    --mount=type=bind,source=pixi.toml,target=pixi.toml \
    pixi install --frozen --skip mad-rl

COPY --chown=${USERNAME}:${USERNAME} . .

RUN --mount=type=cache,target=${PIXI_CACHE_DIR},uid=${UID},gid=${GID} \
    pixi install --locked
