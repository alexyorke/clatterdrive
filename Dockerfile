FROM ghcr.io/astral-sh/uv:0.8.15 AS uv
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    FAKE_HDD_HOST=0.0.0.0 \
    FAKE_HDD_PORT=8080 \
    FAKE_HDD_AUDIO=off \
    FAKE_HDD_BACKING_DIR=/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY fake_hdd_fuse ./fake_hdd_fuse
COPY main.py hdd_model.py smoke.py profile_core.py profile_fragmentation.py generate_audio_samples.py generate_readme_demo_samples.py ./

RUN uv sync --locked --no-dev

EXPOSE 8080
VOLUME ["/data"]

CMD ["uv", "run", "--locked", "python", "-m", "fake_hdd_fuse"]
