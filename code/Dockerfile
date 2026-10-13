# OCC-HTR — medieval Occitan HTR + philological discrepancy detection
# Build (from the repository root):   docker build -t occ-htr ./code
# Run the web interface:              docker run --rm -p 8000:8000 occ-htr
# Run any make target / shell:        docker run --rm occ-htr make <target>
#                                     docker run --rm -it occ-htr bash
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 make git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for locked dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# dependency layer (cached unless the lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# project code + deployed artifacts
COPY src ./src
COPY scripts ./scripts
COPY frontend ./frontend
COPY fonts ./fonts
COPY models ./models
COPY makefile ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PROJECT_ROOT=/app

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["frontend"]
