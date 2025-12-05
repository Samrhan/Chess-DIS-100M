FROM mcr.microsoft.com/devcontainers/python:3.12

# Install Stockfish
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends stockfish \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up the environment
# We install dependencies into the system environment for simplicity in devcontainer,
# or we can rely on `uv sync` creating a .venv and VS Code using it.
# The user asked to "update the devcontainer accordingly".
# A common pattern: let uv manage the environment.
# But for caching layer, we can copy pyproject.toml first.

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Set the default shell to bash
ENV SHELL /bin/bash
