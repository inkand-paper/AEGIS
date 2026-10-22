# The image AEGIS runs model-written code in (see src/aegis/sandbox.py).
# Build it once with:  uv run aegis sandbox build
# It holds Python and pytest only. Nothing else is installed, and the container is started with
# no network, so nothing can be installed at run time either.
FROM python:3.13-slim
RUN pip install --no-cache-dir pytest==9.1.1 \
    && rm -rf /root/.cache
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
