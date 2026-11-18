FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY minivllm ./minivllm
RUN pip install --no-cache-dir ".[server]"

EXPOSE 8000
ENTRYPOINT ["python", "-m", "minivllm.server", "--host", "0.0.0.0"]
