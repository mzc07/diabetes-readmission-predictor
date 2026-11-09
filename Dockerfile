FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
# Incluye el grupo dev (JupyterLab, ipykernel, pytest, ruff)
RUN uv sync --frozen
COPY . .
EXPOSE 8888 8501
# Por defecto: JupyterLab sin token (solo para uso local)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
