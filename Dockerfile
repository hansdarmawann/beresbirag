# syntax=docker/dockerfile:1
FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.title="beresbirag" \
      org.opencontainers.image.description="Berean Standard Bible (BSB) Hybrid RAG System" \
      org.opencontainers.image.authors="Hans Darmawan <work.hansdarmawan001@gmail.com>"

WORKDIR /app

# Copy package manifests first for layer caching
COPY pyproject.toml README.md environment.yml ./
COPY src/ ./src/

# Create the dedicated beresbirag Conda environment
RUN conda env create -f environment.yml -y && \
    conda clean -afy

# Ensure all commands execute inside the beresbirag Conda environment
ENV PATH="/opt/conda/envs/beresbirag/bin:${PATH}" \
    CONDA_DEFAULT_ENV="beresbirag" \
    PYTHONUNBUFFERED="1" \
    QDRANT_MODE="remote" \
    QDRANT_URL="http://qdrant:6333"

COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "beresbirag.api:app", "--host", "0.0.0.0", "--port", "8000"]
