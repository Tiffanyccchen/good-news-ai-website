FROM python:3.10-slim

# Avoid prompts during image build
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy project metadata first (leverages Docker layer caching)
COPY pyproject.toml README.md ./

# Install project (pep517 build via poetry-core) and its dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Copy the rest of the source code
COPY . .

# Expose for HF Spaces (the platform sets $PORT)
CMD ["bash", "-c", "streamlit run run_frontend.py --server.port $PORT --server.address 0.0.0.0"] 