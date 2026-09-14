FROM python:3.12-slim

WORKDIR /app

# CPU-only torch: sentence-transformers never uses CUDA here, and the default
# wheel drags in multi-GB NVIDIA runtime libraries the container cannot use.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements-compose.txt .
RUN pip install --no-cache-dir -r requirements-compose.txt

COPY . .

# Index the corpus into /app/.chroma at build time. Live mode reads this index
# with no host bind mount; fake mode never touches it.
RUN python ingest.py

EXPOSE 8000 8501

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
