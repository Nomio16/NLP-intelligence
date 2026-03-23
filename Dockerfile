FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
#ENV NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=/api
RUN npm run build

FROM python:3.11-slim

# gcc is required to compile hdbscan from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ nodejs npm nginx curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first for HF Space (CPU-only environment)
RUN pip install --no-cache-dir torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY nlp_core/ ./nlp_core/
COPY adapters/ ./adapters/

# Pre-download NER model weights from HuggingFace Hub at build time
# so the first request is fast (no 677MB download on startup)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Nomio4640/ner-mongolian')"

COPY --from=frontend-builder /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY --from=frontend-builder /app/frontend/package*.json ./frontend/
COPY --from=frontend-builder /app/frontend/node_modules ./frontend/node_modules

COPY nginx.conf /etc/nginx/sites-available/default

EXPOSE 7860

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
