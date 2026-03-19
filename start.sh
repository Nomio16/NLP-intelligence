#!/bin/bash
set -e

echo "=== Starting NLP Intelligence ==="

cd /app
PYTHONPATH=/app uvicorn adapters.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 120 &

FASTAPI_PID=$!
echo "FastAPI started (PID $FASTAPI_PID)"

cd /app/frontend
npm run start -- --port 3000 &

NEXTJS_PID=$!
echo "Next.js started (PID $NEXTJS_PID)"

echo "Waiting for FastAPI to be ready..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
        echo "FastAPI is ready (${i}s)"
        break
    fi
    sleep 1
done

echo "Waiting for Next.js to be ready..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:3000/ > /dev/null 2>&1; then
        echo "Next.js is ready (${i}s)"
        break
    fi
    sleep 1
done

nginx -g "daemon off;" &

NGINX_PID=$!
echo "nginx started — app on port 7860"

wait -n $FASTAPI_PID $NEXTJS_PID $NGINX_PID
echo "A process exited — shutting down"
