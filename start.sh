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

sleep 5
nginx -g "daemon off;" &

NGINX_PID=$!
echo "nginx started — app on port 7860"

wait -n $FASTAPI_PID $NEXTJS_PID $NGINX_PID
echo "A process exited — shutting down"
