#!/usr/bin/env bash
# Starts the ChoiceFit API and the ngrok tunnel, and keeps the Mac awake. Stop with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."

ORIGIN="${CHOICEFIT_ORIGINS:-https://yuvaraj-dasarathan.github.io}"
PORT="${PORT:-8000}"

command -v ngrok >/dev/null || { echo "ngrok not found on PATH"; exit 1; }
[ -x .venv/bin/uvicorn ] || { echo "Missing .venv/bin/uvicorn"; exit 1; }

cleanup() { kill "${API_PID:-}" "${NGROK_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

CHOICEFIT_ORIGINS="$ORIGIN" .venv/bin/uvicorn server.api:app --host 127.0.0.1 --port "$PORT" &
API_PID=$!
ngrok http "$PORT" --log=stdout >/tmp/ngrok.log &
NGROK_PID=$!

for _ in $(seq 1 20); do
  URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import sys,json;print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null || true)
  [ -n "$URL" ] && break
  sleep 1
done
echo
echo "API:    http://127.0.0.1:$PORT   (local UI: same address)"
echo "Public: ${URL:-not ready, see /tmp/ngrok.log}"
echo "CORS:   $ORIGIN"
echo "Press Ctrl+C to stop."
caffeinate -i -w "$API_PID"
