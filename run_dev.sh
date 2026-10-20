#!/usr/bin/env bash
# Launch script for Personalised RAG (macOS / Linux).
#
# Activates the virtual environment, runs Django system checks, refuses to
# start if a server is already listening on port 8000, starts the development
# server, and opens the default browser.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}/"

# --- Virtual environment -------------------------------------------------
if [ ! -d ".venv" ]; then
  echo "No .venv found. Create one first:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- Refuse to start a duplicate server ----------------------------------
if command -v curl >/dev/null 2>&1; then
  if curl -s -o /dev/null --max-time 1 "${URL}"; then
    echo "A server is already listening on ${URL}."
    echo "Open that URL in your browser, or stop the other server first."
    exit 1
  fi
fi

# --- Sanity checks -------------------------------------------------------
echo "==> Running Django system checks"
python manage.py check

# --- Start the server ----------------------------------------------------
echo "==> Starting development server at ${URL}"
echo "    Press Ctrl+C to stop it."
( sleep 2 && (xdg-open "${URL}" 2>/dev/null || open "${URL}" 2>/dev/null || true) ) &
exec python manage.py runserver "127.0.0.1:${PORT}"
