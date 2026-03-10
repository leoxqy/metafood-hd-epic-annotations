#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PORT=8000
TARGET="http://127.0.0.1:${PORT}/HD_EPIC_VQA_Interface.html"

echo "Starting local server at ${TARGET}"
echo "Keep this window open while using the interface."
echo
open "${TARGET}"

if command -v python3 >/dev/null 2>&1; then
  python3 -m http.server "${PORT}"
  exit 0
fi

if command -v python >/dev/null 2>&1; then
  python -m http.server "${PORT}"
  exit 0
fi

echo "Python was not found on PATH."
echo "Install Python from https://www.python.org/downloads/ and try again."
read -r -p "Press Enter to close..." _
exit 1
