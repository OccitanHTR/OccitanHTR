#!/bin/sh
set -e

# If the template data folder is mounted (-v .../data:/template/data and
# TEMPLATE_DATA_ROOT=/template), map it onto the pipeline layout once.
if [ -n "$TEMPLATE_DATA_ROOT" ] && [ ! -d /app/data/raw ]; then
    python scripts/setup_data_layout.py || true
fi

cmd="${1:-frontend}"
[ "$#" -gt 0 ] && shift

case "$cmd" in
    frontend)
        exec uvicorn frontend.app:app --host 0.0.0.0 --port 8000 ;;
    make)
        exec make "$@" ;;
    bash|sh)
        exec /bin/bash "$@" ;;
    *)
        exec "$cmd" "$@" ;;
esac
