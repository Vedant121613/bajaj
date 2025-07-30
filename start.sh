#!/bin/bash

# Use Gunicorn with Uvicorn worker and custom timeout
gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT:-10000} \
    --timeout 120
