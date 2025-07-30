#!/bin/bash

# Start FastAPI using uvicorn
uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}

