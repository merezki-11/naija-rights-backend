#!/bin/bash

echo "Checking if ChromaDB is already populated..."

if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo "ChromaDB not found. Running ingestion..."
    python ingest.py
    echo "Ingestion complete."
else
    echo "ChromaDB already exists. Skipping ingestion."
fi

echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port $PORT