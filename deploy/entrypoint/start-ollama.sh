#!/bin/sh

set -e

echo "Starting Ollama server..."

ollama serve &

SERVER_PID=$!

echo "Waiting for Ollama API..."

until ollama list >/dev/null 2>&1
do
  sleep 2
done

echo "Checking model..."

if ! ollama list | grep -q "gemma4:12b"; then
  echo "Downloading gemma4:12b..."
  ollama pull gemma4:12b
else
  echo "gemma4:12b already exists."
fi

echo "Ollama ready."

wait $SERVER_PID