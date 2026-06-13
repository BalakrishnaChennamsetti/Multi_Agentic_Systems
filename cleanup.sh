#!/bin/bash

echo "Removing __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Removing build artifacts..."
rm -rf build
rm -rf dist

echo "Removing egg-info..."
find . -type d -name "*.egg-info" -exec rm -rf {} +

echo "Done."