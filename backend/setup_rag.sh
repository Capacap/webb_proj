#!/bin/bash

# Create resources directory if it doesn't exist
mkdir -p resources/vector_db

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

echo "Setup complete! RAG functionality is ready."
echo "To generate a vector database from scraped files:"
echo "1. Use the vector database creation tool to generate index.faiss and metadata.pkl"
echo "2. Place these files in the backend/resources/vector_db/ directory" 