# Use official Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer — 
# so rebuilds are fast if only your code changes, not dependencies)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run ingest first to build ChromaDB, then start the API
# In production you'd separate these — but for portfolio this is clean
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]