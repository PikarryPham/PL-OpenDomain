FROM python:3.10-slim

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY final_submission/*.py /app/
COPY final_submission/models /app/models/
COPY final_submission/utils /app/utils/

# Install Python dependencies
RUN pip install --no-cache-dir neo4j py2neo

# Configure Neo4j connection
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=password123

# Run the application
CMD ["python", "main.py"]
