# Container image for the Agentic Inference Orchestrator.
# Deployed to Azure Container Apps (free monthly grant covers light use).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir azure-storage-blob azure-search-documents \
       azure-core fastapi uvicorn

COPY . .

# Minimal HTTP wrapper so Container Apps has an endpoint to serve.
EXPOSE 8000
CMD ["uvicorn", "cloud.service:app", "--host", "0.0.0.0", "--port", "8000"]
