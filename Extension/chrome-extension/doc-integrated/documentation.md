# Project Documentation Guide
This README provides guidance on navigating through the documentation files in this project. The documentation is organized to help you understand how to set up, run, and troubleshoot the system efficiently.

## Documentation Files Overview

There are 4 main documentation files:

1. **overall_pipeline.md** - Overview of container reset, restart, and basic API operations
2. **detailed_documentation.md** - Comprehensive setup guide and implementation details
3. **quantization_api_docs.md** - Specific documentation for model quantization APIs
4. **solutions_for_errors.md** - Troubleshooting common errors and recommended processes

## Note about data request
Because the data contain private information of learners so it is only available upon request. Please dm me via email for request data: trang.pham@jaist.ac.jp. 

## Recommended Reading Order

For optimal understanding and successful setup, follow this reading order:

### 1. First: overall_pipeline.md
Start with this file to get a high-level understanding of:
- How to reset and restart Docker containers
- Basic API test commands
- Loading different models
- Checking model status
- Running evaluations

This file gives you the big picture of the system workflow.

### 2. Second: detailed_documentation.md
Once you understand the overall pipeline, move to this file for detailed implementation:
- Complete setup process from scratch
- Database initialization
- Pipeline integration
- Vector database setup
- Task 2 (Embedding Models) implementation details
- Task 3 (AI Summarization) implementation details

This file contains the most comprehensive information about the system.

### 3. Third: solutions_for_errors.md
Review this file to prepare for potential issues:
- Common errors during model loading and fine-tuning
- Recommended error-free processes
- Automatic scripts for streamlined setup
- Troubleshooting steps for specific errors

This will help you avoid or quickly resolve common problems.

### 4. Fourth: quantization_api_docs.md
Finally, read about the advanced model optimization:
- Understanding model quantization
- API documentation for quantization
- Benefits of quantization
- Technical implementation details

## Quick Start Guide

For the fastest setup, follow these steps:

1. **Reset and set up Docker containers**
   ```bash
   cd chatbot/backend
   docker compose down -v
   docker system prune -a
   docker compose build --no-cache
   docker compose up -d
   ```

2. **Create network and deploy**
   ```bash
   docker network create internal-network
   cd chatbot/backend
   docker compose up -d --build
   ```

3. **Set up the database**
   ```bash
   docker exec -it mariadb-tiny bash
   mysql -u root -p
   # Run SQL commands from detailed_documentation.md
   ```

4. **Create vector database collection**
   ```bash
   curl --location 'http://localhost:8000/collection/create' \
     --header 'Content-Type: application/json' \
     --data '{
         "collection_name": "dbpedia"
     }'
   ```

5. **Load and fine-tune models using automated script**
   ```bash
   docker exec -it chatbot-api bash
   chmod +x src/fix_and_finetune.sh
   ./src/fix_and_finetune.sh localhost 8000 my_finetuned_v1
   ```

6. **Sync data to the vector database**
   ```bash
   curl --location 'http://localhost:8000/dbpedia/sync-data' \
     --header 'Content-Type: application/json' \
     --data '{
         "topics": []
     }'
   ```

7. **Index data into the vector database**
   ```bash
   curl --location 'http://localhost:8000/dbpedia/index-data' \
     --header 'Content-Type: application/json' \
     --data '{
         "collection": "dbpedia"
     }'
   ```

## Troubleshooting

If you encounter errors during setup:

1. Check container logs:
   ```bash
   docker logs -f chatbot-api
   docker logs -f chatbot-worker
   ```

2. Use the automated troubleshooting scripts:
   ```bash
   docker exec -it chatbot-api bash
   python src/fix_transformer_models.py --force
   ```

3. Reload models with force download:
   ```bash
   curl -X POST http://localhost:8000/models/load \
     -H "Content-Type: application/json" \
     -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
   ```

4. Review the solutions_for_errors.md file for specific error messages.

## Advanced Features

After basic setup, explore these advanced features:

1. **Model Quantization** - Optimize models for speed and efficiency:
   ```bash
   curl -X POST http://localhost:8000/models/quantize \
     -H "Content-Type: application/json" \
     -d '{"model_type": "roberta"}'
   ```

2. **AI Summarization and Topic Extraction**:
   ```bash
   curl -X POST http://localhost:8000/ai/initialize-models \
     -H "Content-Type: application/json"
   ```

3. **Model Comparison and Evaluation**:
   ```bash
   curl -X POST http://localhost:8000/models/compare \
     -H "Content-Type: application/json" \
     -d '{"path": "data/history_learning_data.json", "sample": 10000}'
   ```

## Key APIs Reference

Here are some of the most important APIs:

- **Load Models**: `POST /models/load`
- **Check Active Models**: `GET /models/active`
- **Fine-tune Models**: `POST /models/fine-tune`
- **Compare Models**: `POST /models/compare`
- **Quantize Models**: `POST /models/quantize`
- **Extract Data**: `POST /dbpedia/extract-data-auto`
- **Sync Data**: `POST /dbpedia/sync-data`
- **Index Data**: `POST /dbpedia/index-data`
- **Summarize Content**: `POST /ai/summarize`
- **Extract Topics**: `POST /ai/extract-topics`

---

For a more detailed understanding of specific components, refer to the corresponding documentation files mentioned above.