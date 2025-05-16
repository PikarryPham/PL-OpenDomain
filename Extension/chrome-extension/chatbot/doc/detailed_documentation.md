# Setup from scratch
1. Create network
```    docker network create internal-network ```

2. Deploy
```
    cd chatbot/backend
    docker ps -a --filter volume=backend_valkey_data (only do it from the 2nd build)
    docker stop <the container_id_in_the_above_step> (only do it from the 2nd build)
    docker rm <the container_id_in_the_above_step> (only do it from the 2nd build)
    docker volume rm backend_valkey_data (only do it from the 2nd build)
    docker compose up -d --build
    docker ps
```
3. Check log
``` 
    docker logs -f chatbot-api
    docker logs -f chatbot-worker (another terminal tab)
```

4. Setup Database

    ```
    docker exec -it mariadb-tiny bash (another terminal tab)
    mysql -u root -p

    CREATE DATABASE rs_dbpedia CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    
    USE rs_dbpedia;

    CREATE TABLE topic (
        id INT NOT NULL AUTO_INCREMENT,
        label VARCHAR(200) NOT NULL DEFAULT '',
        uri VARCHAR(200) NOT NULL DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_topic_uri (uri)
    );
    
    CREATE TABLE category (
        id INT NOT NULL AUTO_INCREMENT,
        label VARCHAR(200) NOT NULL DEFAULT '',
        uri VARCHAR(200) NOT NULL DEFAULT '',
        topic_id INT NOT NULL,
        parent_id INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_category_label (label),
        UNIQUE KEY uq_category_uri (uri),
        FOREIGN KEY (topic_id) REFERENCES topic(id)
    );
    
    CREATE TABLE page (
        id INT NOT NULL AUTO_INCREMENT,
        label VARCHAR(200) NOT NULL DEFAULT '',
        uri VARCHAR(200) NOT NULL DEFAULT '',
        abstract TEXT,
        comment TEXT,
        category_id INT,
        parent_id INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_concept_uri (uri),
        FOREIGN KEY (category_id) REFERENCES category(id)
    );

    SHOW TABLES;
    ```
5. Run the below command to create collection in vector database
```
curl --location 'http://localhost:8000/collection/create' \
        --header 'Content-Type: application/json' \
        --data '{
            "collection_name": "dbpedia"
        }'
```
6. After running the above command, try to access the collection dbpedia on Qdrant
```
    http://localhost:6333/dashboard#/collections/dbpedia
```
7. After running the above command, try to access the worker UI
```
    http://localhost:5555/workers
    user: admin
    pass: CELERY_FLOWER_PASSWORD (env)
```
8. Run the following command to run the integrated pipeline manually (ablation study on module 2. Automatic Summarization & Extract Topic) if you don't know which topics to specify and sync current popular topics which are defined in whitelist_topics
```
    curl --location 'http://localhost:8000/dbpedia/sync-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "topics": []
        }'
```
or if you want to specify your topics as follows
```
    curl --location 'http://localhost:8000/dbpedia/sync-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "topics": ["Data Science","Business","Tourism"]
        }'
```
9. After running the above command successfully, run this command to sync pages to dbpedia collection on Qdrant
```
    curl --location 'http://localhost:8000/dbpedia/index-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "collection": "dbpedia"
        }'
```

# Implementation Guide for Task 2 and Task 3

## Table of Contents
1. **Overview**
2. **Task 2: Comparison of Embedding Models**
   - 2.1. Introduction
   - 2.2. Installation and Configuration
   - 2.3. Loading Models
   - 2.4. Fine-tuning Models
   - 2.5. Quantization
   - 2.6. Evaluation and Comparison of Models
   - 2.7. Common Troubleshooting
3. **Task 3: AI Summarization for Topic Identification**
   - 3.1. Introduction
   - 3.2. Data Extraction Implementation
   - 3.3. Automatic Topic Extraction
   - 3.4. Handling Asynchronous APIs

## 1. Overview

This project includes two main tasks:

- **Task 2**: Comparing the performance of open-source Embedding models
- **Task 3**: Using AI to summarize content and identify related topics

## 2. Task 2: Comparison of Embedding Models

### 2.1. Introduction

In this task, we deploy and compare different embedding models to replace OpenAI:

- **TF-IDF**: Basic model based on term frequency
- **BM25**: Improved version of TF-IDF
- **Transformer Models**: RoBERTa, XLM-RoBERTa, DistilBERT
- **Hybrid Models**: Combining traditional models with transformers

### 2.2. Installation and Configuration

Ensure the Docker system is running:
```bash
docker-compose up -d
```

### 2.3. Loading Models

Check the list of available models:
```bash
curl -X GET http://localhost:8000/models/embedding
```

Load transformer models:
```bash
# Load RoBERTa
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'

# Load XLM-RoBERTa
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": true}'

# Load DistilBERT
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "distilbert", "version": "latest", "force_download": true}'
```

Load traditional and hybrid models:
```bash
# Load TF-IDF
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "tfidf", "version": "latest"}'

# Load BM25
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "bm25", "version": "latest"}'

# Load hybrid models
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "hybrid_tfidf_bert", "version": "latest"}'

curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "hybrid_bm25_bert", "version": "latest"}'
```

### 2.4. Fine-tune Models

Fine-tune models with learning data:
```bash
curl -X POST http://localhost:8000/models/fine-tune \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data_sample.json", "sample": 1000000}'
```

The response will have the format:
```json
{"task_id":"d8a67284-e47d-4e82-b64d-7a43835f2ce1"}
```

Retrieve the results of the fine-tune task:
```bash
curl -X GET http://localhost:8000/models/fine-tune/d8a67284-e47d-4e82-b64d-7a43835f2ce1 \
  -H "Content-Type: application/json"
```

### 2.5. Quantization

#### Introduction to Quantization

Quantization is a technique to reduce the precision of weights in a model to decrease size and increase processing speed. The system uses the **Dynamic Post-Training Quantization (Dynamic PTQ)** method.

```python
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
```

#### Implementing Model Quantization

Quantizing the RoBERTa model:
```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "roberta"}'
```

Quantizing the XLM-RoBERTa model:
```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xlm-roberta"}'
```

Quantizing the DistilBERT model:
```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "distilbert"}'
```

#### Benefits of Quantization

1. Reduces model size by approximately 4 times
2. Increases processing speed
3. Saves energy
4. Maintains prediction quality at an acceptable level

#### Checking Quantization Status

```bash
curl -X GET http://localhost:8000/models/active
```

The result will display the `is_quantized` field for each model:
```json
{
  "roberta": {
    "status": "active",
    "metadata": {
      "model_name": "roberta-base",
      "is_quantized": true
    }
  }
}
```

### 2.6. Evaluation and Comparison of Models

Compare model performance:
```bash
curl -X POST http://localhost:8000/models/compare \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data_sample.json", "sample": 10000}'
```

Run detailed evaluation script:
```bash
# Connect to the container
docker exec -it chatbot-api bash

# Navigate to the src directory
cd /usr/src/app/src

# Run the evaluation script
python evaluate_models.py --data_path data/history_learning_data_sample.json --output_dir results --sample 10000 --limit 15
```

### 2.7. Common Troubleshooting

#### Hugging Face Connection Error
Solution: Use the `force_download: true` parameter when calling the load model API.

#### Error "'torch.dtype' object has no attribute 'data_ptr'"
Solution: Code needs to be modified directly to address compatibility issues with the PyTorch version. Then reload the model with:
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
```

#### Error "'NoneType' object has no attribute 'eval'"
Solution: Check and fix the code in the quantize_model module to ensure null checking before performing eval(). Then restart the container:
```bash
docker restart chatbot-api chatbot-worker
```

## 3. Task 3: AI Summarization for Topic Identification
This section describes the AI Summarization functionality for automatically summarizing content and extracting topics from text. This feature helps improve the process of identifying topics from content without requiring a fixed list.

### Installation

#### 1. Dependencies

All necessary dependencies have been added to the project's `requirements.txt`:
- transformers
- torch
- spacy
- scikit-learn
- numpy

When using Docker, manual installation is not necessary as these dependencies will be installed automatically when the Docker container is rebuilt with the updated requirements.txt.

```bash
# If Docker container needs to be rebuilt
docker-compose build --no-cache chatbot-api
docker-compose up -d chatbot-api
```

#### 2. Installing spaCy model in Docker

For the spaCy model, an additional installation step is required in the container:

```bash
# Connect to the running container
docker exec -it chatbot-api bash

# Install spaCy model from within the container
python -m spacy download en_core_web_sm
```

#### 3. Introduction to setup_ai_models.sh

The `setup_ai_models.sh` file is a support script with functions to:

1. Install the spaCy model for English
2. Create a directory to store transformer models
3. Optionally download the summarization model in advance
4. Optionally start the API server

In a Docker environment:
```bash
# Ensure the script has execution permissions
docker exec -it chatbot-api chmod +x /usr/src/app/src/setup_ai_models.sh

# Run the script from within the container
docker exec -it chatbot-api /usr/src/app/src/setup_ai_models.sh
```

**Note:** In Docker, you may only need to use the spaCy model installation part and not the server startup part (as the container is already running).

#### File Structure

- `ai_summarization.py`: Main module containing the summarization and topic extraction functions
- `setup_ai_models.sh`: Script to install necessary models
- `test_ai_summarization.py`: Script to test the APIs
- APIs in `app.py`:
  - `/ai/summarize`: Summarize content and extract topics
  - `/ai/extract-topics`: Extract topics only
  - `/ai/batch-process`: Batch process multiple entries
  - `/ai/models/status`: Check model status
  - `/ai/initialize-models`: Initialize models

### Usage

#### 1. Initialize and check model status

```bash
# Initialize all AI models
curl -X POST http://localhost:8000/ai/initialize-models \
  -H "Content-Type: application/json"
```

**Input:** No input data required (empty body)

**Output:**
```json
{
  "summarizer": true,
  "topic_extractor": true,
  "nlp": true
}
```


#### API Endpoint for Automated Summarize & Data Extraction
API endpoint to extract data from history learning data using a combination of 3 methods (LDA, DistilBERT, keyword extraction) to automatically extract topics, save topics to the database, then search for categories and link to pages on DBpedia. Strictly follows this process: extract topic -> save to DB -> get categories -> get pages

## Parameters:
- **path**: Path to the data file (default: /usr/src/app/src/data/history_learning_data_sample.json)
- **sample**: Number of samples to process (default: False)
- **limit**: Limit on the number of entries (default: None)
- **embedding_model**: Embedding model to be used (default: distilbert)
- **ensure_count**: Ensures sufficient number of results are processed (default: True)

``` bash
      curl --location 'http://localhost:8000/dbpedia/extract-data-auto' \
    --header 'Content-Type: application/json' \
    --data '{
        "sample": 10000,
        "limit": 15
    }'
```
**Output:**
```json
    {
      "status": "success",
      "message": "Extraction task started",
      "task_id": "c06a24a2-96a5-4346-be74-8eba5ae7aca5",
      "result_endpoint": "/dbpedia/extract-data-auto/c06a24a2-96a5-4346-be74-8eba5ae7aca5",
      "params": {
          "file_path": "/usr/src/app/src/data/history_learning_data_sample.json",
          "sample": 10000,
          "limit": 15,
          "embedding_model": "distilbert",
          "ensure_count": true
      }
  }
```

You can also check the API running progress via the Worker UI or via the GET API
``` bash
      curl --location 'http://localhost:8000/dbpedia/extract-data-auto/c06a24a2-96a5-4346-be74-8eba5ae7aca5' \
  --header 'Content-Type: application/json' \
  --data ''
```

### Detailed Single APIs used to single features
#### 2. Summarize content and extract topics

```bash
# Summarize content and extract topics
curl --location 'http://localhost:8000/ai/summarize' \
--header 'Content-Type: application/json' \
--data '{
   "content": "Rigging forms the backbone of 3D animation, allowing characters and objects to move realistically. Here, you can learn the fundamentals of rigging within Blender, a popular (and totally free) 3D software. Whether you'\''re animating characters for a game, film or personal project, understanding rigging is crucial for bringing your creations to life. In 3D animation, rigging refers to the process of creating a skeletal structure within a 3D model, which allows it to move and deform in a realistic way. Rigging forms the foundation upon which animations are built by providing the framework that allows characters and objects to interact with their environment and each other. Pro tip: Remember to save your work frequently throughout the rigging process."
}'
```

**Input:**
- `content` (string, required): Text content to summarize and extract topics from
- `max_length` (integer, optional): Maximum length of the summary, default is 100 tokens
- `min_length` (integer, optional): Minimum length of the summary, default is 30 tokens

**Output:**
```json
{
    "summary": "Rigging forms the backbone of 3D animation, allowing characters and objects to move realistically. ... In 3D animation, rigging refers to the process of creating a skeletal structure within a 3D model, which allows it to move and deform in a realistic way. ... Pro tip: Remember to save your work frequently throughout the rigging process.",
    "topics": [
        {
            "topic": "Design",
            "score": 0.8197599577418636,
            "method": "distilbert-cosine-similarity"
        },
        {
            "topic": "Engineering",
            "score": 0.666146403590697,
            "method": "distilbert-cosine-similarity"
        },
        {
            "topic": "Technology",
            "score": 0.6519624054453175,
            "method": "distilbert-cosine-similarity"
        },
        {
            "topic": "3D Animation",
            "score": 0.6,
            "method": "keyword-extraction"
        },
        {
            "topic_id": "lda_topic_2",
            "topic": "Rig Animation",
            "keywords": [
                "rig",
                "animation",
                "allow",
                "character",
                "process",
                "object",
                "movement",
                "model",
                "rigging",
                "blender"
            ],
            "score": 0.3175169584645286,
            "method": "lda"
        }
    ],
    "keywords": [
        "3d animation",
        "animation",
        "process",
        "forms",
        "the backbone",
        "characters",
        "refers",
        "the process",
        "a skeletal structure",
        "a 3d model"
    ]
}
```

#### 3. Extract topics only

```bash
# Extract topics only from text
curl -X POST http://localhost:8000/ai/extract-topics \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from data."
  }'
```
**Input:**
- `content` (string, required): Text content to extract topics from
- `threshold` (float, optional): Score threshold to filter topics, default is 0.25
- `max_topics` (integer, optional): Maximum number of topics to return, default is 5

**Output:**
```json
{
  "topics": [
    {
      "topic": "Computer Science",
      "score": 0.89,
      "method": "distilbert-cosine-similarity"
    },
    {
      "topic": "Technology",
      "score": 0.76,
      "method": "distilbert-cosine-similarity"
    },
    {
      "topic": "Mathematics",
      "score": 0.62,
      "method": "lda"
    },
    {
      "topic": "Data",
      "score": 0.58,
      "method": "keyword-extraction"
    }
  ]
}
```
#### 4. Batch processing

```bash
# Process multiple entries simultaneously (synchronous)
curl -X POST http://localhost:8000/ai/batch-process \
  -H "Content-Type: application/json" \
  -d '{
    "entries": [
      {"visible_content": "Data science is an interdisciplinary field that uses scientific methods."},
      {"visible_content": "Machine learning is a subset of artificial intelligence."}
    ]
  }'
```

**Input:**
- `entries` (array, required): Array of objects to process
  - Each object must have a `visible_content` (string) field
- `async` (boolean, optional): Process asynchronously with Celery, default is false
- `summarize` (boolean, optional): Whether to summarize content, default is true
- `extract_topics` (boolean, optional): Whether to extract topics, default is true

**Output (synchronous):**
```json
{
  "processed_entries": [
    {
      "visible_content": "Data science is an interdisciplinary field that uses scientific methods.",
      "summary": "Data science is an interdisciplinary field that uses scientific methods.",
      "ai_topics": [
        "Computer Science",
        "Anthropology",
        "Sociology",
        "History"
      ],
      "ai_keywords": [
        "data science", 
        "an interdisciplinary field", 
        "that", 
        "scientific methods", 
        "data", 
        "science", 
        "field", 
        "method"
      ]
    },
    {
      "visible_content": "Machine learning is a subset of artificial intelligence.",
      "summary": "Machine learning is a subset of artificial intelligence.",
      "ai_topics": [
        "Computer Science",
        "Psychology",
        "History"
      ],
      "ai_keywords": [
        "machine learning", 
        "a subset", 
        "artificial intelligence", 
        "machine", 
        "learning", 
        "subset", 
        "intelligence"
      ]
    }
  ]
}
```
**Output (asynchronous):**
```json
{
  "task_id": "3b38e956-0b45-482a-a0b1-60d5281fe219",
  "status": "Task started",
  "message": "Processing 2 entries asynchronously"
}
```

#### 5. Comparison with the COMMON_TOPICS Method

| Feature | Old Method (COMMON_TOPICS) | New Method (LDA) |
|---------|----------------------------|------------------|
| Predefined topics | Required | Not required |
| Scalability | Limited, requires manual updates | Automatically detects new topics |
| Accuracy | Good with defined topics | Depends on content and LDA quality |
| Processing speed | Faster | Slower due to semantic analysis |
| Flexibility | Low | High, adapts to various content types |

#### 6. Handling Asynchronous APIs

Asynchronous APIs in the system will return a task_id and have corresponding APIs to retrieve results:

| API | Result Retrieval Endpoint |
|-----|---------------------------|
| `/models/fine-tune` | `/models/fine-tune/{task_id}` |
| `/models/compare` | `/models/fine-tune/{task_id}` (uses the same endpoint as fine-tune) |
| `/dbpedia/extract-data` | `/dbpedia/extract-data/{task_id}` |
| `/dbpedia/extract-data-auto-async` | `/dbpedia/extract-data-auto/{task_id}` |
| `/dbpedia/sync-data` | No separate retrieval endpoint, check using other APIs |
| `/ai/batch-process` | No separate retrieval endpoint, check results saved in files |

For asynchronous tasks, if the result is not ready, the system will return a "PENDING" status. When the task is completed, the result will be returned with a "SUCCESS" or "FAILURE" status.

### Issues with /dbpedia/sync-category API
```
"Internal Server Error" when calling the /dbpedia/sync-category API, possible causes:
Topic doesn't exist: The most common error is that the topics "Technology" and "Artificial Intelligence" don't exist in the database yet. The API tries to link the category to a topic_id but can't find it.
```

#### Temporary API to add topics
```
docker exec -it chatbot-api bash -c "cd /usr/src/app/src && python -c \"
from models import get_topic_by_name, insert_topic
if not get_topic_by_name('Technology'):
    insert_topic('Technology')
if not get_topic_by_name('Artificial Intelligence'):
    insert_topic('Artificial Intelligence')
print('Necessary topics created')
```

#### Or try again with a smaller category:
```
curl -X POST http://localhost:8000/dbpedia/sync-category \
  -H "Content-Type: application/json" \
  -d '{
    "categories": [
      {
        "name": "Programming languages",
        "uri": "https://dbpedia.org/page/Category:Programming_languages",
        "topic": "Technology" 
      }
    ]
  }'
```  
#### Explanation of the category synchronization process:

The `/dbpedia/sync-category` API performs the following steps:
1. Check if the category already exists in the database
2. If not, create a new category and link it to the specified topic
3. Initiate an asynchronous task to synchronize pages belonging to that category from DBpedia
4. By default, the synchronization process will go to a depth of 3 (meaning it will retrieve pages belonging to the category, subcategories, and sub-subcategories)

This process helps:
- Import structured batch content from DBpedia based on classification
- Automatically organize content according to hierarchical structure (topic > category > page)
- Build a rich knowledge database without manual input

### Troubleshooting DBpedia Sync APIs

#### 1. "Internal Server Error" when synchronizing categories

**Common causes:**
- The specified topic doesn't exist in the database
- Connection issues to DBpedia
- Asynchronous processing errors in Celery

**Solutions:**
- Ensure topics exist in the database first (can be added via the `/topics` API or checked in the DB)
- Verify the DBpedia URI is correct (try accessing the URI directly in a browser)
- Check container logs:
  ```bash
  docker logs chatbot-api
  docker logs chatbot-worker
  ```
#### 2. Synchronization process takes a long time

Synchronizing a large category can take a lot of time, especially with high depth. You can:
- Reduce the synchronization depth to 1 or 2
- Split into multiple more specific categories
- Monitor progress through the `/dbpedia/extract-data/{task_id}` API

#### 3. Error when not finding enough entities from DBpedia

If the API doesn't return as many entities as expected:
- Try with broader search keywords
- Check if DBpedia contains information about that topic
- Try changing the language of the URI (e.g., use English URI instead of other languages)

### Testing all functions

Use the test script from within the Docker container:

```bash
# Enter the container
docker exec -it chatbot-api bash

# Run all tests
cd /usr/src/app/src
python test_ai_summarization.py
```

**Input:** No input required, the script will generate test content

**Output:** Results of the API tests, showing success/failure status and the responses received

```bash
# Test a specific API
python test_ai_summarization.py --api summarize
python test_ai_summarization.py --api topics
python test_ai_summarization.py --api batch
python test_ai_summarization.py --api sync
python test_ai_summarization.py --api status
python test_ai_summarization.py --api initialize
```

**Input:**
- `--api` (string, optional): Name of the API to test (summarize, topics, batch, sync, status, initialize)
- `--server` (string, optional): Server URL, default is "http://localhost:8000"
- `--content` (string, optional): Text content to test, default is a sample paragraph

**Output:** Detailed results of the specified API

### Troubleshooting

#### 1. Model not loaded

If you receive an error message about a model not being loaded, try:

```bash
# Reinitialize models
curl -X POST http://localhost:8000/ai/initialize-models
```

Or check the logs for more details:
```bash
docker logs chatbot-api
```

#### 2. Error loading model

If you encounter errors when loading models from Hugging Face, it could be due to network connectivity. Make sure the Docker container has a stable internet connection.

#### 3. Memory errors

If you encounter memory errors (out of memory), you can:
- Increase memory for the Docker container in docker-compose.yml:
  ```yaml
  services:
    chatbot-api:
      mem_limit: 4g  # Increase memory limit
  ```
- Reduce batch size
- Use a smaller model version (can be modified in `ai_summarization.py`)

#### Customization and Extension

You can customize the AI Summarization functionality by:

1. **Changing the Summarization model**: Edit the `summarizer_name` variable in the `init_models()` function of the `ai_summarization.py` file to use a different model (e.g., "t5-small", "facebook/bart-large-cnn", etc.)

2. **Adding topics to the COMMON_TOPICS list**: Edit the `COMMON_TOPICS` variable in the `ai_summarization.py` file

3. **Adjusting score thresholds**: Change the threshold value `0.25` in the topic extraction functions to filter topics with higher or lower scores