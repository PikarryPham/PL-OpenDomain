# Example chatbot project

## Create network
    docker network create internal-network

## Deploy
    cd chatbot/backend
    docker volume rm backend_valkey_data
    docker compose up -d --build

## Check log
    docker logs -f chatbot-api
    docker logs -f chatbot-worker
    

## Database
Run command in file chatbot/mariadb/README.md

    docker exec -it mariadb-tiny bash
    mysql -u root -p

    CREATE DATABASE demo_bot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    
    USE demo_bot;

    CREATE TABLE chat_conversations (
        id INT NOT NULL AUTO_INCREMENT,
        conversation_id VARCHAR(50) NOT NULL DEFAULT '',
        bot_id VARCHAR(100) NOT NULL,
        user_id VARCHAR(100) NOT NULL,
        message TEXT,
        is_request BOOLEAN DEFAULT TRUE,
        completed BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id)
    );

    CREATE TABLE document (
        id INT NOT NULL AUTO_INCREMENT,
        title VARCHAR(100) NOT NULL,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id)
    );

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

### Vector DB
    http://localhost:6333/dashboard#/collections/llm
    http://localhost:6333/dashboard#/collections/dbpedia

    Online: http://34.42.208.64:6333/dashboard#/collections/llm


### Chat UI
    http://localhost:8051/


### Task UI
    http://localhost:5555/workers
    
    user: admin
    pass: CELERY_FLOWER_PASSWORD (env)


### Task UI
    http://localhost:5555/
    User: admin
    Pass: 123asdas12


### Test curl
    curl --location 'http://localhost:8000/collection/create' \
        --header 'Content-Type: application/json' \
        --data '{
            "collection_name": "dbpedia"
        }'

    curl --location 'http://localhost:8000/dbpedia/sync-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "topics": ["Software Development"]
        }'

    curl --location 'http://localhost:8000/dbpedia/sync-category' \
        --header 'task_id: 752d4033-c696-4134-a899-c6e12b1957b2' \
        --header 'Content-Type: application/json' \
        --data '{
            "categories": [
                {
                    "name": "Video_game_publishers",
                    "uri": "http://dbpedia.org/resource/Category:Video_game_publishers",
                    "topic": "Proprietary_software"
                }
            ]
        }'

    curl --location 'http://localhost:8000/dbpedia/index-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "collection": "dbpedia"
        }'

    curl --location 'http://localhost:8000/dbpedia/index-category' \
        --header 'Content-Type: application/json' \
        --data '{
            "collection": "dbpedia",
            "categories": ["Software_licenses"]
        }'

    curl --location 'http://localhost:8000/dbpedia/get-vector-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "id": 1
        }'

    curl --location 'http://localhost:8000/dbpedia/search-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "keywords": ["Software"],
            "limit": 10
        }'

## Extract data

Example 1:

    curl --location 'http://localhost:8000/dbpedia/extract-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "sample": 10,
            "limit": 2
        }'


Example 2: get more 10 items

    curl --location 'http://localhost:8000/dbpedia/extract-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "sample": 11,
            "limit": 1000
        }'

    {
        "task_id": "604aea2a-c7d4-4f77-b7b6-186a4872c088"
    }

    curl --location --request GET 'http://localhost:8000/dbpedia/extract-data/604aea2a-c7d4-4f77-b7b6-186a4872c088' \
        --header 'Content-Type: application/json' \
        --data '{
        }'


## Notebook:
    https://colab.research.google.com/drive/1-uodfFUM-MbjdYa2R97sdmHcPVmM1abO?usp=sharing#scrollTo=Q8gNgMZw8BfB


## API:
    - Search Data:
    curl --location 'http://localhost:8000/dbpedia/search-data' \
        --header 'Content-Type: application/json' \
        --data '{
            "keywords": ["Software", "Development"]
        }'


## References
- https://fastapi.tiangolo.com/tutorial/first-steps/
- https://derlin.github.io/introduction-to-fastapi-and-celery/03-celery/
- https://testdriven.io/courses/fastapi-celery/getting-started/