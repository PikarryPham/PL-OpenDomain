# # Guide to Deploying the Educational Temporal Knowledge Graph (ETKG) Application with Docker

## Introduction

This document guides how to deploy the Educational Temporal Knowledge Graph (ETKG) application using Docker and Docker Compose. Using Docker simplifies the installation and deployment process, ensuring the application runs consistently across all environments.

## System Requirements

- Docker Engine (version 19.03 or higher)
- Docker Compose (version 1.27 or higher)
- At least 2GB of RAM and 10GB of free disk space

## Folder structure

```
tkg-docker/
├── Dockerfile                # Docker configuration for the Python application
├── docker-compose.yml        # Docker Compose configuration for the entire system
├── run_docker.sh             # Script to automate the deployment process
├── final_submission/         # Application source code
│   ├── main.py               # Entry point
│   ├── config.py             # Configuration
│   ├── db_connector.py       # Connection to Neo4j
│   ├── node_manager.py       # Node management
│   ├── relationship_manager.py # Relationship management
│   ├── models/               # Data models
│   └── utils/                # Utilities
└── upload/                   # Directory containing input data
    ├── users_data.json
    ├── question_data.json
    ├── options_data.json
    ├── history_learning_data.json
    └── final_sample_output.json
```

## Deployment Steps

### 1. Prepare the Environment

Ensure Docker and Docker Compose are installed on your system:

```bash
docker --version
docker-compose --version
```

### 2. Prepare the Data

Ensure the JSON data files are placed in the upload/ directory:

- users_data.json
- question_data.json
- options_data.json
- history_learning_data.json
- final_sample_output.json

### 3. Automated Deployment

The easiest way to deploy is to use the run_docker.sh script:

```
bash
./run_docker.sh
```

This script will:
- Check the Docker and Docker Compose installations
- Create the upload directory if it doesn't exist
- Copy the data files into the upload directory
- Build and start the containers
- Display the status of the containers
- Tail the application logs

### 4. Manual Deployment

If you want to deploy manually, follow these steps:

```bash
# Build and start the containers
docker-compose up --build -d

# Check the status of the containers
docker-compose ps

# View application logs
docker-compose logs -f app

# View Neo4j logs
docker-compose logs -f neo4j
```

## Accessing the Neo4j Browser

Once deployment is successful, you can access the Neo4j Browser to view and query the Knowledge Graph:

1. Open your web browser and go to: http://localhost:7474
2. Log in with:
   - Username: neo4j
   - Password: password123
3. You can run Cypher queries to explore the Knowledge Graph

## Useful Cypher Queries

Below are some useful Cypher queries to explore the Knowledge Graph:

```cypher
// View all nodes
MATCH (n) RETURN n LIMIT 100

// View all Users
MATCH (u:User) RETURN u

// View all Entries of a User
MATCH (u:User {user_id: "f8ccf0cf-bd98-4bad-8dcf-9d98f6ad8361"})-[r:HAS]->(e:Entry)
RETURN u, r, e LIMIT 10

// View all Topics
MATCH (t:Topic) RETURN t

// View relationships between nodes
MATCH p=()-[r]->() RETURN p LIMIT 25
```

## Container Management

### Stopping Containers

```bash
docker-compose stop
```

### Starting Containers

```bash
docker-compose start
```

### Stopping and Removing Containers

```bash
docker-compose down
```

### Stopping, Removing Containers, and Deleting Volumes

```bash
docker-compose down -v
```

## Troubleshooting

### Neo4j Fails to Start

If Neo4j fails to start, check the logs:

```bash
docker-compose logs neo4j
```

You may need to increase Docker's memory allocation or adjust Neo4j's configuration in the docker-compose.yml file.

### Application Cannot Connect to Neo4j

Check the application logs:

```bash
docker-compose logs app
```

Make sure Neo4j has fully started before the application attempts to connect. You may need to adjust the healthcheck timeout.

## Conclusion

Deploying the Temporal Knowledge Graph application with Docker simplifies the installation process and ensures consistency across environments. By using Docker Compose, we can easily manage both the Python application and the Neo4j database in a single, containerized environment.
