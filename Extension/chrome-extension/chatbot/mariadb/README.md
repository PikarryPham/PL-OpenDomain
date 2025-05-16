## Create .env file
    MYSQL_ROOT_PASSWORD='root_password'

## Run docker
    docker compose up -d

## Stop docker
    docker compose down

## Access
    docker exec -it mariadb-tiny bash
    mysql -u root -p

    CREATE DATABASE rs_dbpedia CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
