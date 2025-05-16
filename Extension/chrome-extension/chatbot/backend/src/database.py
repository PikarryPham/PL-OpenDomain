import logging
import os

# import clickhouse_connect (Comment lại để tránh lỗi kết nối)

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "0.0.0.0")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3308")

# Celery settings
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379")

# MySQL database configuration
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/rs_dbpedia"
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Improve connection resilience
)

mysql_config = {
    "host": MYSQL_HOST,
    "port": MYSQL_PORT,
    "database": "rs_dbpedia",
    "user": MYSQL_USER,
    "password": MYSQL_PASSWORD,
}

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clickhouse - Comment lại để tránh lỗi kết nối
# CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
# clickhouse_client = clickhouse_connect.get_client(
#      host='h8tw70myst.ap-northeast-1.aws.clickhouse.cloud',
#      user='default',
#      password=CLICKHOUSE_PASSWORD,
#     secure=True
# )

# Dummy clickhouse_client để thay thế
clickhouse_client = None


def get_user_learning_history(user_id):
    # Comment lại code gọi ClickHouse
    # logging.info("Result:", clickhouse_client.query("SELECT 1").result_set[0][0])
    # output = clickhouse_client.query("SELECT 1").result_set[0][0]

    # Trả về dict rỗng thay vì gọi đến ClickHouse
    logging.info(
        f"ClickHouse connection disabled, returning empty result for user {user_id}"
    )
    return {}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_celery_app(name):
    # Create a Celery app instance
    app = Celery(
        name,
        broker=CELERY_BROKER_URL,  # Redis as the message broker
        backend=CELERY_RESULT_BACKEND,  # Redis as the result backend
    )

    # Optionally, you can configure additional settings here
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Ho_Chi_Minh",  # Set to a city in UTC+7
        enable_utc=True,
    )

    # Configure Celery logging
    app.conf.update(
        worker_hijack_root_logger=False,
        worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
        worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    )

    return app
