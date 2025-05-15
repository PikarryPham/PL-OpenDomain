import logging
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Boolean,
    DateTime,
    Integer,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError

from cache import get_conversation_id
from utils import setup_logging
from database import engine

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)
setup_logging()
logger = logging.getLogger(__name__)


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(50), nullable=False, default="")
    bot_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    message = Column(String)  # Assuming TextField is equivalent to String in SQLAlchemy
    is_request = Column(Boolean, default=True)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )


def load_conversation(conversation_id: str):
    with SessionLocal() as session:
        return (
            session.query(ChatConversation)
            .filter(ChatConversation.conversation_id == conversation_id)
            .order_by(ChatConversation.created_at)
            .all()
        )


def convert_conversation_to_openai_messages(user_conversations):
    conversation_list = [
        {"role": "system", "content": "You are an amazing virtual assistant"}
    ]

    for conversation in user_conversations:
        role = "assistant" if not conversation.is_request else "user"
        content = str(conversation.message)
        conversation_list.append({"role": role, "content": content})

    logging.info(f"Create conversation to {conversation_list}")

    return conversation_list


def update_chat_conversation(
    bot_id: str, user_id: str, message: str, is_request: bool = True
):
    # Step 1: Create a new ChatConversation instance
    conversation_id = get_conversation_id(bot_id, user_id)

    new_conversation = ChatConversation(
        conversation_id=conversation_id,
        bot_id=bot_id,
        user_id=user_id,
        message=message,
        is_request=is_request,
        completed=not is_request,
    )
    with SessionLocal() as session:
        try:
            session.add(new_conversation)
            session.commit()
            session.refresh(new_conversation)
            logger.info(f"Create message for conversation {conversation_id}")
            return conversation_id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating chat conversation: {e}")
            raise


def get_conversation_messages(conversation_id):
    user_conversations = load_conversation(conversation_id)
    return convert_conversation_to_openai_messages(user_conversations)


class Document(Base):
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False, default="")
    content = Column(String)  # Assuming TextField is equivalent to String in SQLAlchemy
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )


def insert_document(title: str, content: str):
    # Step 1: Create a new Document instance
    new_doc = Document(
        title=title,
        content=content,
    )
    with SessionLocal() as session:
        try:
            session.add(new_doc)
            session.commit()
            session.refresh(new_doc)
            logger.info(f"Create document successfully {new_doc}")
            return new_doc
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error inserting document: {e}")
            raise


class Topic(Base):
    __tablename__ = "topic"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(200), nullable=False, default="")
    uri = Column(String(200), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("uri", name="uq_topic_uri"),)


def get_topics():
    with SessionLocal() as session:
        return session.query(Topic).all()


def get_topic_by_id(id: int):
    with SessionLocal() as session:
        return session.query(Topic).filter(Topic.id == id).first()


def get_topic_by_name(label: str):
    with SessionLocal() as session:
        return session.query(Topic).filter(Topic.label == label).first()


def insert_topic(label: str, uri: str):
    with SessionLocal() as session:
        try:
            existing_topic = session.query(Topic).filter(Topic.uri == uri).first()
            if existing_topic:
                existing_topic.label = label
                session.commit()
                session.refresh(existing_topic)
                logger.info(f"Updated topic successfully {existing_topic}")
                return existing_topic
            else:
                new_topic = Topic(label=label, uri=uri)
                session.add(new_topic)
                session.commit()
                session.refresh(new_topic)
                logger.info(f"Create topic successfully {new_topic}")
                return new_topic
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error inserting topic: {e}")
            raise


class Category(Base):
    __tablename__ = "category"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(200), nullable=False, default="")
    uri = Column(String(200), nullable=False, default="")
    topic_id = Column(Integer, ForeignKey("topic.id"), nullable=False)
    parent_id = Column(Integer, ForeignKey("category.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("uri", name="uq_category_uri"),)


def get_category_by_name(label):
    with SessionLocal() as session:
        return session.query(Category).filter(Category.label == label).first()


def get_categories(topic_id):
    with SessionLocal() as session:
        return session.query(Category).filter(Category.topic_id == topic_id).all()


def get_category_by_id(id: int):
    with SessionLocal() as session:
        return session.query(Category).filter(Category.id == id).first()


def insert_category(label: str, uri: str, topic_id: int = None, parent_id: int = None):
    with SessionLocal() as session:
        try:
            existing_category = (
                session.query(Category).filter(Category.uri == uri).first()
            )
            if existing_category:
                logger.info(f"Updated category successfully {existing_category}")
                return existing_category
            else:
                new_category = Category(
                    label=label, uri=uri, topic_id=topic_id, parent_id=parent_id
                )
                session.add(new_category)
                session.commit()
                session.refresh(new_category)
                logger.info(f"Create category successfully {new_category}")
                return new_category
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error inserting category: {e}")
            raise


class Page(Base):
    __tablename__ = "page"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(200), nullable=False, default="")
    uri = Column(String(200), nullable=False, default="")
    abstract = Column(String)
    comment = Column(String)
    category_id = Column(Integer, ForeignKey("category.id"), nullable=True)
    parent_id = Column(Integer, ForeignKey("page.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("uri", name="uq_concept_uri"),)


def get_pages(category_id):
    with SessionLocal() as session:
        return session.query(Page).filter(Page.category_id == category_id).all()


def insert_page(
    label: str,
    uri: str,
    abstract: str,
    comment: str,
    category_id: int,
    parent_id: int = None,
):
    with SessionLocal() as session:
        try:
            existing_page = session.query(Page).filter(Page.uri == uri).first()
            if existing_page:
                logger.info(f"Page already exists {existing_page}")
                return existing_page
            else:
                new_page = Page(
                    label=label,
                    uri=uri,
                    abstract=abstract,
                    comment=comment,
                    category_id=category_id,
                    parent_id=parent_id,
                )
                session.add(new_page)
                session.commit()
                session.refresh(new_page)
                logger.info(f"Create page successfully {new_page}")
                return new_page
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error inserting page: {e}")
            raise


class Task(Base):
    __tablename__ = "task"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(200), nullable=False, unique=True)
    task_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="PENDING")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )
    result_file = Column(String(500), nullable=True)
    params = Column(String)  # Lưu tham số dưới dạng JSON string


def insert_task(task_info):
    """
    Lưu thông tin task vào database

    Args:
        task_info: Dict chứa thông tin task

    Returns:
        Task object
    """
    import json

    # Chuyển đổi params thành JSON string nếu cần
    params = task_info.get("params", {})
    if isinstance(params, dict):
        params_json = json.dumps(params)
    else:
        params_json = str(params)

    # Tạo task mới
    new_task = Task(
        task_id=task_info.get("task_id"),
        task_type=task_info.get("task_type"),
        status=task_info.get("status", "PENDING"),
        result_file=task_info.get("result_file"),
        params=params_json,
    )

    with SessionLocal() as session:
        try:
            session.add(new_task)
            session.commit()
            session.refresh(new_task)
            logger.info(f"Task created successfully: {new_task.task_id}")
            return new_task
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error inserting task: {e}")
            raise


def get_task_by_id(task_id):
    """
    Lấy thông tin task từ database theo task_id

    Args:
        task_id: ID của task

    Returns:
        Dict chứa thông tin task hoặc None nếu không tìm thấy
    """
    import json

    with SessionLocal() as session:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            return None

        # Chuyển đổi params từ JSON string thành dict
        try:
            params = json.loads(task.params) if task.params else {}
        except:
            params = {}

        return {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "result_file": task.result_file,
            "params": params,
        }


def update_task_status(task_id, status):
    """
    Cập nhật trạng thái của task

    Args:
        task_id: ID của task
        status: Trạng thái mới (SUCCESS, FAILURE, PENDING)

    Returns:
        Task object đã được cập nhật hoặc None nếu không tìm thấy
    """
    with SessionLocal() as session:
        try:
            task = session.query(Task).filter(Task.task_id == task_id).first()
            if not task:
                logger.error(f"Task not found: {task_id}")
                return None

            task.status = status
            task.updated_at = datetime.now()

            session.commit()
            session.refresh(task)
            logger.info(f"Task status updated: {task_id} -> {status}")
            return task
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating task status: {e}")
            raise
