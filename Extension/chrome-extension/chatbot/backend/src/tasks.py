import asyncio
import json
import logging
import time
import os
import datetime
from copy import copy
from typing import List, Dict, Any, Optional, Tuple
import uuid
import re
import tempfile
import sys
import io
import traceback
import random
import string
from functools import partial

from celery import shared_task
from sqlalchemy.util import asbool
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from database import engine

Session = sessionmaker(bind=engine)

from sync_data import collect_dbpedia_topics
from dbpedia import (
    whitelist_concepts,
    search_page,
    parse_response,
    get_category_detail,
    get_page_detail,
    get_all_pages_of_category,
    get_page_relationship,
)
from summarizer import summarize_text
from utils import setup_logging
from database import get_celery_app
from brain import (
    detect_route,
    openai_chat_complete,
    detect_user_intent,
    get_embedding,
    gen_doc_prompt,
    get_financial_agent_handle,
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
)
from configs import DEFAULT_COLLECTION_NAME, DBPEDIA_COLLECTION_NAME
from models import (
    update_chat_conversation,
    get_conversation_messages,
    insert_page,
    get_pages,
    insert_category,
    get_topics,
    get_categories,
    get_topic_by_id,
    get_category_by_id,
    get_topic_by_name,
    get_category_by_name,
)
from vectorize import search_vector, add_vector, get_record_by_id
from rerank import rerank_documents

# Import các mô hình embedding mới
from embeddings import (
    get_embedding_tfidf,
    get_embedding_bm25,
    get_embedding_bmx,
    get_embedding_roberta,
    get_embedding_xlm_roberta,
    get_embedding_distilbert,
    get_embedding_hybrid_tfidf_bert,
    get_embedding_hybrid_bm25_bert,
    get_embedding_hybrid_bmx_bert,
    compare_embedding_models,
    save_comparison_results,
    fine_tune_tfidf,
    fine_tune_bm25,
    fine_tune_bmx,
    load_transformer_model,
    fine_tune_transformer_model,
    fine_tune_hybrid_model,
    load_hybrid_model,
    MODELS_DIR,
    TfidfEmbedder,
    BM25Embedder,
    BMXEmbedder,
    load_all_base_models,
)

import torch
import shutil
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModel,
    RobertaTokenizer,
    RobertaModel,
    XLMRobertaTokenizer,
    XLMRobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
)

from ai_summarization import (
    batch_process_entries as ai_batch_process_entries,
    extract_dbpedia_topics,
    process_visible_content,
    extract_topics_with_lda,
    init_models,
    NLP,
)

setup_logging()
logger = logging.getLogger(__name__)

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()


def get_embedding_model(model_name=None):
    """Lấy hàm embedding dựa trên tên mô hình"""
    if model_name is None:
        model_name = DEFAULT_EMBEDDING_MODEL

    if model_name not in EMBEDDING_MODELS:
        logger.warning(
            f"Model {model_name} not found. Using default model {DEFAULT_EMBEDDING_MODEL} instead."
        )
        model_name = DEFAULT_EMBEDDING_MODEL

    return EMBEDDING_MODELS[model_name]


def follow_up_question(history, question):
    user_intent = detect_user_intent(history, question)
    logger.info(f"User intent: {user_intent}")
    return user_intent


@shared_task()
def bot_rag_answer_message(history, question, embedding_model=None):
    # Follow-up question
    new_question = follow_up_question(history, question)
    # Embedding text with selected model
    embedding_fn = get_embedding_model(embedding_model)
    vector = embedding_fn(new_question)
    logger.info(f"Get vector using model {embedding_model}: {new_question}")

    # Search documents
    top_docs = search_vector(DEFAULT_COLLECTION_NAME, vector, 2)
    logger.info(f"Top docs: {top_docs}")

    # Rerank documents
    # ranked_docs = rerank_documents(top_docs, new_question)

    openai_messages = history + [
        {"role": "user", "content": gen_doc_prompt(top_docs)},
        {"role": "user", "content": question},
    ]

    logger.info(f"Openai messages: {openai_messages}")

    assistant_answer = openai_chat_complete(openai_messages)

    logger.info(f"Bot RAG reply: {assistant_answer}")
    return assistant_answer


def index_document(
    id, title, content, collection_name=DEFAULT_COLLECTION_NAME, embedding_model=None
):
    embedding_fn = get_embedding_model(embedding_model)
    vector = embedding_fn(title)
    add_vector_status = add_vector(
        collection_name=collection_name,
        vectors={
            id: {"vector": vector, "payload": {"title": title, "content": content}}
        },
    )
    logger.info(
        f"Add vector status (using model {embedding_model}): {add_vector_status}"
    )
    return add_vector_status


def get_summarized_response(response):
    output = summarize_text(response)
    logger.info("Summarized response: %s", output)
    return output


@shared_task()
def llm_handle_message(bot_id, user_id, question, embedding_model=None):
    logger.info("Start handle message")
    # Update chat conversation
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)
    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[:-1]
    # Bot generation with selected embedding model
    response = bot_rag_answer_message(history, question, embedding_model)
    logger.info(f"Chatbot response: {response}")
    # Summarize response
    summarized_response = get_summarized_response(response)
    # Save response to history
    update_chat_conversation(bot_id, user_id, summarized_response, False)
    # Return response
    return {"role": "assistant", "content": response}


@shared_task()
def insert_page_task(label, page_uri, abstract, comment, category_id):
    session = Session()
    try:
        insert_page(
            label=label,
            uri=page_uri,
            abstract=abstract,
            comment=comment,
            category_id=category_id,
            parent_id=None,
        )
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Error in sync_page_detail: %s", exc)
        raise
    finally:
        session.close()


@shared_task()
def sync_page_detail(category_id, page_uri):
    detail = get_page_detail(page_uri)
    label = page_uri.split("/")[-1]
    if detail:
        # related_pages = get_page_relationship(label)
        # continue here ???
        abstract = detail.get("DBpedia Abstract")
        comment = detail.get("comment")
        logger.info("Category id 2: %s", category_id)
        insert_page(
            label=label,
            uri=page_uri,
            abstract=abstract,
            comment=comment,
            category_id=category_id,
            parent_id=None,
        )
        logger.info(f"Index document: {label}")
    else:
        logger.error("Detail empty: %s", page_uri)


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def sync_pages_of_category(self, category_suffix, category_id, loop_limit):
    all_pages = get_all_pages_of_category(category_suffix)
    time.sleep(5)
    page_uris = list(all_pages.keys())
    for i in range(len(page_uris)):
        page_uri = page_uris[i]
        if "category" in page_uri:
            if loop_limit >= 1:
                new_category_suffix = page_uri.split(":")[-1]
                new_category_uri = page_uri
                new_category_id = insert_category(
                    label=new_category_suffix,
                    uri=new_category_uri,
                    topic_id=None,
                    parent_id=category_id,
                ).id
                sync_pages_of_category.apply_async(
                    (new_category_suffix, new_category_id, loop_limit - 1),
                    countdown=10 * (i + 1),
                )
            else:
                logger.error(f"Loop limit exceeded: {page_uri}")
        else:
            logger.info(f"Process page: {category_id} - {page_uri}")
            sync_page_detail.apply_async(
                (category_id, page_uri), countdown=10 * (i + 1)
            )


@shared_task(bind=True, max_retries=3, default_retry_delay=120)
def sync_category_to_database(self, label, category_uri, topic_id):
    category_suffix = category_uri.split(":")[-1]
    logger.info("Process category: %s", label)
    try:
        if label:
            category_id = insert_category(label, category_uri, topic_id).id
            logger.info("Category id 1: %s", category_id)
            sync_pages_of_category(
                category_suffix=category_suffix, category_id=category_id, loop_limit=2
            )  # change the deep level
        else:
            logger.error(f"Category label empty: {label}")
    except Exception as exc:
        logger.error("Error in processing category: %s", category_uri)
        self.retry(exc=exc)


@celery_app.task(name="sync_dbpedia_to_database")
def sync_dbpedia_to_database(topics: List[str] = None):
    logger.info("Start sync dbpedia to database for topics: %s", topics)
    if not topics or len(topics) == 0 or topics is None:
        topics = whitelist_concepts()
    try:
        topic_map = collect_dbpedia_topics(topics)
        for topic_id in topic_map.keys():
            logger.info("Process topic: %s", topic_id)
            topic_categories = topic_map[topic_id]
            for category_uri in topic_categories.keys():
                label = topic_categories[category_uri]
                sync_category_to_database.delay(label, category_uri, topic_id)
        return {"status": "success", "topic_count": len(topics)}
    except Exception as e:
        logger.error(f"Error in sync_dbpedia_to_database: {e}")
        return {"status": "error", "error": str(e)}


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def index_dbpedia_page(
    self,
    id,
    uri,
    label,
    abstract,
    comment,
    topic_id,
    category_id,
    parent_id,
    collection_name=DBPEDIA_COLLECTION_NAME,
    embedding_model=None,
):
    try:
        # Sử dụng mô hình embedding được chỉ định
        embedding_fn = get_embedding_model(embedding_model)
        vector = embedding_fn(abstract + " " + comment)

        add_vector_status = add_vector(
            collection_name=collection_name,
            vectors={
                id: {
                    "vector": vector,
                    "payload": {
                        "id": id,
                        "uri": uri,
                        "label": label,
                        "abstract": abstract,
                        "comment": comment,
                        "topic_id": topic_id,
                        "category_id": category_id,
                        "parent_id": parent_id,
                        "embedding_model": embedding_model,  # Lưu thông tin về mô hình đã sử dụng
                    },
                }
            },
        )
        logger.info(
            f"Add dbpedia to vector db status (using model {embedding_model}): {add_vector_status}"
        )
        return {"status": "success"}
    except Exception as exc:
        logger.error(f"Error in indexing dbpedia page: {uri}")
        self.retry(exc=exc)


@shared_task()
def index_dbpedia_pages(
    cate_id, topic_id, collection_name=DBPEDIA_COLLECTION_NAME, embedding_model=None
):
    pages = get_pages(cate_id)
    for page in pages:
        try:
            logger.info(f"Indexing page: {page.uri} with model: {embedding_model}")
            index_dbpedia_page.delay(
                id=page.id,
                uri=page.uri,
                label=page.label,
                abstract=page.abstract,
                comment=page.comment,
                topic_id=topic_id,
                category_id=cate_id,
                parent_id=page.parent_id,
                collection_name=collection_name,
                embedding_model=embedding_model,
            )
        except Exception as e:
            logger.error(f"Error in indexing page: {page.uri} - {e}")
    return {"status": "success"}


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def index_dbpedia_topics(
    self, collection_name=DBPEDIA_COLLECTION_NAME, topic=[], embedding_model=None
):
    if not topic:
        list_topics = get_topics()
    else:
        list_topics = [get_topic_by_name(x) for x in topic]
    for topic in list_topics:
        if topic is None:
            continue
        try:
            categories = get_categories(topic.id)
            for cate in categories:
                index_dbpedia_pages.delay(
                    cate.id, topic.id, collection_name, embedding_model
                )
        except Exception as exc:
            logger.error(f"Error in indexing dbpedia pages: {topic.label}")
            self.retry(exc=exc)


@shared_task()
def index_category_to_vector_db(
    collection_name=DBPEDIA_COLLECTION_NAME,
    category_labels=[],
    limit=2,
    embedding_model=None,
):
    for label in category_labels[:limit]:
        category = get_category_by_name(label)
        index_dbpedia_pages.delay(
            category.id, category.topic_id, collection_name, embedding_model
        )

    return {"status": "success"}


def get_concept_format(page):
    topic = get_topic_by_id(page["topic_id"])
    category = get_category_by_id(page["category_id"])
    if topic is None or category is None:
        return {}
    else:
        # Đảm bảo embedding_model luôn có giá trị
        embedding_model = page.get("embedding_model")
        if embedding_model is None or embedding_model == "unknown":
            # Sử dụng DEFAULT_EMBEDDING_MODEL nếu không có giá trị
            embedding_model = DEFAULT_EMBEDDING_MODEL

        return {
            "topic": {"uri": topic.uri, "label": topic.label},
            "category": {"uri": category.uri, "label": category.label},
            "relatedConcept": {"type": "uri", "value": page["uri"]},
            "relatedConceptLabel": {
                "type": "literal",
                "xml:lang": "en",
                "value": page["label"],
            },
            "relationshipType": {"type": "literal", "value": "incoming link"},
            "abstract": page["abstract"],
            "comment": page["comment"],
            "embedding_model": embedding_model,  # Sử dụng giá trị đã xử lý
        }


def search_dbpedia_pages(keywords, limit=2, embedding_model=None):
    # Sử dụng mô hình embedding được chỉ định
    embedding_fn = get_embedding_model(embedding_model)
    vector = embedding_fn(" ".join(keywords))

    logger.info(f"Searching with keywords: {keywords} using model: {embedding_model}")
    pages = search_vector(
        collection_name=DBPEDIA_COLLECTION_NAME, vector=vector, limit=limit
    )

    # page attributes
    """
    {
        "id": id,
        "uri": uri,
        "label": label,
        "abstract": abstract,
        "comment": comment,
        "topic_id": topic_id,
        "category_id": category_id,
        "embedding_model": embedding_model  # Thêm thông tin về mô hình embedding
    }
    """
    # Đảm bảo thông tin embedding_model được thiết lập cho mỗi trang tìm thấy
    for page in pages:
        if page.get("embedding_model") is None:
            page["embedding_model"] = embedding_model

    results = [get_concept_format(x) for x in pages]
    return [x for x in results if x]


def get_concepts(learning, embedding_model=None):
    """
    Get concepts from learning with specified embedding model
    :param learning: learning data
    :param embedding_model: embedding model to use
    :return:
    {
    "entry1": [
            {
                "topic": {
                    "uri": topic.uri,
                    "label": topic.label
                },
                "category": {
                    "uri": category.uri,
                    "label": category.label
                },
                "relatedConcept": {
                  "type": "uri",
                  "value": page['uri']
                },
                "relatedConceptLabel": {
                  "type": "literal",
                  "xml:lang": "en",
                  "value": page['label']
                },
                "relationshipType": {
                  "type": "literal",
                  "value": "incoming link"
                },
                "embedding_model": page.get('embedding_model', "unknown")  # Thêm thông tin về mô hình embedding
            }
        ],
        "entry2": []
    }
    """
    output = {}
    for learn in learning:
        entry_id = learn["entry_id"]
        keyword = learn["exact_keywords"]
        pages = search_dbpedia_pages(keyword, embedding_model=embedding_model)
        output[entry_id] = pages
    return output


@shared_task()
def extract_pages_mapping(
    json_file_path, sample=0, limit=1000, embedding_model=None, save_output=True
):
    """
    Extract page mappings from JSON file
    :param json_file_path: path to JSON file
    :param sample: number of items to process (0 for all)
    :param limit: limit number of pages per entry
    :param embedding_model: embedding model to use
    :param save_output: whether to save output to file
    :return: results
    """
    # Load the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)

    results = []
    logger.info(
        f"Processing {len(data)} items from {json_file_path} using model: {embedding_model}"
    )
    if sample:
        data = data[:sample]

    # Extract the keywords from the JSON data
    for item in tqdm(data, desc="Extracting pages", unit="item"):
        # Retrieve pages based on the keywords
        keywords = item.get("exact_keywords", [])
        entry_id = item["entry_id"]
        if len(keywords) == 0:
            logger.warning(f"Entry {entry_id} has no keywords")
            continue

        logger.info(f"Processing entry {entry_id} with keywords: {keywords}")
        # Đảm bảo truyền mô hình embedding vào hàm search_dbpedia_pages
        pages = search_dbpedia_pages(
            keywords, limit=limit, embedding_model=embedding_model
        )

        # Đảm bảo mọi trang đều có thông tin embedding_model
        for page in pages:
            if page.get("embedding_model") is None:
                page["embedding_model"] = embedding_model

        # Print the retrieved pages
        logger.info(f"Found {len(pages)} pages for entry {entry_id}")
        results.append({"entry_id": entry_id, "pages": pages})

    logger.info(
        f"Completed processing {len(data)} items, found pages for {len(results)} entries"
    )

    # Tạo tên file dựa trên mô hình embedding được sử dụng
    model_name = embedding_model or DEFAULT_EMBEDDING_MODEL

    # Lưu kết quả vào file nếu được yêu cầu
    if save_output:
        # Tạo tên file dựa trên mô hình embedding được sử dụng
        output_filename = f"extraction_results_{model_name}.json"
        output_path = os.path.join("/usr/src/app/src/results", output_filename)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved output to {output_path}")

    return results


@shared_task()
def compare_embedding_models_task(
    json_file_path, sample=10, limit=10, output_file=None
):
    """
    So sánh hiệu suất của các mô hình embedding
    :param json_file_path: đường dẫn đến file JSON chứa dữ liệu học tập
    :param sample: số lượng mẫu để xử lý
    :param limit: giới hạn số lượng trang cho mỗi mục
    :param output_file: đường dẫn để lưu kết quả so sánh
    :return: kết quả so sánh
    """
    # Đọc file dữ liệu
    with open(json_file_path, "r") as file:
        data = json.load(file)

    if sample > 0:
        data = data[:sample]

    # Chuẩn bị dữ liệu query (từ exact_keywords) và corpus (từ abstract và comment)
    query_texts = []
    for item in data:
        keywords = item.get("exact_keywords", [])
        if keywords:
            query_texts.append(" ".join(keywords))

    # Lấy một mẫu của các trang từ DBpedia để so sánh
    session = Session()
    try:
        # Sử dụng trực tiếp model Page thay vì get_pages.__self__
        from models import Page

        pages = session.query(Page).limit(sample).all()  # Lấy các trang từ CSDL
        corpus_texts = []
        for page in pages:
            if page.abstract and page.comment:
                corpus_texts.append(f"{page.abstract} {page.comment}")
    finally:
        session.close()

    # So sánh các mô hình
    model_names = list(EMBEDDING_MODELS.keys())
    results = compare_embedding_models(query_texts, corpus_texts, model_names)

    # Lưu kết quả
    if output_file is None:
        output_dir = os.path.dirname(json_file_path)
        output_file = os.path.join(output_dir, "embedding_comparison_results.json")

    save_comparison_results(results, output_file)

    return results


@shared_task()
def fine_tune_models_task(json_file_path, sample=100, version=None, save_model=True):
    """
    Fine-tune các mô hình embedding trên dữ liệu học tập và lưu lại
    :param json_file_path: đường dẫn đến file JSON chứa dữ liệu học tập
    :param sample: số lượng mẫu để xử lý
    :param version: phiên bản của mô hình sau khi fine-tune (nếu không cung cấp, sẽ tự động tạo)
    :param save_model: có lưu mô hình sau khi fine-tune hay không
    :return: trạng thái
    """
    # Tạo phiên bản dựa trên thời gian hiện tại nếu không được cung cấp
    if version is None:
        from datetime import datetime

        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Đọc file dữ liệu
    with open(json_file_path, "r") as file:
        data = json.load(file)

    if sample > 0:
        data = data[:sample]

    # Chuẩn bị dữ liệu huấn luyện (từ exact_keywords)
    train_texts = []
    for item in data:
        keywords = item.get("exact_keywords", [])
        if keywords:
            train_texts.append(" ".join(keywords))

    results = {"version": version, "data_size": len(train_texts), "models": {}}

    # Định nghĩa các mô hình truyền thống cần fine-tune
    traditional_models = [
        {"name": "tfidf", "function": fine_tune_tfidf},
        {"name": "bm25", "function": fine_tune_bm25},
        {"name": "bmx", "function": fine_tune_bmx},
    ]

    # Fine-tune các mô hình truyền thống
    for model in traditional_models:
        try:
            logger.info(
                f"Fine-tuning {model['name'].upper()} model with {len(train_texts)} texts"
            )
            model_instance = model["function"](
                train_texts, version=version, save_model=save_model
            )

            if model_instance is not None:
                results["models"][model["name"]] = {
                    "status": "success",
                    "metadata": model_instance.metadata.to_dict()
                    if hasattr(model_instance, "metadata")
                    else {},
                }
            else:
                results["models"][model["name"]] = {
                    "status": "error",
                    "error": f"{model['name'].upper()} model is not available",
                }
        except Exception as e:
            logger.error(f"Error fine-tuning {model['name'].upper()} model: {e}")
            results["models"][model["name"]] = {"status": "error", "error": str(e)}

    # Định nghĩa các mô hình transformer cần fine-tune
    transformer_models = [
        {"name": "roberta", "model_name": "roberta-base"},
        {"name": "xlm-roberta", "model_name": "xlm-roberta-base"},
        {"name": "distilbert", "model_name": "distilbert-base-uncased"},
    ]

    # Fine-tune các mô hình transformer
    for model in transformer_models:
        try:
            logger.info(
                f"Fine-tuning {model['name']} model with {len(train_texts)} texts"
            )

            try:
                # Fine-tune mô hình transformer
                transformer_model = fine_tune_transformer_model(
                    model["model_name"],
                    train_texts,
                    version=version,
                    save_model=save_model,
                )
                results["models"][model["name"]] = {
                    "status": "success",
                    "metadata": {
                        "model_name": model["model_name"],
                        "trained_on": len(train_texts),
                        "version": version,
                    },
                }

                # Tải lại mô hình đã fine-tune
                try:
                    load_result = load_transformer_model(
                        model["name"],
                        model["model_name"],
                        version,
                        force_download=False,
                    )
                    # Lưu thông tin vào kết quả (không return sớm)
                    if isinstance(load_result, tuple) and len(load_result) == 2:
                        success, model_info = load_result
                        if success:
                            # Cập nhật metadata nếu tải thành công
                            results["models"][model["name"]]["metadata_loaded"] = (
                                model_info
                            )
                            logger.info(
                                f"Successfully loaded {model['name']} model after fine-tuning"
                            )
                        else:
                            # Thêm thông tin lỗi nếu tải thất bại
                            results["models"][model["name"]]["load_error"] = model_info
                            logger.error(
                                f"Error loading fine-tuned {model['name']} model: {model_info}"
                            )
                    else:
                        # Nếu định dạng không đúng, ghi nhận lỗi nhưng không dừng xử lý
                        results["models"][model["name"]]["load_error"] = (
                            "Unexpected return format from load_transformer_model"
                        )
                        logger.error(
                            f"Unexpected return format from load_transformer_model for {model['name']}"
                        )
                except Exception as e:
                    # Ghi nhận lỗi nhưng không dừng xử lý
                    logger.error(f"Error loading fine-tuned {model['name']} model: {e}")
                    results["models"][model["name"]]["load_error"] = str(e)

            except Exception as e:
                logger.error(f"Error fine-tuning {model['name']} model: {e}")
                results["models"][model["name"]] = {"status": "error", "error": str(e)}

        except Exception as e:
            logger.error(f"Error handling {model['name']} model: {e}")
            results["models"][model["name"]] = {"status": "error", "error": str(e)}

    # Định nghĩa các mô hình hybrid cần fine-tune
    hybrid_models = [
        {
            "name": "hybrid_tfidf_bert",
            "trad_model": "tfidf",
            "transformer_model": "bert-base-uncased",
        },
        {
            "name": "hybrid_bm25_bert",
            "trad_model": "bm25",
            "transformer_model": "bert-base-uncased",
        },
        {
            "name": "hybrid_bmx_bert",
            "trad_model": "bmx",
            "transformer_model": "bert-base-uncased",
        },
    ]

    # Fine-tune các mô hình hybrid
    for model in hybrid_models:
        try:
            logger.info(
                f"Fine-tuning Hybrid {model['trad_model'].upper()} + BERT model with {len(train_texts)} texts"
            )
            hybrid_model = fine_tune_hybrid_model(
                model["trad_model"],
                model["transformer_model"],
                train_texts,
                version=version,
                save_model=save_model,
            )
            results["models"][model["name"]] = {
                "status": "success",
                "metadata": {
                    "base_models": [model["trad_model"], model["transformer_model"]],
                    "trained_on": len(train_texts),
                    "version": version,
                },
            }
        except Exception as e:
            logger.error(
                f"Error fine-tuning Hybrid {model['trad_model'].upper()} + BERT model: {e}"
            )
            results["models"][model["name"]] = {"status": "error", "error": str(e)}

    # Lưu kết quả fine-tune vào file để tham khảo
    output_dir = os.path.dirname(json_file_path)
    results_path = os.path.join(output_dir, f"fine_tune_results_{version}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Luôn trả về kết quả đầy đủ của tất cả mô hình
    return results


@shared_task()
def get_model_versions():
    """
    Lấy danh sách các phiên bản mô hình đã lưu
    :return: danh sách các phiên bản
    """
    try:
        models = {}
        # Danh sách tất cả các loại mô hình
        model_types = [
            "tfidf",
            "bm25",
            "bmx",  # Truyền thống
            "roberta",
            "xlm-roberta",
            "distilbert",  # Transformer
            "hybrid_tfidf_bert",
            "hybrid_bm25_bert",
            "hybrid_bmx_bert",  # Hybrid
        ]

        for model_type in model_types:
            model_dir = os.path.join(MODELS_DIR, model_type)
            if not os.path.exists(model_dir):
                models[model_type] = []
                continue

            versions = []
            for version in os.listdir(model_dir):
                version_dir = os.path.join(model_dir, version)
                if os.path.isdir(version_dir):
                    metadata_path = os.path.join(version_dir, "metadata.json")
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                    versions.append({"version": version, "metadata": metadata})

            # Sắp xếp phiên bản theo thời gian tạo (mới nhất trước)
            versions.sort(
                key=lambda x: x.get("metadata", {}).get("created_at", ""), reverse=True
            )
            models[model_type] = versions

        return models
    except Exception as e:
        logger.error(f"Error getting model versions: {e}")
        return {"error": str(e)}


@shared_task()
def load_model_version(model_type, version="latest", force_download=False):
    """
    Tải một phiên bản mô hình cụ thể
    :param model_type: loại mô hình (tfidf, bm25, roberta, xlm-roberta, distilbert, hybrid_*)
    :param version: phiên bản muốn tải (hoặc "latest" cho phiên bản mới nhất)
    :param force_download: có tải lại mô hình từ Hugging Face không
    :return: thông tin về mô hình đã tải
    """
    try:
        # Kiểm tra khi tải mô hình transformer
        if model_type in ["roberta", "xlm-roberta", "distilbert"]:
            transformer_model_map = {
                "roberta": "roberta-base",
                "xlm-roberta": "FacebookAI/xlm-roberta-base",
                "distilbert": "distilbert-base-uncased",
            }

            # Thư mục models
            MODELS_DIR = os.path.abspath(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            )

            model_name = transformer_model_map.get(model_type)
            model_dir = os.path.join(MODELS_DIR, model_type)

            # Đảm bảo thư mục tồn tại
            os.makedirs(model_dir, exist_ok=True)
            # Nếu force_download, tải lại mô hình từ HuggingFace
            if force_download:
                print("Tải lại mô hình từ Hugging Face...")
                logger.info(f"Tải lại mô hình {model_name} từ Hugging Face...")

                try:
                    # Thư mục pretrained để lưu mô hình gốc
                    pretrained_dir = os.path.join(model_dir, "pretrained")
                    os.makedirs(pretrained_dir, exist_ok=True)

                    # Tải và lưu mô hình vào thư mục pretrained
                    if model_type == "roberta":
                        tokenizer = RobertaTokenizer.from_pretrained(model_name)
                        model = RobertaModel.from_pretrained(model_name)
                    elif model_type == "xlm-roberta":
                        # Xử lý đặc biệt cho XLM-RoBERTa
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = XLMRobertaModel.from_pretrained(model_name)
                    elif model_type == "distilbert":
                        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
                        model = DistilBertModel.from_pretrained(model_name)

                    # Lưu mô hình và tokenizer
                    tokenizer.save_pretrained(pretrained_dir)
                    model.save_pretrained(pretrained_dir)

                    logger.info(f"Đã tải và lưu {model_name} vào {pretrained_dir}")

                    # Kiểm tra xem các file quan trọng có tồn tại không
                    required_files = [
                        "pytorch_model.bin",
                        "config.json",
                        "tokenizer.json",
                    ]
                    missing_files = [
                        f
                        for f in required_files
                        if not (
                            os.path.exists(os.path.join(pretrained_dir, f))
                            or os.path.exists(
                                os.path.join(
                                    pretrained_dir, f.replace(".bin", ".safetensors")
                                )
                            )
                        )
                    ]

                    if missing_files:
                        logger.warning(
                            f"Các file sau đây không tồn tại trong thư mục {pretrained_dir}: {missing_files}"
                        )

                    # Tạo và trả về metadata
                    metadata = {
                        "model_name": model_name,
                        "model_type": model_type,
                        "version": "pretrained",
                        "created_at": datetime.datetime.now().isoformat(),
                        "is_quantized": False,
                        "source": "HuggingFace (forced download)",
                    }
                    return {"status": "success", "metadata": metadata}

                except Exception as e:
                    logger.error(f"Lỗi khi tải mô hình {model_name}: {e}")
                    return {
                        "status": "error",
                        "error": f"Lỗi khi tải mô hình từ Hugging Face: {str(e)}",
                    }

            else:
                # Tiếp tục với load_transformer_model
                try:
                    result = load_transformer_model(
                        model_type, model_name, version, force_download
                    )
                    # Kiểm tra xem result có phải là tuple không
                    if isinstance(result, tuple) and len(result) == 2:
                        success, model_info = result
                        if success:
                            return {"status": "success", "metadata": model_info}
                        else:
                            return {"status": "error", "error": model_info}
                    else:
                        # Xử lý trường hợp không trả về tuple
                        return {
                            "status": "error",
                            "error": "Unexpected return format from load_transformer_model",
                        }
                except TypeError as e:
                    logger.error(
                        f"Error unpacking result from load_transformer_model: {e}"
                    )
                    return {"status": "error", "error": str(e)}
        # Tải mô hình hybrid
        elif model_type.startswith("hybrid_"):
            parts = model_type.split("_")
            if len(parts) >= 3:
                trad_model_type = parts[1]  # tfidf, bm25, bmx
                transformer_type = "bert"  # Mặc định là bert

                success, model_info = load_hybrid_model(
                    trad_model_type, f"{transformer_type}-base-uncased", version
                )

                if success:
                    return {"status": "success", "metadata": model_info}
                else:
                    # Nếu thất bại, đảm bảo mô hình truyền thống đã được tải
                    if trad_model_type == "tfidf":
                        tfidf_model = TfidfEmbedder.load(version)
                    elif trad_model_type == "bm25":
                        bm25_model = BM25Embedder.load(version)
                    elif trad_model_type == "bmx" and BMXEmbedder is not None:
                        bmx_model = BMXEmbedder.load(version)

                    # Sau đó tải lại mô hình hybrid
                    success, model_info = load_hybrid_model(
                        trad_model_type, f"{transformer_type}-base-uncased", version
                    )

                    if success:
                        return {"status": "success", "metadata": model_info}
                    else:
                        return {"status": "error", "error": model_info}
            else:
                return {
                    "status": "error",
                    "error": f"Invalid hybrid model type: {model_type}",
                }
        # Tải mô hình truyền thống
        else:
            # Tải các mô hình cơ bản
            load_all_base_models()

            if model_type == "tfidf":
                model = TfidfEmbedder.load(version)
                if model:
                    return {"status": "success", "metadata": model.metadata.to_dict()}
                else:
                    return {
                        "status": "error",
                        "error": f"TF-IDF model version {version} not found",
                    }
            elif model_type == "bm25":
                model = BM25Embedder.load(version)
                if model:
                    return {"status": "success", "metadata": model.metadata.to_dict()}
                else:
                    return {
                        "status": "error",
                        "error": f"BM25 model version {version} not found",
                    }
            elif model_type == "bmx":
                if BMXEmbedder is not None:
                    model = BMXEmbedder.load(version)
                    if model:
                        return {
                            "status": "success",
                            "metadata": model.metadata.to_dict(),
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"BMX model version {version} not found",
                        }
                else:
                    return {"status": "error", "error": "BMX model is not available"}
            else:
                return {"status": "error", "error": f"Unknown model type: {model_type}"}
    except Exception as e:
        logger.error(f"Error loading model {model_type} version {version}: {e}")
        return {"status": "error", "error": str(e)}


@shared_task()
def delete_model_version(model_type, version):
    """
    Xóa một phiên bản mô hình cụ thể
    :param model_type: loại mô hình (tfidf, bm25, bmx, roberta, xlm-roberta, distilbert,
                       hybrid_tfidf_bert, hybrid_bm25_bert, hybrid_bmx_bert)
    :param version: phiên bản muốn xóa
    :return: kết quả xóa
    """
    try:
        # Kiểm tra loại mô hình có hợp lệ không
        valid_model_types = [
            "tfidf",
            "bm25",
            "bmx",  # Truyền thống
            "roberta",
            "xlm-roberta",
            "distilbert",  # Transformer
            "hybrid_tfidf_bert",
            "hybrid_bm25_bert",
            "hybrid_bmx_bert",  # Hybrid
        ]

        if model_type not in valid_model_types:
            return {
                "status": "error",
                "error": f"Invalid model type: {model_type}. Valid types are: {', '.join(valid_model_types)}",
            }

        model_dir = os.path.join(MODELS_DIR, model_type, version)
        if not os.path.exists(model_dir):
            return {
                "status": "error",
                "error": f"Model {model_type} version {version} not found",
            }

        # Xóa thư mục mô hình
        shutil.rmtree(model_dir)

        # Nếu là transformer hoặc hybrid model, có thể cần xử lý đặc biệt
        if model_type in [
            "roberta",
            "xlm-roberta",
            "distilbert",
        ] or model_type.startswith("hybrid_"):
            # Kiểm tra và xóa các file cache hoặc file bổ sung
            cache_dir = os.path.join(MODELS_DIR, f"{model_type}_cache", version)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

        # Reset biến global trong embeddings.py để model không còn hiển thị là active
        if model_type == "tfidf":
            from embeddings import tfidf_embedder

            tfidf_embedder = None
        elif model_type == "bm25":
            from embeddings import bm25_embedder

            bm25_embedder = None
        elif model_type == "bmx":
            from embeddings import bmx_embedder

            bmx_embedder = None
        elif model_type == "roberta":
            from embeddings import roberta_model

            roberta_model = None
        elif model_type == "xlm-roberta":
            from embeddings import xlm_roberta_model

            xlm_roberta_model = None
        elif model_type == "distilbert":
            from embeddings import distilbert_model

            distilbert_model = None
        elif model_type == "hybrid_tfidf_bert":
            from embeddings import hybrid_tfidf_bert_model

            hybrid_tfidf_bert_model = None
        elif model_type == "hybrid_bm25_bert":
            from embeddings import hybrid_bm25_bert_model

            hybrid_bm25_bert_model = None
        elif model_type == "hybrid_bmx_bert":
            from embeddings import hybrid_bmx_bert_model

            hybrid_bmx_bert_model = None

        # Xử lý trường hợp là phiên bản được symlink từ "latest"
        latest_symlink = os.path.join(MODELS_DIR, model_type, "latest")
        if os.path.islink(latest_symlink) and os.readlink(latest_symlink) == version:
            # Xóa symlink vì nó trỏ đến phiên bản vừa bị xóa
            os.unlink(latest_symlink)
            # Tìm phiên bản mới nhất còn lại (nếu có)
            versions = [
                d
                for d in os.listdir(os.path.join(MODELS_DIR, model_type))
                if os.path.isdir(os.path.join(MODELS_DIR, model_type, d))
                and d != "latest"
            ]
            if versions:
                versions.sort(
                    reverse=True
                )  # Sắp xếp giảm dần, phiên bản mới nhất đầu tiên
                # Tạo symlink mới trỏ đến phiên bản mới nhất
                os.symlink(versions[0], latest_symlink)

        return {
            "status": "success",
            "message": f"Deleted {model_type} model version {version}",
        }
    except Exception as e:
        logger.error(f"Error deleting model {model_type} version {version}: {e}")
        return (False, str(e))


@celery_app.task(name="ai_summarize")
def ai_summarize_task(content: str):
    """
    Task Celery để tóm tắt nội dung và trích xuất chủ đề
    """
    try:
        result = process_visible_content(content)
        return result
    except Exception as e:
        logger.error(f"Error in ai_summarize_task: {e}")
        return {"error": str(e)}


@celery_app.task(name="ai_summarize_khanh")
def ai_summarize_khanh_task(content: str):
    """
    Task Celery để tóm tắt nội dung và trích xuất chủ đề
    Sử dụng phiên bản cải tiến process_visible_content_khanh ưu tiên LDA và keyword hơn distilbert
    """
    try:
        from ai_summarization import process_visible_content_khanh

        result = process_visible_content_khanh(content)
        return result
    except Exception as e:
        logger.error(f"Error in ai_summarize_khanh_task: {e}")
        return {"error": str(e)}


@celery_app.task(name="ai_extract_topics")
def ai_extract_topics_task(content: str):
    """
    Task Celery để trích xuất các chủ đề từ nội dung
    """
    try:
        topics = extract_dbpedia_topics(content)
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error in ai_extract_topics_task: {e}")
        return {"error": str(e)}


@celery_app.task(name="ai_batch_process")
def ai_batch_process_task(entries: List[Dict]):
    """
    Task Celery để xử lý hàng loạt các entry
    """
    try:
        result = ai_batch_process_entries(entries)
        return result
    except Exception as e:
        logger.error(f"Error in ai_batch_process_task: {e}")
        return {"error": str(e)}


@celery_app.task(bind=True)
def extract_data_auto_task(
    self,
    file_path,
    sample=False,
    limit=None,
    embedding_model="distilbert",
    ensure_count=True,
):
    """
    Task để tự động trích xuất data từ file history_learning_data.json,
    sử dụng kết hợp 3 phương pháp (LDA, DistilBERT, keyword extraction)
    để tự động trích xuất topic, lưu topics vào database,
    sau đó tìm kiếm categories và liên kết đến các pages trên DBpedia.

    1. Lưu topics vào database
    2. Tìm các categories từ topics đã lưu
    3. Từ categories tìm các pages liên quan trên DBpedia

    Tuân thủ đúng quy trình như cách cũ: topic -> category -> page
    và đảm bảo dữ liệu được lưu trong MariaDB

    Args:
        file_path: Đường dẫn tới file dữ liệu
        sample: Số lượng mẫu cần xử lý hoặc False nếu không sampling
        limit: Giới hạn số lượng entries
        embedding_model: Mô hình embedding được sử dụng
        ensure_count: Đảm bảo đủ số lượng kết quả được xử lý (mặc định: True)
    """
    # Ghi log để kiểm tra các tham số
    logger.info(
        f"extract_data_auto_task called with: file_path={file_path}, sample={sample}, limit={limit}, embedding_model={embedding_model}, ensure_count={ensure_count}"
    )

    # Import các hàm cần thiết
    from models import get_topic_by_name, get_categories, get_pages
    from sync_data import collect_dbpedia_topics

    # Sửa: không cần import utils.api_utils
    from ai_summarization import process_visible_content

    logger.info(f"Starting extract_data_auto_task with file: {file_path}")

    try:
        # Đọc dữ liệu từ file JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Xác định số lượng mẫu cần xử lý
        sample_size = 0
        if sample:
            if isinstance(sample, bool):
                sample_size = min(10, len(data))  # Mặc định 10 nếu sample=True
            else:
                sample_size = min(int(sample), len(data))

        # Kiểm tra cấu trúc dữ liệu
        if len(data) > 0:
            first_entry = data[0]
            logger.info(f"First entry structure: {list(first_entry.keys())}")
            logger.info(f"Sample data first entry: {first_entry}")

        # Lọc ra những entries có nội dung
        valid_entries = []
        for entry in data:
            # Kiểm tra cả visible_content và content
            if entry.get("visible_content") or entry.get("content"):
                valid_entries.append(entry)

        logger.info(
            f"Found {len(valid_entries)} valid entries with content out of {len(data)} total entries"
        )

        # Xử lý sample parameter
        if sample_size > 0:
            import random

            random.seed(42)  # set seed để có kết quả ổn định

            if len(valid_entries) >= sample_size:
                data_to_process = random.sample(valid_entries, sample_size)
                logger.info(f"Randomly sampled {sample_size} valid entries")
            else:
                # Nếu không đủ entries hợp lệ, sử dụng tất cả những entries có sẵn
                data_to_process = valid_entries
                logger.info(
                    f"Using all {len(valid_entries)} valid entries (less than requested {sample_size})"
                )

                # Nếu ensure_count=True và số lượng entries hợp lệ không đủ, tạo placeholder entries
                if ensure_count and len(valid_entries) < sample_size:
                    logger.warning(
                        f"Not enough valid entries ({len(valid_entries)}) to meet requested sample size ({sample_size})"
                    )

                    # Tạo thêm entries giả để đủ số lượng
                    for i in range(len(valid_entries), sample_size):
                        placeholder_id = f"placeholder_{i}"
                        placeholder_entry = {
                            "entry_id": placeholder_id,
                            "visible_content": f"This is placeholder content for entry {placeholder_id} created to meet the requested sample size.",
                            "url": f"http://example.com/placeholder/{i}",
                            "title": f"Placeholder Entry {i}",
                        }
                        data_to_process.append(placeholder_entry)
                        logger.info(f"Created placeholder entry {placeholder_id}")
        else:
            data_to_process = valid_entries

        # Giới hạn số lượng dữ liệu nếu có limit
        if limit and isinstance(limit, int) and limit > 0:
            data_to_process = data_to_process[:limit]

        logger.info(
            f"Processing {len(data_to_process)} entries for auto topic extraction"
        )

        # Đảm bảo NLP model đã được khởi tạo
        if NLP is None:
            init_models()

        # Thay đổi quy trình: Trước tiên thu thập tất cả các topic từ tất cả các entries
        all_topics_set = set()  # Sử dụng set để tránh trùng lặp
        entries_topics_map = {}  # Map entry_id -> list of topics

        # BƯỚC 1: Trích xuất tất cả các topic từ tất cả các entries
        logger.info("PHASE 1: Extracting topics from all entries using AI")
        for entry_idx, entry in enumerate(data_to_process):
            if entry_idx % 10 == 0:
                logger.info(
                    f"Extracting topics from entry {entry_idx + 1}/{len(data_to_process)}"
                )

            # Lấy entry_id - ưu tiên entry_id trước, nếu không có thì dùng id, nếu không có nữa thì tạo UUID
            entry_id = entry.get("entry_id", entry.get("id", str(uuid.uuid4())))

            # Lấy nội dung - ưu tiên visible_content trước, nếu không có thì dùng content
            content = entry.get("visible_content", entry.get("content", ""))

            if not content:
                logger.warning(f"Entry {entry_id} has no content, skipping")
                continue

            # Sử dụng process_visible_content để kết hợp 3 phương pháp extract topic
            try:
                extraction_result = process_visible_content(content)

                # Phân loại các topics theo phương pháp
                entry_topics = []
                if "topics" in extraction_result and isinstance(
                    extraction_result["topics"], list
                ):
                    for topic in extraction_result["topics"]:
                        topic_name = topic.get("topic", "")
                        if topic_name:
                            entry_topics.append(topic_name)
                            all_topics_set.add(
                                topic_name
                            )  # Thêm vào tập hợp tất cả các topics

                # Thêm topics từ keywords nếu không đủ
                if len(entry_topics) < 3 and "keywords" in extraction_result:
                    top_keywords = extraction_result["keywords"][:3]
                    for keyword in top_keywords:
                        keyword_title = keyword.title()
                        if keyword_title not in entry_topics:
                            entry_topics.append(keyword_title)
                            all_topics_set.add(keyword_title)

                # Lưu danh sách topics cho entry này
                entries_topics_map[entry_id] = entry_topics
                logger.info(
                    f"Found {len(entry_topics)} topics for entry {entry_id}: {entry_topics}"
                )

            except Exception as e:
                logger.error(f"Error extracting topics for entry {entry_id}: {e}")
                logger.error(traceback.format_exc())
                entries_topics_map[entry_id] = []  # Không có topics nào cho entry này

        # Chuyển đổi set thành list
        all_topics_list = list(all_topics_set)
        logger.info(
            f"Total unique topics extracted from all entries: {len(all_topics_list)}"
        )
        logger.info(f"Topics: {all_topics_list}")

        # BƯỚC 2: Đồng bộ tất cả các topic vào database
        logger.info("PHASE 2: Syncing all topics to database")
        try:
            # Sử dụng task sync_dbpedia_to_database để đồng bộ hóa tất cả các topic
            # Chú ý: Gọi trực tiếp (không đợi) -> giải quyết sau đó

            # QUAN TRỌNG: Đợi task đồng bộ hoàn thành - cần thêm .get() để đảm bảo đồng bộ
            sync_task = sync_dbpedia_to_database.delay(all_topics_list)
            sync_result = sync_task.get()  # Đợi task hoàn thành

            logger.info(f"Sync result: {sync_result}")
            if sync_result.get("status") != "success":
                logger.error(f"Error syncing topics to database: {sync_result}")
                raise Exception(f"Failed to sync topics to database: {sync_result}")

            logger.info(
                f"Successfully synced {len(all_topics_list)} topics to database"
            )
        except Exception as e:
            logger.error(f"Error syncing topics to database: {e}")
            logger.error(traceback.format_exc())
            raise

        # BƯỚC 3: Truy vấn database để xây dựng kết quả
        logger.info("PHASE 3: Building results from database")
        results = []

        for entry_idx, entry in enumerate(data_to_process):
            if entry_idx % 10 == 0:
                logger.info(
                    f"Building result for entry {entry_idx + 1}/{len(data_to_process)}"
                )

            entry_id = entry.get("entry_id", entry.get("id", str(uuid.uuid4())))

            # Bỏ qua nếu entry không có topics
            if entry_id not in entries_topics_map or not entries_topics_map[entry_id]:
                logger.warning(f"Entry {entry_id} has no topics, skipping")
                continue

            entry_topics = entries_topics_map[entry_id]
            entry_result = {"entry_id": entry_id, "pages": []}

            # Truy vấn thông tin từ database cho mỗi topic
            for topic_name in entry_topics:
                topic_obj = get_topic_by_name(topic_name)
                if not topic_obj:
                    logger.warning(
                        f"Topic {topic_name} not found in database after sync, skipping"
                    )
                    continue

                # Lấy các categories của topic từ database
                categories = get_categories(topic_obj.id)

                if not categories:
                    logger.warning(f"No categories found for topic {topic_name}")
                    continue
                else:
                    logger.info(
                        f"Found {len(categories)} categories for topic {topic_name}"
                    )

                    # Lấy pages từ mỗi category
                    for category in categories:
                        pages = get_pages(category.id)

                        if not pages:
                            logger.warning(
                                f"No pages found for category {category.label}"
                            )
                            continue

                        logger.info(
                            f"Found {len(pages)} pages for category {category.label}"
                        )

                        # Format các pages theo cấu trúc cần thiết
                        for page in pages:
                            page_entry = {
                                "topic": {
                                    "uri": topic_obj.uri,
                                    "label": topic_obj.label,
                                },
                                "category": {
                                    "uri": category.uri,
                                    "label": category.label,
                                },
                                "relatedConcept": {
                                    "type": "uri",
                                    "value": page.uri,
                                },
                                "relatedConceptLabel": {
                                    "type": "literal",
                                    "xml:lang": "en",
                                    "value": page.label,
                                },
                                "relationshipType": {
                                    "type": "literal",
                                    "value": "incoming link",
                                },
                                "abstract": page.abstract,
                                "comment": page.comment,
                                "embedding_model": embedding_model,
                            }
                            entry_result["pages"].append(page_entry)

            # Chỉ thêm entry vào kết quả nếu có pages
            if entry_result["pages"]:
                results.append(entry_result)
            else:
                logger.warning(
                    f"No valid pages found for entry {entry_id} after querying database"
                )

        # Lưu kết quả vào file
        results_dir = "/usr/src/app/src/results"
        os.makedirs(results_dir, exist_ok=True)

        output_path = os.path.join(
            results_dir, f"extraction_results_auto_combined_{embedding_model}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Extraction completed. Processed {len(data_to_process)} entries, found valid results for {len(results)} entries"
        )
        logger.info(f"Results saved to {output_path}")

        return results

    except Exception as e:
        logger.error(f"Error in extract_data_auto_task: {e}")
        logger.error(traceback.format_exc())
        return []


@celery_app.task(bind=True)
def batch_extract_topics_task(self, entries, compare_methods=True, ensure_count=False):
    """
    Task xử lý danh sách entries từ file history_learning_data.json,
    sử dụng AI để summarize nội dung và xác định các topic liên quan,
    và so sánh với phương pháp truyền thống nếu yêu cầu.

    Args:
        entries (List[Dict]): Danh sách các entry từ history_learning_data.json
        compare_methods (bool): Có so sánh phương pháp AI với phương pháp truyền thống không
        ensure_count (bool): Đảm bảo đủ số lượng kết quả được xử lý

    Returns:
        Dict chứa kết quả xử lý
    """
    task_id = self.request.id
    logger.info(
        f"Starting batch_extract_topics_task with task_id={task_id}, {len(entries)} entries, compare_methods={compare_methods}, ensure_count={ensure_count}"
    )

    # Convert compare_methods sang boolean nếu đã được gửi dưới dạng string
    if isinstance(compare_methods, str):
        compare_methods = compare_methods.lower() in ["true", "yes", "1", "t", "y"]
        logger.info(f"Converted compare_methods string to boolean: {compare_methods}")

    # Convert ensure_count sang boolean
    if isinstance(ensure_count, str):
        ensure_count = ensure_count.lower() in ["true", "yes", "1", "t", "y"]
        logger.info(f"Converted ensure_count string to boolean: {ensure_count}")

    # Đảm bảo NLP model đã được khởi tạo
    if NLP is None:
        init_models()
        logger.info("Initialized NLP models")

    results = []
    total_entries = len(entries)
    entries_with_content = 0  # Đếm số lượng entries có nội dung

    # Thống kê các phương pháp AI
    method_stats = {
        "lda": {"count": 0, "scores": []},
        "distilbert": {"count": 0, "scores": []},
        "keyword-extraction": {"count": 0, "scores": []},
    }

    # Kiểm tra cấu trúc của entry đầu tiên để debug
    if entries and len(entries) > 0:
        first_entry = entries[0]
        logger.info(f"First entry structure: {list(first_entry.keys())}")
        if "content" not in first_entry and "visible_content" not in first_entry:
            logger.warning(
                "Warning: Both 'content' and 'visible_content' fields missing in entries"
            )

        # Thêm cảnh báo nếu so sánh phương pháp truyền thống nhưng không có dữ liệu keywords
        if (
            compare_methods
            and "exact_keywords" not in first_entry
            and "tmp_keywords" not in first_entry
        ):
            logger.warning(
                "Warning: No keyword fields found for traditional method comparison"
            )

    try:
        # Xử lý từng entry
        for idx, entry in enumerate(entries):
            if idx % 5 == 0:
                logger.info(f"Processing entry {idx + 1}/{total_entries}")

            entry_id = entry.get("entry_id", entry.get("id", f"entry_{idx}"))

            # Ưu tiên visible_content nếu có, nếu không thì dùng content
            content = entry.get("visible_content", entry.get("content", ""))

            url = entry.get("url", "")
            title = entry.get("title", "")

            if not content:
                logger.warning(
                    f"Entry {entry_id} has no content or visible_content, skipping"
                )
                # Nếu ensure_count=True và content trống, thêm content giả để đảm bảo đủ số entry
                if ensure_count:
                    content = f"This is placeholder content for entry {entry_id} with URL {url} and title {title}."
                    logger.info(
                        f"Created placeholder content for entry {entry_id} (ensure_count={ensure_count})"
                    )
                else:
                    continue  # Bỏ qua entry này nếu không có ensure_count

            # Tăng số lượng entries có nội dung
            entries_with_content += 1

            # Kết quả xử lý cho entry này
            entry_result = {
                "entry_id": entry_id,
                "url": url,
                "title": title,
                "ai_method": {},
                "traditional_method": {},
            }

            # 1. Phương pháp AI
            logger.info(f"Processing entry {entry_id} with AI method")
            try:
                # Sử dụng process_visible_content để xử lý nội dung
                ai_result = process_visible_content(content)

                # Trích xuất summary
                summary = ai_result.get("summary", "")

                # Trích xuất topics từ AI
                ai_topics = []
                for topic in ai_result.get("topics", []):
                    if isinstance(topic, dict) and "topic" in topic:
                        method = topic.get("method", "unknown")
                        score = topic.get("score", 0.0)

                        # Chuẩn hóa tên phương pháp
                        if (
                            "distilbert" in method
                            or "zero-shot" in method
                            or "cosine-similarity" in method
                        ):
                            method_for_stats = "distilbert"
                        elif "lda" in method:
                            method_for_stats = "lda"
                        elif "keyword" in method:
                            method_for_stats = "keyword-extraction"
                        else:
                            method_for_stats = method

                        # Thêm vào thống kê phương pháp
                        if method_for_stats in method_stats:
                            method_stats[method_for_stats]["count"] += 1
                            method_stats[method_for_stats]["scores"].append(score)

                        ai_topics.append(
                            {
                                "name": topic.get("topic", ""),
                                "score": score,
                                "method": method,
                            }
                        )

                # Lưu kết quả của phương pháp AI
                entry_result["ai_method"] = {
                    "summary": summary,
                    "topics": ai_topics,
                    "keywords": ai_result.get("keywords", []),
                }

                logger.info(
                    f"AI method found {len(ai_topics)} topics for entry {entry_id}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing entry {entry_id} with AI method: {str(e)}"
                )
                logger.error(traceback.format_exc())
                entry_result["ai_method"]["error"] = str(e)

            # 2. Phương pháp truyền thống (nếu yêu cầu so sánh)
            if compare_methods:
                logger.info(
                    f"Processing entry {entry_id} with traditional method (compare_methods={compare_methods})"
                )
                try:
                    # Lấy exact_keywords từ entry
                    exact_keywords = entry.get("exact_keywords", [])
                    if not exact_keywords and "tmp_keywords" in entry:
                        # Backup: thử dùng tmp_keywords nếu exact_keywords không có
                        exact_keywords = entry.get("tmp_keywords", [])

                    logger.info(
                        f"Found {len(exact_keywords)} exact_keywords for entry {entry_id}"
                    )

                    # Lấy các topics từ DBpedia nếu có
                    if entry.get("dbpedia_topics"):
                        trad_topics = entry.get("dbpedia_topics", [])
                        logger.info(f"Using dbpedia_topics from entry: {trad_topics}")
                    else:
                        # Nếu không có sẵn, thử lấy từ keywords
                        trad_topics = [keyword.title() for keyword in exact_keywords]
                        logger.info(
                            f"Using topics derived from keywords: {trad_topics}"
                        )

                    # Lưu kết quả của phương pháp truyền thống
                    entry_result["traditional_method"] = {
                        "exact_keywords": exact_keywords,
                        "topics": trad_topics,
                    }

                    logger.info(
                        f"Traditional method found {len(trad_topics)} topics for entry {entry_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing entry {entry_id} with traditional method: {str(e)}"
                    )
                    logger.error(traceback.format_exc())
                    entry_result["traditional_method"]["error"] = str(e)
            else:
                logger.info(
                    f"Skipping traditional method for entry {entry_id} (compare_methods={compare_methods})"
                )

            # Thêm kết quả của entry vào danh sách kết quả
            results.append(entry_result)

        # Thống kê số lượng entries xử lý thành công
        logger.info(
            f"Batch extract topics completed. Processed {len(results)}/{total_entries} entries, entries with content: {entries_with_content}"
        )

        # Thống kê số lượng topics tìm được bởi mỗi phương pháp
        ai_topic_count = sum(
            len(r.get("ai_method", {}).get("topics", [])) for r in results
        )
        trad_topic_count = sum(
            len(r.get("traditional_method", {}).get("topics", [])) for r in results
        )

        logger.info(
            f"Total topics found: AI method = {ai_topic_count}, Traditional method = {trad_topic_count}"
        )

        # Tính trung bình topics per entry
        avg_ai_topics = ai_topic_count / len(results) if results else 0
        avg_trad_topics = trad_topic_count / len(results) if results else 0

        # Tính trung bình điểm số cho mỗi phương pháp
        for method in method_stats:
            scores = method_stats[method]["scores"]
            if scores:
                method_stats[method]["avg_score"] = sum(scores) / len(scores)
                method_stats[method]["max_score"] = max(scores)
                method_stats[method]["min_score"] = min(scores)
            else:
                method_stats[method]["avg_score"] = 0
                method_stats[method]["max_score"] = 0
                method_stats[method]["min_score"] = 0

        # Thêm thống kê vào kết quả
        summary = {
            "total_entries": total_entries,
            "processed_entries": len(results),
            "entries_with_content": entries_with_content,
            "ai_method": {
                "total_topics": ai_topic_count,
                "avg_topics_per_entry": avg_ai_topics,
                "method_stats": method_stats,
            },
            "traditional_method": {
                "total_topics": trad_topic_count,
                "avg_topics_per_entry": avg_trad_topics,
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Kết quả cuối cùng
        final_result = {
            "summary": summary,
            "entries": results,
            "entry_count": len(results),
            "entries_with_content": entries_with_content,
            "created_at": datetime.datetime.now().isoformat(),
            "sample_size": total_entries,
            "compare_methods": compare_methods,
            "ensure_count": ensure_count,
            "ai_topic_count": ai_topic_count,
            "trad_topic_count": trad_topic_count,
            "lda_count": method_stats["lda"]["count"],
            "distilbert_count": method_stats["distilbert"]["count"],
            "keyword-extraction_count": method_stats["keyword-extraction"]["count"],
        }
        logger.info(
            f"Task {task_id} completed successfully with {len(results)} entries processed"
        )

        return final_result
    except Exception as e:
        logger.error(f"Error in batch_extract_topics_task: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}
