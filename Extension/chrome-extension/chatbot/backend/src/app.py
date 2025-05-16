import logging
import time
import os
import json
from typing import Dict, Optional, List
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
import datetime
# from flask import request, jsonify

from brain import get_embedding
from models import (
    insert_document,
    get_category_by_name,
    insert_category,
    get_topic_by_name,
)
from utils import setup_logging
from tasks import (
    llm_handle_message,
    index_document,
    sync_dbpedia_to_database,
    search_dbpedia_pages,
    extract_pages_mapping,
    sync_pages_of_category,
    index_dbpedia_topics,
    index_category_to_vector_db,
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    compare_embedding_models_task,
    fine_tune_models_task,
    get_model_versions,
    load_model_version,
    delete_model_version,
    ai_summarize_task,
    ai_extract_topics_task,
    ai_batch_process_task,
    extract_data_auto_task,
    batch_extract_topics_task,
)
from vectorize import create_collection, get_record_by_id
from dbpedia import get_data, whitelist_concepts
from ai_summarization import (
    init_models as init_ai_models,
    process_visible_content,
    extract_dbpedia_topics,
    batch_process_entries,
    extract_topics_with_lda,
    summarize_text,
    extract_topics_with_zero_shot,
    extract_keywords,
    process_visible_content_ai,
)

setup_logging()
logger = logging.getLogger(__name__)

# Khởi tạo tất cả các mô hình embedding
try:
    from embeddings import (
        load_all_base_models,
        load_transformer_model,
        load_hybrid_model,
        quantize_model,
        roberta_model,
        xlm_roberta_model,
    )

    # Tải các mô hình cơ bản (TF-IDF, BM25, BMX)
    logger.info("Loading base embedding models...")
    load_all_base_models()

    # Tải các mô hình transformer
    logger.info("Loading transformer models...")
    transformer_models = {
        "roberta": "roberta-base",
        "xlm-roberta": "xlm-roberta-base",
        "distilbert": "distilbert-base-uncased",
    }

    # Đảm bảo biến toàn cục đã được khởi tạo
    loaded_models = {}

    # Hàm để tải model và thực hiện quantization với retry
    def load_and_quantize_model(model_type, model_name, max_retries=3):
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Loading {model_type} model (attempt {attempt + 1}/{max_retries})..."
                )
                success, model_info = load_transformer_model(model_type, model_name)

                if not success:
                    logger.error(f"Failed to load {model_type} model: {model_info}")
                    continue

                # Đợi một chút để đảm bảo biến toàn cục đã được cập nhật
                time.sleep(0.5)

                # Lấy tham chiếu đến biến global model tương ứng
                if model_type == "roberta":
                    model_ref = roberta_model
                elif model_type == "xlm-roberta":
                    model_ref = xlm_roberta_model
                else:
                    model_ref = None

                # Áp dụng quantization nếu loại model hỗ trợ và model đã được tải thành công
                if (
                    success
                    and model_type in ["roberta", "xlm-roberta"]
                    and model_ref is not None
                ):
                    logger.info(f"Applying quantization to {model_type} model...")
                    quantized_model = quantize_model(model_ref)

                    # Kiểm tra kết quả quantization
                    if quantized_model is not None and hasattr(
                        quantized_model, "is_quantized"
                    ):
                        logger.info(f"Successfully quantized {model_type} model")

                        # Cập nhật biến toàn cục với model đã quantize
                        if model_type == "roberta":
                            from embeddings import roberta_model as rm

                            globals()["roberta_model"] = quantized_model
                        elif model_type == "xlm-roberta":
                            from embeddings import xlm_roberta_model as xrm

                            globals()["xlm_roberta_model"] = quantized_model

                        return True
                    else:
                        logger.warning(
                            f"Quantization did not complete successfully for {model_type} model"
                        )
                elif success:
                    logger.info(
                        f"Model {model_type} loaded successfully but not eligible for quantization"
                    )
                    return True

                # Nếu model vẫn là None, thử lại
                if model_ref is None:
                    logger.warning(
                        f"Model {model_type} reference is None after loading, retrying..."
                    )
                    time.sleep(1)  # Đợi thêm thời gian trước khi thử lại
                else:
                    return True

            except Exception as e:
                logger.error(f"Error in load_and_quantize_model for {model_type}: {e}")
                time.sleep(1)  # Đợi trước khi thử lại

        return False

    # Tải và quantize các mô hình
    for model_type, model_name in transformer_models.items():
        load_and_quantize_model(model_type, model_name)

    # Tải các mô hình hybrid
    logger.info("Loading hybrid models...")
    hybrid_models = [
        ("tfidf", "bert-base-uncased"),
        ("bm25", "bert-base-uncased"),
        ("bmx", "bert-base-uncased"),
    ]

    for trad_model, transformer_model in hybrid_models:
        try:
            logger.info(f"Loading hybrid model {trad_model}+{transformer_model}...")
            load_hybrid_model(trad_model, transformer_model)
        except Exception as e:
            logger.error(
                f"Error loading hybrid model {trad_model}+{transformer_model}: {e}"
            )

except Exception as e:
    logger.error(f"Error initializing embedding models: {e}")

app = FastAPI()


class CompleteRequest(BaseModel):
    bot_id: Optional[str] = "botPedia"
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False
    embedding_model: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models/embedding")
async def get_embedding_models():
    """API endpoint để lấy danh sách các mô hình embedding có sẵn"""
    return {
        "available_models": list(EMBEDDING_MODELS.keys()),
        "default_model": DEFAULT_EMBEDDING_MODEL,
    }


@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    embedding_model = data.embedding_model
    logger.info(
        f"Complete chat from user {user_id} to {bot_id}: {user_message} (using model: {embedding_model})"
    )

    if not user_message or not user_id:
        raise HTTPException(
            status_code=400, detail="User id and user message are required"
        )

    if data.sync_request:
        response = llm_handle_message(bot_id, user_id, user_message, embedding_model)
        return {"response": str(response)}
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message, embedding_model)
        return {"task_id": task.id}


@app.get("/chat/complete/{task_id}")
async def get_response(task_id: str):
    start_time = time.time()
    while True:
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        logger.info(f"Task result: {task_result.result}")

        if task_status == "PENDING":
            if time.time() - start_time > 60:  # 60 seconds timeout
                return {
                    "task_id": task_id,
                    "task_status": task_result.status,
                    "task_result": task_result.result,
                    "error_message": "Service timeout, retry please",
                }
            else:
                time.sleep(0.5)  # sleep for 0.5 seconds before retrying
        else:
            result = {
                "task_id": task_id,
                "task_status": task_result.status,
                "task_result": task_result.result,
            }
            return result


@app.post("/collection/create")
async def create_vector_collection(data: Dict):
    collection_name = data.get("collection_name")
    create_status = create_collection(collection_name)
    logging.info(f"Create collection {collection_name} status: {create_status}")
    return {"status": create_status is not None}


@app.post("/document/create")
async def create_document(data: Dict):
    doc_id = data.get("id")
    title = data.get("title")
    content = data.get("content")
    embedding_model = data.get("embedding_model")
    create_status = insert_document(title, content)
    logging.info(f"Create document status: {create_status}")
    index_status = index_document(
        doc_id, title, content, embedding_model=embedding_model
    )
    return {"status": create_status is not None, "index_status": index_status}


@app.post("/dbpedia/query")
async def query_dbpedia(data: Dict):
    query = data.get("query")
    logging.info(f"Query: {query}")
    data = get_data(query)
    return {"data": data}


@app.post("/dbpedia/sync-data")
async def sync_data(data: Dict):
    topics = data.get("topics", [])

    # Sử dụng AI để trích xuất topic nếu có visible_content
    visible_content = data.get("visible_content")
    if visible_content and (not topics or len(topics) == 0):
        try:
            ai_topics = extract_dbpedia_topics(visible_content)
            if ai_topics and len(ai_topics) > 0:
                topics = ai_topics
                logger.info(f"Using AI-extracted topics: {topics}")
        except Exception as e:
            logger.error(f"Error extracting AI topics: {e}")

    # Sử dụng whitelist_concepts nếu không có topics nào được cung cấp
    if len(topics) == 0:
        topics = whitelist_concepts()

    task = sync_dbpedia_to_database.delay(topics)
    return {"task_id": task.id}


@app.post("/dbpedia/sync-category")
async def sync_category_data(data: Dict):
    """
    [
        {
            "name": "Category Name",
            "uri": "https://dbpedia.org/page/Category:Computer_animation
        },
        ...
    ]
    """
    categories = data.get("categories", [])
    tasks = []
    for category in categories:
        if not category.get("name") and not category.get("uri"):
            raise HTTPException(
                status_code=400, detail="Category name or uri are required"
            )
        if not category.get("name"):
            category_name = category.get("uri").split(":")[-1]
        else:
            category_name = category.get("name")
        category_obj = get_category_by_name(category_name)
        if category_obj:
            category_id = category_obj.id
        else:
            topic_id = get_topic_by_name(category.get("topic")).id
            category_id = insert_category(
                label=category_name, uri=category.get("uri"), topic_id=topic_id
            ).id
        task = sync_pages_of_category.apply_async(
            (category_name, category_id, 3), countdown=1
        )
        tasks.append(task.id)
    return {"task_ids": [str(task_id) for task_id in tasks]}


@app.post("/dbpedia/get-vector-data")
async def get_vector_data(data: Dict):
    id = data.get("id")
    collection = data.get("collection", "dbpedia")
    task = get_record_by_id(collection, id)
    return {"result": task}


@app.post("/dbpedia/index-data")
async def index_dbpedia(data: Dict):
    collection = data.get("collection", "dbpedia")
    topics = data.get("topic_names", [])
    embedding_model = data.get("embedding_model")
    task = index_dbpedia_topics.delay(collection, topics, embedding_model)
    return {"task_id": task.id}


@app.post("/dbpedia/index-category")
async def index_category(data: Dict):
    collection = data.get("collection", "dbpedia")
    categories = data.get("categories", [])
    limit = data.get("limit", 3)
    embedding_model = data.get("embedding_model")
    task = index_category_to_vector_db.delay(
        collection, categories, limit, embedding_model
    )
    return {"task_id": task.id}


@app.post("/dbpedia/search-data")
async def search_dbpedia(data: Dict):
    keywords = data.get("keywords", [])
    limit = data.get("limit", 3)
    embedding_model = data.get("embedding_model")
    return {"pages": search_dbpedia_pages(keywords, limit, embedding_model)}


@app.post("/dbpedia/extract-data")
async def extract_data(data: Dict):
    file_path = data.get("path", "data/history_learning_data.json")
    json_file_path = f"/usr/src/app/src/{file_path}"
    sample = data.get("sample", 0)
    limit = data.get("limit", 1000)
    embedding_model = data.get("embedding_model")
    save_output = data.get("save_output", True)

    # Gọi hàm extract_pages_mapping và trả kết quả
    if sample < 10:
        result = extract_pages_mapping(
            json_file_path=json_file_path,
            sample=sample,
            limit=limit,
            embedding_model=embedding_model,
            save_output=save_output,
        )
        return {"result": result}
    else:
        task = extract_pages_mapping.delay(
            json_file_path=json_file_path,
            sample=sample,
            limit=limit,
            embedding_model=embedding_model,
            save_output=save_output,
        )
        return {"task_id": task.id}


@app.get("/dbpedia/extract-data/{task_id}")
async def extract_data_response(task_id: str):
    start_time = time.time()
    while True:
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        logger.info(f"Task result: {task_result.result}")

        if task_status == "PENDING":
            if time.time() - start_time > 60:  # 60 seconds timeout
                return {
                    "task_id": task_id,
                    "task_status": task_result.status,
                    "task_result": task_result.result,
                    "error_message": "Service timeout, retry please",
                }
            else:
                time.sleep(5)  # sleep for 5 seconds before retrying
        else:
            # Lưu kết quả vào file JSON khi task hoàn thành
            if task_status == "SUCCESS" and task_result.result:
                output_path = "/usr/src/app/src/data/final_output_v1.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"result": task_result.result}, f, ensure_ascii=False, indent=4
                    )
            result = {
                "task_id": task_id,
                "task_status": task_result.status,
                "task_result": task_result.result,
            }
            return result


@app.post("/models/compare")
async def compare_models(data: Dict):
    """
    API endpoint để so sánh hiệu suất của các mô hình embedding
    """
    file_path = data.get("path", "data/history_learning_data.json")
    json_file_path = f"/usr/src/app/src/{file_path}"
    sample = data.get("sample", 10)
    limit = data.get("limit", 10)
    output_file = data.get("output_file")

    task = compare_embedding_models_task.delay(
        json_file_path=json_file_path,
        sample=sample,
        limit=limit,
        output_file=output_file,
    )

    return {"task_id": task.id}


@app.post("/models/fine-tune")
async def fine_tune_models(data: Dict):
    """
    API endpoint để fine-tune các mô hình embedding trên dữ liệu học tập
    """
    file_path = data.get("path", "data/history_learning_data.json")
    json_file_path = f"/usr/src/app/src/{file_path}"
    sample = data.get("sample", 100)
    version = data.get("version")
    save_model = data.get("save_model", True)

    task = fine_tune_models_task.delay(
        json_file_path=json_file_path,
        sample=sample,
        version=version,
        save_model=save_model,
    )

    return {"task_id": task.id}


@app.get("/models/versions")
async def get_versions():
    """
    API endpoint để lấy danh sách các phiên bản mô hình đã lưu
    """
    results = get_model_versions()
    return results


@app.post("/models/load")
def load_version(data: Dict):
    model_type = data.get("model_type")
    version = data.get("version", "latest")
    force_download = data.get("force_download", False)

    try:
        logger.info(
            f"Loading model {model_type} version {version} with force_download: {force_download}"
        )
        result = load_model_version(model_type, version, force_download)

        # Kiểm tra kết quả từ load_model_version
        if isinstance(result, dict):
            # load_model_version trả về dict trực tiếp
            if result.get("status") == "success":
                return {"status": "success", "model": result.get("metadata", {})}
            else:
                return {
                    "status": "error",
                    "error_message": result.get("error", "Unknown error"),
                }
        elif isinstance(result, tuple) and len(result) == 2:
            # Xử lý trường hợp khi hàm vẫn trả về tuple (bool, str) như cũ
            success, error_or_info = result
            if success:
                return {"status": "success", "model": error_or_info}
            else:
                return {"status": "error", "error_message": error_or_info}
        else:
            # Trường hợp không xác định
            logger.error(f"Kết quả không xác định từ load_model_version: {result}")
            return {
                "status": "error",
                "error_message": f"Invalid result format: {result}",
            }
    except Exception as e:
        logger.exception(f"Lỗi khi tải mô hình {model_type}: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "stack_trace": traceback.format_exc(),
        }


@app.delete("/models/{model_type}/{version}")
async def delete_version(model_type: str, version: str):
    """
    API endpoint để xóa một phiên bản mô hình cụ thể
    """
    result = delete_model_version(model_type, version)
    return result


@app.get("/models/active")
async def get_active_models():
    """
    API endpoint để lấy thông tin về các mô hình đang hoạt động
    """
    try:
        # In log để debug
        logger.info("Starting get_active_models API call")

        # Import các biến toàn cục hoặc module cần thiết
        import sys
        import importlib
        import inspect

        # Cách tiếp cận mới: Đọc trực tiếp từ các file trong thư mục models/
        models_dir = "/usr/src/app/models"
        import os
        import json

        models = {
            "tfidf": {"status": "unknown", "metadata": None},
            "bm25": {"status": "unknown", "metadata": None},
            "bmx": {"status": "unknown", "metadata": None},
            "roberta": {"status": "unknown", "metadata": None},
            "xlm-roberta": {"status": "unknown", "metadata": None},
            "distilbert": {"status": "unknown", "metadata": None},
            "hybrid_tfidf_bert": {"status": "unknown", "metadata": None},
            "hybrid_bm25_bert": {"status": "unknown", "metadata": None},
            "hybrid_bmx_bert": {"status": "unknown", "metadata": None},
        }

        # Import từ module embeddings
        try:
            from embeddings import (
                tfidf_embedder,
                bm25_embedder,
                bmx_embedder,
                roberta_model,
                xlm_roberta_model,
                distilbert_model,
                hybrid_tfidf_bert_model,
                hybrid_bm25_bert_model,
                hybrid_bmx_bert_model,
            )

            # Debug thông tin về các mô hình
            logger.info(f"tfidf_embedder: {tfidf_embedder}")
            logger.info(f"bm25_embedder: {bm25_embedder}")
            logger.info(f"bmx_embedder: {bmx_embedder}")
            logger.info(f"roberta_model: {roberta_model}")
            logger.info(f"xlm_roberta_model: {xlm_roberta_model}")
            logger.info(f"distilbert_model: {distilbert_model}")
            logger.info(f"hybrid_tfidf_bert_model: {hybrid_tfidf_bert_model}")
            logger.info(f"hybrid_bm25_bert_model: {hybrid_bm25_bert_model}")
            logger.info(f"hybrid_bmx_bert_model: {hybrid_bmx_bert_model}")

            # Kiểm tra và cập nhật trạng thái các mô hình

            # Mô hình truyền thống
            # TF-IDF
            if tfidf_embedder is not None:
                if hasattr(tfidf_embedder, "metadata"):
                    models["tfidf"] = tfidf_embedder.metadata.to_dict()
                else:
                    models["tfidf"] = {"status": "active", "metadata": None}
            else:
                models["tfidf"] = {"status": "not loaded", "metadata": None}

            # BM25
            if bm25_embedder is not None:
                if hasattr(bm25_embedder, "metadata"):
                    models["bm25"] = bm25_embedder.metadata.to_dict()
                else:
                    models["bm25"] = {"status": "active", "metadata": None}
            else:
                models["bm25"] = {"status": "not loaded", "metadata": None}

            # BMX
            if bmx_embedder is not None:
                if hasattr(bmx_embedder, "metadata"):
                    models["bmx"] = bmx_embedder.metadata.to_dict()
                else:
                    models["bmx"] = {"status": "active", "metadata": None}
            else:
                models["bmx"] = {"status": "not available", "metadata": None}

            # Mô hình transformer
            # RoBERTa
            if roberta_model is not None:
                models["roberta"] = {
                    "status": "active",
                    "metadata": {
                        "model_name": "roberta-base",
                        "is_quantized": getattr(roberta_model, "is_quantized", False),
                    },
                }
            else:
                models["roberta"] = {"status": "not loaded", "metadata": None}

            # XLM-RoBERTa
            if xlm_roberta_model is not None:
                models["xlm-roberta"] = {
                    "status": "active",
                    "metadata": {
                        "model_name": "xlm-roberta-base",
                        "is_quantized": getattr(
                            xlm_roberta_model, "is_quantized", False
                        ),
                    },
                }
            else:
                models["xlm-roberta"] = {"status": "not loaded", "metadata": None}

            # DistilBERT
            if distilbert_model is not None:
                models["distilbert"] = {
                    "status": "active",
                    "metadata": {
                        "model_name": "distilbert-base-uncased",
                        "is_quantized": getattr(
                            distilbert_model, "is_quantized", False
                        ),
                    },
                }
            else:
                models["distilbert"] = {"status": "not loaded", "metadata": None}

            # Mô hình hybrid
            # Hybrid TF-IDF + BERT
            if hybrid_tfidf_bert_model is not None:
                models["hybrid_tfidf_bert"] = {
                    "status": "active",
                    "metadata": {"base_models": ["tfidf", "bert-base-uncased"]},
                }
                if (
                    isinstance(hybrid_tfidf_bert_model, dict)
                    and "metadata" in hybrid_tfidf_bert_model
                ):
                    models["hybrid_tfidf_bert"]["metadata"] = hybrid_tfidf_bert_model[
                        "metadata"
                    ]
            else:
                models["hybrid_tfidf_bert"] = {"status": "not loaded", "metadata": None}

            # Hybrid BM25 + BERT
            if hybrid_bm25_bert_model is not None:
                models["hybrid_bm25_bert"] = {
                    "status": "active",
                    "metadata": {"base_models": ["bm25", "bert-base-uncased"]},
                }
                if (
                    isinstance(hybrid_bm25_bert_model, dict)
                    and "metadata" in hybrid_bm25_bert_model
                ):
                    models["hybrid_bm25_bert"]["metadata"] = hybrid_bm25_bert_model[
                        "metadata"
                    ]
            else:
                models["hybrid_bm25_bert"] = {"status": "not loaded", "metadata": None}

            # Hybrid BMX + BERT
            if hybrid_bmx_bert_model is not None:
                models["hybrid_bmx_bert"] = {
                    "status": "active",
                    "metadata": {"base_models": ["bmx", "bert-base-uncased"]},
                }
                if (
                    isinstance(hybrid_bmx_bert_model, dict)
                    and "metadata" in hybrid_bmx_bert_model
                ):
                    models["hybrid_bmx_bert"]["metadata"] = hybrid_bmx_bert_model[
                        "metadata"
                    ]
            else:
                models["hybrid_bmx_bert"] = {"status": "not loaded", "metadata": None}

        except Exception as e:
            logger.error(f"Error importing model variables: {e}")

        # Bổ sung thông tin từ file metadata nếu có
        try:
            for model_name in models.keys():
                model_dir = os.path.join(models_dir, model_name)
                if os.path.exists(model_dir):
                    # Tìm phiên bản mới nhất
                    versions = [
                        d
                        for d in os.listdir(model_dir)
                        if os.path.isdir(os.path.join(model_dir, d))
                    ]
                    if versions:
                        versions.sort(reverse=True)
                        latest_version = versions[0]
                        metadata_path = os.path.join(
                            model_dir, latest_version, "metadata.json"
                        )
                        if os.path.exists(metadata_path):
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                                # Cập nhật trạng thái nếu không phải là active hoặc metadata chỉ có thông tin cơ bản
                                if models[model_name].get("status") != "active" or (
                                    model_name == "distilbert"
                                    and len(models[model_name].get("metadata", {})) <= 2
                                ):
                                    models[model_name] = metadata
                                    models[model_name]["status"] = "available"
        except Exception as e:
            logger.error(f"Error reading metadata files: {e}")

        logger.info(f"Final models status: {models}")
        return models
    except Exception as e:
        logger.error(f"Error getting active models: {e}")
        return {"error": str(e)}


@app.get("/models/fine-tune/{task_id}")
async def get_fine_tune_result(task_id: str):
    """
    API endpoint để lấy kết quả của fine-tune task
    """
    start_time = time.time()
    while True:
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        logger.info(f"Fine-tune task result: {task_result.result}")

        if task_status == "PENDING":
            if time.time() - start_time > 60:  # 60 seconds timeout
                return {
                    "task_id": task_id,
                    "task_status": task_result.status,
                    "error_message": "Service timeout, retry please",
                }
            else:
                time.sleep(5)  # sleep for 5 seconds before retrying
        else:
            result = {
                "task_id": task_id,
                "task_status": task_result.status,
                "task_result": task_result.result,
            }
            return result


@app.post("/ai/summarize")
async def summarize_content(data: Dict):
    """
    API endpoint để tóm tắt nội dung và trích xuất chủ đề
    """
    content = data.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    try:
        # Log độ dài và loại nội dung đầu vào để dễ gỡ lỗi
        content_preview = content[:100] + "..." if len(content) > 100 else content
        logger.info(
            f"Summarizing content (length: {len(content)}, preview: {content_preview})"
        )

        # Sử dụng process_visible_content_ai để tóm tắt và trích xuất chủ đề
        result = process_visible_content_ai(content)

        # Kiểm tra chất lượng tóm tắt
        summary = result.get("summary", "")
        if not summary or summary == content or len(summary.split()) < 3:
            logger.warning(
                "Summary appears to be low quality, trying direct summarization"
            )
            # Thử lại với tùy chọn khác như tăng min_length
            from ai_summarization import summarize_text

            summary = summarize_text(content, min_length=30, max_length=200)
            result["summary"] = summary

        logger.info(f"Summary result: {summary[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error in summarize_content: {e}", exc_info=True)
        # Trả về thông báo lỗi chi tiết hơn cho client
        return {
            "error": str(e),
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "topics": [],
            "keywords": [],
            "status": "error",
            "message": "Failed to summarize content due to an error. Please check the input format.",
        }


@app.post("/ai/extract-topics")
async def extract_topics(data: Dict):
    """
    API endpoint để trích xuất các chủ đề từ nội dung với thông tin chi tiết về score và method
    """
    content = data.get("content", "")
    threshold = data.get("threshold", 0.25)
    max_topics = data.get("max_topics", 5)

    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    try:
        # Sử dụng hàm process_visible_content để lấy thông tin đầy đủ
        result = process_visible_content(content)

        # Kiểm tra và ghi log các topics có trong kết quả
        logger.info(
            f"All topics from process_visible_content: {result.get('topics', [])}"
        )
        logger.info(
            f"All keywords from process_visible_content: {result.get('keywords', [])}"
        )

        # Phân loại các topics theo phương pháp
        distilbert_topics = []
        lda_topics = []

        if "topics" in result and isinstance(result["topics"], list):
            for topic_item in result["topics"]:
                if isinstance(topic_item, dict) and "score" in topic_item:
                    # Kiểm tra và log thông tin về method
                    logger.info(
                        f"Topic method: {topic_item.get('method')}, score: {topic_item.get('score')}"
                    )

                    if (
                        topic_item.get("method") == "distilbert-cosine-similarity"
                        and topic_item["score"] >= threshold
                    ):
                        distilbert_topics.append(topic_item)
                    elif topic_item.get("method") == "lda":
                        # Đảm bảo LDA topics có trường "topic" để hiển thị đúng
                        if "keywords" in topic_item and topic_item["keywords"]:
                            if not topic_item.get("topic"):
                                topic_item["topic"] = topic_item["keywords"][0].title()
                        # Giảm ngưỡng điểm cho LDA topics để đảm bảo chúng được chọn
                        lda_topics.append(topic_item)
                        logger.info(f"Found LDA topic: {topic_item}")

        # Sắp xếp các danh sách theo điểm số
        distilbert_topics.sort(key=lambda x: x["score"], reverse=True)
        lda_topics.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Filtered distilbert topics: {distilbert_topics}")
        logger.info(f"Filtered LDA topics: {lda_topics}")

        # Kết hợp các topics từ các phương pháp khác nhau
        final_topics = []

        # 1. Luôn thêm ít nhất 1 LDA topic vào đầu danh sách (nếu có)
        if lda_topics:
            lda_topic = lda_topics[0]
            final_topics.append(lda_topic)
            logger.info(f"Added LDA topic to final result: {lda_topic}")

        # 2. Thêm các distilbert topics (tối đa 3 topics)
        max_distilbert = min(3, len(distilbert_topics))
        for topic in distilbert_topics[:max_distilbert]:
            # Kiểm tra trùng lặp với các topics đã có dựa trên topic text
            if not any(
                t.get("topic", "").lower() == topic.get("topic", "").lower()
                for t in final_topics
            ):
                final_topics.append(topic)

        # 3. Thêm LDA topic thứ hai nếu có
        if len(lda_topics) > 1 and len(final_topics) < max_topics:
            lda_topic = lda_topics[1]
            if not any(
                t.get("topic", "").lower() == lda_topic.get("topic", "").lower()
                for t in final_topics
            ):
                final_topics.append(lda_topic)
                logger.info(f"Added second LDA topic to final result: {lda_topic}")

        # 4. Thêm keyword topics nếu cần
        if (
            len(final_topics) < max_topics
            and "keywords" in result
            and result["keywords"]
        ):
            remaining_slots = max_topics - len(final_topics)
            top_keywords = result["keywords"][:remaining_slots]

            # Tạo ít nhất 1 keyword topic
            if top_keywords:
                keyword = top_keywords[0]
                keyword_score = 0.6  # Đặt điểm cố định cao để đảm bảo hiển thị

                # Kiểm tra không trùng với topics đã có
                if not any(
                    t.get("topic", "").lower() == keyword.title().lower()
                    for t in final_topics
                ):
                    final_topics.append(
                        {
                            "topic": keyword.title(),
                            "score": keyword_score,
                            "method": "keyword-extraction",
                        }
                    )

                # Thêm các từ khóa còn lại
                for idx, keyword in enumerate(top_keywords[1:], 1):
                    keyword_score = 0.6 - (idx * 0.05)

                    if not any(
                        t.get("topic", "").lower() == keyword.title().lower()
                        for t in final_topics
                    ):
                        final_topics.append(
                            {
                                "topic": keyword.title(),
                                "score": keyword_score,
                                "method": "keyword-extraction",
                            }
                        )

        # Đảm bảo không vượt quá số lượng topics tối đa
        if len(final_topics) > max_topics:
            # Sắp xếp lại theo điểm số trước khi cắt bớt
            final_topics.sort(key=lambda x: x["score"], reverse=True)

            # Đảm bảo có ít nhất 1 LDA topic trong kết quả cuối cùng
            has_lda = any(t.get("method") == "lda" for t in final_topics[:max_topics])
            if not has_lda and any(t.get("method") == "lda" for t in final_topics):
                # Tìm LDA topic đầu tiên
                lda_idx = next(
                    i for i, t in enumerate(final_topics) if t.get("method") == "lda"
                )
                lda_topic = final_topics.pop(lda_idx)
                # Chèn vào vị trí max_topics-1 để đảm bảo nó nằm trong kết quả
                final_topics.insert(max_topics - 1, lda_topic)

            final_topics = final_topics[:max_topics]

        logger.info(f"Final topics to return: {final_topics}")
        return {"topics": final_topics}
    except Exception as e:
        logger.error(f"Error in extract_topics: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/ai/batch-process")
async def batch_process(data: Dict):
    """
    API endpoint để xử lý hàng loạt các entry
    """
    entries = data.get("entries", [])
    if not entries:
        raise HTTPException(status_code=400, detail="Entries are required")

    async_process = data.get("async", False)

    if async_process:
        # Sử dụng Celery cho xử lý bất đồng bộ
        task = ai_batch_process_task.delay(entries)
        return {"task_id": task.id}
    else:
        try:
            result = batch_process_entries(entries)
            return {"processed_entries": result}
        except Exception as e:
            logger.error(f"Error in batch_process: {e}")
            return {"error": str(e)}


@app.get("/ai/models/status")
async def get_ai_models_status():
    """
    API endpoint để kiểm tra trạng thái của các mô hình AI
    """
    from ai_summarization import SUMMARIZER, TOPIC_EXTRACTOR, NLP

    status = {
        "summarizer": SUMMARIZER is not None,
        "topic_extractor": TOPIC_EXTRACTOR is not None,
        "nlp": NLP is not None,
    }

    return status


@app.post("/ai/initialize-models")
async def initialize_ai_models():
    """
    API endpoint để khởi tạo các mô hình AI
    """
    try:
        init_ai_models()
        from ai_summarization import SUMMARIZER, TOPIC_EXTRACTOR, NLP

        status = {
            "summarizer": SUMMARIZER is not None,
            "topic_extractor": TOPIC_EXTRACTOR is not None,
            "nlp": NLP is not None,
        }

        return {"status": "success", "models": status}
    except Exception as e:
        logger.error(f"Error initializing AI models: {e}")
        return {"status": "error", "error_message": str(e)}


# Khởi tạo các model AI sau khi startup
@app.on_event("startup")
async def startup_event():
    # Khởi tạo các model AI
    try:
        logger.info("Initializing AI models...")
        init_ai_models()
    except Exception as e:
        logger.error(f"Error initializing AI models: {e}")


@app.post("/models/quantize")
def quantize_model_endpoint(data: Dict):
    """
    API endpoint để quantize các mô hình transformer
    """
    model_type = data.get("model_type")

    if model_type not in ["roberta", "xlm-roberta", "distilbert"]:
        return {"status": "error", "message": f"Unsupported model type: {model_type}"}

    try:
        # Import model variables
        from embeddings import (
            roberta_model,
            xlm_roberta_model,
            distilbert_model,
            quantize_model,
        )

        # Get the appropriate model
        model_to_quantize = None
        if model_type == "roberta":
            model_to_quantize = roberta_model
        elif model_type == "xlm-roberta":
            model_to_quantize = xlm_roberta_model
        elif model_type == "distilbert":
            model_to_quantize = distilbert_model

        # Check if model exists
        if model_to_quantize is None:
            # Try to load it first
            success, _ = load_model_version(model_type, "latest")
            if not success:
                return {
                    "status": "error",
                    "message": f"Model {model_type} not loaded yet or failed to load",
                }

            # Get the model again after loading
            if model_type == "roberta":
                model_to_quantize = roberta_model
            elif model_type == "xlm-roberta":
                model_to_quantize = xlm_roberta_model
            elif model_type == "distilbert":
                model_to_quantize = distilbert_model

            # Check again
            if model_to_quantize is None:
                return {
                    "status": "error",
                    "message": f"Failed to load model {model_type}",
                }

        # Check if model is already quantized
        is_already_quantized = (
            hasattr(model_to_quantize, "is_quantized")
            and model_to_quantize.is_quantized
        )
        if is_already_quantized:
            return {
                "status": "success",
                "message": f"Model {model_type} is already quantized",
                "already_quantized": True,
            }

        # Apply quantization
        logger.info(f"Applying quantization to {model_type} model...")
        quantized_model = quantize_model(model_to_quantize)

        if (
            quantized_model is not None
            and hasattr(quantized_model, "is_quantized")
            and quantized_model.is_quantized
        ):
            return {
                "status": "success",
                "message": f"Successfully quantized {model_type} model",
                "is_quantized": True,
            }
        else:
            return {
                "status": "error",
                "message": f"Quantization did not complete successfully for {model_type} model",
            }

    except Exception as e:
        logger.exception(f"Error in quantize_model_endpoint: {e}")
        return {"status": "error", "message": f"Error during quantization: {str(e)}"}


@app.post("/ai/test-lda")
async def test_lda(data: Dict):
    """
    API endpoint tạm thời để kiểm tra trực tiếp hàm extract_topics_with_lda
    """
    content = data.get("content", "")
    num_topics = data.get("num_topics", 5)

    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    try:
        # Import trực tiếp hàm extract_topics_with_lda
        from ai_summarization import extract_topics_with_lda, init_models, NLP

        # Đảm bảo mô hình NLP đã được khởi tạo
        if NLP is None:
            init_models()

        # Gọi trực tiếp hàm extract_topics_with_lda
        lda_topics = extract_topics_with_lda(content, num_topics=num_topics)

        # Xử lý các chủ đề trùng lặp
        processed_topics = []
        topic_count = {}  # Đếm số lần xuất hiện của mỗi chủ đề

        for i, topic in enumerate(lda_topics):
            topic_name = topic.get("topic", "")
            topic_keywords = tuple(topic.get("keywords", []))

            # Tạo ID duy nhất cho mỗi chủ đề
            topic_id = f"lda_topic_{i}"
            topic["topic_id"] = topic_id

            # Kiểm tra xem chủ đề này đã xuất hiện chưa
            if topic_name in topic_count:
                # Nếu đã xuất hiện, tăng số đếm và đổi tên
                topic_count[topic_name] += 1
                original_name = topic_name
                # Thêm số vào tên chủ đề để phân biệt
                topic["topic"] = f"{original_name} (variant {topic_count[topic_name]})"
                # Thêm thông tin về sự trùng lặp
                topic["is_duplicate"] = True
                topic["original_topic"] = original_name
            else:
                topic_count[topic_name] = 1
                topic["is_duplicate"] = False

            processed_topics.append(topic)

        # Log kết quả để kiểm tra
        logger.info(f"LDA topics directly from function: {processed_topics}")

        return {"lda_topics": processed_topics}
    except Exception as e:
        logger.error(f"Error in test_lda: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/ai/extract-topics-auto")
async def extract_topics_auto(data: Dict):
    """
    Tự động trích xuất topics từ content sử dụng kết hợp 3 phương pháp:
    1. LDA (Latent Dirichlet Allocation)
    2. DistilBERT Cosine Similarity
    3. Keyword Extraction

    Không cần danh sách topics định nghĩa sẵn như hàm extract_topics
    """
    try:
        content = data.get("content", "")
        threshold = data.get("threshold", 0.4)  # Ngưỡng mặc định: 0.4
        max_topics = data.get("max_topics", 5)  # Số lượng topic tối đa: 5

        if not content:
            raise HTTPException(status_code=400, detail="No content provided")

        # Bước 1: Tóm tắt content
        from ai_summarization import summarize_text

        summary = summarize_text(content)

        # Bước 2: Sử dụng process_visible_content để trích xuất topics
        from ai_summarization import (
            process_visible_content,
            extract_topics_with_lda,
            extract_topics_with_zero_shot,
            extract_keywords,
            process_visible_content_ai,
        )

        # Trích xuất topics từ mỗi phương pháp riêng biệt thay vì dùng process_visible_content
        # 1. Trích xuất từ DistilBERT
        distilbert_topics = extract_topics_with_zero_shot(
            content, max_topics=max_topics
        )
        logger.info(f"Extracted {len(distilbert_topics)} topics from DistilBERT")

        # 2. Trích xuất từ LDA (tăng num_topics để có nhiều lựa chọn hơn)
        lda_topics = extract_topics_with_lda(
            content, num_topics=max(5, max_topics // 2)
        )
        logger.info(f"Extracted {len(lda_topics)} topics from LDA")

        # 3. Trích xuất từ keywords
        keywords = extract_keywords(content, top_n=max(10, max_topics))
        logger.info(f"Extracted {len(keywords)} keywords")

        # Tạo topics từ keywords
        keyword_topics = []
        for idx, keyword in enumerate(keywords):
            if len(keyword) > 2:  # Đảm bảo keyword đủ dài để làm topic
                keyword_topics.append(
                    {
                        "topic": keyword.title(),
                        "score": 0.6 - (idx * 0.02),  # Giảm nhẹ hơn để có thêm topics
                        "method": "keyword-extraction",
                    }
                )

        # Kết hợp tất cả topics từ các phương pháp khác nhau
        all_topics = []
        all_topics.extend(distilbert_topics)
        all_topics.extend(lda_topics)
        all_topics.extend(keyword_topics)

        # Loại bỏ các topics trùng lặp, ưu tiên giữ lại topic có score cao hơn
        unique_topics = {}
        for topic in all_topics:
            topic_name = topic.get("topic", "")
            if not topic_name:
                continue

            # Nếu topic chưa có trong từ điển hoặc có score cao hơn topic cũ
            if (
                topic_name not in unique_topics
                or topic["score"] > unique_topics[topic_name]["score"]
            ):
                unique_topics[topic_name] = topic

        # Chuyển từ từ điển về danh sách
        filtered_topics = list(unique_topics.values())

        # Sắp xếp theo score giảm dần
        sorted_topics = sorted(
            filtered_topics, key=lambda x: x.get("score", 0), reverse=True
        )

        # Lọc theo ngưỡng và giới hạn số lượng
        final_topics = [t for t in sorted_topics if t.get("score", 0) >= threshold][
            :max_topics
        ]

        # Đếm số lượng topics từ mỗi phương pháp
        lda_count = sum(1 for t in final_topics if t.get("method") == "lda")
        distilbert_count = sum(
            1 for t in final_topics if t.get("method") == "distilbert-cosine-similarity"
        )
        keyword_count = sum(
            1 for t in final_topics if t.get("method") == "keyword-extraction"
        )

        logger.info(
            f"Final topics: {len(final_topics)} (LDA: {lda_count}, DistilBERT: {distilbert_count}, Keyword: {keyword_count})"
        )

        # Trả về kết quả
        return {
            "status": "success",
            "summary": summary,
            "topics": final_topics,
            "keywords": keywords[:5],  # Giới hạn số lượng keywords trả về
        }

    except Exception as e:
        logger.error(f"Error in extract_topics_auto: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error extracting topics: {str(e)}"
        )


@app.post("/dbpedia/extract-data-auto")
async def extract_data_auto(data: Dict):
    """
    API endpoint để trích xuất dữ liệu từ history learning data
    sử dụng kết hợp 3 phương pháp (LDA, DistilBERT, keyword extraction)
    để tự động trích xuất topic, lưu topics vào database,
    sau đó tìm kiếm categories và liên kết đến các pages trên DBpedia.

    Tuân thủ đúng quy trình: extract topic -> save to DB -> get categories -> get pages

    Tham số:
        path: Đường dẫn đến file dữ liệu (mặc định: /usr/src/app/src/data/history_learning_data.json)
        sample: Số lượng mẫu cần xử lý (mặc định: False)
        limit: Giới hạn số lượng entries (mặc định: None)
        embedding_model: Mô hình embedding được sử dụng (mặc định: distilbert)
        ensure_count: Đảm bảo đủ số lượng kết quả được xử lý (mặc định: True)
    """
    try:
        file_path = data.get("path", "/usr/src/app/src/data/history_learning_data.json")
        sample = data.get("sample", False)
        limit = data.get("limit")
        embedding_model = data.get("embedding_model", "distilbert")
        ensure_count = data.get("ensure_count", True)

        # Convert ensure_count sang boolean nếu là string
        if isinstance(ensure_count, str):
            ensure_count = ensure_count.lower() in ["true", "yes", "1", "t", "y"]

        # Ghi log tham số
        logger.info(
            f"extract_data_auto called with parameters: sample={sample}, limit={limit}, embedding_model={embedding_model}, ensure_count={ensure_count}"
        )

        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Gọi task extract_data_auto_task
        task = extract_data_auto_task.apply_async(
            args=[file_path, sample, limit, embedding_model, ensure_count]
        )

        # Tạo endpoint result để kiểm tra kết quả
        result_endpoint = f"/dbpedia/extract-data-auto/{task.id}"

        # Trả về thông tin task
        return {
            "status": "success",
            "message": "Extraction task started",
            "task_id": task.id,
            "result_endpoint": result_endpoint,
            "params": {
                "file_path": file_path,
                "sample": sample,
                "limit": limit,
                "embedding_model": embedding_model,
                "ensure_count": ensure_count,
            },
        }

    except Exception as e:
        logger.error(f"Error in extract_data_auto: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error starting extraction task: {str(e)}"
        )


@app.get("/dbpedia/extract-data-auto/{task_id}")
async def extract_data_auto_response(task_id: str):
    """
    API endpoint để lấy kết quả của task extract_data_auto_task
    Kết quả trả về sẽ theo định dạng giống như API '/dbpedia/extract-data'.
    """
    try:
        # Lấy thông tin task
        task = extract_data_auto_task.AsyncResult(task_id)

        if task.state == "PENDING":
            response = {"status": "pending", "message": "Task is pending"}
        elif task.state == "SUCCESS":
            if task.result:
                # Tìm file kết quả
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                results_dir = "/usr/src/app/src/results"

                # Nếu task.result là list, nghĩa là task đã hoàn thành và trả về kết quả
                if isinstance(task.result, list):
                    # Lưu kết quả vào file
                    output_path = os.path.join(
                        results_dir,
                        f"extraction_results_auto_combined_{timestamp}.json",
                    )

                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(task.result, f, ensure_ascii=False, indent=2)

                    # Trả về kết quả gốc từ task mà không thay đổi cấu trúc
                    response = {
                        "status": "success",
                        "message": "Extraction completed successfully",
                        "result_file": output_path,
                        "result": task.result,
                    }
                else:
                    # Nếu task.result không phải list, nghĩa là có lỗi
                    response = {
                        "status": "error",
                        "message": "Task completed but returned invalid result",
                        "result": task.result,
                    }
            else:
                response = {
                    "status": "error",
                    "message": "Task completed but returned no result",
                }
        elif task.state == "FAILURE":
            response = {
                "status": "error",
                "message": "Task failed",
                "error": str(task.result),
            }
        else:
            response = {
                "status": "processing",
                "message": f"Task is in state {task.state}",
            }

        return response

    except Exception as e:
        logger.error(f"Error in extract_data_auto_response: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error retrieving task result: {str(e)}"
        )


@app.post("/ai/batch-extract-topics-khanh")
def batch_extract_topics_ai(
    path: str = "data/history_learning_data.json", sample: int = 10
):
    """
    Simplified version of batch-extract-topics API to test Celery worker task processing
    """
    try:
        logger.info(f"batch_extract_topics_ai called with sample_size={sample}")

        # Đọc dữ liệu từ file
        full_path = os.path.join("/usr/src/app/src", path)

        if not os.path.exists(full_path):
            logger.error(f"File not found: {full_path}")
            return {"error": f"File not found: {path}"}

        logger.info(f"Reading data from {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} entries from {full_path}")

        # Lọc entry có nội dung
        valid_entries = []
        for entry in data:
            # Ưu tiên visible_content
            content = entry.get("visible_content", "")
            if not content:
                content = entry.get("content", "")

            if content:
                valid_entries.append(entry)

        logger.info(f"Found {len(valid_entries)} valid entries with content")

        # Random sampling
        import random  # Thêm import ở đây

        if sample > 0 and sample < len(valid_entries):
            random.seed(42)  # Đảm bảo kết quả luôn giống nhau
            valid_entries = random.sample(valid_entries, sample)
            logger.info(f"Randomly sampled {len(valid_entries)} valid entries")

        # Tạo task với sample
        task = batch_extract_topics_task.delay(valid_entries, True, False)
        task_id = task.id
        logger.info(
            f"Created batch_extract_topics_task with ID: {task_id}, compare_methods=True"
        )

        # Tạo file tạm thời để lưu thông tin task
        result_file = f"/usr/src/app/src/results/batch_extract_topics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Khởi tạo file kết quả
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "created_at": datetime.datetime.now().isoformat(),
                    "sample_size": sample,
                    "status": "PENDING",
                    "entries": [],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Lưu thông tin task
        task_info_file = f"/usr/src/app/src/results/task_{task_id}.json"
        with open(task_info_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "task_type": "batch_extract_topics",
                    "created_at": datetime.datetime.now().isoformat(),
                    "status": "PENDING",
                    "result_file": result_file,
                    "params": {
                        "file_path": full_path,
                        "sample_size": sample,
                        "compare_methods": True,
                        "valid_entries_count": len(valid_entries),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Created task {task_id}. Info saved to {task_info_file}")

        return {
            "task_id": task_id,
            "result_endpoint": f"/ai/batch-extract-topics/{task_id}",
            "message": f"Processing {len(valid_entries)} entries with batch extract topics",
        }
    except Exception as e:
        logger.error(f"Error in batch_extract_topics_ai: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/ai/summarize-khanh")
async def summarize_content_ai(data: Dict):
    """
    API endpoint để tóm tắt nội dung và trích xuất chủ đề
    Sử dụng phiên bản cải tiến process_visible_content_ai để ưu tiên LDA và keyword hơn distilbert
    """
    content = data.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    try:
        # Log độ dài và loại nội dung đầu vào để dễ gỡ lỗi
        content_preview = content[:100] + "..." if len(content) > 100 else content
        logger.info(
            f"Summarizing content with khanh method (length: {len(content)}, preview: {content_preview})"
        )

        # Sử dụng process_visible_content_ai để tóm tắt và trích xuất chủ đề
        result = process_visible_content_ai(content)

        # Kiểm tra chất lượng tóm tắt
        summary = result.get("summary", "")
        if not summary or summary == content or len(summary.split()) < 3:
            logger.warning(
                "Summary appears to be low quality, trying direct summarization"
            )
            # Thử lại với tùy chọn khác như tăng min_length
            from ai_summarization import summarize_text

            summary = summarize_text(content, min_length=30, max_length=200)
            result["summary"] = summary

        logger.info(f"Summary result: {summary[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error in summarize_content_ai: {e}", exc_info=True)
        # Trả về thông báo lỗi chi tiết hơn cho client
        return {
            "error": str(e),
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "topics": [],
            "keywords": [],
            "status": "error",
            "message": "Failed to summarize content due to an error. Please check the input format.",
        }


@app.post("/ai/extract-topics-auto-khanh")
async def extract_topics_auto_ai(data: Dict):
    """
    Tự động trích xuất topics từ content sử dụng kết hợp 3 phương pháp:
    1. LDA (Latent Dirichlet Allocation)
    2. DistilBERT Cosine Similarity
    3. Keyword Extraction

    Sử dụng phiên bản cải tiến process_visible_content_ai để ưu tiên LDA và keyword hơn distilbert
    """
    try:
        content = data.get("content", "")
        threshold = data.get("threshold", 0.4)  # Ngưỡng mặc định: 0.4
        max_topics = data.get("max_topics", 5)  # Số lượng topic tối đa: 5

        if not content:
            raise HTTPException(status_code=400, detail="No content provided")

        # Đầu tiên, trích xuất topics từ mỗi phương pháp riêng biệt
        from ai_summarization import (
            process_visible_content_ai,
            extract_topics_with_lda,
            extract_topics_with_zero_shot,
            extract_keywords,
            summarize_text,
        )

        # 1. Tóm tắt nội dung
        summary = summarize_text(content)

        # 2. Trích xuất từ LDA (tăng num_topics để có nhiều lựa chọn hơn)
        lda_topics_raw = extract_topics_with_lda(
            content, num_topics=max(5, max_topics // 2)
        )
        logger.info(f"Extracted {len(lda_topics_raw)} topics from LDA")

        # Tăng score cho các topics LDA để đảm bảo chúng vượt qua ngưỡng và cao hơn distilbert
        lda_topics = []
        for topic in lda_topics_raw:
            # Tăng score thêm 0.2 để đảm bảo LDA topics có đủ score
            topic["score"] = min(0.9, topic["score"] + 0.2)
            lda_topics.append(topic)
            logger.info(
                f"Enhanced LDA topic: {topic['topic']} with new score: {topic['score']}"
            )

        # 3. Xử lý content bằng phiên bản cải tiến của process_visible_content
        result = process_visible_content_ai(content, max_topics=max_topics)

        # Lấy summary và topics từ kết quả
        topics = result.get("topics", [])
        keywords = result.get("keywords", [])

        # Đếm số lượng topics từ mỗi phương pháp trong kết quả hiện tại
        lda_count_current = sum(1 for t in topics if t.get("method") == "lda")
        logger.info(f"LDA topics in current result: {lda_count_current}")

        # Nếu không có LDA topics trong kết quả, thêm vào ít nhất 1 topic từ LDA
        if lda_count_current == 0 and lda_topics:
            # Sắp xếp LDA topics theo score để lấy topic tốt nhất
            best_lda_topics = sorted(
                lda_topics, key=lambda x: x.get("score", 0), reverse=True
            )

            # Thêm topic LDA tốt nhất vào đầu danh sách
            if best_lda_topics:
                logger.info(f"Adding LDA topic manually: {best_lda_topics[0]['topic']}")
                topics.insert(0, best_lda_topics[0])

        # Sắp xếp lại các topics theo score sau khi thêm LDA topic
        topics = sorted(topics, key=lambda x: x.get("score", 0), reverse=True)

        # Tách topics thành hai nhóm: LDA và non-LDA
        lda_topics_from_result = [t for t in topics if t.get("method") == "lda"]
        non_lda_topics = [t for t in topics if t.get("method") != "lda"]

        # Lọc non-LDA topics theo ngưỡng
        filtered_non_lda = [t for t in non_lda_topics if t.get("score", 0) >= threshold]

        # Luôn giữ lại ít nhất 1 LDA topic với điểm cao nhất, bất kể ngưỡng
        if lda_topics_from_result:
            best_lda = sorted(
                lda_topics_from_result, key=lambda x: x.get("score", 0), reverse=True
            )[0]
            filtered_topics = [best_lda] + filtered_non_lda
        else:
            # Nếu không có LDA topics từ kết quả, sử dụng LDA topics đã tăng cường
            if lda_topics:
                best_lda = sorted(
                    lda_topics, key=lambda x: x.get("score", 0), reverse=True
                )[0]
                filtered_topics = [best_lda] + filtered_non_lda
            else:
                filtered_topics = filtered_non_lda

        # Sắp xếp lại và giới hạn số lượng topics
        filtered_topics = sorted(
            filtered_topics, key=lambda x: x.get("score", 0), reverse=True
        )[:max_topics]

        # Đếm số lượng topics từ mỗi phương pháp trong kết quả cuối cùng
        lda_count = sum(1 for t in filtered_topics if t.get("method") == "lda")
        distilbert_count = sum(
            1
            for t in filtered_topics
            if t.get("method") == "distilbert-cosine-similarity"
        )
        keyword_count = sum(
            1 for t in filtered_topics if t.get("method") == "keyword-extraction"
        )

        logger.info(
            f"Final topics: {len(filtered_topics)} (LDA: {lda_count}, DistilBERT: {distilbert_count}, Keyword: {keyword_count})"
        )

        # Trả về kết quả
        return {
            "status": "success",
            "summary": summary,
            "topics": filtered_topics,
            "keywords": keywords[:5],  # Giới hạn số lượng keywords trả về
        }
    except Exception as e:
        logger.error(f"Error in extract_topics_auto_ai: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error extracting topics: {str(e)}"
        )


@app.get("/ai/batch-extract-topics/{task_id}")
async def get_batch_extract_topics_result(task_id: str):
    """
    API để lấy kết quả của batch extract topics task.

    Args:
        task_id: ID của task

    Returns:
        JSON response với status của task và kết quả nếu đã hoàn thành
    """
    # Đọc thông tin task từ file tạm thay vì database
    temp_file = f"/usr/src/app/src/results/task_{task_id}.json"

    if not os.path.exists(temp_file):
        # Tạo response cơ bản nếu không tìm thấy file
        logger.warning(f"Task info file not found: {temp_file}")
        task_info = {
            "task_id": task_id,
            "result_file": f"/usr/src/app/src/results/batch_extract_topics_{task_id}.json",
        }
    else:
        # Đọc thông tin task từ file
        with open(temp_file, "r", encoding="utf-8") as f:
            task_info = json.load(f)
        logger.info(f"Task info loaded from {temp_file}: {task_info}")

    # Kiểm tra trạng thái của task
    task = batch_extract_topics_task.AsyncResult(task_id)
    logger.info(f"Task {task_id} status: {task.state}")

    if task.state == "PENDING":
        response = {"status": "pending", "message": "Task is still running"}
    elif task.state == "FAILURE":
        error_info = str(task.info) if task.info else "Unknown error"
        logger.error(f"Task {task_id} failed with error: {error_info}")
        response = {"status": "FAILURE", "error": error_info}
    else:
        # Task đã hoàn thành
        try:
            # Lấy kết quả từ task
            result = task.get()
            logger.info(f"Task {task_id} completed successfully")

            # Kiểm tra xem result có phải là dict không
            if not isinstance(result, dict):
                logger.error(f"Task result is not a dict: {type(result)}")
                response = {
                    "status": "ERROR",
                    "message": f"Task completed but returned invalid result type: {type(result)}",
                }
                return response

            # Kiểm tra xem kết quả có chứa summary không
            if "summary" not in result:
                logger.error(
                    f"Task result missing 'summary' field: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}"
                )

            # Kiểm tra xem kết quả có chứa entries không
            if "entries" not in result:
                logger.error(
                    f"Task result missing 'entries' field: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}"
                )

            # Lưu kết quả vào file
            result_file = task_info.get("result_file")
            if result_file:
                try:
                    # Đảm bảo thư mục tồn tại
                    os.makedirs(os.path.dirname(result_file), exist_ok=True)

                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved batch extract topics result to {result_file}")

                    # Cập nhật trạng thái task trong file tạm
                    task_info["status"] = "SUCCESS"
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(task_info, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    logger.error(f"Error saving result to file: {str(e)}")
                    logger.error(traceback.format_exc())

            # Tạo response với thông tin chi tiết hơn
            response = {
                "status": "SUCCESS",
                "result_file": result_file,
                "summary": result.get("summary", {}),
                "entry_count": len(result.get("entries", [])),
                "entries_with_content": result.get("entries_with_content", 0),
                "created_at": task_info.get("created_at"),
                "sample_size": task_info.get("params", {}).get("sample_size", 0),
                "compare_methods": task_info.get("params", {}).get(
                    "compare_methods", True
                ),
            }

            # Thêm thông tin về phương pháp AI và truyền thống
            ai_method = result.get("summary", {}).get("ai_method", {})
            trad_method = result.get("summary", {}).get("traditional_method", {})

            response["ai_topic_count"] = ai_method.get("total_topics", 0)
            response["trad_topic_count"] = trad_method.get("total_topics", 0)

            # Thêm thông tin về số lượng topics được tìm thấy bởi mỗi phương pháp
            method_stats = ai_method.get("method_stats", {})
            for method, stats in method_stats.items():
                response[f"{method}_count"] = stats.get("count", 0)

        except Exception as e:
            logger.error(f"Error processing task result: {str(e)}")
            logger.error(traceback.format_exc())
            response = {
                "status": "ERROR",
                "message": f"Error processing task result: {str(e)}",
            }

    return response


@app.post("/ai/batch-extract-topics")
async def batch_extract_topics(data: Dict):
    """
    API để đọc dữ liệu từ file history_learning_data.json,
    xử lý các entry sử dụng AI để summarize nội dung và xác định các topic liên quan,
    và so sánh với phương pháp truyền thống.

    Tham số:
        path: Đường dẫn đến file history_learning_data.json (mặc định: data/history_learning_data.json)
        sample: Số lượng entry cần xử lý (mặc định: 10)
        compare_methods: Có so sánh giữa phương pháp AI và phương pháp truyền thống không (mặc định: True)

    Returns:
        JSON response với task_id và result_endpoint
    """
    # Lấy các tham số
    file_path = data.get("path", "data/history_learning_data.json")
    # Đảm bảo path được xử lý đúng cách
    json_file_path = (
        f"/usr/src/app/src/{file_path}"
        if not file_path.startswith("/usr/src/app")
        else file_path
    )

    sample_size = int(data.get("sample", 10))
    compare_methods = data.get("compare_methods", True)
    if isinstance(compare_methods, str):
        compare_methods = compare_methods.lower() in ["true", "yes", "1", "t", "y"]

    # Thêm log để kiểm tra giá trị của tham số
    logger.info(
        f"batch_extract_topics called with sample_size={sample_size}, compare_methods={compare_methods}"
    )

    # Check file tồn tại
    if not os.path.exists(json_file_path):
        logger.error(f"File not found: {json_file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {json_file_path}")

    # Đọc file
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            all_entries = json.load(f)

        logger.info(f"Loaded {len(all_entries)} entries from {json_file_path}")

        # Kiểm tra cấu trúc của dữ liệu
        if len(all_entries) > 0:
            first_entry = all_entries[0]
            logger.info(f"First entry structure: {list(first_entry.keys())}")

            # Cảnh báo nếu thiếu các trường quan trọng
            if "content" not in first_entry and "visible_content" not in first_entry:
                logger.warning(
                    "Warning: Both 'content' and 'visible_content' fields not found in entries"
                )

        # Chỉ lấy những entries có nội dung
        valid_entries = []
        for entry in all_entries:
            if entry.get("visible_content") or entry.get("content"):
                valid_entries.append(entry)

        logger.info(f"Found {len(valid_entries)} valid entries with content")

        # Lấy mẫu với số lượng yêu cầu
        import random

        random.seed(42)  # Đảm bảo kết quả nhất quán

        # Nếu có đủ entries hợp lệ, lấy mẫu từ đó
        if len(valid_entries) >= sample_size:
            entries = random.sample(valid_entries, sample_size)
            logger.info(f"Randomly sampled {sample_size} valid entries")
        else:
            # Nếu không đủ entries hợp lệ, sử dụng tất cả những entries có sẵn
            entries = valid_entries
            logger.info(
                f"Using all {len(entries)} valid entries (less than requested {sample_size})"
            )

        # Tạo task để xử lý bất đồng bộ
        task = batch_extract_topics_task.delay(entries, compare_methods)
        logger.info(
            f"Created batch_extract_topics_task with ID: {task.id}, compare_methods={compare_methods}"
        )

        # Tạo timestamp để đặt tên file kết quả
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"/usr/src/app/src/results/batch_extract_topics_{timestamp}.json"

        # Đảm bảo thư mục results tồn tại
        os.makedirs("/usr/src/app/src/results", exist_ok=True)

        # Tạo cấu trúc file kết quả ban đầu
        initial_result = {
            "summary": {
                "total_entries": len(entries),
                "processed_entries": 0,
                "ai_method": {
                    "total_topics": 0,
                    "avg_topics_per_entry": 0,
                    "method_stats": {
                        "lda": {
                            "count": 0,
                            "scores": [],
                            "avg_score": 0,
                            "max_score": 0,
                            "min_score": 0,
                        },
                        "distilbert": {
                            "count": 0,
                            "scores": [],
                            "avg_score": 0,
                            "max_score": 0,
                            "min_score": 0,
                        },
                        "keyword-extraction": {
                            "count": 0,
                            "scores": [],
                            "avg_score": 0,
                            "max_score": 0,
                            "min_score": 0,
                        },
                    },
                },
                "traditional_method": {"total_topics": 0, "avg_topics_per_entry": 0},
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "entries": [],
        }

        # Lưu trạng thái ban đầu vào file kết quả
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(initial_result, f, ensure_ascii=False, indent=2)
        logger.info(f"Created initial result file: {result_file}")

        # Lưu thông tin task vào file tạm thay vì database
        temp_info = {
            "task_id": task.id,
            "task_type": "batch_extract_topics",
            "created_at": datetime.datetime.now().isoformat(),
            "status": "PENDING",
            "result_file": result_file,
            "params": {
                "file_path": json_file_path,
                "sample_size": sample_size,
                "compare_methods": compare_methods,
                "valid_entries_count": len(valid_entries),
            },
        }

        # Lưu thông tin task vào file tạm
        temp_file = f"/usr/src/app/src/results/task_{task.id}.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(temp_info, f, ensure_ascii=False, indent=2)

        logger.info(f"Created task {task.id}. Info saved to {temp_file}")

        # Trả về response
        return {
            "task_id": task.id,
            "result_endpoint": f"/ai/batch-extract-topics/{task.id}",
            "message": f"Processing {len(entries)} entries with batch extract topics task",
            "valid_entries_found": len(valid_entries),
            "requested_sample_size": sample_size,
        }

    except Exception as e:
        logger.error(f"Error in batch_extract_topics: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test")
def test_api():
    """
    Endpoint đơn giản để kiểm tra API server
    """
    return {"status": "OK", "message": "API server is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
        access_log=True,
    )
