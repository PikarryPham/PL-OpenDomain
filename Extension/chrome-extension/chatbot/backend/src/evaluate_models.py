#!/usr/bin/env python
"""
Script đánh giá và so sánh hiệu suất các mô hình embedding khác nhau
Cách sử dụng:
    python evaluate_models.py --data_path data/history_learning_data.json --output_dir results
"""

import os
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Thiết lập đường dẫn cho NLTK data
nltk_data_path = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_path  # Thiết lập biến môi trường


# Định nghĩa hàm verify_nltk_resources trước khi sử dụng
def verify_nltk_resources():
    """Kiểm tra xem tài nguyên NLTK đã được tải thành công chưa"""
    try:
        import nltk
        from nltk.data import find

        resources = ["tokenizers/punkt", "corpora/stopwords"]
        missing = []

        for resource in resources:
            try:
                find(resource)
                print(f"✓ Tài nguyên NLTK '{resource}' đã được tải thành công.")
            except LookupError:
                missing.append(resource)
                print(f"⨯ Thiếu tài nguyên NLTK: '{resource}'")

        if missing:
            print("CẢNH BÁO: Một số tài nguyên NLTK cần thiết chưa được tải đúng cách!")
            print("Đang thử tải lại...")

            for resource in missing:
                if "punkt" in resource:
                    nltk.download("punkt", quiet=False, download_dir=nltk_data_path)
                elif "stopwords" in resource:
                    nltk.download("stopwords", quiet=False, download_dir=nltk_data_path)

            # Kiểm tra lại sau khi tải
            for resource in missing:
                try:
                    find(resource)
                    print(f"✓ Tài nguyên '{resource}' đã được tải thành công.")
                except LookupError:
                    print(
                        f"⨯ Vẫn không thể tải '{resource}'. Có thể ảnh hưởng đến quá trình đánh giá."
                    )
        else:
            print("Tất cả tài nguyên NLTK cần thiết đã sẵn sàng!")

        return len(missing) == 0
    except Exception as e:
        print(f"Lỗi khi kiểm tra tài nguyên NLTK: {e}")
        return False


# Import hàm download_nltk_resources từ file download_nltk_resources.py
try:
    from download_nltk_resources import download_nltk_resources

    # Tải các tài nguyên NLTK cần thiết ngay khi khởi động script
    print("Kiểm tra và tải các tài nguyên NLTK cần thiết...")
    download_nltk_resources()
except ImportError:
    print(
        "Không thể import hàm download_nltk_resources. Cố gắng tải NLTK resources theo cách khác..."
    )
    import nltk

    try:
        nltk.download("punkt", quiet=False, download_dir=nltk_data_path)
        nltk.download("stopwords", quiet=False, download_dir=nltk_data_path)
        nltk.download("wordnet", quiet=False, download_dir=nltk_data_path)
        nltk.download("omw-1.4", quiet=False, download_dir=nltk_data_path)
        print("Đã tải các tài nguyên NLTK cần thiết.")
    except Exception as e:
        print(f"Lỗi khi tải NLTK resources: {e}")

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
)
from tasks import extract_pages_mapping, search_dbpedia_pages
from brain import get_embedding, EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Kiểm tra tài nguyên NLTK sau khi đã import
print("\nKiểm tra tài nguyên NLTK đã tải đầy đủ chưa...")
verify_nltk_resources()
print("Tiếp tục đánh giá các mô hình...")

# Định nghĩa các mô hình embedding để so sánh
COMPARE_MODELS = {
    "openai": get_embedding,
    "tfidf": get_embedding_tfidf,
    "bm25": get_embedding_bm25,
    "distilbert": get_embedding_distilbert,
    "roberta": get_embedding_roberta,
    "xlm-roberta": get_embedding_xlm_roberta,
    "hybrid_tfidf_bert": get_embedding_hybrid_tfidf_bert,
    "hybrid_bm25_bert": get_embedding_hybrid_bm25_bert,
}


def load_data(file_path: str, sample: int = 0) -> List[Dict]:
    """Đọc dữ liệu từ file JSON"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if sample > 0 and sample < len(data):
        return data[:sample]
    return data


def extract_keywords(data: List[Dict]) -> List[str]:
    """Trích xuất keywords từ dữ liệu"""
    keywords_list = []
    for item in data:
        exact_keywords = item.get("exact_keywords", [])
        if exact_keywords:
            keywords_list.append(" ".join(exact_keywords))
    return keywords_list


def get_corpus_from_pages(pages: List[Dict]) -> List[str]:
    """Lấy corpus từ danh sách các trang"""
    corpus = []
    for page in pages:
        abstract = page.get("abstract", "")
        comment = page.get("comment", "")
        if abstract or comment:
            corpus.append(f"{abstract} {comment}")
    return corpus


def measure_embedding_time(model_name: str, texts: List[str]) -> Dict[str, Any]:
    """Đo thời gian embedding của một mô hình"""
    embedding_fn = COMPARE_MODELS[model_name]

    # Đo thời gian
    start_time = time.time()
    for text in texts:
        _ = embedding_fn(text)
    total_time = time.time() - start_time
    avg_time = total_time / len(texts)

    return {
        "model": model_name,
        "total_time": total_time,
        "avg_time": avg_time,
        "num_texts": len(texts),
    }


def evaluate_extraction_quality(
    data_path: str, output_dir: str, sample: int = 10, limit: int = 3
) -> Dict[str, Any]:
    """Đánh giá chất lượng trích xuất của các mô hình"""
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Kết quả đánh giá
    evaluation_results = {"models": {}, "comparison": {}, "summary": {}}

    # Trích xuất dữ liệu cho mỗi mô hình
    for model_name in tqdm(COMPARE_MODELS.keys(), desc="Evaluating Models"):
        model_results = extract_pages_mapping(
            json_file_path=data_path,
            sample=sample,
            limit=limit,
            embedding_model=model_name,
            save_output=False,
        )

        # Lưu kết quả
        model_output_path = os.path.join(
            output_dir, f"extraction_results_{model_name}.json"
        )
        with open(model_output_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2)

        # Phân tích kết quả
        num_entries = len(model_results)
        total_pages = sum(len(entry.get("pages", [])) for entry in model_results)
        avg_pages_per_entry = total_pages / num_entries if num_entries > 0 else 0

        # Thêm vào kết quả đánh giá
        evaluation_results["models"][model_name] = {
            "num_entries": num_entries,
            "total_pages": total_pages,
            "avg_pages_per_entry": avg_pages_per_entry,
        }

    # Tính độ tương đồng giữa các mô hình
    model_names = list(COMPARE_MODELS.keys())
    for i, model_i in enumerate(model_names):
        evaluation_results["comparison"][model_i] = {}
        for j, model_j in enumerate(model_names):
            if i == j:
                evaluation_results["comparison"][model_i][model_j] = 1.0
                continue

            # Đọc kết quả từ hai mô hình
            results_i = json.load(
                open(
                    os.path.join(output_dir, f"extraction_results_{model_i}.json"), "r"
                )
            )
            results_j = json.load(
                open(
                    os.path.join(output_dir, f"extraction_results_{model_j}.json"), "r"
                )
            )

            # Tính Jaccard similarity giữa các tập kết quả
            similarity = calculate_result_similarity(results_i, results_j)
            evaluation_results["comparison"][model_i][model_j] = similarity

    # Tạo biểu đồ so sánh
    create_comparison_charts(evaluation_results, output_dir)

    # Lưu kết quả tổng hợp
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)

    return evaluation_results


def calculate_result_similarity(results_a: List[Dict], results_b: List[Dict]) -> float:
    """Tính độ tương đồng giữa hai tập kết quả sử dụng Jaccard similarity"""
    # Ánh xạ kết quả theo entry_id
    entries_a = {entry.get("entry_id"): entry.get("pages", []) for entry in results_a}
    entries_b = {entry.get("entry_id"): entry.get("pages", []) for entry in results_b}

    # Tính similarity cho mỗi entry
    similarities = []
    for entry_id in set(entries_a.keys()).intersection(set(entries_b.keys())):
        pages_a = set(
            page.get("relatedConcept", {}).get("value", "")
            for page in entries_a[entry_id]
        )
        pages_b = set(
            page.get("relatedConcept", {}).get("value", "")
            for page in entries_b[entry_id]
        )

        union_size = len(pages_a.union(pages_b))
        intersection_size = len(pages_a.intersection(pages_b))

        if union_size > 0:
            similarity = intersection_size / union_size
            similarities.append(similarity)

    # Trả về trung bình của tất cả các similarities
    return sum(similarities) / len(similarities) if similarities else 0


def create_comparison_charts(results: Dict[str, Any], output_dir: str):
    """Tạo biểu đồ so sánh các mô hình"""
    # Tạo dataframe từ kết quả
    model_metrics = []
    for model_name, metrics in results["models"].items():
        model_metrics.append(
            {
                "Model": model_name,
                "Số lượng entry": metrics["num_entries"],
                "Tổng số trang trả về": metrics["total_pages"],
                "Số trang trung bình/entry": metrics["avg_pages_per_entry"],
            }
        )

    df = pd.DataFrame(model_metrics)

    # Biểu đồ so sánh số trang trung bình/entry
    plt.figure(figsize=(12, 6))
    plt.bar(df["Model"], df["Số trang trung bình/entry"], color="skyblue")
    plt.title("So sánh số trang trung bình trả về cho mỗi entry")
    plt.xlabel("Mô hình")
    plt.ylabel("Số trang trung bình")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_pages_comparison.png"))

    # Biểu đồ so sánh tổng số trang
    plt.figure(figsize=(12, 6))
    plt.bar(df["Model"], df["Tổng số trang trả về"], color="lightgreen")
    plt.title("So sánh tổng số trang trả về")
    plt.xlabel("Mô hình")
    plt.ylabel("Tổng số trang")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_pages_comparison.png"))

    # Biểu đồ heatmap độ tương đồng giữa các mô hình
    similarity_matrix = np.zeros(
        (len(results["comparison"]), len(results["comparison"]))
    )
    model_names = list(results["comparison"].keys())

    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            similarity_matrix[i, j] = results["comparison"][model_i].get(model_j, 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Jaccard Similarity")
    plt.title("Độ tương đồng giữa các mô hình embedding")
    plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
    plt.yticks(np.arange(len(model_names)), model_names)

    # Thêm giá trị vào heatmap
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text = plt.text(
                j,
                i,
                f"{similarity_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_similarity_heatmap.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare embedding models"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/history_learning_data.json",
        help="Path to learning history data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--sample", type=int, default=20, help="Number of samples to use for evaluation"
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Limit number of pages per entry"
    )
    args = parser.parse_args()

    logger.info(f"Starting evaluation with data path: {args.data_path}")
    logger.info(f"Sample size: {args.sample}, limit: {args.limit}")

    # Kiểm tra tài nguyên NLTK
    if verify_nltk_resources():
        # Đánh giá chất lượng trích xuất
        results = evaluate_extraction_quality(
            data_path=args.data_path,
            output_dir=args.output_dir,
            sample=args.sample,
            limit=args.limit,
        )

        logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

        # Hiển thị tóm tắt kết quả
        print("\nTÓM TẮT KẾT QUẢ ĐÁNH GIÁ:")
        print("=" * 50)
        for model_name, metrics in results["models"].items():
            print(f"Mô hình: {model_name}")
            print(f"  - Số lượng entry: {metrics['num_entries']}")
            print(f"  - Tổng số trang: {metrics['total_pages']}")
            print(
                f"  - Số trang trung bình/entry: {metrics['avg_pages_per_entry']:.2f}"
            )
            print("-" * 30)
    else:
        print("Đánh giá không thể thực hiện được do thiếu tài nguyên NLTK.")


if __name__ == "__main__":
    main()
