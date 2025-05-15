#!/usr/bin/env python3
"""
Script để kiểm tra chức năng AI Summarization
"""
import argparse
import json
import requests
import time

def test_summarize(server_url, content):
    """
    Test API summarize content
    """
    print(f"\n===== Kiểm tra API tóm tắt nội dung =====")
    url = f"{server_url}/ai/summarize"
    data = {"content": content}
    
    response = requests.post(url, json=data)
    result = response.json()
    
    print(f"Status code: {response.status_code}")
    print(f"Summary: {result.get('summary', '')}")
    print(f"Số lượng topics: {len(result.get('topics', []))}")
    print(f"Topics đầu tiên: {result.get('topics', [])[:2]}")
    print(f"Keywords: {result.get('keywords', [])}")
    
    return result


def test_extract_topics(server_url, content):
    """
    Test API extract topics
    """
    print(f"\n===== Kiểm tra API trích xuất chủ đề =====")
    url = f"{server_url}/ai/extract-topics"
    data = {"content": content}
    
    response = requests.post(url, json=data)
    result = response.json()
    
    print(f"Status code: {response.status_code}")
    print(f"Topics: {result.get('topics', [])}")
    
    return result


def test_batch_process(server_url, entries, use_async=False):
    """
    Test API batch process
    """
    print(f"\n===== Kiểm tra API xử lý hàng loạt {'(bất đồng bộ)' if use_async else ''} =====")
    url = f"{server_url}/ai/batch-process"
    data = {"entries": entries, "async": use_async}
    
    response = requests.post(url, json=data)
    result = response.json()
    
    print(f"Status code: {response.status_code}")
    
    if use_async:
        task_id = result.get("task_id")
        print(f"Task ID: {task_id}")
        print("Đang chờ kết quả...")
        
        # Kiểm tra kết quả sau 30 giây (thực tế cần tạo API riêng để check kết quả task)
        time.sleep(30)
        print("Lưu ý: Trong môi trường thực tế, cần cung cấp API để kiểm tra kết quả task")
    else:
        processed = result.get("processed_entries", [])
        print(f"Đã xử lý {len(processed)} entries")
        for i, entry in enumerate(processed[:2]):
            print(f"Entry {i+1} - summary: {entry.get('summary', '')[:100]}...")
    
    return result


def test_integrated_sync_data(server_url, content):
    """
    Test API sync_data với visible_content
    """
    print(f"\n===== Kiểm tra tích hợp với sync_data API =====")
    url = f"{server_url}/dbpedia/sync-data"
    data = {"visible_content": content}
    
    response = requests.post(url, json=data)
    result = response.json()
    
    print(f"Status code: {response.status_code}")
    print(f"Task ID: {result.get('task_id')}")
    
    return result


def test_models_status(server_url):
    """
    Test API models status
    """
    print(f"\n===== Kiểm tra trạng thái các mô hình AI =====")
    url = f"{server_url}/ai/models/status"
    
    response = requests.get(url)
    result = response.json()
    
    print(f"Status code: {response.status_code}")
    print(f"Summarizer: {result.get('summarizer', False)}")
    print(f"Topic extractor: {result.get('topic_extractor', False)}")
    print(f"NLP: {result.get('nlp', False)}")
    
    return result


def test_initialize_models(server_url):
    """
    Test API initialize models
    """
    print(f"\n===== Khởi tạo các mô hình AI =====")
    url = f"{server_url}/ai/initialize-models"
    
    response = requests.post(url)
    result = response.json()
    
    print(f"Status code: {response.status_code}")
    print(f"Status: {result.get('status')}")
    print(f"Models: {result.get('models')}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test AI Summarization APIs")
    parser.add_argument("--server", "-s", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--content", "-c", type=str, default="", help="Content to process")
    parser.add_argument("--file", "-f", type=str, default="", help="File containing content to process")
    parser.add_argument("--api", "-a", type=str, default="all", help="API to test: all, summarize, topics, batch, sync, status, initialize")
    
    args = parser.parse_args()
    
    # Lấy nội dung cần xử lý
    content = args.content
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            return
    
    if not content and args.api not in ["status", "initialize"]:
        content = """
        Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data, and apply knowledge from data across a broad range of application domains. Data science is related to data mining, machine learning and big data. Data science is a "concept to unify statistics, data analysis, informatics, and their related methods" in order to "understand and analyze actual phenomena" with data.
        It uses techniques and theories drawn from many fields within the context of mathematics, statistics, computer science, information science, and domain knowledge. However, data science is different from computer science and information science. Turing Award winner Jim Gray imagined data science as a "fourth paradigm" of science (empirical, theoretical, computational, and now data-driven) and asserted that "everything about science is changing because of the impact of information technology" and the data deluge.
        """
    
    # Kiểm tra các API
    server_url = args.server.rstrip("/")
    
    if args.api in ["all", "status"]:
        test_models_status(server_url)
    
    if args.api in ["all", "initialize"]:
        test_initialize_models(server_url)
    
    if args.api in ["all", "summarize"]:
        test_summarize(server_url, content)
    
    if args.api in ["all", "topics"]:
        test_extract_topics(server_url, content)
    
    if args.api in ["all", "batch"]:
        entries = [{"visible_content": content}, {"visible_content": content[:200]}]
        test_batch_process(server_url, entries)
    
    if args.api in ["all", "sync"]:
        test_integrated_sync_data(server_url, content)


if __name__ == "__main__":
    main() 