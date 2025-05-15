#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script sửa lỗi 'NoneType' object has no attribute 'eval' trong hàm quantize_model
"""

import os
import sys
import logging
import shutil
import datetime
import re

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Kiểm tra xem có đang chạy trong Docker container hay không
IN_DOCKER = os.path.exists("/.dockerenv")


def find_embeddings_file():
    """Tìm file embeddings.py trong dự án"""
    search_paths = []

    # Nếu chạy trong Docker, tìm trong đường dẫn Docker
    if IN_DOCKER:
        search_paths.append("/usr/src/app/src/embeddings.py")

    # Tìm từ thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(current_dir, "embeddings.py"))

    # Tìm từ thư mục cha của thư mục hiện tại
    parent_dir = os.path.dirname(current_dir)
    search_paths.append(os.path.join(parent_dir, "src", "embeddings.py"))

    # Kiểm tra các đường dẫn
    for path in search_paths:
        if os.path.exists(path):
            return path

    return None


def patch_quantize_function(file_path):
    """Sửa đổi hàm quantize_model để kiểm tra model là None"""
    if not os.path.exists(file_path):
        logger.error(f"Không tìm thấy file {file_path}")
        return False

    # Tạo bản sao để backup
    backup_file = (
        file_path + ".bak." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    shutil.copy2(file_path, backup_file)
    logger.info(f"Đã tạo bản sao {backup_file}")

    # Đọc nội dung file
    with open(file_path, "r") as f:
        content = f.read()

    # Tìm hàm quantize_model
    quantize_pattern = re.compile(r"def\s+quantize_model\s*\(\s*model\s*\)\s*:")
    match = quantize_pattern.search(content)

    if not match:
        logger.warning("Không tìm thấy hàm quantize_model trong file")
        return False

    start_pos = match.start()

    # Tìm vị trí của phần "try:"
    try_pattern = re.compile(r"(\s+)try\s*:", re.MULTILINE)
    try_match = try_pattern.search(content, start_pos)

    if not try_match:
        logger.warning("Không tìm thấy khối try trong hàm quantize_model")
        return False

    indent = try_match.group(1)  # Lấy khoảng trắng đầu dòng

    # Tạo đoạn mã để kiểm tra model là None
    check_none_code = f'{indent}# Kiểm tra model có tồn tại không\n{indent}if model is None:\n{indent}    logger.error("Model is None, cannot quantize")\n{indent}    return None\n\n'

    # Chèn đoạn mã vào vị trí ngay sau try:
    new_content = (
        content[: try_match.end()] + "\n" + check_none_code + content[try_match.end() :]
    )

    # Ghi nội dung đã sửa vào file
    with open(file_path, "w") as f:
        f.write(new_content)

    logger.info("Đã sửa hàm quantize_model để kiểm tra model là None")
    return True


def main():
    """Hàm chính của script"""
    embeddings_file = find_embeddings_file()

    if not embeddings_file:
        logger.error(
            "Không tìm thấy file embeddings.py. Vui lòng chạy script trong thư mục dự án."
        )
        return False

    logger.info(f"Tìm thấy file embeddings.py tại: {embeddings_file}")

    success = patch_quantize_function(embeddings_file)

    if success:
        logger.info("Đã sửa hàm quantize_model thành công")
        print("\n=================================================================")
        print("  ĐÃ SỬA LỖI THÀNH CÔNG")
        print("  Lỗi 'NoneType' object has no attribute 'eval' đã được sửa.")
        print("  File embeddings.py đã được cập nhật.")
        print("  Vui lòng khởi động lại container để áp dụng thay đổi.")
        print("=================================================================\n")
        return True
    else:
        logger.error("Không thể sửa hàm quantize_model")
        print("\n=================================================================")
        print("  KHÔNG THỂ SỬA LỖI")
        print("  Vui lòng kiểm tra lại file embeddings.py hoặc sửa thủ công.")
        print("=================================================================\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
