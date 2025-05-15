#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chuẩn bị các mô hình Hugging Face cho chế độ offline.
Script này nên được chạy khi có kết nối internet để tải xuống và lưu vào cache.
"""

import os
import logging
import argparse
from pathlib import Path
import torch
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

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Danh sách các mô hình cần tải
MODELS = [
    {
        "name": "roberta-base",
        "tokenizer_class": RobertaTokenizer,
        "model_class": RobertaModel,
    },
    {
        "name": "xlm-roberta-base",
        "tokenizer_class": XLMRobertaTokenizer,
        "model_class": XLMRobertaModel,
    },
    {
        "name": "distilbert-base-uncased",
        "tokenizer_class": DistilBertTokenizer,
        "model_class": DistilBertModel,
    },
    {
        "name": "bert-base-uncased",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModel,
    },
]


def download_model(model_info, cache_dir=None, save_local=False, local_dir=None):
    """
    Tải xuống mô hình và tokenizer, đồng thời cache chúng.

    Tham số:
    - model_info: Thông tin về mô hình (tên, lớp tokenizer, lớp model)
    - cache_dir: Thư mục cache cho Hugging Face
    - save_local: Có lưu mô hình vào thư mục local không
    - local_dir: Đường dẫn đến thư mục local để lưu mô hình
    """
    try:
        logger.info(f"Tải xuống {model_info['name']}...")

        # Tải tokenizer
        tokenizer = model_info["tokenizer_class"].from_pretrained(
            model_info["name"], use_fast=True, cache_dir=cache_dir
        )

        # Tải model
        model = model_info["model_class"].from_pretrained(
            model_info["name"], cache_dir=cache_dir
        )

        logger.info(f"Đã tải xuống {model_info['name']} thành công!")

        # Lưu vào thư mục local nếu được yêu cầu
        if save_local and local_dir:
            model_dir = os.path.join(local_dir, model_info["name"])
            os.makedirs(model_dir, exist_ok=True)

            logger.info(f"Lưu {model_info['name']} vào {model_dir}...")
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            logger.info(f"Đã lưu {model_info['name']} vào thư mục local!")

        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải {model_info['name']}: {str(e)}")
        return False


def setup_offline_mode(models_dir):
    """Tạo file .no_download trong thư mục các mô hình để ngăn tải xuống."""
    try:
        for model_info in MODELS:
            model_dir = os.path.join(models_dir, model_info["name"])
            if os.path.exists(model_dir):
                no_download_file = os.path.join(model_dir, ".no_download")
                with open(no_download_file, "w") as f:
                    f.write("Prevent downloading when in offline mode")
                logger.info(f"Đã tạo file .no_download trong {model_dir}")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập chế độ offline: {str(e)}")


def test_models(models_dir):
    """Kiểm tra xem các mô hình đã được tải xuống có hoạt động không."""
    try:
        # Lưu biến môi trường hiện tại
        original_offline = os.environ.get("TRANSFORMERS_OFFLINE", "0")

        # Thiết lập chế độ offline
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        for model_info in MODELS:
            try:
                model_dir = os.path.join(models_dir, model_info["name"])
                if os.path.exists(model_dir):
                    logger.info(
                        f"Kiểm tra {model_info['name']} trong chế độ offline..."
                    )

                    # Tải tokenizer và model từ thư mục local
                    tokenizer = model_info["tokenizer_class"].from_pretrained(model_dir)
                    model = model_info["model_class"].from_pretrained(model_dir)

                    # Thử encode một câu để kiểm tra
                    text = (
                        "Kiểm tra xem mô hình có hoạt động trong chế độ offline không."
                    )
                    inputs = tokenizer(text, return_tensors="pt")

                    with torch.no_grad():
                        outputs = model(**inputs)

                    logger.info(
                        f"{model_info['name']} hoạt động tốt trong chế độ offline!"
                    )
                else:
                    logger.warning(
                        f"Không tìm thấy {model_info['name']} trong thư mục local."
                    )
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra {model_info['name']}: {str(e)}")

        # Khôi phục biến môi trường
        if original_offline != "0":
            os.environ["TRANSFORMERS_OFFLINE"] = original_offline
        else:
            del os.environ["TRANSFORMERS_OFFLINE"]

    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra các mô hình: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị mô hình Hugging Face cho chế độ offline"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Thư mục cache cho Hugging Face (mặc định: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./models",
        help="Thư mục local để lưu các mô hình",
    )
    parser.add_argument(
        "--save_local", action="store_true", help="Lưu mô hình vào thư mục local"
    )
    parser.add_argument(
        "--setup_offline", action="store_true", help="Thiết lập chế độ offline"
    )
    parser.add_argument(
        "--test", action="store_true", help="Kiểm tra các mô hình đã tải xuống"
    )

    args = parser.parse_args()

    # Tạo thư mục local nếu cần
    if args.save_local:
        os.makedirs(args.local_dir, exist_ok=True)

    # Tải các mô hình
    for model_info in MODELS:
        download_model(
            model_info,
            cache_dir=args.cache_dir,
            save_local=args.save_local,
            local_dir=args.local_dir,
        )

    # Thiết lập chế độ offline nếu được yêu cầu
    if args.setup_offline:
        setup_offline_mode(args.local_dir)

    # Kiểm tra các mô hình nếu được yêu cầu
    if args.test:
        test_models(args.local_dir)

    logger.info("Hoàn thành việc chuẩn bị mô hình cho chế độ offline!")


if __name__ == "__main__":
    main()
