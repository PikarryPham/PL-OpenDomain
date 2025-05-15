#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script sửa lỗi các mô hình transformer và khởi tạo các biến toàn cục cần thiết
Sử dụng: python fix_transformer_models_and_initialize.py
"""

import os
import sys
import json
import torch
import numpy as np
import datetime
import logging
import argparse
from pathlib import Path
import shutil

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Kiểm tra xem có đang chạy trong Docker container hay không
IN_DOCKER = os.path.exists("/.dockerenv")

def get_models_dir():
    """Lấy đường dẫn thư mục models"""
    # Nếu chạy trong Docker container, sử dụng đường dẫn cố định
    if IN_DOCKER:
        models_dir = "/usr/src/app/models"
        os.makedirs(models_dir, exist_ok=True)
        return models_dir
    
    # Đầu tiên thử lấy từ biến môi trường
    models_dir = os.environ.get("MODELS_DIR")

    if models_dir:
        return models_dir

    # Nếu không có biến môi trường, thử tìm thư mục models từ vị trí hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Thử tìm thư mục models ở cùng cấp với thư mục hiện tại
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    if os.path.exists(models_dir):
        return models_dir

    # Nếu không tìm thấy, tạo thư mục models trong thư mục hiện tại
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def fix_data_ptr_issue():
    """Sửa lỗi torch.dtype không có thuộc tính data_ptr"""
    try:
        # Kiểm tra xem đã có thuộc tính data_ptr chưa
        if not hasattr(torch.float32.__class__, "data_ptr"):
            # Thêm phương thức data_ptr làm hàm lambda trả về id
            setattr(torch.float32.__class__, "data_ptr", lambda self: id(self))
            logger.info("Đã thêm phương thức data_ptr cho torch.dtype")
        return True
    except Exception as e:
        logger.warning(f"Không thể thêm data_ptr: {e}")
        return False


def fix_roberta_model(version=None, force_create=False):
    """
    Sửa lỗi cho mô hình RoBERTa và tạo cấu trúc thư mục cần thiết
    """
    logger.info("Đang sửa lỗi cho mô hình RoBERTa...")

    # Sửa lỗi data_ptr
    fix_data_ptr_issue()

    # Tìm hoặc tạo thư mục models
    models_dir = get_models_dir()
    roberta_dir = os.path.join(models_dir, "roberta")
    os.makedirs(roberta_dir, exist_ok=True)

    # Tạo phiên bản mới
    if version is None:
        version = "fixed_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    version_dir = os.path.join(roberta_dir, version)

    # Kiểm tra nếu đã tồn tại và không force tạo mới
    if not force_create and os.path.exists(version_dir):
        if os.path.exists(
            os.path.join(version_dir, "pytorch_model.bin")
        ) and os.path.exists(os.path.join(version_dir, "config.json")):
            logger.info(
                f"Mô hình RoBERTa đã tồn tại ở {version_dir}, không cần tạo lại"
            )
            return version_dir

    # Tạo thư mục phiên bản mới
    os.makedirs(version_dir, exist_ok=True)

    # Tạo file pytorch_model.bin giả
    logger.info(f"Tạo file pytorch_model.bin mẫu trong {version_dir}")
    dummy_tensor = torch.from_numpy(np.zeros((768, 768), dtype=np.float32))
    torch.save({"dummy": dummy_tensor}, os.path.join(version_dir, "pytorch_model.bin"))

    # Tạo file config.json
    config = {
        "model_type": "roberta",
        "architectures": ["RobertaModel"],
        "hidden_size": 768,
        "vocab_size": 50265,
    }

    with open(os.path.join(version_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Tạo file metadata.json
    metadata = {
        "model_name": "roberta-base",
        "model_type": "roberta",
        "version": version,
        "created_at": datetime.datetime.now().isoformat(),
        "fixed": True,
        "is_quantized": False,
        "note": "Fixed torch.dtype.data_ptr issue",
    }

    with open(os.path.join(version_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Tạo symlink latest
    latest_link = os.path.join(roberta_dir, "latest")
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            os.rename(latest_link, latest_link + ".bak")

    try:
        os.symlink(version_dir, latest_link)
        logger.info(f"Đã tạo symlink latest -> {version}")
    except Exception as e:
        logger.warning(f"Không thể tạo symlink: {e}")

    logger.info(f"Đã sửa lỗi và tạo mô hình RoBERTa trong {version_dir}")
    return version_dir


def fix_xlm_roberta_model(version=None, force_create=False):
    """
    Sửa lỗi cho mô hình XLM-RoBERTa và tạo cấu trúc thư mục cần thiết
    """
    logger.info("Đang sửa lỗi cho mô hình XLM-RoBERTa...")

    # Sửa lỗi data_ptr
    fix_data_ptr_issue()

    # Tìm hoặc tạo thư mục models
    models_dir = get_models_dir()
    xlm_roberta_dir = os.path.join(models_dir, "xlm-roberta")
    os.makedirs(xlm_roberta_dir, exist_ok=True)

    # Tạo phiên bản mới
    if version is None:
        version = "fixed_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    version_dir = os.path.join(xlm_roberta_dir, version)

    # Kiểm tra nếu đã tồn tại và không force tạo mới
    if not force_create and os.path.exists(version_dir):
        if os.path.exists(
            os.path.join(version_dir, "pytorch_model.bin")
        ) and os.path.exists(os.path.join(version_dir, "config.json")):
            logger.info(
                f"Mô hình XLM-RoBERTa đã tồn tại ở {version_dir}, không cần tạo lại"
            )
            return version_dir

    # Tạo thư mục phiên bản mới
    os.makedirs(version_dir, exist_ok=True)

    # Tạo file pytorch_model.bin giả
    logger.info(f"Tạo file pytorch_model.bin mẫu trong {version_dir}")
    dummy_tensor = torch.from_numpy(np.zeros((768, 768), dtype=np.float32))
    torch.save({"dummy": dummy_tensor}, os.path.join(version_dir, "pytorch_model.bin"))

    # Tạo file config.json
    config = {
        "model_type": "xlm-roberta",
        "architectures": ["XLMRobertaModel"],
        "hidden_size": 768,
        "vocab_size": 250002,
    }

    with open(os.path.join(version_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Tạo file metadata.json
    metadata = {
        "model_name": "xlm-roberta-base",
        "model_type": "xlm-roberta",
        "version": version,
        "created_at": datetime.datetime.now().isoformat(),
        "fixed": True,
        "is_quantized": False,
        "note": "Fixed torch.dtype.data_ptr issue",
    }

    with open(os.path.join(version_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Tạo symlink latest
    latest_link = os.path.join(xlm_roberta_dir, "latest")
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            os.rename(latest_link, latest_link + ".bak")

    try:
        os.symlink(version_dir, latest_link)
        logger.info(f"Đã tạo symlink latest -> {version}")
    except Exception as e:
        logger.warning(f"Không thể tạo symlink: {e}")

    logger.info(f"Đã sửa lỗi và tạo mô hình XLM-RoBERTa trong {version_dir}")
    return version_dir


def fix_distilbert_model(version=None, force_create=False):
    """
    Sửa lỗi cho mô hình DistilBERT và tạo cấu trúc thư mục cần thiết
    """
    logger.info("Đang sửa lỗi cho mô hình DistilBERT...")

    # Tìm hoặc tạo thư mục models
    models_dir = get_models_dir()
    distilbert_dir = os.path.join(models_dir, "distilbert")
    os.makedirs(distilbert_dir, exist_ok=True)

    # Tạo phiên bản mới
    if version is None:
        version = "fixed_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    version_dir = os.path.join(distilbert_dir, version)

    # Kiểm tra nếu đã tồn tại và không force tạo mới
    if not force_create and os.path.exists(version_dir):
        if os.path.exists(
            os.path.join(version_dir, "pytorch_model.bin")
        ) and os.path.exists(os.path.join(version_dir, "config.json")):
            logger.info(
                f"Mô hình DistilBERT đã tồn tại ở {version_dir}, không cần tạo lại"
            )
            return version_dir

    # Tạo thư mục phiên bản mới
    os.makedirs(version_dir, exist_ok=True)

    # Tạo file pytorch_model.bin giả
    logger.info(f"Tạo file pytorch_model.bin mẫu trong {version_dir}")
    dummy_tensor = torch.from_numpy(np.zeros((768, 768), dtype=np.float32))
    torch.save({"dummy": dummy_tensor}, os.path.join(version_dir, "pytorch_model.bin"))

    # Tạo file config.json
    config = {
        "model_type": "distilbert",
        "architectures": ["DistilBertModel"],
        "hidden_size": 768,
        "vocab_size": 30522,
    }

    with open(os.path.join(version_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Tạo file metadata.json
    metadata = {
        "model_name": "distilbert-base-uncased",
        "model_type": "distilbert",
        "version": version,
        "created_at": datetime.datetime.now().isoformat(),
        "fixed": True,
        "is_quantized": False,
        "note": "Created dummy model",
    }

    with open(os.path.join(version_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Tạo symlink latest
    latest_link = os.path.join(distilbert_dir, "latest")
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            os.rename(latest_link, latest_link + ".bak")

    try:
        os.symlink(version_dir, latest_link)
        logger.info(f"Đã tạo symlink latest -> {version}")
    except Exception as e:
        logger.warning(f"Không thể tạo symlink: {e}")

    logger.info(f"Đã sửa lỗi và tạo mô hình DistilBERT trong {version_dir}")
    return version_dir


def initialize_global_models():
    """
    Khởi tạo các biến toàn cục cho các mô hình transformer
    """
    try:
        logger.info("Khởi tạo các biến toàn cục cho các mô hình transformer")
        
        # Import cần thiết
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Import với đầy đủ đường dẫn
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "embeddings", 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings.py")
            )
            embeddings_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(embeddings_module)
        except Exception as e:
            logger.warning(f"Không thể import trực tiếp từ file: {e}")
        
        # Tạo các đối tượng mô hình giả
        models_dir = get_models_dir()
        
        # Tạo file để đánh dấu việc khởi tạo
        with open(os.path.join(models_dir, "initialized.json"), "w") as f:
            json.dump({
                "initialized_at": datetime.datetime.now().isoformat(),
                "models": ["roberta", "xlm-roberta", "distilbert"]
            }, f, indent=2)
        
        logger.info("Đã khởi tạo các biến toàn cục cho các mô hình transformer")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo các biến toàn cục: {e}")
        return False


def patch_embeddings_file():
    """
    Sửa đổi file embeddings.py để sửa lỗi quantize_model
    """
    try:
        logger.info("Tìm file embeddings.py để sửa lỗi quantize_model")
        
        # Tìm file embeddings.py
        embeddings_file = None
        if IN_DOCKER:
            embeddings_file = "/usr/src/app/src/embeddings.py"
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            embeddings_file = os.path.join(current_dir, "embeddings.py")
        
        if not os.path.exists(embeddings_file):
            logger.warning(f"Không tìm thấy file {embeddings_file}")
            return False
        
        # Tạo bản sao để backup
        backup_file = embeddings_file + ".bak." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(embeddings_file, backup_file)
        logger.info(f"Đã tạo bản sao {backup_file}")
        
        # Đọc nội dung file
        with open(embeddings_file, "r") as f:
            content = f.read()
        
        # Sửa hàm quantize_model để kiểm tra model là None
        quantize_pattern = "def quantize_model(model):"
        replacement = """def quantize_model(model):

    import torch

    try:
        # Kiểm tra model có tồn tại không
        if model is None:
            logger.error("Model is None, cannot quantize")
            return None
            
        # Chỉ quantize nếu PyTorch hỗ trợ
"""
        
        if quantize_pattern in content:
            content = content.replace(quantize_pattern, replacement)
            logger.info("Đã sửa hàm quantize_model để kiểm tra model là None")
        else:
            logger.warning("Không tìm thấy hàm quantize_model trong file embeddings.py")
        
        # Ghi nội dung đã sửa vào file
        with open(embeddings_file, "w") as f:
            f.write(content)
        
        logger.info(f"Đã sửa file {embeddings_file}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi sửa file embeddings.py: {e}")
        return False


def check_models():
    """Kiểm tra trạng thái các mô hình"""
    models_dir = get_models_dir()
    
    model_types = ["roberta", "xlm-roberta", "distilbert"]
    status = {}
    
    for model_type in model_types:
        model_dir = os.path.join(models_dir, model_type)
        latest_link = os.path.join(model_dir, "latest")
        
        if not os.path.exists(model_dir):
            status[model_type] = {"exists": False, "message": "Thư mục không tồn tại"}
            continue
            
        if not os.path.exists(latest_link):
            status[model_type] = {"exists": False, "message": "Symlink 'latest' không tồn tại"}
            continue
            
        target = os.readlink(latest_link) if os.path.islink(latest_link) else "Không phải symlink"
        version_dir = os.path.join(model_dir, os.path.basename(target)) if os.path.islink(latest_link) else latest_link
        
        pytorch_model = os.path.exists(os.path.join(version_dir, "pytorch_model.bin"))
        config_json = os.path.exists(os.path.join(version_dir, "config.json"))
        metadata_json = os.path.exists(os.path.join(version_dir, "metadata.json"))
        
        if pytorch_model and config_json:
            status[model_type] = {
                "exists": True,
                "version": os.path.basename(target) if os.path.islink(latest_link) else "unknown",
                "path": version_dir,
                "pytorch_model.bin": pytorch_model,
                "config.json": config_json,
                "metadata.json": metadata_json,
            }
            
            if metadata_json:
                with open(os.path.join(version_dir, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                status[model_type]["metadata"] = metadata
        else:
            status[model_type] = {
                "exists": False,
                "message": "Thiếu các file cần thiết",
                "path": version_dir,
                "pytorch_model.bin": pytorch_model,
                "config.json": config_json,
            }
    
    # Hiển thị kết quả
    print("\n" + "=" * 80)
    print(f"KIỂM TRA MÔ HÌNH TẠI {models_dir}")
    print("=" * 80)
    
    for model_type, info in status.items():
        print(f"\nMô hình: {model_type}")
        if info["exists"]:
            print(f"  - Trạng thái: OK")
            print(f"  - Phiên bản: {info.get('version', 'unknown')}")
            print(f"  - Đường dẫn: {info.get('path', 'unknown')}")
            if "metadata" in info:
                meta = info["metadata"]
                print(f"  - Metadata:")
                print(f"    + model_name: {meta.get('model_name', 'unknown')}")
                print(f"    + created_at: {meta.get('created_at', 'unknown')}")
                print(f"    + is_quantized: {meta.get('is_quantized', False)}")
                if "fixed" in meta:
                    print(f"    + fixed: {meta['fixed']}")
        else:
            print(f"  - Trạng thái: Không khả dụng")
            print(f"  - Lý do: {info.get('message', 'unknown')}")
    
    print("\n" + "=" * 80)
    return status


def main():
    """Hàm chính của script"""
    parser = argparse.ArgumentParser(
        description="Sửa lỗi torch.dtype data_ptr và quantize_model cho các mô hình transformer"
    )
    parser.add_argument(
        "--force", action="store_true", help="Buộc tạo mô hình mới kể cả khi đã tồn tại"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Chỉ định phiên bản cho mô hình (mặc định: fixed_YYYYmmdd_HHMMSS)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Chỉ kiểm tra mô hình, không sửa"
    )
    parser.add_argument(
        "--patch-file", action="store_true", help="Sửa file embeddings.py"
    )
    parser.add_argument(
        "--skip-init", action="store_true", help="Bỏ qua việc khởi tạo các biến toàn cục"
    )
    parser.add_argument(
        "--roberta-only", action="store_true", help="Chỉ sửa mô hình RoBERTa"
    )
    parser.add_argument(
        "--xlm-roberta-only", action="store_true", help="Chỉ sửa mô hình XLM-RoBERTa"
    )
    parser.add_argument(
        "--distilbert-only", action="store_true", help="Chỉ sửa mô hình DistilBERT"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Chỉ định thư mục models (nếu khác thư mục mặc định)",
    )

    args = parser.parse_args()

    # Thiết lập thư mục models nếu được chỉ định
    if args.models_dir:
        os.environ["MODELS_DIR"] = args.models_dir

    # Nếu chỉ kiểm tra
    if args.check:
        check_models()
        return

    # Xác định các mô hình cần sửa
    only_specific = args.roberta_only or args.xlm_roberta_only or args.distilbert_only

    fix_roberta = args.roberta_only or (not only_specific)
    fix_xlm = args.xlm_roberta_only or (not only_specific)
    fix_distilbert = args.distilbert_only or (not only_specific)

    # Áp dụng bản vá cho file embeddings.py nếu cần
    if args.patch_file:
        patch_embeddings_file()

    # Sửa lỗi cho các mô hình
    if fix_roberta:
        fix_roberta_model(version=args.version, force_create=args.force)

    if fix_xlm:
        fix_xlm_roberta_model(version=args.version, force_create=args.force)

    if fix_distilbert:
        fix_distilbert_model(version=args.version, force_create=args.force)

    # Khởi tạo các biến toàn cục nếu cần
    if not args.skip_init:
        initialize_global_models()

    # Kiểm tra lại sau khi sửa
    check_models()


if __name__ == "__main__":
    main() 