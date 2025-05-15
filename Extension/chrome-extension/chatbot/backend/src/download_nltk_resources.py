#!/usr/bin/env python
"""
Script để tải xuống tất cả các tài nguyên NLTK cần thiết cho quá trình đánh giá mô hình embedding
"""

import nltk
import logging
import os

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Đảm bảo thư mục nltk_data tồn tại
nltk_data_path = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)


def download_nltk_resources():
    """Tải xuống tất cả các tài nguyên NLTK cần thiết"""
    resources = [
        "punkt",  # Tokenizer
        "stopwords",  # Stopwords
        "wordnet",  # WordNet synsets
        "omw-1.4",  # Open Multilingual WordNet
    ]

    success = True
    for resource in resources:
        try:
            logger.info(f"Đang tải xuống tài nguyên NLTK: {resource}")
            nltk.download(resource)
            logger.info(f"Đã tải xuống thành công: {resource}")
        except Exception as e:
            logger.error(f"Lỗi khi tải xuống {resource}: {e}")
            success = False

    # Đặc biệt tạo liên kết tới punkt_tab từ punkt để giải quyết lỗi
    try:
        logger.info("Kiểm tra và cấu hình tài nguyên punkt_tab...")
        import os
        from nltk.data import find

        # Tìm thư mục punkt trong NLTK data
        try:
            punkt_dir = os.path.dirname(find("tokenizers/punkt"))
            punkt_tab_dir = os.path.join(
                os.path.dirname(punkt_dir), "punkt_tab", "english"
            )
            os.makedirs(punkt_tab_dir, exist_ok=True)

            # Tạo liên kết hoặc copy file từ punkt sang punkt_tab
            punkt_english_dir = os.path.join(punkt_dir, "english")
            for file in os.listdir(punkt_english_dir):
                source = os.path.join(punkt_english_dir, file)
                target = os.path.join(punkt_tab_dir, file)
                if not os.path.exists(target):
                    # Cố gắng tạo liên kết hoặc copy file
                    try:
                        os.symlink(source, target)
                        logger.info(f"Đã tạo liên kết từ {source} tới {target}")
                    except:
                        import shutil

                        shutil.copy2(source, target)
                        logger.info(f"Đã sao chép từ {source} tới {target}")

            logger.info("Đã cấu hình thành công tài nguyên punkt_tab")
        except Exception as e:
            logger.error(f"Không thể cấu hình punkt_tab: {e}")

            # Thử cách khác: tải trực tiếp PunktTokenizer
            try:
                from nltk.tokenize import PunktTokenizer

                # Chỉ để đảm bảo class được tải
                tokenizer = PunktTokenizer()
                logger.info("Đã tải PunktTokenizer trực tiếp")
            except Exception as e2:
                logger.error(f"Không thể tải PunktTokenizer: {e2}")
                success = False
    except Exception as e:
        logger.error(f"Lỗi khi xử lý punkt_tab: {e}")
        success = False

    return success


if __name__ == "__main__":
    logger.info("Bắt đầu tải xuống các tài nguyên NLTK...")
    success = download_nltk_resources()

    if success:
        logger.info("Đã tải xuống tất cả các tài nguyên NLTK thành công!")
    else:
        logger.warning("Có lỗi xảy ra khi tải một số tài nguyên NLTK.")
