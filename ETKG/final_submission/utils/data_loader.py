import json
import logging
from datetime import datetime
import sys
sys.path.append('/home/ubuntu')
from config import *

# Cấu hình logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='tkg_builder.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

def load_json_file(file_path):
    """
    Đọc dữ liệu từ file JSON
    
    Args:
        file_path (str): Đường dẫn đến file JSON
        
    Returns:
        dict or list: Dữ liệu từ file JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Đã đọc dữ liệu từ {file_path}")
        return data
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        raise

def load_users_data():
    """
    Đọc dữ liệu người dùng từ file JSON
    
    Returns:
        list: Danh sách người dùng
    """
    return load_json_file(USERS_DATA_PATH)

def load_questions_data():
    """
    Đọc dữ liệu câu hỏi từ file JSON
    
    Returns:
        list: Danh sách câu hỏi
    """
    return load_json_file(QUESTION_DATA_PATH)

def load_options_data():
    """
    Đọc dữ liệu lựa chọn từ file JSON
    
    Returns:
        list: Danh sách lựa chọn
    """
    return load_json_file(OPTIONS_DATA_PATH)

def load_history_learning_data():
    """
    Đọc dữ liệu lịch sử học tập từ file JSON
    
    Returns:
        list: Danh sách lịch sử học tập
    """
    return load_json_file(HISTORY_LEARNING_DATA_PATH)

def load_final_sample_output():
    """
    Đọc dữ liệu kết quả mẫu từ file JSON
    
    Returns:
        dict: Dữ liệu kết quả mẫu
    """
    return load_json_file(FINAL_SAMPLE_OUTPUT_PATH)

def batch_data(data, batch_size=BATCH_SIZE):
    """
    Chia dữ liệu thành các batch nhỏ hơn
    
    Args:
        data (list): Dữ liệu cần chia
        batch_size (int): Kích thước của mỗi batch
        
    Returns:
        list: Danh sách các batch
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def get_current_timestamp():
    """
    Lấy thời gian hiện tại theo định dạng ISO
    
    Returns:
        str: Thời gian hiện tại theo định dạng ISO
    """
    return datetime.now().isoformat()
