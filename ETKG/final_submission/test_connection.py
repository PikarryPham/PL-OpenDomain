import logging
import sys
sys.path.append('/home/ubuntu')

from config import *

# Cấu hình logging hiển thị ra console
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Thêm StreamHandler để hiển thị log ra console
    ]
)
logger = logging.getLogger(__name__)

# Kiểm tra kết nối Neo4j
try:
    from neo4j import GraphDatabase
    
    # Thử kết nối với Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN 'Kết nối Neo4j thành công!' as message")
        for record in result:
            print(record["message"])
    driver.close()
    
    print("Đã kết nối thành công với Neo4j")
except Exception as e:
    print(f"Lỗi khi kết nối với Neo4j: {str(e)}")
    
# Kiểm tra đọc dữ liệu từ file JSON
try:
    import json
    
    # Đọc file users_data_sample.json
    with open(USERS_DATA_PATH, 'r', encoding='utf-8') as file:
        users_data = json.load(file)
        print(f"Đã đọc thành công file users_data_sample.json, số lượng user: {len(users_data)}")
    
    # Đọc file question_data_sample.json
    with open(QUESTION_DATA_PATH, 'r', encoding='utf-8') as file:
        questions_data = json.load(file)
        print(f"Đã đọc thành công file question_data_sample.json, số lượng question: {len(questions_data)}")
    
    # Đọc file options_data_sample.json
    with open(OPTIONS_DATA_PATH, 'r', encoding='utf-8') as file:
        options_data = json.load(file)
        print(f"Đã đọc thành công file options_data_sample.json, số lượng option: {len(options_data)}")
    
    # Đọc file history_learning_data_sample.json
    with open(HISTORY_LEARNING_DATA_PATH, 'r', encoding='utf-8') as file:
        entries_data = json.load(file)
        print(f"Đã đọc thành công file history_learning_data_sample.json, số lượng entry: {len(entries_data)}")
    
    # Đọc file final_sample_output.json
    with open(FINAL_SAMPLE_OUTPUT_PATH, 'r', encoding='utf-8') as file:
        concepts_data = json.load(file)
        print(f"Đã đọc thành công file final_sample_output.json")
        
    print("Đã kiểm tra thành công việc đọc dữ liệu từ các file JSON")
except Exception as e:
    print(f"Lỗi khi đọc dữ liệu từ file JSON: {str(e)}")
