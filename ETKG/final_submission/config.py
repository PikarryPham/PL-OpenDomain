# Cấu hình kết nối Neo4j
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"  # Đã thay đổi mật khẩu từ mặc định

# Đường dẫn đến các file dữ liệu
USERS_DATA_PATH = "/app/upload/users_data_sample.json"
QUESTION_DATA_PATH = "/app/upload/question_data_sample.json"
OPTIONS_DATA_PATH = "/app/upload/options_data_sample.json"
HISTORY_LEARNING_DATA_PATH = "/app/upload/history_learning_data_sample.json"
FINAL_SAMPLE_OUTPUT_PATH = "/app/upload/final_sample_output.json"

# Cấu hình batch size cho xử lý dữ liệu lớn
BATCH_SIZE = 100

# Cấu hình logging
LOG_LEVEL = "INFO"
LOG_FILE = "/app/tkg_builder.log"
