import general_lib
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import json
from datetime import datetime
import uuid
import logging
import sys
from io import StringIO
from io import BytesIO
import tempfile
import heapq
import os
from datetime import timedelta
from pprint import pprint
import re
import hashlib
import base64

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = True


ACCOUNT_NAME = "13f45cadls"
AZURE_DATALAKE_STORAGE_KEY =  ""
CLICK_HOUSE_HOST = "h8tw70myst.ap-northeast-1.aws.clickhouse.cloud"
CLICK_HOUSE_USER = "default"
CLICK_HOUSE_PASSWORD = ""
SOURCE_CONTAINER = "00_1_land"
DESTINATION_CONTAINER = "02bronze"
BASE_PATH = "streaming-sources"

PROBLEM_SOURCE_FILE_BASE_PATH = "batch-sources/junyi/junyi_ProblemLog_original/json/"
DESTINATION_BASE_PATH = "batch-sources/junyi/preprocessing_batch_sources_users/json"

def load_user():
    JOB_DATE = datetime.now().strftime('%Y%m%d')
    file_path = f"{PROBLEM_SOURCE_FILE_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_ProblemLog_original.json"
    # extract problem log
    service_client = general_lib.get_azure_service_client_by_account_key(ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    source_file_system_client = service_client.get_file_system_client(SOURCE_CONTAINER)
    source_file_client = source_file_system_client.get_file_client(file_path)
    
    df = general_lib.read_azure_datalake_storage(SOURCE_CONTAINER, file_path, ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    logger.info(f"Loaded {len(df)} row of problem log")
    return df

def hash_password(user_id):
    #password = ('junyi' + str(user_id)).encode('utf-8')
    password = ('junyi123456').encode('utf-8')
    salt = os.urandom(16)  # Generate a random salt
    
    # Use SHA-256 with salt for hashing (not as secure as bcrypt but available)
    hashed = hashlib.pbkdf2_hmac('sha256', password, salt, 100)
    
    # Convert to string for storage
    return base64.b64encode(salt + hashed).decode('utf-8')

def user_transform(input_df):
    # 3. Giữ nguyên trường user_id, các trường còn lại drop column
    print("\n3. Keeping only the user_id column...")
    user_profiles = input_df[['user_id']].copy()
    user_profiles['user_id'] = pd.to_numeric(user_profiles['user_id'], errors='coerce').astype('Int64')

    print(f"Columns after filtering: {user_profiles.columns.tolist()}")

    # Remove duplicate user_id to get unique users
    print("Removing duplicate user_ids...")
    user_profiles = user_profiles.drop_duplicates()
    print(f"Number of unique users: {user_profiles.shape[0]}")

    # 4. Thêm mới 1 cột có tên là username: Concate 2 chuỗi "'junyi" + giá trị user_id
    print("\n4. Adding username column...")
    user_profiles['username'] = 'junyi' + user_profiles['user_id'].astype(str)
    print("Sample usernames:")
    print(user_profiles['username'].head())

    user_profiles['password'] = user_profiles['user_id'].apply(hash_password)
    print("Sample hashed passwords:")
    print(user_profiles['password'].head())

    print("\n5-6. Adding created_time and updated_time columns...")
    current_time = datetime.now()
    user_profiles['updated_time'] = current_time
    user_profiles['created_time_original'] = current_time
    print(f"Timestamp set for both created_time_original and updated_time: {current_time}")

    # 7. Thêm cột preferred_content_type - MODIFIED to store as array of objects
    print("\n7. Adding preferred_content_type column...")
    # Define as a Python list of dictionaries (will remain as objects in JSON)
    preferred_content_type = [
        {
            "order": 1,  # Removed quotes to keep as number
            "option": "Tests/Quizzes"
        }
    ]
    # Store directly as the list object, not as a JSON string
    user_profiles['preferred_content_type'] = [preferred_content_type] * len(user_profiles)

    # 8. Thêm cột preferred_learn_styles - MODIFIED to store as array of objects
    print("\n8. Adding preferred_learn_styles column...")
    # Define as a Python list of dictionaries
    preferred_learn_styles = [
        {
            "order": 1,  # Removed quotes to keep as number
            "option": "Learn through videos and images"
        }
    ]
    # Store directly as the list object, not as a JSON string
    user_profiles['preferred_learn_styles'] = [preferred_learn_styles] * len(user_profiles)

    # 9. Thêm cột education_lv - MODIFIED to store as array of objects
    print("\n9. Adding education_lv column...")
    # Define as a Python list of dictionaries
    education_lv = [
        {
            "order": 1,  # Removed quotes to keep as number
            "option": "None of the above"
        }
    ]
    # Store directly as the list object, not as a JSON string
    user_profiles['education_lv'] = [education_lv] * len(user_profiles)

    # 10. Thêm cột preferred_areas - MODIFIED to store as array of objects
    print("\n10. Adding preferred_areas column...")
    # Define as a Python list of dictionaries
    preferred_areas = [
        {
            "order": 1,  # Removed quotes to keep as number
            "option": "STEM"
        },
        {
            "order": 2,  # Removed quotes to keep as number
            "option": "Technology and Computer Science"
        },
        {
            "order": 3,  # Removed quotes to keep as number
            "option": "Business and Finance"
        }
    ]
    # Store directly as the list object, not as a JSON string
    user_profiles['preferred_areas'] = [preferred_areas] * len(user_profiles)

    # 11. Thêm 6 cột theo yêu cầu
    print("\n11. Adding 6 additional columns...")
    user_profiles['source_name'] = 'junyi-batch-data'
    user_profiles['source_id'] = 2
    user_profiles['is_update'] = 0
    user_profiles['is_delete'] = 0
    user_profiles['created_time'] = current_time  # Sử dụng lại timestamp đã tạo ở trên
    user_profiles['created_date'] = current_time.date()

    # 12. Đảm bảo không có thông tin user trùng nhau
    print("\n12. Checking for duplicate user information...")
    # Đã xử lý ở bước 3 khi loại bỏ duplicate user_id
    # Kiểm tra thêm lần nữa để đảm bảo username và password không trùng nhau
    duplicate_username = user_profiles.duplicated(subset=['username']).sum()
    duplicate_password = user_profiles.duplicated(subset=['password']).sum()

    print(f"Number of duplicate usernames: {duplicate_username}")
    print(f"Number of duplicate passwords: {duplicate_password}")

    if duplicate_username > 0 or duplicate_password > 0:
        print("Removing duplicates...")
        user_profiles = user_profiles.drop_duplicates(subset=['user_id', 'username', 'password'])

    # 13. Hiển thị thông tin dữ liệu đã được pre-process
    print("\n13. Displaying processed data information:")
    print(f"Final number of users: {user_profiles.shape[0]}")
    print(f"Columns in processed data: {user_profiles.columns.tolist()}")
    print("\nSample processed data (first 5 rows):")
    print(user_profiles.head())
    print("\nData types:")
    print(user_profiles.dtypes)

    # 14. Lưu dữ liệu đã được xử lý vào file JSON
    print("\n14. Saving processed data to user_profiles.json...")
 
    return user_profiles

def write_user(input_df, destination_file_path):
    # LOAD
    general_lib.write_dls(input_df, "json", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
                DESTINATION_CONTAINER, 
                destination_file_path
            )

def main():
    print("Preprocessing Users")
    df = load_user()
    nan_count = df['user_id'].isna().sum()
    print("Number of NaNs in user_id:", nan_count)

    df = user_transform(df)
    JOB_DATE = datetime.now().strftime('%Y%m%d')
    file_path = f"{DESTINATION_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_users_process.json"
    write_user(df,file_path )
  

if __name__ == '__main__':
    main()