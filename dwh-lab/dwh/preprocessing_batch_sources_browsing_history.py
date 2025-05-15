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

EXERCISE_SOURCE_FILE_BASE_PATH = "batch-sources/junyi/junyi_Exercise_table_trans/json/"
PROBLEM_SOURCE_FILE_BASE_PATH = "batch-sources/junyi/junyi_ProblemLog_original/json/"
DESTINATION_BASE_PATH = "batch-sources/junyi/preprocessing_batch_sources_browsing_history/json"
JOB_DATE = "20250502"


# new feature here
TEMP_SORTED_FILES = []
def process_and_save_sorted_chunk(df):
    # json_str = '\n'.join(lines)
    # df = pd.read_json(StringIO(json_str), lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values(by='timestamp')

    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json")
    df_sorted.to_json(temp_file.name, orient='records', lines=True, force_ascii=False, date_format='iso')
    TEMP_SORTED_FILES.append(temp_file)

def merge_and_upload_to_azure(dest_file_client):
    file_handles = [open(f.name, 'r', encoding='utf-8') for f in TEMP_SORTED_FILES]
    def gen(file):
        for line in file:
            obj = pd.read_json(StringIO(line), lines=True).iloc[0]
            obj['timestamp'] = pd.to_datetime(obj['timestamp'])
            yield (obj['timestamp'], line)
    merged = heapq.merge(*[gen(f) for f in file_handles])
    offset = 0
    for _, line in merged:
        data_bytes = line.encode("utf-8")
        dest_file_client.append_data(data=data_bytes, offset=offset, length=len(data_bytes))
        offset += len(data_bytes)
    dest_file_client.flush_data(offset)

    for f in file_handles:
        f.close()

    for f in TEMP_SORTED_FILES:
        os.remove(f.name)

def ETL_problem_log_without_chunks(exercise_table_df):
    logger.info(f"PROCESSING ELT_PROBLEM_LOG...")
    JOB_DATE = datetime.now().strftime('%Y%m%d')
    file_path = f"{PROBLEM_SOURCE_FILE_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_ProblemLog_original.json"
    # extract problem log
    service_client = general_lib.get_azure_service_client_by_account_key(ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    source_file_system_client = service_client.get_file_system_client(SOURCE_CONTAINER)
    source_file_client = source_file_system_client.get_file_client(file_path)
    dest_file_system_client = service_client.get_file_system_client(DESTINATION_CONTAINER)
    destination_file_path = f"{DESTINATION_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_browsing_process.json"
    dest_file_client = dest_file_system_client.get_file_client(destination_file_path)
    dest_file_client.create_file() 
    
    df = general_lib.read_azure_datalake_storage(SOURCE_CONTAINER, file_path, ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    logger.info(f"Loaded {len(df)} row of problem log")
    # 👉 Your processing logic here
    transform_df = transform_problem_log(df, exercise_table_df)
    logger.info(f"rc after transform {len(transform_df)} row of problem log")

    # 👉 Stage 3 here

    # Then convert timestamp to datetime
    transform_df['timestamp'] = pd.to_datetime(transform_df['timestamp'])

    seen_combinations = set()
    duplicate_columns = [
        'user_id', 'pageview_count', 'referrer_page', 'search_keyword', 
        'timestamp', 'title', 'url', 'tmp_keywords', 'visible_content'
    ]
    dedup_rows = []
    for _, row in transform_df.iterrows():
        key_tuple = tuple(str(row[col]) for col in duplicate_columns)
        if key_tuple not in seen_combinations:
            seen_combinations.add(key_tuple)
            dedup_rows.append(row)

    if dedup_rows:
        df_to_write = pd.DataFrame(dedup_rows)
        logger.info(f"rc after dedup {len(df_to_write)} row")

    # print("before return ETL_problem_log_without_chunks")
    # visible_content_counts = df_to_write['visible_content'].value_counts()
    # pprint(visible_content_counts.to_dict())  


    return df_to_write


def ETL_problem_log(exercise_table_df):
    logger.info(f"PROCESSING ELT_PROBLEM_LOG...")
    file_path = f"{PROBLEM_SOURCE_FILE_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_ProblemLog_original.json"

    # extract problem log
    service_client = general_lib.get_azure_service_client_by_account_key(ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    source_file_system_client = service_client.get_file_system_client(SOURCE_CONTAINER)
    source_file_client = source_file_system_client.get_file_client(file_path)

    dest_file_system_client = service_client.get_file_system_client(DESTINATION_CONTAINER)
    
    destination_file_path = f"{DESTINATION_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_browsing_process.json"
    dest_file_client = dest_file_system_client.get_file_client(destination_file_path)
    dest_file_client.create_file() 

    download = source_file_client.download_file(timeout=300)
    chunks = download.chunks()

    index = 0
    buffer = ""
    line_buffer = []
    offset = 0
    CHUNK_SIZE_LINES = 10000 

    # 🔧 Init deduplication logic
    seen_combinations = set()
    duplicate_columns = [
        'user_id', 'pageview_count', 'referrer_page', 'search_keyword', 
        'timestamp', 'title', 'url', 'tmp_keywords', 'visible_content'
    ]


    for chunk in chunks:
        text_chunk = chunk.decode('utf-8')
        buffer += text_chunk
        lines = buffer.split('\n')
        buffer = lines.pop()  # last line might be incomplete
        line_buffer.extend(lines)
        while len(line_buffer) >= CHUNK_SIZE_LINES:

            chunk_lines = line_buffer[:CHUNK_SIZE_LINES]
            line_buffer = line_buffer[CHUNK_SIZE_LINES:]
            json_str = '\n'.join(chunk_lines)
            chunk_df = pd.read_json(StringIO(json_str), lines=True)

            # 👉 Your processing logic here
            chunk_transform_df = transform_problem_log(chunk_df, exercise_table_df)

            # 👉 Stage 3 here
            process_and_save_sorted_chunk(chunk_transform_df)

            # 🔧 Deduplicate logic
            dedup_rows = []
            for _, row in chunk_transform_df.iterrows():
                key_tuple = tuple(str(row[col]) for col in duplicate_columns)
                if key_tuple not in seen_combinations:
                    seen_combinations.add(key_tuple)
                    dedup_rows.append(row)

            if not dedup_rows:
                logger.info("⚠️ No unique rows left after deduplication, skipping write.")
                continue



             # WRITE TO DLS json
            json_buffer = StringIO()
            chunk_transform_df.to_json(json_buffer, orient="records", lines=True, force_ascii=False, default_handler=str)
            json_data = json_buffer.getvalue()
            # Encode to bytes for correct length count
            data_bytes = json_data.encode("utf-8")
            dest_file_client.append_data(data=data_bytes, offset=offset, length=len(data_bytes))
            offset += len(data_bytes)
            dest_file_client.flush_data(offset)

            logger.info(f"✅ Processed chunk of {len(chunk_df)} rows")


        index += 1
        if index == 1:
            break

    # Process remaining lines
    if line_buffer:
        

        json_str = '\n'.join(line_buffer)
        chunk_df = pd.read_json(StringIO(json_str), lines=True)
        chunk_transform_df = transform_problem_log(chunk_df, exercise_table_df)

        # new code here
        process_and_save_sorted_chunk(chunk_transform_df)
        # new code here

        # 🔧 Deduplicate final chunk
        dedup_rows = []
        for _, row in chunk_transform_df.iterrows():
            key_tuple = tuple(str(row[col]) for col in duplicate_columns)
            if key_tuple not in seen_combinations:
                seen_combinations.add(key_tuple)
                dedup_rows.append(row)

        if dedup_rows:
            df_to_write = pd.DataFrame(dedup_rows)
            json_buffer = StringIO()
            df_to_write.to_json(json_buffer, orient="records", lines=True, force_ascii=False, default_handler=str)
            json_data = json_buffer.getvalue()
            data_bytes = json_data.encode("utf-8")
            dest_file_client.append_data(data=data_bytes, offset=offset, length=len(data_bytes))
            offset += len(data_bytes)
            dest_file_client.flush_data(offset)

            logger.info(f"✅ Processed final chunk of {len(df_to_write)} unique rows")
        else:
            logger.info("⚠️ Final chunk contains only duplicates. Skipping write.")
    
    logger.info("Merging sorted data")
    merged_destination_file_path = f"{DESTINATION_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_browsing_process_merged.json"
    merged_dest_file_client = dest_file_system_client.get_file_client(merged_destination_file_path)
    merged_dest_file_client.create_file() 
    merge_and_upload_to_azure(merged_dest_file_client)

def read_full_exercise():
    logger.info(f"LOADING FULL EXERCISE DATA...")
    JOB_DATE = datetime.now().strftime('%Y%m%d')
    file_path = f"{EXERCISE_SOURCE_FILE_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_Exercise_table_trans.json"
    df = general_lib.read_azure_datalake_storage(SOURCE_CONTAINER, file_path, ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    logger.info(f"✅ Total Loaded: {len(df)} rows")
    print(df)
    return df

exercise_table_df = read_full_exercise()
exercise_info = {}
for _, row in exercise_table_df.iterrows():
    name = row['name']
    topic = row['topic'] if pd.notna(row['topic']) and row['topic'] != '' else None
    area = row['area'] if pd.notna(row['area']) and row['area'] != '' else None
    prerequisites = row['prerequisites'] if pd.notna(row['prerequisites']) and row['prerequisites'] != '' else None
    
    # Lưu thông tin vào dictionary
    exercise_info[name] = {
        'topic': topic,
        'area': area,
        'prerequisites': prerequisites
    }

# Predefine a reusable clean-up function
def clean_text(text):
    if text is None:
        return ""
    return text.replace('_', ' ').replace('-', ' ').replace('@', ' ')

# Optimized tmp_keywords generator
def create_tmp_keywords(title):
    info = exercise_info.get(title, {})
    return list(filter(None, [
        clean_text(title),
        clean_text(info['topic']) if 'topic' in info else None,
        clean_text(info['area']) if 'area' in info else None,
        clean_text(info['prerequisites']) if 'prerequisites' in info else None
    ]))


def transform_problem_log(input_df, exercise_table_df):
    history_learning_data = input_df
    history_learning_data['user_id'] = history_learning_data['user_id'].astype(str)
    history_learning_data['entry_id'] = [str(uuid.uuid4()) for _ in range(len(history_learning_data))]
    history_learning_data.rename(columns={'problem_number': 'pageview_count'}, inplace=True)
    history_learning_data['pageview_count'] = history_learning_data['pageview_count'].astype('Int64')

    merged = history_learning_data.merge(
        exercise_table_df[['name', 'prerequisites']],
        left_on='exercise',
        right_on='name',
        how='left'
    )
    merged['referrer_page'] = merged['prerequisites'].where(
        merged['prerequisites'].notna() & (merged['prerequisites'] != ''),
        ''
    ).apply(lambda x: f"https://www.junyiacademy.org/exercise/{x}" if x else '')
    history_learning_data['referrer_page'] = merged['referrer_page']
    history_learning_data['search_keyword'] = [[] for _ in range(len(history_learning_data))]

    if 'creation_date' in history_learning_data.columns:
        history_learning_data.rename(columns={'creation_date': 'timestamp'}, inplace=True)
        # Cố gắng chuyển đổi timestamp sang datetime nếu có thể
        try:
            history_learning_data['timestamp'] = pd.to_datetime(history_learning_data['timestamp'])
        except:
            print("Warning: Could not convert timestamp to datetime format")
    elif 'time_done' in history_learning_data.columns:
        # Nếu không có creation_date nhưng có time_done, sử dụng time_done
        history_learning_data.rename(columns={'time_done': 'timestamp'}, inplace=True)
        # Chuyển đổi unix timestamp sang datetime
        history_learning_data['timestamp'] = pd.to_datetime(history_learning_data['timestamp'], unit='us')

    print(history_learning_data['exercise'].unique())
    history_learning_data.rename(columns={'exercise': 'title'}, inplace=True)

    history_learning_data['title'] = history_learning_data['title'].astype(str)
    history_learning_data['url'] = "https://www.junyiacademy.org/exercise/" + history_learning_data['title']
    history_learning_data['tmp_keywords'] = history_learning_data['title'].map(create_tmp_keywords)

    exercise_table_df['name'] = exercise_table_df['name'].astype(str)
    history_learning_data['title'] = history_learning_data['title'].astype(str)
    def clean_text(s):
        return re.sub(r'\s+', '', str(s)).strip()

    exercise_table_df['name'] = exercise_table_df['name'].map(clean_text)
    history_learning_data['title'] = history_learning_data['title'].map(clean_text)

   

    exercise_to_display = dict(zip(exercise_table_df['name'], exercise_table_df['short_display_name_english']))
    pprint(exercise_to_display)
    print("test ========= ",  exercise_to_display.get("division_0.9"))
    def get_visible_content(title):
        if exercise_to_display.get(title, '') == '':
            print(f"[Missing] '{title}'") 
        return exercise_to_display.get(title, '')
    title_counts = history_learning_data['title'].value_counts()
    pprint(title_counts.to_dict())  

    history_learning_data['visible_content'] = history_learning_data['title'].apply(get_visible_content)

    # visible_content_counts = history_learning_data['visible_content'].value_counts()
    # pprint(visible_content_counts.to_dict())  

    current_time = datetime.now()
    current_date = current_time.date()
    history_learning_data['source_name'] = 'junyi-batch-data'
    history_learning_data['source_id'] = 2
    history_learning_data['is_update'] = 0
    history_learning_data['is_delete'] = 0
    history_learning_data['created_time'] = current_time
    history_learning_data['created_date'] = current_date
    columns_to_keep = [
        'user_id', 'entry_id', 'pageview_count', 'referrer_page', 'search_keyword',
        'timestamp', 'title', 'url', 'tmp_keywords', 'visible_content',
        'source_name', 'source_id', 'is_update', 'is_delete', 'created_time', 'created_date'
    ]

    # Lọc ra những cột cần giữ lại
    final_columns = [col for col in columns_to_keep if col in history_learning_data.columns]
    history_learning_data = history_learning_data[final_columns]


    # print("============ before return")
    # visible_content_counts = history_learning_data['visible_content'].value_counts()
    # pprint(visible_content_counts.to_dict())  

    return history_learning_data

def parse_timestamp(timestamp_str):
    """Parse timestamp strings in various formats to datetime objects"""
    if pd.isna(timestamp_str):
        return None
        
    # If the timestamp is already a datetime object, return it as is
    if isinstance(timestamp_str, datetime):
        return timestamp_str

    timestamp_formats = [
        '%Y-%m-%dT%H:%M:%S',  # ISO format
        '%Y-%m-%d %H:%M:%S',  # Standard format
        '%Y-%m-%d %H:%M:%S.%f'  # Format with microseconds
    ]
    
    # Handle nanoseconds by truncating them to microseconds
    if isinstance(timestamp_str, str) and len(timestamp_str.split('.')[-1]) > 6:
        timestamp_str = timestamp_str[:timestamp_str.find('.') + 7]
        
    # Try parsing with the defined formats
    for fmt in timestamp_formats:
        try:
            return datetime.strptime(str(timestamp_str), fmt)
        except ValueError:
            continue
    
    # If all formats fail, return None instead of raising an exception
    print(f"Warning: Unable to parse timestamp: {timestamp_str}")
    return None

# Helper function to process time window fields
def process_timestamp(dt):
    """Calculate window_time fields based on timestamp hour"""
    if dt is None:
        return {
            'window_time_details': 0,
            'window_time_details_meaning': "0ham",
            'window_time_overall': 0,
            'window_time_overall_meaning': "(0h - 6h) am"
        }
        
    hour = dt.hour
    
    # Calculate window_time_details and window_time_details_meaning
    window_time_details = hour
    am_pm = "am" if hour < 12 else "pm"
    window_time_details_meaning = f"{hour}h{am_pm}"
    
    # Calculate window_time_overall and window_time_overall_meaning
    if 0 <= hour < 6:
        window_time_overall = 0
        window_time_overall_meaning = "(0h - 6h) am"
    elif 6 <= hour < 12:
        window_time_overall = 6
        window_time_overall_meaning = "(6h - 12h) am"
    elif 12 <= hour < 18:
        window_time_overall = 12
        window_time_overall_meaning = "(12h - 18h) pm"
    else:  # 18 <= hour < 24
        window_time_overall = 18
        window_time_overall_meaning = "(18h - 24h) pm"
        
    return {
        'window_time_details': window_time_details,
        'window_time_details_meaning': window_time_details_meaning,
        'window_time_overall': window_time_overall,
        'window_time_overall_meaning': window_time_overall_meaning
    }

# Helper function to process keywords
def process_keywords(tmp_keywords, visible_content):
    """Extract keywords from tmp_keywords and visible_content"""
    keywords = []
    
    # Process tmp_keywords if available
    if isinstance(tmp_keywords, str) and tmp_keywords.strip():
        try:
            # Handle different formats of tmp_keywords
            if isinstance(tmp_keywords, str):
                if tmp_keywords.startswith('[') and tmp_keywords.endswith(']'):
                    # Try to parse as JSON list
                    try:
                        tmp_keywords_list = json.loads(tmp_keywords.replace("'", '"'))
                        if isinstance(tmp_keywords_list, list):
                            keywords.extend(tmp_keywords_list)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try eval (safer alternative would be ast.literal_eval)
                        try:
                            tmp_keywords_list = eval(tmp_keywords)
                            if isinstance(tmp_keywords_list, list):
                                keywords.extend(tmp_keywords_list)
                        except:
                            # Fall back to treating as comma-separated string
                            tmp_keywords_list = [k.strip() for k in tmp_keywords.split(',')]
                            keywords.extend(tmp_keywords_list)
                else:
                    # Treat as comma-separated string
                    tmp_keywords_list = [k.strip() for k in tmp_keywords.split(',')]
                    keywords.extend(tmp_keywords_list)
        except Exception as e:
            pass
    
    # Process visible_content if available
    if pd.notna(visible_content):
        try:
            # Split content by whitespace and extract meaningful words
            content = visible_content.lower()
            
            # Extract individual words (longer than 2 characters)
            words = [w for w in re.findall(r'\b\w+\b', content) if len(w) > 2]
            keywords.extend(words)
            
            # Extract phrases (2-3 consecutive words) - Limited to save memory
            word_list = content.split()[:100]  # Limit to first 100 words to save memory
            
            # Pairs of words (limit to first 20 pairs to save memory)
            for i in range(min(20, len(word_list) - 1)):
                if len(word_list[i]) > 2 and len(word_list[i+1]) > 2:
                    keywords.append(f"{word_list[i]} {word_list[i+1]}")
            
            # Triplets of words (limit to first 10 triplets to save memory)
            for i in range(min(10, len(word_list) - 2)):
                if len(word_list[i]) > 2 and len(word_list[i+1]) > 2 and len(word_list[i+2]) > 2:
                    keywords.append(f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}")
        except Exception:
            pass
    
    # Remove duplicates and return unique keywords (limit to 50 keywords to save space)
    return list(set(keywords))[:20]

def load_browsing_history(input_df, destination_file_path):
    # LOAD
    general_lib.write_dls(input_df, "json", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
                DESTINATION_CONTAINER, 
                destination_file_path
            )


def ETL_problem_log_without_chunks_stage3(input_df):
    # Define constants
    SESSION_TIMEOUT = 30 * 60  # 30 minutes in seconds
    MAX_TIME_PER_PAGE = 60 * 60  # 60 minutes (3600 seconds)
    DEFAULT_LAST_PAGE_TIME = 30 * 60  # 30 minutes (1800 seconds) for last page
    print("Starting large-scale data processing...")
    columns_to_keep = ['user_id', 'entry_id', 'timestamp', 'timestamp_dt', 'pageview_count', 
                   'referrer_page', 'search_keyword', 'title', 'url']
    
    


    input_df['timestamp_dt'] = input_df['timestamp'].apply(parse_timestamp)
    input_df["user_id"] = (
        pd.to_numeric(input_df["user_id"], errors="coerce") 
        .fillna(0)
        .astype(int)
    )


    # Filter out rows with invalid timestamps
    input_df = input_df[input_df['timestamp_dt'].notna()]

    # Convert timestamp_dt to string for storage
    input_df['timestamp_dt'] = input_df['timestamp_dt'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    input_df['timestamp_dt'] = pd.to_datetime(input_df['timestamp_dt'])

    # print("stage 3 input_df")
    # visible_content_counts = input_df['visible_content'].value_counts()
    # pprint(visible_content_counts.to_dict())  



    original_df = input_df.copy()


    # Sort the data
    print("Sorting data...")
    sorted_df = input_df.sort_values('timestamp_dt')
    sorted_df['timestamp_dt'] = pd.to_datetime(sorted_df['timestamp_dt'])

    original_df.set_index('entry_id', inplace=True)
    # Track session and previous timestamp
    current_session = 1
    previous_timestamp = None
    records_processed = 0
    result_records = []

    # print("stage 3 original_df")
    # visible_content_counts = original_df['visible_content'].value_counts()
    # pprint(visible_content_counts.to_dict())  

    # print("stage 3 sorted_df")
    # visible_content_counts = sorted_df['visible_content'].value_counts()
    # pprint(visible_content_counts.to_dict())  

    for i, row in sorted_df.iterrows():
        records_processed += 1

        # Session logic
        if previous_timestamp is not None:
            time_diff = (row['timestamp_dt'] - previous_timestamp).total_seconds()
            if time_diff > SESSION_TIMEOUT:
                current_session += 1

        # Determine next timestamp
        if i == len(sorted_df) - 1:
            next_timestamp = row['timestamp_dt'] + timedelta(seconds=DEFAULT_LAST_PAGE_TIME)
        else:
            next_timestamp = sorted_df.iloc[i + 1]['timestamp_dt']

        # Time calculations
        raw_time_on_page = (next_timestamp - row['timestamp_dt']).total_seconds()
        capped_time_on_page = min(raw_time_on_page, MAX_TIME_PER_PAGE)

        # Time window processing
        time_windows = process_timestamp(row['timestamp_dt'])

        # Lookup original data
        try:
            original_data = original_df.loc[row['entry_id']]
        except KeyError:
            continue  # Skip if entry_id not found

        visible_content = original_data.get('visible_content', '')
        tmp_keywords = original_data.get('tmp_keywords', '')
        exact_keywords = process_keywords(tmp_keywords, visible_content)

        # Construct record
        record = {
            'entry_id': row['entry_id'],
            'exit_page': '',
            'pageview_count': int(row['pageview_count']),
            'referrer_page': row['referrer_page'],
            'search_keyword': row['search_keyword'] if pd.notna(row['search_keyword']) else '',
            'timestamp': row['timestamp_dt'].strftime('%Y-%m-%dT%H:%M:%S'),
            'title': row['title'] if pd.notna(row['title']) else '',
            'tmp_keywords':tmp_keywords if isinstance(tmp_keywords, (list,str)) and tmp_keywords else [],
            'url': row['url'] if pd.notna(row['url']) else '',
            'user_id': row['user_id'],
            'visible_content': visible_content if pd.notna(visible_content) else '',
            'window_time_details': time_windows['window_time_details'],
            'window_time_details_meaning': time_windows['window_time_details_meaning'],
            'window_time_overall': time_windows['window_time_overall'],
            'window_time_overall_meaning': time_windows['window_time_overall_meaning'],
            'exact_keywords': exact_keywords if isinstance(exact_keywords, (list,str)) and exact_keywords else [],
            'session_id': current_session,
            'raw_time_on_page': float(raw_time_on_page),
            'capped_time_on_page': float(capped_time_on_page),
            'source_name': original_data.get('source_name', ''),
            'source_id': int(original_data.get('source_id', 0)),
            'is_update': bool(original_data.get('is_update', False)),
            'is_delete': bool(original_data.get('is_delete', False)),
            'created_time': original_data.get('created_time', ''),
            'created_date': original_data.get('created_date', '')
        }
        result_records.append(record)

        # Update previous timestamp
        previous_timestamp = row['timestamp_dt']

    result_df = pd.DataFrame(result_records)
    print(f"\nData processing complete.")

    return result_df




def main():
    print("stage 1")
    # ETL_problem_log(exercise_table_df)
    rs = ETL_problem_log_without_chunks(exercise_table_df)
    rs = ETL_problem_log_without_chunks_stage3(rs)
    JOB_DATE = datetime.now().strftime('%Y%m%d')
    file_path = f"{DESTINATION_BASE_PATH}/{JOB_DATE}/{JOB_DATE}_junyi_browsing_process.json"
    load_browsing_history(rs,file_path )

if __name__ == '__main__':
    main()