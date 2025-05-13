import os
import pandas as pd
from io import StringIO
from io import BytesIO
from datetime import datetime
from azure.storage.filedatalake import (
    DataLakeServiceClient,
    DataLakeDirectoryClient,
    FileSystemClient
)
from azure.identity import DefaultAzureCredential
import smtplib
from email.message import EmailMessage
from pathlib import Path
from email.mime.base import MIMEBase
from email import encoders
import clickhouse_connect
import uuid
import argparse

import general_lib

ACCOUNT_NAME = "13f45cadls"
AZURE_DATALAKE_STORAGE_KEY =  "+oB+LkoL2KPaMvZbChL9vKVr/3lFJyDjmHI2cpyFJFDlMFW2pEzPN1zAQbmx9ovFE0hX1vvfll66+ASthCJINQ=="
CLICK_HOUSE_HOST = "h8tw70myst.ap-northeast-1.aws.clickhouse.cloud"
CLICK_HOUSE_USER = "default"
CLICK_HOUSE_PASSWORD = "YAWU5r~485Xr~"

# Source from  
# Get all directory from batch_sources
# Get all entities from each directory --> Full Load
# WRITE CSV, JSON, DELTA to daily path yyyyMMdd_entityname.format
# WRITE TO 01-land-zone/batch_sources/directory
# ARCHIVE 00_FS/batch_sources/archives/junyi
# FAIL --> MOVE TO LOG
# DATA QUALITY CHECK 

SOURCE_CONTAINER = "00fs"
DESTINATION_CONTAINER = "01landzone"
SOURCE_BASE_PATH =  "batch-sources/junyi"

# GET ENTITIES FROM batch_sources

service_client = general_lib.get_azure_service_client_by_account_key(ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
file_system_client = service_client.get_file_system_client(file_system= SOURCE_CONTAINER)

# paths = file_system_client.get_paths(path='/{}'.format(SOURCE_BASE_PATH))
# paths = [path.name for path in paths]


def main(entity_path):
    print("Processing: {}".format(entity_path))

    current_date = datetime.now().strftime('%Y%m%d')
    entity_name = entity_path.split("/")[-1].split(".")[0]
    general_lib.read_chunk_and_writle_dls(SOURCE_CONTAINER, entity_path, DESTINATION_CONTAINER, 
            "{}/{}/{}/{}/{}_{}.{}".format(SOURCE_BASE_PATH, entity_name,"json", current_date, current_date, entity_name, "json"),
            ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, "archives/{}/{}/{}/{}/{}_{}.{}".format(SOURCE_BASE_PATH, entity_name,"parquet",current_date, current_date, entity_name, "parquet"))
    
    # # EXTRACT
    # df = general_lib.read_azure_datalake_storage(SOURCE_CONTAINER, entity_path, ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    
    # # LOAD
    # current_date = datetime.now().strftime('%Y%m%d')
    # entity_name = entity_path.split("/")[-1].split(".")[0]
    # general_lib.write_dls(df, "json", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
    #                         DESTINATION_CONTAINER, 
    #                         "{}/{}/{}/{}/{}_{}.{}".format(SOURCE_BASE_PATH, entity_name,"json", current_date, current_date, entity_name, "json")
    #                     )

    # # ARCHIVE
    # general_lib.write_dls(df, "parquet", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
    #                         SOURCE_CONTAINER, 
    #                         "archives/{}/{}/{}/{}/{}_{}.{}".format(SOURCE_BASE_PATH, entity_name,"parquet",current_date, current_date, entity_name, "parquet")
    #                     )
    
    
        
if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Ingest Batch Sources")
    parser.add_argument('--entity_path', type=str, required=True, help="entity_path to ingest")
    
    # Parse the arguments
    args = parser.parse_args()
    main(args.entity_path)
        
