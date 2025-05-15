import general_lib
import argparse
from datetime import datetime
import pandas as pd


ACCOUNT_NAME = "13f45cadls"
AZURE_DATALAKE_STORAGE_KEY =  ""
CLICK_HOUSE_HOST = "h8tw70myst.ap-northeast-1.aws.clickhouse.cloud"
CLICK_HOUSE_USER = "default"
CLICK_HOUSE_PASSWORD = ""
SOURCE_CONTAINER = "00_1_land"
DESTINATION_CONTAINER = "02bronze"
BASE_PATH = "streaming-sources"

def add_additional_columns(df: pd.DataFrame) -> pd.DataFrame:
    current_time = datetime.now()
    df = df.copy()  # avoid modifying the original DataFrame

    df['source_name'] = 'clickhouse-streaming-data'
    df['source_id'] = 1
    df['is_update'] = False
    df['is_delete'] = False
    df['created_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
    df['created_date'] = current_time.date().strftime('%Y-%m-%d')

    return df


def main(entity_name):
    
    current_date = datetime.now().strftime('%Y%m%d')
    # READ DATA FROM
    entity_path = f"{BASE_PATH}/{entity_name}/json/{current_date}/{current_date}_{entity_name}.json"
    df = general_lib.read_azure_datalake_storage(SOURCE_CONTAINER, entity_path, ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    df = add_additional_columns(df)

    # LOAD
    general_lib.write_dls(df, "json", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
                            DESTINATION_CONTAINER, 
                            "{}/{}/{}/{}/{}_{}.{}".format(BASE_PATH, entity_name,"json", current_date, current_date, entity_name, "json")
                        )

   
        
if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_name', type=str, required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    main(args.entity_name)
        
