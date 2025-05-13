import general_lib
import argparse
from datetime import datetime

ACCOUNT_NAME = "13f45cadls"
AZURE_DATALAKE_STORAGE_KEY =  "+oB+LkoL2KPaMvZbChL9vKVr/3lFJyDjmHI2cpyFJFDlMFW2pEzPN1zAQbmx9ovFE0hX1vvfll66+ASthCJINQ=="
CLICK_HOUSE_HOST = "h8tw70myst.ap-northeast-1.aws.clickhouse.cloud"
CLICK_HOUSE_USER = "default"
CLICK_HOUSE_PASSWORD = "YAWU5r~485Xr~"
SOURCE_CONTAINER = "00fs"
DESTINATION_CONTAINER = "01landzone"
BASE_PATH = "streaming-sources"


def main(entity_name):
    click_house_client =  general_lib.get_click_house_client( CLICK_HOUSE_HOST, CLICK_HOUSE_USER, CLICK_HOUSE_PASSWORD)

    # full_load_table_list = [
    #     "options",
    #     "questions"
    # ]
    

    df = general_lib.read_click_house(click_house_client, "SELECT * FROM {}".format(entity_name))
    
    # LOAD
    current_date = datetime.now().strftime('%Y%m%d')
    general_lib.write_dls(df, "jsonline", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
                            DESTINATION_CONTAINER, 
                            "{}/{}/{}/{}/{}_{}.{}".format(BASE_PATH, entity_name,"json", current_date, current_date, entity_name, "json")
                        )

    # ARCHIVE
    df = df.applymap(general_lib.convert_uuid)
    general_lib.write_dls(df, "parquet", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
        SOURCE_CONTAINER, 
    "archives/{}/{}/{}/{}/{}_{}.{}".format(BASE_PATH, entity_name,"parquet",current_date, current_date, entity_name, "parquet")
)
        
if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_name', type=str, required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    main(args.entity_name)
        
