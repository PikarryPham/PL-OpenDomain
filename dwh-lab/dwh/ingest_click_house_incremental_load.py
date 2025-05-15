import general_lib
import argparse
from datetime import datetime

ACCOUNT_NAME = "13f45cadls"
AZURE_DATALAKE_STORAGE_KEY =  ""
CLICK_HOUSE_HOST = "h8tw70myst.ap-northeast-1.aws.clickhouse.cloud"
CLICK_HOUSE_USER = "default"
CLICK_HOUSE_PASSWORD = ""
SOURCE_CONTAINER = "00fs"
DESTINATION_CONTAINER = "01landzone"
BASE_PATH = "streaming-sources"



WATERMARK_CONTAINER =  "00fs"
WATERMARK_PATH =  "watermark_table.csv"
BASE_PATH = "streaming-sources"

def main(entity_name):
    click_house_client =  general_lib.get_click_house_client( CLICK_HOUSE_HOST, CLICK_HOUSE_USER, CLICK_HOUSE_PASSWORD)
    # READ WATERMARK
    watermark_df = general_lib.read_azure_datalake_storage(WATERMARK_CONTAINER, WATERMARK_PATH, ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY)
    df_filtered = watermark_df[watermark_df['table_name'] == entity_name]
    watermark_value = df_filtered['watermark_value'].iloc[0] if not df_filtered.empty else None
    update_column = df_filtered['update_column'].iloc[0] if not df_filtered.empty else None
    
    # EXTRACT
    df = general_lib.read_click_house(
        click_house_client, "SELECT * FROM {} where {} > '{}'".format(entity_name, update_column, watermark_value))

    # UPDATE WATERMARK VALUE
    if not df.empty:
        new_watermark_value = df[update_column].max()
        watermark_df.loc[watermark_df['table_name'] == entity_name, 'watermark_value'] = new_watermark_value
        general_lib.write_dls(watermark_df, "csv", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
                            WATERMARK_CONTAINER, 
                            WATERMARK_PATH)
        print(f"Watermark updated to {new_watermark_value}")
    else:
        print("No data extracted, watermark not updated.")
        
    
    # LOAD
    current_date = datetime.now().strftime('%Y%m%d')
    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    
    general_lib.write_dls(df, "json", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
                    DESTINATION_CONTAINER, 
                    "{}/{}/{}/{}/{}_{}.{}".format(BASE_PATH, entity_name,"json", current_date, current_datetime, entity_name, "json")
                )

    # ARCHIVE
    df = df.applymap(general_lib.convert_uuid)
    general_lib.write_dls(df, "parquet", ACCOUNT_NAME, AZURE_DATALAKE_STORAGE_KEY, 
        SOURCE_CONTAINER, 
    "archives/{}/{}/{}/{}/{}_{}.{}".format(BASE_PATH, entity_name,"parquet", current_date, current_datetime, entity_name, "parquet")
   )
    
        
if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_name', type=str, required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    main(args.entity_name)
