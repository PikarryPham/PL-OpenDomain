import argparse
import os
import sys
import logging
import subprocess
import tempfile
from datetime import datetime
import general_lib

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

ACCOUNT_NAME = "13f45cadls"
AZURE_DATALAKE_STORAGE_KEY = os.environ.get("AZURE_DATALAKE_STORAGE_KEY", "")
DESTINATION_CONTAINER = "03gold"
DESTINATION_BASE_PATH = "ETKG/json"

def load_learning_history(job_date: str):
    rel_path = f"{DESTINATION_BASE_PATH}/{job_date}/{job_date}_integrated_learning_history_data.json"
    logger.info(f"Loading integrated learning history from {rel_path} ...")
    df = general_lib.read_azure_datalake_storage(
        DESTINATION_CONTAINER,
        rel_path,
        ACCOUNT_NAME,
        AZURE_DATALAKE_STORAGE_KEY
    )
    logger.info(f"Đã load {len(df):,} bản ghi")
    return df

# --- Hàm gọi Docker để build KG ---
def run_kg_builder(input_json_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    """
    Chạy Docker container chứa code build ETKG lên Neo4j.
    """
    docker_image = "neo4j:latest"     
    docker_cmd = [
        "docker", "run", "--rm",
        "-e", f"NEO4J_URI={neo4j_uri}",
        "-e", f"NEO4J_USER={neo4j_user}",
        "-e", f"NEO4J_PASSWORD={neo4j_password}",
        "-v", f"{os.path.abspath(input_json_path)}:/app/input.json:ro",
        docker_image,
        "python", "/app/main.py",  #
        "--input", "/app/input.json"
    ]
    logger.info("Starting Docker container to build ETKG…")
    subprocess.run(docker_cmd, check=True)
    logger.info("ETKG build completed.")

def main():
    parser = argparse.ArgumentParser(description="ETL + ETKG Builder for Integrated Learning History")
    parser.add_argument("--job-date",
                        help="Ngày chạy dưới dạng YYYYMMDD (mặc định: hôm nay)",
                        default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--neo4j-uri",
                        help="Bolt URI cho Neo4j (ví dụ: bolt://localhost:7687)",
                        required=True)
    parser.add_argument("--neo4j-user",
                        help="Username Neo4j",
                        required=True)
    parser.add_argument("--neo4j-password",
                        help="Password Neo4j",
                        required=True)
    args = parser.parse_args()

    
    df_history = load_learning_history(args.job_date)

    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
        df_history.to_json(tmp, orient="records", lines=True)
    logger.info(f"Wrote temp JSON to {tmp_path}")

    
    run_kg_builder(
        input_json_path=tmp_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )

    try:
        os.remove(tmp_path)
        logger.info(f"Removed temp file {tmp_path}")
    except OSError:
        logger.warning(f"Could not remove temp file {tmp_path}")

if __name__ == "__main__":
    main()
