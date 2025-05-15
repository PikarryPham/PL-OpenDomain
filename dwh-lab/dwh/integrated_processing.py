#!/usr/bin/env python3
import argparse
import subprocess
import os
import logging
import sys
from datetime import datetime

# --- Setup logger ---
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(
        description="Run integrated learning-history ETL pipeline inside Docker"
    )
    parser.add_argument(
        "--job-date",
        help="Job date in YYYYMMDD (default: today)",
        default=datetime.now().strftime("%Y%m%d")
    )
    parser.add_argument(
        "--account-name",
        help="Azure Data Lake account name",
        default="13f45cadls"
    )
    parser.add_argument(
        "--azure-key",
        help="Azure Data Lake storage key",
        default=os.environ.get("AZURE_DATALAKE_STORAGE_KEY", "")
    )
    parser.add_argument(
        "--source-container",
        help="Source ADLS container",
        default="00_1_land"
    )
    parser.add_argument(
        "--destination-container",
        help="Destination ADLS container",
        default="02silver"
    )
    parser.add_argument(
        "--docker-image",
        help="Docker image that contains the ETL + KG builder",
        default="valkey-db:latest"
    )
    args = parser.parse_args()

    # Các biến môi trường sẽ được container sử dụng
    env = os.environ.copy()
    env.update({
        "ACCOUNT_NAME": args.account_name,
        "AZURE_DATALAKE_STORAGE_KEY": args.azure_key,
        "SOURCE_CONTAINER": args.source_container,
        "DESTINATION_CONTAINER": args.destination_container,
        "JOB_DATE": args.job_date,
        "DESTINATION_BASE_PATH": "integrated_learning_history/json"
    })

    # Build lệnh docker run
    cmd = [
        "docker", "run", "--rm",
        # truyền toàn bộ env vars vào container
    ]
    for k in ("ACCOUNT_NAME",
              "AZURE_DATALAKE_STORAGE_KEY",
              "SOURCE_CONTAINER",
              "DESTINATION_CONTAINER",
              "JOB_DATE",
              "DESTINATION_BASE_PATH"):
        cmd += ["-e", f"{k}={env[k]}"]

    cmd += [args.docker_image]

    logger.info("Running pipeline in Docker:")
    logger.info("  %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, env=env)
        logger.info("Pipeline completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Pipeline failed (exit code %s).", e.returncode)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
    # Chạy hàm main