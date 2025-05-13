#!/bin/bash

# Check if the script is run as root
echo "Kiểm tra cài đặt Docker và Docker Compose..."
if ! command -v docker &> /dev/null; then
    echo "Docker chưa được cài đặt. Vui lòng cài đặt Docker trước khi tiếp tục."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose chưa được cài đặt. Vui lòng cài đặt Docker Compose trước khi tiếp tục."
    exit 1
fi

# Create a directory for the upload files
mkdir -p upload

# Copy the data files to the upload directory
echo "Sao chép các file dữ liệu vào thư mục upload..."
cp /home/ubuntu/upload/*.json upload/

# Copy the Docker Compose file to the current directory
echo "Xây dựng và khởi chạy các container..."
docker-compose up --build -d

# Check if the containers are running
echo "Kiểm tra trạng thái của các container..."
docker-compose ps

echo "Đang theo dõi log của ứng dụng..."
docker-compose logs -f app
