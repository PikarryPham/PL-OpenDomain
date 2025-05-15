#!/bin/bash

# Script để cài đặt các mô hình cần thiết cho chức năng AI Summarization

# Cài đặt mô hình spaCy cho tiếng Anh
echo "Cài đặt mô hình spaCy cho tiếng Anh..."
python -m spacy download en_core_web_sm

# Tạo thư mục lưu trữ model transformer
echo "Tạo thư mục lưu trữ model transformer..."
mkdir -p /usr/src/app/models/summarization

# Tải trước model summarization (nếu cần)
echo "Bạn có muốn tải trước model summarization không? (y/n)"
read tải_model

if [ "$tải_model" = "y" ]; then
    echo "Đang tải model summarization..."
    python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6'); model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')"
    echo "Đã tải xong model summarization."
else
    echo "Bỏ qua tải model summarization."
fi

# Khởi động API server
echo "Bạn có muốn khởi động API server không? (y/n)"
read khởi_động

if [ "$khởi_động" = "y" ]; then
    echo "Khởi động API server..."
    cd /usr/src/app/src && python app.py
else
    echo "Đã cài đặt xong các model cần thiết."
fi 