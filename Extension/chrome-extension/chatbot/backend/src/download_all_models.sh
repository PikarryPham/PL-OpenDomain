#!/bin/bash

# Script tải tất cả các mô hình embedding cần thiết với force_download
# Sử dụng: ./download_all_models.sh [host] [port]

HOST=${1:-localhost}
PORT=${2:-8000}
API_URL="http://$HOST:$PORT"

echo "Bắt đầu tải tất cả các mô hình embedding..."

# Kiểm tra kết nối API
echo "Kiểm tra kết nối API..."
if ! curl -s "$API_URL" > /dev/null; then
    echo "Không thể kết nối đến API tại $API_URL. Vui lòng đảm bảo API đang chạy."
    exit 1
fi

# Hàm tải một mô hình
download_model() {
    model_type=$1
    force=${2:-false}
    version=${3:-latest}
    
    echo "Đang tải mô hình $model_type (force_download: $force, version: $version)..."
    
    response=$(curl -s -X POST "$API_URL/models/load" \
        -H "Content-Type: application/json" \
        -d "{\"model_type\": \"$model_type\", \"version\": \"$version\", \"force_download\": $force}")
    
    if echo "$response" | grep -q "\"status\": \"success\""; then
        echo "✅ Tải thành công mô hình $model_type"
    else
        error=$(echo "$response" | grep -o "\"error\": \"[^\"]*\"" | cut -d'"' -f4)
        echo "❌ Lỗi khi tải mô hình $model_type: $error"
        
        # Nếu là lỗi không có model.bin hoặc không kết nối được, thử lại với force=true
        if [ "$force" = "false" ] && (echo "$error" | grep -q "pytorch_model.bin\|connect to 'https://huggingface.co'\|torch.dtype"); then
            echo "🔄 Thử lại với force_download=true..."
            download_model "$model_type" true "$version"
        fi
    fi
}

# Tải mô hình gốc (không transformer)
echo "1️⃣ Tải các mô hình cơ bản..."
download_model "tfidf" false
download_model "bm25" false

# Tải các mô hình transformer với force_download
echo "2️⃣ Tải các mô hình transformer..."
download_model "roberta" true
download_model "xlm-roberta" true
download_model "distilbert" true

# Tải các mô hình hybrid
echo "3️⃣ Tải các mô hình hybrid..."
download_model "hybrid_tfidf_bert" false
download_model "hybrid_bm25_bert" false
download_model "hybrid_bmx_bert" false

# Kiểm tra trạng thái các mô hình
echo "4️⃣ Kiểm tra trạng thái các mô hình..."
curl -s -X GET "$API_URL/models/active" | python -m json.tool

echo "Hoàn thành tải tất cả các mô hình!"
echo "Bây giờ bạn có thể tiến hành fine-tune các mô hình với lệnh:"
echo "curl -X POST \"$API_URL/models/fine-tune\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"path\": \"data/history_learning_data.json\", \"sample\": 100, \"version\": \"custom_v1\"}'"

# Đặt quyền thực thi cho script
chmod +x "$0" 