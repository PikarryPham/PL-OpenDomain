#!/bin/bash

# Script để chạy fix_transformer_models_and_initialize.py
# Sử dụng: ./run_model_fix.sh [--force] [--patch-file]

echo "===== CÔNG CỤ SỬA LỖI MÔ HÌNH TRANSFORMER ====="
echo "Script này sẽ sửa lỗi các mô hình transformer trong container Docker"
echo "======================================================"

# Kiểm tra xem đang chạy trong Docker container hay không
if [ ! -f "/.dockerenv" ]; then
    echo "CẢNH BÁO: Không phát hiện môi trường Docker. Script này nên được chạy trong container Docker."
    read -p "Bạn có muốn tiếp tục không? (y/n): " continue_outside_docker
    if [ "$continue_outside_docker" != "y" ]; then
        echo "Đã hủy."
        exit 1
    fi
fi

# Thiết lập đường dẫn
if [ -d "/usr/src/app" ]; then
    APP_DIR="/usr/src/app"
elif [ -d "/app" ]; then
    APP_DIR="/app"
else
    APP_DIR=$(pwd)
fi

SRC_DIR="$APP_DIR/src"
MODELS_DIR="$APP_DIR/models"

# Kiểm tra xem file Python đã tồn tại chưa
PYTHON_SCRIPT="$SRC_DIR/fix_transformer_models_and_initialize.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Không tìm thấy script $PYTHON_SCRIPT!"
    echo "Hãy đảm bảo bạn đã tạo file này trước."
    exit 1
fi

# Cấp quyền thực thi cho script Python
chmod +x "$PYTHON_SCRIPT"

# Xử lý các tham số dòng lệnh
FORCE_FLAG=""
PATCH_FILE_FLAG=""
VERSION_FLAG=""

for arg in "$@"; do
    case $arg in
        --force)
            FORCE_FLAG="--force"
            ;;
        --patch-file)
            PATCH_FILE_FLAG="--patch-file"
            ;;
        --version=*)
            VERSION_FLAG="--version=${arg#*=}"
            ;;
    esac
done

# Thiết lập đường dẫn models
export MODELS_DIR="$MODELS_DIR"

echo "Đường dẫn models: $MODELS_DIR"
echo "Kiểm tra trạng thái mô hình hiện tại..."

# Kiểm tra trạng thái mô hình trước khi sửa
python "$PYTHON_SCRIPT" --check

# Hỏi người dùng có muốn tiếp tục sửa không
read -p "Bạn có muốn tiếp tục sửa các mô hình transformer? (y/n): " continue_fix
if [ "$continue_fix" != "y" ]; then
    echo "Đã hủy."
    exit 0
fi

# Hỏi người dùng có muốn sửa file embeddings.py không
if [ -z "$PATCH_FILE_FLAG" ]; then
    read -p "Bạn có muốn sửa file embeddings.py để xử lý lỗi 'NoneType' object has no attribute 'eval'? (y/n): " patch_file
    if [ "$patch_file" = "y" ]; then
        PATCH_FILE_FLAG="--patch-file"
    fi
fi

# Hỏi người dùng có muốn force tạo mô hình mới không
if [ -z "$FORCE_FLAG" ]; then
    read -p "Bạn có muốn buộc tạo mô hình mới (kể cả khi đã tồn tại)? (y/n): " force_create
    if [ "$force_create" = "y" ]; then
        FORCE_FLAG="--force"
    fi
fi

# Chạy script Python với các tham số đã chọn
echo "Đang chạy script sửa lỗi mô hình..."
CMD="python $PYTHON_SCRIPT $FORCE_FLAG $PATCH_FILE_FLAG $VERSION_FLAG"
echo "Lệnh thực thi: $CMD"
eval $CMD

# Kiểm tra kết quả
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "===== HOÀN THÀNH ====="
    echo "Script đã thực hiện thành công. Hãy khởi động lại container để áp dụng các thay đổi."
    echo "Lệnh khởi động lại container:"
    echo "docker restart chatbot-api chatbot-worker"
else
    echo "===== LỖI ====="
    echo "Script gặp lỗi khi thực thi. Mã lỗi: $EXIT_CODE"
fi

# Xác nhận khởi động lại container
read -p "Bạn có muốn khởi động lại containers ngay bây giờ? (y/n): " restart_now
if [ "$restart_now" = "y" ]; then
    echo "Đang khởi động lại containers..."
    if command -v docker &> /dev/null; then
        docker restart chatbot-api chatbot-worker
        echo "Đã khởi động lại containers thành công."
    else
        echo "Lệnh docker không khả dụng. Vui lòng khởi động lại containers thủ công."
    fi
fi

exit 0 