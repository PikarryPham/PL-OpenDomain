# Lấy danh sách các mô hình có sẵn
curl -X GET http://localhost:8000/models/embedding

Mô hình chưa có sẵn: Các mô hình transformer (RoBERTa, XLM-RoBERTa, DistilBERT) cần được tải từ HuggingFace trước khi sử dụng, điều này có thể mất thời gian và không tự động.
# Tải mô hình transformer bằng API:

## Tải mô hình với chế độ thông thường:
(Sẽ tải phiên bản đã lưu cục bộ nếu có, nếu không sẽ tải từ Hugging Face và lưu lại)
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest"}'
```

## Tải mô hình và bắt buộc tải lại từ Hugging Face:
(Tham số `force_download: true` sẽ luôn tải lại mô hình từ Hugging Face, ngay cả khi đã có phiên bản cục bộ. Phiên bản mới tải về sẽ được lưu dưới một tên version mới (ví dụ: `hf_YYYYMMDD_HHMMSS`) nếu version được đặt là "latest", hoặc ghi đè lên version cũ nếu version cụ thể được chỉ định. Symlink `latest` sẽ được cập nhật để trỏ đến phiên bản mới nhất.)
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
```
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": true}'
```
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "distilbert", "version": "latest", "force_download": true}'
```

## Tải các mô hình khác:
(Các mô hình hybrid và truyền thống không hỗ trợ `force_download` trực tiếp qua API này, nhưng việc tải lại các mô hình transformer thành phần (nếu cần) sẽ cập nhật chúng.)
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "tfidf", "version": "latest"}'

curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "bm25", "version": "latest"}'

curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "hybrid_tfidf_bert", "version": "latest"}'

curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "hybrid_bm25_bert", "version": "latest"}'

curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "hybrid_bmx_bert", "version": "latest"}'
```
# Fine-tune các mô hình trên dữ liệu học tập
curl -X POST http://localhost:8000/models/fine-tune \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 100}'
# Test model active
curl -X GET http://localhost:8000/models/active \
  -H "Content-Type: application/json"
# Test model version
curl -X GET http://localhost:8000/models/versions \
  -H "Content-Type: application/json"
# Sử dụng một mô hình cụ thể để trích xuất dữ liệu
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 5, "embedding_model": "distilbert"}'

  # Sử dụng một mô hình cụ thể để trích xuất dữ liệu
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 5, "embedding_model": "bmx"}'

# So sánh các mô hình embedding
curl -X POST http://localhost:8000/models/compare \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 10}'



# Chạy script đánh giá:

# Kết nối vào container chatbot-api
docker exec -it chatbot-api bash

# Di chuyển đến thư mục src
cd /usr/src/app/src

# Chạy script đánh giá mô hình (sẽ tự động tải NLTK resources)
python evaluate_models.py --data_path data/history_learning_data.json --output_dir results --sample 20

# Đăng nhập vào container
docker exec -it chatbot-api bash

# Tải các mô hình
python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('roberta-base'); model = AutoModel.from_pretrained('roberta-base'); print('Đã tải xong roberta-base')"

python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base'); model = AutoModel.from_pretrained('xlm-roberta-base'); print('Đã tải xong xlm-roberta-base')"

python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased'); model = AutoModel.from_pretrained('distilbert-base-uncased'); print('Đã tải xong distilbert-base-uncased')"

# Xử lý lỗi trong quá trình tải và fine-tune mô hình

## Lỗi thường gặp và cách xử lý

### 1. Lỗi kết nối tới Hugging Face

Nếu gặp lỗi như:
```
"We couldn't connect to 'https://huggingface.co' to load the files..."
```
hoặc các lỗi timeout khác khi tải mô hình transformer.

**Giải pháp:**
- Đảm bảo máy chủ có kết nối internet ổn định tới Hugging Face.
- Sử dụng tham số `force_download: true` khi gọi API `POST /models/load` để thử tải lại. Lệnh này sẽ cố gắng tải lại từ nguồn và lưu một phiên bản mới.
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
```

### 2. Lỗi "'torch.dtype' object has no attribute 'data_ptr'" (Đã được xử lý bởi fix_transformer_models.py)

Lỗi này thường xảy ra với các phiên bản PyTorch cũ hơn hoặc cấu hình môi trường không đúng khi tải RoBERTa hoặc XLM-RoBERTa.

**Giải pháp:**
- Chạy script `fix_transformer_models.py` trước khi tải hoặc fine-tune.
- **Quan trọng:** Sử dụng API load với `force_download: true` **sau khi đã chạy script fix**. API `load` với `force_download` sẽ tải lại mô hình từ Hugging Face, đảm bảo bạn có phiên bản tương thích nhất.
```bash
# Bước 1: Chạy script fix (nếu chưa chạy)
# python src/fix_transformer_models.py --force

# Bước 2: Tải lại mô hình với force_download
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
```

### 3. Lỗi "does not appear to have a file named pytorch_model.bin..."

Thường xảy ra khi mô hình chưa được tải xuống đầy đủ hoặc bị lỗi trong quá trình tải/lưu cục bộ.

**Giải pháp:**
- Sử dụng API load với `force_download: true` để tải lại toàn bộ mô hình từ Hugging Face.
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": true}'
```

### 4. Lỗi "BMX model is not available"

**Giải pháp:**
- BMX là thư viện tùy chọn (optional dependency). Nếu bạn không cần sử dụng mô hình `bmx` hoặc `hybrid_bmx_bert`, bạn có thể bỏ qua lỗi này. Hệ thống vẫn hoạt động với các mô hình khác.
- Nếu bạn *cần* sử dụng BMX, hãy đảm bảo đã cài đặt thư viện `baguetter` chính xác theo hướng dẫn của họ. Việc cài đặt có thể yêu cầu các bước biên dịch C++.

### 5. Lỗi "'bool' object is not iterable" khi gọi /models/load

Lỗi này xảy ra do hàm xử lý API trả về giá trị boolean thay vì định dạng JSON mong đợi.

**Giải pháp:**
- **Đã được khắc phục** trong phiên bản code mới nhất bằng cách sửa hàm `load_transformer_model` trong `embeddings.py` để luôn trả về tuple `(success, result)`. Đảm bảo bạn đang chạy code đã được cập nhật.
- Nếu vẫn gặp lỗi, hãy kiểm tra lại các thay đổi trong `embeddings.py` và đảm bảo Docker container đã được build lại với code mới nhất.

## Quy trình tải và fine-tune không gặp lỗi (Đề xuất)

1.  **(Tùy chọn nhưng khuyến nghị)** Chạy script sửa lỗi transformer trước:
    ```bash
    python src/fix_transformer_models.py --force
    ```
2.  Tải các mô hình cơ bản (TF-IDF, BM25):
    ```bash
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "tfidf", "version": "latest"}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "bm25", "version": "latest"}'
    ```
3.  Tải các mô hình transformer với `force_download: true` (chỉ cần chạy lần đầu hoặc khi muốn cập nhật/sửa lỗi):
    ```bash
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "roberta", "version": "latest", "force_download": false}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": false}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "distilbert", "version": "latest", "force_download": false}'
    ```
4.  Tải các mô hình kết hợp (hybrid):
    (Lưu ý: Các lệnh này sẽ sử dụng các phiên bản *mới nhất* của mô hình thành phần đã được tải ở các bước trên)
    ```bash
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_tfidf_bert", "version": "latest"}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_bm25_bert", "version": "latest"}'
    # curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_bmx_bert", "version": "latest"}' # Nếu đã cài BMX
    ```
5.  Tiến hành fine-tune (sẽ sử dụng các phiên bản mô hình mới nhất hiện có):
    ```bash
    curl -X POST http://localhost:8000/models/fine-tune \
      -H "Content-Type: application/json" \
      -d '{"path": "data/history_learning_data.json", "sample": 100, "version": "custom_v1"}'
    ```
6.  Kiểm tra trạng thái mô hình:
    ```bash
    curl -X GET http://localhost:8000/models/active
    ```

# Cách đơn giản nhất: Sử dụng script tự động

## Cách 1: Tải tất cả các mô hình (Sử dụng download_all_models.sh)
Script này sẽ tự động gọi API `/models/load` cho tất cả các mô hình, bao gồm cả việc sử dụng `force_download: true` cho các mô hình transformer để đảm bảo chúng được tải đúng cách.

```bash
# Cấp quyền thực thi cho script
chmod +x src/download_all_models.sh

# Chạy script (tham số tùy chọn: [host] [port])
./src/download_all_models.sh localhost 8000
```

## Cách 2: Sửa lỗi mô hình transformer cụ thể (Sử dụng fix_transformer_models.py)
Script này giúp tạo cấu trúc thư mục và file cần thiết để khắc phục một số lỗi khi tải RoBERTa và XLM-RoBERTa. **Nên chạy script này trước khi tải các mô hình transformer lần đầu.**

```bash
# Cấp quyền thực thi cho script
chmod +x src/fix_transformer_models.py

# Chạy script để sửa lỗi (tạo file giả và cấu trúc thư mục)
python src/fix_transformer_models.py --force

# Kiểm tra xem các mô hình đã được "sửa" chưa (kiểm tra cấu trúc thư mục)
python src/fix_transformer_models.py --check
```
**Lưu ý:** Sau khi chạy script này, bạn vẫn cần gọi API `/models/load` với `force_download: true` để thực sự tải dữ liệu mô hình từ Hugging Face.

## Cách 3: Tự động cả quá trình (Sử dụng fix_and_finetune.sh - Cách tốt nhất)
Script này kết hợp việc chạy `fix_transformer_models.py`, `download_all_models.sh` (tải tất cả mô hình với `force_download`), và sau đó tự động gọi API fine-tune.

```bash
# Cấp quyền thực thi cho script
chmod +x src/fix_and_finetune.sh

# Chạy script (tham số tùy chọn: [host] [port] [finetune_version])
./src/fix_and_finetune.sh localhost 8000 my_finetuned_v1
```

Script này sẽ:
1.  Chạy `fix_transformer_models.py --force`.
2.  Chạy `download_all_models.sh` để tải tất cả mô hình (với `force_download` cho transformers).
3.  Gọi API `/models/fine-tune` với phiên bản được chỉ định.
4.  Theo dõi và hiển thị kết quả của tác vụ fine-tune.
5.  Kiểm tra lại trạng thái cuối cùng của các mô hình qua API `/models/active`.

Đây là cách đơn giản và toàn diện nhất để đảm bảo các mô hình được thiết lập đúng cách và fine-tune.