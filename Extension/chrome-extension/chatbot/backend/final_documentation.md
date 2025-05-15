# Setup từ đầu 
Đọc file README.md 
# Hướng dẫn triển khai Task 2 và Task 3

## Mục lục
1. **Tổng quan**
2. **Task 2: So sánh các mô hình Embedding**
   - 2.1. Giới thiệu
   - 2.2. Cài đặt và cấu hình
   - 2.3. Tải các mô hình
   - 2.4. Fine-tune các mô hình
   - 2.5. Lượng tử hóa (Quantization)
   - 2.6. Đánh giá và so sánh các mô hình
   - 2.7. Xử lý lỗi thường gặp
3. **Task 3: AI Summarization cho việc xác định chủ đề**
   - 3.1. Giới thiệu
   - 3.2. Thực hiện trích xuất dữ liệu
   - 3.3. Trích xuất chủ đề tự động (LDA)
   - 3.4. Xử lý các API bất đồng bộ

## 1. Tổng quan

Dự án này bao gồm hai task chính:

- **Task 2**: So sánh hiệu suất của các mô hình Embedding nguồn mở thay vì sử dụng model của OpenAI
- **Task 3**: Sử dụng AI để tóm tắt nội dung và xác định các chủ đề liên quan

## 2. Task 2: So sánh các mô hình Embedding

### 2.1. Giới thiệu

Trong task này, chúng ta triển khai và so sánh các mô hình nhúng (embedding models) khác nhau để thay thế cho OpenAI:

- **TF-IDF**: Mô hình cơ bản dựa trên tần suất từ
- **BM25**: Phiên bản cải tiến của TF-IDF
- **Transformer Models**: RoBERTa, XLM-RoBERTa, DistilBERT
- **Hybrid Models**: Kết hợp mô hình truyền thống với transformer

### 2.2. Cài đặt và cấu hình

Đảm bảo hệ thống Docker đã được khởi động:
```bash
docker-compose up -d
```

### 2.3. Tải các mô hình

Kiểm tra danh sách mô hình có sẵn:
```bash
curl -X GET http://localhost:8000/models/embedding
```

Tải các mô hình transformer:
```bash
# Tải RoBERTa
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'

# Tải XLM-RoBERTa
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": true}'

# Tải DistilBERT
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "distilbert", "version": "latest", "force_download": true}'
```

Tải các mô hình truyền thống và hybrid:
```bash
# Tải TF-IDF
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "tfidf", "version": "latest"}'

# Tải BM25
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "bm25", "version": "latest"}'

# Tải các mô hình kết hợp
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "hybrid_tfidf_bert", "version": "latest"}'

curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "hybrid_bm25_bert", "version": "latest"}'
```

### 2.4. Fine-tune các mô hình

Fine-tune mô hình với dữ liệu học tập:
```bash
curl -X POST http://localhost:8000/models/fine-tune \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 100}'
```

Kết quả trả về sẽ có dạng:
```json
{"task_id":"d8a67284-e47d-4e82-b64d-7a43835f2ce1"}
```

Truy xuất kết quả của task fine-tune:
```bash
curl -X GET http://localhost:8000/models/fine-tune/d8a67284-e47d-4e82-b64d-7a43835f2ce1 \
  -H "Content-Type: application/json"
```

### 2.5. Lượng tử hóa (Quantization)

#### Giới thiệu về lượng tử hóa

Lượng tử hóa là kỹ thuật giảm độ chính xác của các trọng số trong mô hình để giảm kích thước và tăng tốc độ xử lý. Hệ thống sử dụng phương pháp **Dynamic Post-Training Quantization (Dynamic PTQ)**.

```python
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
```

#### Thực hiện lượng tử hóa mô hình

Lượng tử hóa mô hình RoBERTa:
```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "roberta"}'
```

Lượng tử hóa mô hình XLM-RoBERTa:
```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xlm-roberta"}'
```

Lượng tử hóa mô hình DistilBERT:
```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "distilbert"}'
```

#### Lợi ích của lượng tử hóa

1. Giảm kích thước mô hình khoảng 4 lần
2. Tăng tốc độ xử lý
3. Tiết kiệm năng lượng
4. Duy trì chất lượng dự đoán ở mức chấp nhận được

#### Kiểm tra trạng thái lượng tử hóa

```bash
curl -X GET http://localhost:8000/models/active
```

Kết quả sẽ hiển thị trường `is_quantized` cho mỗi mô hình:
```json
{
  "roberta": {
    "status": "active",
    "metadata": {
      "model_name": "roberta-base",
      "is_quantized": true
    }
  }
}
```

### 2.6. Đánh giá và so sánh các mô hình

So sánh hiệu suất các mô hình:
```bash
curl -X POST http://localhost:8000/models/compare \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 10}'
```

Chạy script đánh giá chi tiết:
```bash
# Kết nối vào container
docker exec -it chatbot-api bash

# Di chuyển đến thư mục src
cd /usr/src/app/src

# Chạy script đánh giá
python evaluate_models.py --data_path data/history_learning_data.json --output_dir results --sample 20
```

### 2.7. Xử lý lỗi thường gặp

#### Lỗi kết nối tới Hugging Face
Giải pháp: Sử dụng tham số `force_download: true` khi gọi API load model.

#### Lỗi "'torch.dtype' object has no attribute 'data_ptr'"
Giải pháp: Cần chỉnh sửa code trực tiếp để xử lý các vấn đề tương thích với phiên bản PyTorch. Sau đó tải lại mô hình với:
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
```

#### Lỗi "'NoneType' object has no attribute 'eval'"
Giải pháp: Cần kiểm tra và sửa code trong module quantize_model để đảm bảo kiểm tra null trước khi thực hiện eval(). Sau đó khởi động lại container:
```bash
docker restart chatbot-api chatbot-worker
```

## 3. Task 3: AI Summarization cho việc xác định chủ đề

### 3.1. Giới thiệu

Task này sử dụng AI để:
1. Tóm tắt nội dung visible_content của mỗi entry trong learning history data
2. Xác định các chủ đề (topics) liên quan từ bản tóm tắt

### 3.2. Thực hiện trích xuất dữ liệu

Sử dụng API để trích xuất dữ liệu với mô hình cụ thể:

```bash
# Sử dụng DistilBERT
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 5, "embedding_model": "distilbert"}'

# Sử dụng BMX
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 5, "embedding_model": "bmx"}'
```

Khi thực hiện với lượng dữ liệu lớn (sample > 10), API sẽ trả về task_id:
```json
{"task_id":"604aea2a-c7d4-4f77-b7b6-186a4872c088"}
```

Truy xuất kết quả của task extract-data:
```bash
curl -X GET http://localhost:8000/dbpedia/extract-data/604aea2a-c7d4-4f77-b7b6-186a4872c088 \
  -H "Content-Type: application/json"
```

Kết quả sau khi trích xuất sẽ được lưu vào file "final_output_v1.json" theo định dạng:
- entry_id: ID của dữ liệu học tập
- topics: Danh sách các chủ đề chính
- categories: Danh sách các danh mục con
- related_concepts: Các khái niệm liên quan

### 3.3. Trích xuất chủ đề tự động (LDA)

Hệ thống có thêm các API để tự động trích xuất chủ đề từ nội dung mà không cần danh sách chủ đề định nghĩa trước (COMMON_TOPICS), sử dụng thuật toán LDA (Latent Dirichlet Allocation).

#### 3.3.1. Trích xuất chủ đề tự động từ nội dung

API này giúp tự động trích xuất chủ đề từ nội dung bất kỳ, không phụ thuộc vào danh sách chủ đề định nghĩa sẵn:

```bash
curl -X POST http://localhost:8000/ai/extract-topics-auto \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs.",
    "num_topics": 3
  }'
```

Kết quả trả về:
```json
{
  "topics": [
    {
      "topic": "Intelligence Recognition",
      "keywords": ["intelligence", "recognition", "machine", "human", "artificial"],
      "score": 0.72,
      "method": "lda-auto"
    },
    {
      "topic": "Language Translation",
      "keywords": ["language", "translation", "natural", "mapping", "input"],
      "score": 0.65,
      "method": "lda-auto"
    }
  ],
  "message": "Topics extracted automatically using LDA without predefined topic list"
}
```

#### 3.3.2. Xử lý dữ liệu history learning đồng bộ

API đồng bộ để xử lý history learning data với phương pháp LDA tự động:

```bash
curl -X POST http://localhost:8000/dbpedia/extract-data-auto \
  -H "Content-Type: application/json" \
  -d '{
    "path": "data/history_learning_data.json",
    "sample": 5,
    "limit": 10,
    "embedding_model": "openai"
  }'
```

Kết quả được xử lý ngay lập tức và trả về trực tiếp trong response. Phù hợp cho việc xử lý số lượng nhỏ dữ liệu (sample < 10).

#### 3.3.3. Xử lý dữ liệu history learning bất đồng bộ

API bất đồng bộ để xử lý lượng lớn dữ liệu:

```bash
curl -X POST http://localhost:8000/dbpedia/extract-data-auto-async \
  -H "Content-Type: application/json" \
  -d '{
    "path": "data/history_learning_data.json",
    "sample": 10,
    "limit": 10,
    "embedding_model": "openai"
  }'
```

API này trả về một task_id:
```json
{"task_id":"fa2db846-c9e1-48f5-a551-2f3b1df8f32e"}
```

Để lấy kết quả:
```bash
curl -X GET http://localhost:8000/dbpedia/extract-data-auto/fa2db846-c9e1-48f5-a551-2f3b1df8f32e
```

Kết quả được lưu trong file "final_output_auto_lda_openai.json" (tên file phụ thuộc vào mô hình embedding sử dụng), định dạng kết quả:
```json
{
  "result": [
    {
      "entry_id": "entry123",
      "auto_topics": ["Data Analysis", "Machine Learning", "Software Engineering"],
      "pages": [/* Danh sách các trang DBpedia liên quan */]
    }
  ]
}
```

#### 3.3.4. So sánh với phương pháp sử dụng COMMON_TOPICS

| Tính năng | Phương pháp cũ (COMMON_TOPICS) | Phương pháp mới (LDA) |
|-----------|--------------------------------|----------------------|
| Định nghĩa trước topics | Yêu cầu | Không yêu cầu |
| Khả năng mở rộng | Hạn chế, cần cập nhật thủ công | Tự động phát hiện chủ đề mới |
| Độ chính xác | Tốt với chủ đề đã định nghĩa | Phụ thuộc vào nội dung và chất lượng LDA |
| Tốc độ xử lý | Nhanh hơn | Chậm hơn do cần phân tích ngữ nghĩa |
| Linh hoạt | Thấp | Cao, tự thích ứng với nhiều loại nội dung |

### 3.4. Xử lý các API bất đồng bộ

Các API bất đồng bộ trong hệ thống sẽ trả về task_id và có API tương ứng để truy xuất kết quả:

| API | Endpoint truy xuất kết quả |
|-----|----------------------------|
| `/models/fine-tune` | `/models/fine-tune/{task_id}` |
| `/models/compare` | `/models/fine-tune/{task_id}` (sử dụng cùng endpoint với fine-tune) |
| `/dbpedia/extract-data` | `/dbpedia/extract-data/{task_id}` |
| `/dbpedia/extract-data-auto-async` | `/dbpedia/extract-data-auto/{task_id}` |
| `/dbpedia/sync-data` | Không có endpoint truy xuất riêng, kiểm tra bằng API khác |
| `/ai/batch-process` | Không có endpoint truy xuất riêng, kiểm tra kết quả lưu trong file |

Đối với các task bất đồng bộ, nếu kết quả chưa sẵn sàng, hệ thống sẽ trả về trạng thái "PENDING". Khi task hoàn thành, kết quả sẽ được trả về với trạng thái "SUCCESS" hoặc "FAILURE".
