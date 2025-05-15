# AI Summarization cho việc xác định Topics

Tài liệu này mô tả chức năng AI Summarization để tự động tóm tắt nội dung và trích xuất chủ đề từ văn bản. Chức năng này giúp cải thiện quá trình xác định các topic từ nội dung mà không cần sử dụng danh sách cố định.

## Tổng quan

Chức năng AI Summarization bao gồm hai quá trình chính:
1. **Tóm tắt nội dung** (Summarization): Sử dụng mô hình DistilBART-CNN để tóm tắt văn bản dài thành văn bản ngắn gọn.
2. **Trích xuất chủ đề** (Topic Extraction): Kết hợp nhiều phương pháp để trích xuất chủ đề từ văn bản, bao gồm:
   - Zero-shot classification với DistilBERT
   - LDA (Latent Dirichlet Allocation) để phân tích chủ đề không giám sát
   - Phân tích từ khóa dựa trên NLP

## Cài đặt

### 1. Dependencies

Tất cả dependencies cần thiết đã được thêm vào `requirements.txt` của dự án:
- transformers
- torch
- spacy
- scikit-learn
- numpy

Khi sử dụng với Docker, không cần cài đặt thủ công vì dependencies này sẽ được cài đặt tự động khi Docker container được build lại với requirements.txt cập nhật.

```bash
# Nếu cần rebuild Docker container
docker-compose build --no-cache chatbot-api
docker-compose up -d chatbot-api
```

### 2. Cài đặt model spaCy trong Docker

Đối với mô hình spaCy, cần thực hiện thêm bước cài đặt trong container:

```bash
# Kết nối tới container đang chạy
docker exec -it chatbot-api bash

# Cài đặt mô hình spaCy từ trong container
python -m spacy download en_core_web_sm
```

### 3. Giới thiệu về setup_ai_models.sh

File `setup_ai_models.sh` là một script hỗ trợ có các chức năng:

1. Cài đặt mô hình spaCy cho tiếng Anh
2. Tạo thư mục lưu trữ model transformer
3. Tùy chọn tải trước model summarization
4. Tùy chọn khởi động API server

Trong môi trường Docker:
```bash
# Đảm bảo script có quyền thực thi
docker exec -it chatbot-api chmod +x /usr/src/app/src/setup_ai_models.sh

# Chạy script từ trong container
docker exec -it chatbot-api /usr/src/app/src/setup_ai_models.sh
```

**Lưu ý:** Trong Docker, bạn có thể chỉ cần sử dụng phần cài đặt mô hình spaCy và không cần phần khởi động server (vì container đã chạy).

## Cấu trúc file

- `ai_summarization.py`: Module chính chứa các hàm tóm tắt và trích xuất chủ đề
- `setup_ai_models.sh`: Script để cài đặt các model cần thiết
- `test_ai_summarization.py`: Script để test các API
- Các API trong `app.py`:
  - `/ai/summarize`: Tóm tắt nội dung và trích xuất chủ đề
  - `/ai/extract-topics`: Chỉ trích xuất chủ đề
  - `/ai/batch-process`: Xử lý hàng loạt các entry
  - `/ai/models/status`: Kiểm tra trạng thái các model
  - `/ai/initialize-models`: Khởi tạo các model

## Cách sử dụng

### 1. Khởi tạo và kiểm tra trạng thái model

```bash
# Khởi tạo tất cả các model AI
curl -X POST http://localhost:8000/ai/initialize-models \
  -H "Content-Type: application/json"
```

**Input:** Không yêu cầu dữ liệu đầu vào (body rỗng)

**Output:**
```json
{
  "summarizer": true,
  "topic_extractor": true,
  "nlp": true
}
```

### 2. Tóm tắt nội dung và trích xuất chủ đề

```bash
# Tóm tắt nội dung và trích xuất chủ đề
curl -X POST http://localhost:8000/ai/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data, and apply knowledge from data across a broad range of application domains. Data science is related to data mining, machine learning and big data."
  }'
```

**Input:**
- `content` (string, bắt buộc): Nội dung văn bản cần tóm tắt và trích xuất chủ đề
- `max_length` (integer, tùy chọn): Độ dài tối đa của bản tóm tắt, mặc định là 100 tokens
- `min_length` (integer, tùy chọn): Độ dài tối thiểu của bản tóm tắt, mặc định là 30 tokens

**Output:**
```json
{
  "summary": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data, and apply knowledge from data across a broad range of application domains. Data science is related to data mining, machine learning and big data.",
  "topics": [
    {
      "topic": "Computer Science",
      "score": 0.5140364518531549,
      "method": "distilbert-cosine-similarity"
    },
    {
      "topic": "Linguistics",
      "score": 0.35284635764183125,
      "method": "distilbert-cosine-similarity"
    },
    {
      "topic": "Anthropology",
      "score": 0.3490498196807106,
      "method": "distilbert-cosine-similarity"
    },
    {
      "topic_id": "lda_topic_4",
      "keywords": ["datum", "knowledge", "science", "use", "application", "apply", "big", "broad", "data", "domain"],
      "score": 0.029999486696799325,
      "method": "lda"
    }
  ],
  "keywords": ["datum", "knowledge", "data science", "data", "science", "an interdisciplinary field", "that", "scientific methods", "processes", "algorithms"]
}
```

### 3. Chỉ trích xuất chủ đề

```bash
# Chỉ trích xuất các chủ đề từ văn bản
curl -X POST http://localhost:8000/ai/extract-topics \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from data."
  }'
```

**Input:**
- `content` (string, bắt buộc): Nội dung văn bản cần trích xuất chủ đề
- `threshold` (float, tùy chọn): Ngưỡng điểm số để lọc các chủ đề, mặc định là 0.25
- `max_topics` (integer, tùy chọn): Số lượng chủ đề tối đa trả về, mặc định là 5

**Output:**
```json
{
  "topics": [
    {
      "topic": "Computer Science",
      "score": 0.89,
      "method": "distilbert-cosine-similarity"
    },
    {
      "topic": "Technology",
      "score": 0.76,
      "method": "distilbert-cosine-similarity"
    },
    {
      "topic": "Mathematics",
      "score": 0.62,
      "method": "lda"
    },
    {
      "topic": "Data",
      "score": 0.58,
      "method": "keyword-extraction"
    }
  ]
}
```

### 4. Xử lý hàng loạt

```bash
# Xử lý nhiều entries cùng lúc (đồng bộ)
curl -X POST http://localhost:8000/ai/batch-process \
  -H "Content-Type: application/json" \
  -d '{
    "entries": [
      {"visible_content": "Data science is an interdisciplinary field that uses scientific methods."},
      {"visible_content": "Machine learning is a subset of artificial intelligence."}
    ]
  }'
```

**Input:**
- `entries` (array, bắt buộc): Mảng các đối tượng cần xử lý
  - Mỗi đối tượng phải có trường `visible_content` (string)
- `async` (boolean, tùy chọn): Xử lý bất đồng bộ với Celery, mặc định là false
- `summarize` (boolean, tùy chọn): Có tóm tắt nội dung hay không, mặc định là true
- `extract_topics` (boolean, tùy chọn): Có trích xuất chủ đề hay không, mặc định là true

**Output (đồng bộ):**
```json
{
  "processed_entries": [
    {
      "visible_content": "Data science is an interdisciplinary field that uses scientific methods.",
      "summary": "Data science is an interdisciplinary field that uses scientific methods.",
      "ai_topics": [
        "Computer Science",
        "Anthropology",
        "Sociology",
        "History"
      ],
      "ai_keywords": [
        "data science", 
        "an interdisciplinary field", 
        "that", 
        "scientific methods", 
        "data", 
        "science", 
        "field", 
        "method"
      ]
    },
    {
      "visible_content": "Machine learning is a subset of artificial intelligence.",
      "summary": "Machine learning is a subset of artificial intelligence.",
      "ai_topics": [
        "Computer Science",
        "Psychology",
        "History"
      ],
      "ai_keywords": [
        "machine learning", 
        "a subset", 
        "artificial intelligence", 
        "machine", 
        "learning", 
        "subset", 
        "intelligence"
      ]
    }
  ]
}
```

**Output (bất đồng bộ):**
```json
{
  "task_id": "3b38e956-0b45-482a-a0b1-60d5281fe219",
  "status": "Task started",
  "message": "Processing 2 entries asynchronously"
}
```

### 5. Tích hợp với DBpedia sync

```bash
# Sử dụng visible_content để tự động trích xuất topic cho DBpedia sync
curl -X POST http://localhost:8000/dbpedia/sync-data \
  -H "Content-Type: application/json" \
  -d '{
    "visible_content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge."
  }'
```

**Input:**
- `visible_content` (string, bắt buộc khi không có topics): Nội dung văn bản để trích xuất chủ đề
- `topics` (array, tùy chọn): Danh sách các chủ đề được chỉ định thủ công
  - Nếu cung cấp cả `visible_content` và `topics`, `topics` sẽ được ưu tiên sử dụng
  - Nếu không cung cấp `topics`, hệ thống sẽ tự động trích xuất từ `visible_content`

**Output:**
```json
{
  "task_id": "8334de79-80fe-48c3-b240-ad158c3b29c4"
}
```

#### Theo dõi tiến trình đồng bộ

API `/dbpedia/sync-data` thực hiện quá trình đồng bộ bất đồng bộ và chỉ trả về `task_id`. Để theo dõi tiến trình và xác nhận thành công:

```bash
# Theo dõi logs của worker để xem tiến trình đồng bộ
docker logs -f chatbot-worker
```

Bạn có thể kiểm tra kết quả đồng bộ bằng cách:
1. Tìm kiếm các entities đã được đồng bộ qua API `/dbpedia/search-data`:
   ```bash
   curl -X POST http://localhost:8000/dbpedia/search-data \
     -H "Content-Type: application/json" \
     -d '{"keywords": ["Data Science", "Algorithm"]}'
   ```
2. Kiểm tra trực tiếp trong database:
   ```bash
   docker exec -it mariadb-tiny bash
   mysql -u root -p
   USE dbpedia;
   SELECT * FROM topic ORDER BY created_at DESC LIMIT 10;
   ```

#### Giải thích về quá trình đồng bộ DBpedia:

API `/dbpedia/sync-data` thực hiện các bước sau:
1. Trích xuất chủ đề từ nội dung (nếu không có topics được chỉ định)
2. Sử dụng các chủ đề để tìm thông tin liên quan từ DBpedia
3. Tạo các liên kết trong cơ sở dữ liệu giữa nội dung và các entity từ DBpedia
4. Đánh chỉ mục (index) thông tin vào vector database để phục vụ tìm kiếm ngữ nghĩa

Quá trình này giúp hệ thống có thể:
- Tự động phát hiện các chủ đề từ nội dung
- Làm giàu nội dung với thông tin từ nguồn kiến thức DBpedia
- Cải thiện khả năng tìm kiếm và gợi ý thông tin liên quan

### 6. Đồng bộ dữ liệu danh mục từ DBpedia

```bash
# Đồng bộ hóa các danh mục (categories) từ DBpedia
curl -X POST http://localhost:8000/dbpedia/sync-category \
  -H "Content-Type: application/json" \
  -d '{
    "categories": [
      {
        "name": "Computer Science",
        "uri": "https://dbpedia.org/page/Category:Computer_science",
        "topic": "Technology"
      },
      {
        "name": "Machine Learning",
        "uri": "https://dbpedia.org/page/Category:Machine_learning",
        "topic": "Artificial Intelligence"
      }
    ]
  }'
```

**Input:**
- `categories` (array, bắt buộc): Mảng các đối tượng danh mục cần đồng bộ
  - `name` (string, tùy chọn nếu có uri): Tên danh mục
  - `uri` (string, tùy chọn nếu có name): URI DBpedia của danh mục
  - `topic` (string, bắt buộc): Tên topic mà danh mục này thuộc về

**Output:**
```json
{
  "task_ids": [
    "a1b2c3d4-e5f6-7890-abcd-123456789012", 
    "b2c3d4e5-f6g7-8901-bcde-2345678901234"
  ]
}
```
# Vấn đề với API /dbpedia/sync-category
Lỗi "Internal Server Error" khi gọi API /dbpedia/sync-category, nguyên nhân có thể là:
Topic không tồn tại: Lỗi phổ biến nhất là topics "Technology" và "Artificial Intelligence" chưa tồn tại trong cơ sở dữ liệu. API cố gắng liên kết category với topic_id nhưng không tìm thấy.

# Tạo API tạm thời để thêm topics
docker exec -it chatbot-api bash -c "cd /usr/src/app/src && python -c \"
from models import get_topic_by_name, insert_topic
if not get_topic_by_name('Technology'):
    insert_topic('Technology')
if not get_topic_by_name('Artificial Intelligence'):
    insert_topic('Artificial Intelligence')
print('Đã tạo các topics cần thiết')
\""
# Hoặc Thử lại với một category nhỏ hơn:

curl -X POST http://localhost:8000/dbpedia/sync-category \
  -H "Content-Type: application/json" \
  -d '{
    "categories": [
      {
        "name": "Programming languages",
        "uri": "https://dbpedia.org/page/Category:Programming_languages",
        "topic": "Technology" 
      }
    ]
  }'
  
#### Giải thích về quá trình đồng bộ danh mục:

API `/dbpedia/sync-category` thực hiện các bước sau:
1. Kiểm tra danh mục (category) đã tồn tại trong cơ sở dữ liệu chưa
2. Nếu chưa có, tạo danh mục mới và liên kết với topic được chỉ định
3. Khởi động tác vụ bất đồng bộ để đồng bộ các trang (pages) thuộc danh mục đó từ DBpedia
4. Mặc định, quá trình đồng bộ sẽ đi đến độ sâu 3 (có nghĩa là sẽ lấy các trang thuộc danh mục, các danh mục con, và các danh mục cháu)

Quá trình này giúp:
- Nhập hàng loạt nội dung có cấu trúc từ DBpedia dựa trên phân loại
- Tự động tổ chức nội dung theo cấu trúc phân cấp (topic > category > page)
- Xây dựng cơ sở dữ liệu kiến thức phong phú mà không cần nhập thủ công

### Xử lý lỗi với các API đồng bộ DBpedia

#### 1. Lỗi "Internal Server Error" khi đồng bộ danh mục

**Nguyên nhân thường gặp:**
- Topic được chỉ định không tồn tại trong cơ sở dữ liệu
- Vấn đề kết nối đến DBpedia
- Lỗi xử lý bất đồng bộ trong Celery

**Cách khắc phục:**
- Đảm bảo các topic đã tồn tại trong cơ sở dữ liệu trước (có thể thêm bằng API `/topics` hoặc kiểm tra trong DB)
- Kiểm tra URI DBpedia chính xác (thử truy cập URI trực tiếp trong trình duyệt)
- Kiểm tra logs của container:
  ```bash
  docker logs chatbot-api
  docker logs chatbot-worker
  ```

#### 2. Quá trình đồng bộ tốn nhiều thời gian

Việc đồng bộ một danh mục lớn có thể tốn nhiều thời gian, đặc biệt với độ sâu cao. Bạn có thể:
- Giảm độ sâu đồng bộ xuống còn 1 hoặc 2
- Chia nhỏ thành nhiều danh mục cụ thể hơn
- Theo dõi tiến trình thông qua API `/dbpedia/extract-data/{task_id}`

#### 3. Lỗi khi không tìm thấy đủ entity từ DBpedia

Nếu API không trả về đủ entity như mong đợi:
- Thử với từ khóa tìm kiếm rộng hơn
- Kiểm tra liệu DBpedia có chứa thông tin về chủ đề đó không
- Thử thay đổi ngôn ngữ của URI (ví dụ: dùng URI tiếng Anh thay vì ngôn ngữ khác)

## Kiểm tra tất cả các chức năng

Sử dụng script test từ trong container Docker:

```bash
# Vào container
docker exec -it chatbot-api bash

# Chạy tất cả các test
cd /usr/src/app/src
python test_ai_summarization.py
```

**Input:** Không yêu cầu đầu vào, script sẽ tự tạo nội dung test

**Output:** Kết quả của các API test, hiển thị trạng thái thành công/thất bại và phản hồi nhận được

```bash
# Test một API cụ thể
python test_ai_summarization.py --api summarize
python test_ai_summarization.py --api topics
python test_ai_summarization.py --api batch
python test_ai_summarization.py --api sync
python test_ai_summarization.py --api status
python test_ai_summarization.py --api initialize
```

**Input:**
- `--api` (string, tùy chọn): Tên API cần test (summarize, topics, batch, sync, status, initialize)
- `--server` (string, tùy chọn): URL máy chủ, mặc định là "http://localhost:8000"
- `--content` (string, tùy chọn): Nội dung văn bản để test, mặc định là đoạn văn mẫu

**Output:** Kết quả chi tiết của API được chỉ định

## Khắc phục sự cố

### 1. Model không được tải

Nếu bạn nhận được thông báo lỗi về việc model chưa được tải, hãy thử:

```bash
# Khởi tạo lại các model
curl -X POST http://localhost:8000/ai/initialize-models
```

Hoặc kiểm tra log để biết thêm chi tiết:
```bash
docker logs chatbot-api
```

### 2. Lỗi khi tải model

Nếu gặp lỗi khi tải model từ Hugging Face, có thể do kết nối mạng. Hãy đảm bảo rằng container Docker có kết nối internet ổn định.

### 3. Lỗi bộ nhớ

Nếu gặp lỗi về bộ nhớ (out of memory), bạn có thể:
- Tăng bộ nhớ cho Docker container trong docker-compose.yml:
  ```yaml
  services:
    chatbot-api:
      mem_limit: 4g  # Tăng giới hạn bộ nhớ
  ```
- Giảm kích thước batch
- Sử dụng phiên bản model nhỏ hơn (có thể sửa trong `ai_summarization.py`)

## Tùy chỉnh và mở rộng

Bạn có thể tùy chỉnh chức năng AI Summarization bằng cách:

1. **Thay đổi mô hình Summarization**: Sửa biến `summarizer_name` trong hàm `init_models()` của file `ai_summarization.py` để sử dụng mô hình khác (ví dụ: "t5-small", "facebook/bart-large-cnn", etc.)

2. **Thêm chủ đề vào danh sách COMMON_TOPICS**: Sửa biến `COMMON_TOPICS` trong file `ai_summarization.py`

3. **Điều chỉnh ngưỡng điểm số**: Thay đổi giá trị ngưỡng `0.25` trong các hàm trích xuất chủ đề để lọc ra các chủ đề có điểm cao hơn hoặc thấp hơn 