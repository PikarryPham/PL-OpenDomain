# Reset all docker containers
```bash
cd chatbot/backend
docker compose down -v
docker system prune -a
```
# Start lai docker 
docker compose build --no-cache  # Build khoang 750s 
docker compose up -d
docker compose ps # -> Tất cả container đều ở statsu Up (healthy)
docker compose logs -f chatbot-api

-> Lúc này source đã tự động tải các model từ hugging face về theo 2 mục hf_timestamps và folder latest.
# Import data
Có thể import lại cũng được nhưng chưa cần thiết cho viết test các data mới

# Test API
- Có thể run lệnh curl trực tiếp trên terminal 
## Remove folder models
sudo rm -rf models
# 1. Test load model models/load 
Những model chưa có trong foldel models tức là là cần fine tune mô hình trước khi tải như tfidf 
```
curl -X POST http://localhost:8000/models/fine-tune \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 100}'
```
sau khi fine tune xong có thể tải  lại mô hình
```
curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "tfidf", "version": "latest"}'
```

Tải các mô hình transformer với `force_download: true` (chỉ cần chạy lần đầu hoặc khi muốn cập nhật/sửa lỗi):

    ```
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "roberta", "version": "latest", "force_download": false}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": false}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "distilbert", "version": "latest", "force_download": false}'
    ```

Tải các mô hình kết hợp (hybrid):
    (Lưu ý: Các lệnh này sẽ sử dụng các phiên bản *mới nhất* - theo timestamp của model của mô hình thành phần đã được tải ở các bước trên)
```
curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_tfidf_bert", "version": "latest"}'
curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_bm25_bert", "version": "latest"}'
# curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_bmx_bert", "version": "latest"}' # Nếu đã cài BMX
```

# 2. Kiểm tra trạng thái mô hình:
```
curl -X GET http://localhost:8000/models/active
```
# 3 Lấy ra/check tất cả version của tất cả model (bao gồm cả thông tin metadata)
```
curl -X GET http://localhost:8000/models/versions \
  -H "Content-Type: application/json"
```
Delete một model version cụ thể 
```
curl -X DELETE http://localhost:8000/models/roberta/hf_20250416_163351
```

# 4 So sánh các mô hình embedding
```
curl -X POST http://localhost:8000/models/compare \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 10}'
```
```
curl -X GET http://localhost:8000/models/fine-tune/4474c114-032b-4fd0-a328-6e1dd90e8e65
```
```
curl -X GET http://localhost:8000/dbpedia/extract-data/c51a3967-f5d1-455a-9591-4f20dbb1d091 \
  -H "Content-Type: application/json"
```
# 5 Chạy script đánh giá:
Kết nối vào container chatbot-api
```
docker exec -it chatbot-api bash
```
Di chuyển đến thư mục src
```
cd /usr/src/app/src
```
Chạy script đánh giá mô hình (sẽ tự động tải NLTK resources)
```
python evaluate_models.py --data_path data/history_learning_data.json --output_dir results --sample 60 --limit 15
```

# 4. Sử dụng một mô hình cụ thể để trích xuất dữ liệu (sau khi đã load và index data)
```
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 5, "embedding_model": "roberta"}'
```