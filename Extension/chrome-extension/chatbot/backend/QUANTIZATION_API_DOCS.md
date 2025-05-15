# Tài liệu API lượng tử hóa mô hình (Model Quantization)

## Giới thiệu

Tài liệu này mô tả chức năng API mới để quản lý việc lượng tử hóa (quantization) các mô hình transformer trong hệ thống. Lượng tử hóa là quá trình giảm độ chính xác của các trọng số trong mô hình nhằm giảm kích thước mô hình và tăng tốc độ inference mà vẫn duy trì chất lượng dự đoán ở mức chấp nhận được.

## API mới: `/models/quantize`

### Mô tả
API này cho phép người dùng chủ động kích hoạt quá trình lượng tử hóa cho một mô hình transformer cụ thể, giúp giảm kích thước mô hình và tăng tốc độ inference.

### Phương thức
POST

### Tham số
Yêu cầu body JSON với cấu trúc:
```json
{
  "model_type": "<tên_mô_hình>"
}
```

Trong đó `<tên_mô_hình>` có thể là một trong các giá trị sau:
- `roberta` - Mô hình RoBERTa
- `xlm-roberta` - Mô hình XLM-RoBERTa
- `distilbert` - Mô hình DistilBERT

### Phản hồi

#### Thành công - Mô hình được lượng tử hóa
```json
{
  "status": "success",
  "message": "Successfully quantized <tên_mô_hình> model",
  "is_quantized": true
}
```

#### Thành công - Mô hình đã được lượng tử hóa trước đó
```json
{
  "status": "success",
  "message": "Model <tên_mô_hình> is already quantized",
  "already_quantized": true
}
```

#### Lỗi - Mô hình không tồn tại hoặc không được hỗ trợ
```json
{
  "status": "error",
  "message": "Model <tên_mô_hình> not found or not supported for quantization"
}
```

#### Lỗi - Lỗi trong quá trình lượng tử hóa
```json
{
  "status": "error",
  "message": "Error during quantization: <chi_tiết_lỗi>"
}
```

## Cách kiểm tra trạng thái lượng tử hóa

### Sử dụng API `/models/active`

Gửi yêu cầu GET đến API `/models/active` và kiểm tra trường `is_quantized` trong metadata của mô hình:

```json
{
  "roberta": {
    "status": "active",
    "metadata": {
      "model_name": "roberta-base",
      "is_quantized": true
    }
  },
  "xlm-roberta": {
    "status": "active",
    "metadata": {
      "model_name": "xlm-roberta-base",
      "is_quantized": false
    }
  }
}
```

## Các thay đổi kỹ thuật

### Cải tiến quá trình lượng tử hóa

1. **Tách biệt quá trình lượng tử hóa**: Việc lượng tử hóa giờ đây được xử lý thông qua API riêng biệt thay vì được thực hiện tự động trong quá trình tải mô hình.

2. **Xử lý race condition**: Các sửa đổi giúp tránh race condition trong quá trình khởi động container, khi nhiều mô hình cố gắng lượng tử hóa cùng lúc.

3. **Quản lý lỗi tốt hơn**: Hệ thống ghi log chi tiết hơn về quá trình lượng tử hóa và các lỗi liên quan.

4. **Kiểm tra trạng thái trước khi lượng tử hóa**: Hệ thống kiểm tra xem mô hình đã được lượng tử hóa chưa trước khi tiến hành quá trình lượng tử hóa, tránh thực hiện dư thừa.

### Thực hiện lượng tử hóa

Hệ thống sử dụng phương pháp Dynamic Post-Training Quantization (Dynamic PTQ) thông qua thư viện `torch.quantization`:

```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

Phương pháp này chỉ lượng tử hóa các lớp Linear trong mô hình với kiểu dữ liệu `torch.qint8` (8-bit integer), giúp:
- Giảm kích thước mô hình khoảng 4 lần
- Tăng tốc độ inference
- Duy trì chất lượng dự đoán ở mức hợp lý

### Lưu trữ metadata

Khi một mô hình được lượng tử hóa, hệ thống cập nhật trường `is_quantized` trong file metadata.json của mô hình:

```json
{
  "model_name": "roberta-base",
  "model_type": "roberta",
  "version": "hf_20250417_084022",
  "created_at": "2025-04-17T08:40:22.516901",
  "is_quantized": true,
  "source": "HuggingFace (downloaded)",
  "dimensions": 768
}
```

## Sử dụng API quantize

### Ví dụ: Lượng tử hóa mô hình RoBERTa

```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "roberta"}'
```

### Ví dụ: Lượng tử hóa mô hình XLM-RoBERTa

```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xlm-roberta"}'
```

## Lợi ích của lượng tử hóa

1. **Tiết kiệm bộ nhớ**: Mô hình lượng tử hóa có kích thước nhỏ hơn, giảm yêu cầu về RAM khi chạy.
2. **Tăng tốc độ xử lý**: Tính toán với số nguyên nhanh hơn so với số thực, dẫn đến tốc độ inference nhanh hơn.
3. **Hiệu quả năng lượng**: Tiêu thụ ít năng lượng hơn, phù hợp cho các thiết bị có giới hạn về nguồn điện.
4. **Hiệu suất tương đương**: Trong nhiều trường hợp, mô hình lượng tử hóa vẫn duy trì hiệu suất gần tương đương với mô hình gốc. 