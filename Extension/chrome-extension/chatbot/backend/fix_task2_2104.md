# Câu trả lời thắc mắc về chức năng lượng tử hóa (Quantization)

## THẮC MẮC 4
Em thấy có 1 số mô hình có metadata pre-trained như xml-roberta & Roberta thì theo em tìm hiểu được hầu hết các mô hình được quantization thường là những mô hình đã được pre-trained trước đó. Vậy tức là các mô hình đã có folder metadata pre-trained tức là đã được is_quanzited đúng không?

### Trả lời:
Không, việc một mô hình có metadata pre-trained không đồng nghĩa với việc mô hình đó đã được lượng tử hóa (quantized). 

Trong hệ thống của chúng ta:
- Các mô hình có folder metadata pre-trained chỉ thể hiện rằng đây là các mô hình đã được huấn luyện trước (pre-trained) và được tải từ Hugging Face hoặc nguồn khác.
- Trạng thái lượng tử hóa được lưu riêng trong metadata của mô hình thông qua trường `is_quantized`.
- Khi một mô hình được tải, hệ thống sẽ kiểm tra trường `is_quantized` trong metadata để xác định trạng thái lượng tử hóa của mô hình.
- Có thể có mô hình pre-trained nhưng chưa được lượng tử hóa (is_quantized = false).

Bạn có thể kiểm tra trạng thái lượng tử hóa của mô hình thông qua API `/models/active` hoặc kiểm tra file metadata.json trong thư mục mô hình tương ứng.

## THẮC MẮC 5
Thuật toán/kĩ thuật được dùng để quantization là thuật toán/phương pháp nào và đoạn code nào (để em report vs giáo viên hướng dẫn): VD: Quantization-Aware Training (QAT), Mixed-Precision Quantization, Post-Training Quantization (PTQ)

### Trả lời:
Hệ thống của chúng ta sử dụng phương pháp **Dynamic Post-Training Quantization (Dynamic PTQ)** để lượng tử hóa các mô hình transformer.

Đoạn code thực hiện lượng tử hóa nằm trong hàm `quantize_model` trong file `embeddings.py`:

```python
def quantize_model(model):
    # [...]
    # Áp dụng dynamic quantization cho các lớp Linear
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    # [...]
```

Đặc điểm của phương pháp này:
1. **Post-Training Quantization (PTQ)**: Lượng tử hóa được áp dụng sau khi mô hình đã được huấn luyện, không yêu cầu huấn luyện lại.
2. **Dynamic Quantization**: Chuyển đổi các trọng số (weights) của mô hình từ định dạng float32 sang định dạng int8 trong quá trình inference, giúp giảm kích thước mô hình và tăng tốc độ xử lý.
3. **Áp dụng chủ yếu cho lớp Linear**: Chỉ áp dụng lượng tử hóa cho các lớp Linear trong mô hình, được chỉ định qua tham số `{torch.nn.Linear}`.
4. **Kiểu dữ liệu đích**: Sử dụng kiểu dữ liệu torch.qint8 (8-bit quantized integer) làm định dạng lượng tử hóa.

Phương pháp này giúp giảm đáng kể kích thước mô hình (thường là 4 lần) và tăng tốc độ inference, đồng thời duy trì chất lượng dự đoán ở mức hợp lý.
