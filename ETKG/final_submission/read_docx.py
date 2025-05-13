import docx
import json

# Đọc file docx
doc_path = "/home/ubuntu/upload/KG Requirement Doc - V1 - Deadline_ 11_00 A.M 14.4.2025.docx"
try:
    doc = docx.Document(doc_path)
    content = []
    for para in doc.paragraphs:
        if para.text.strip():
            content.append(para.text)
    
    # Lưu nội dung vào file text để dễ đọc
    with open("/home/ubuntu/requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(content))
    
    print("Đã đọc và lưu nội dung tài liệu thành công!")
except Exception as e:
    print(f"Lỗi khi đọc file docx: {e}")
