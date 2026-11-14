# Tài liệu Dự án Day 7 — Agentic RAG / Retrieval Context

## 1. Tài liệu chuẩn của Môn học
- [Hướng dẫn chung](../GUIDE.md) · [Bộ tiêu chí Benchmark](../BENCHMARK.md) · [Thang điểm Rubric](../RUBRIC.md)
- [Mã nguồn Starter](../starter/README.md) · [Định dạng dữ liệu chuẩn](../data/README.md)
- [Quy tắc thực hiện](../RULES.md) · [Hướng dẫn nộp bài](../SUBMISSION.md)
- [Báo cáo cá nhân](../report/individual/2A202603471_NguyenVanTung.md)

---

## 2. Thông tin Dữ liệu & Benchmark (Bước 1)

### A. Dữ liệu Corpus & Chunks
- **Chủ đề:** Quy chế Đào tạo Trình độ Đại học theo Hệ thống Tín chỉ (Chính quy tập trung).
- **Văn bản gốc:** 36 trang nội dung hoàn chỉnh (loại bỏ bìa/mục lục), tổng cộng 35.010 ký tự Unicode NFC chuẩn hóa. Nguồn công khai (Public Domain theo Điều 19 Luật Sở hữu trí tuệ).
- **Cấu trúc Chunks:** Kế thừa từ Strategy B của Day 6 (cắt theo ranh giới đoạn văn bản `\n\n` hoặc điều khoản/khoản luật, `max_size = 800` ký tự).
  - Tổng số chunks: **73 chunks**.
  - File lưu trữ: [`data/chunks.json`](../data/chunks.json).
  - Metadata mỗi chunk: `id`, `source_id`, `source`, `page`, `chapter`, `section`, `article`, `char_length`.

### B. Vector Store & Embedding Model
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (chạy local qua PyTorch, embedding vector **384 chiều**, chuẩn hóa L2 norm).
- **Vector Database:** **ChromaDB v1.5.9** (Persistent Client lưu tại `starter/data/chroma_db`, HNSW index với Cosine similarity).
- **Chi phí phát sinh:** **$0.00** (chạy hoàn toàn trên máy local, không tốn phí API).

### C. Thiết kế Bộ câu hỏi Benchmark (24 câu)
Toàn bộ ca đánh giá được lưu tại [`data/cases.json`](../data/cases.json):

1. **Tập Dev (17 câu):**
   - **15 câu có đáp án:** `q01` đến `q15` (thuộc [`data/questions.json`](../data/questions.json)).
   - **02 câu thiếu bằng chứng (Unanswerable):**
     - `q_unans_dev_01`: Thời gian đào tạo tiêu chuẩn chương trình Tiến sĩ (quy chế chỉ có đại học, không có tiến sĩ).
     - `q_unans_dev_02`: Điểm rèn luyện nhận học bổng du học toàn phần Nhật Bản (không có trong quy chế).

2. **Tập Test (07 câu độc lập):**
   - **05 câu có đáp án:** `q16` đến `q20` (thuộc [`data/questions.json`](../data/questions.json)).
   - **02 câu thiếu bằng chứng (Unanswerable):**
     - `q_unans_test_01`: Mức phạt tiền mặt khi làm mất sách thư viện (không quy định biểu phí phạt thư viện).
     - `q_unans_test_02`: Số tín chỉ tích lũy tối thiểu nhận bằng Thạc sĩ (quy chế chỉ áp dụng bậc đại học).

---

## 3. Lệnh chạy và Kiểm thử Dữ liệu

```bash
# 1. Chạy sinh và xác thực dữ liệu từ Day 6
python starter/prepare_data.py

# 2. Chạy thử nghiệm baseline TF-IDF trên tập Dev
python starter/baseline.py --chunks data/chunks.json --questions data/questions.json --split dev --output starter/results/dev-tfidf

# 3. Chạy toàn bộ Unit Tests kiểm tra code
python -m unittest discover -s tests -v
```
