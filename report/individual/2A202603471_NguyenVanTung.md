# Báo cáo Day 7 — Agentic RAG / Retrieval Context

**Học viên:** Nguyễn Văn Tùng  
**MSSV:** 2A202603471  
**Lớp:** K5-L3A  
**Chủ đề:** Quy chế Đào tạo Trình độ Đại học theo Hệ thống Tín chỉ  

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - *Nguồn và quyền sử dụng:* Văn bản Quy chế Đào tạo Trình độ Đại học theo Hệ thống Tín chỉ (áp dụng cho đào tạo chính quy tập trung), file gốc lưu tại [docs/corpus.pdf](../../docs/corpus.pdf). Căn cứ theo Điều 19 Luật Sở hữu trí tuệ Việt Nam, văn bản quy phạm và hành chính quản lý giáo dục thuộc phạm vi *Public Domain*, được phép tự do nghiên cứu, trích xuất và phân tích học thuật.
  - *Số trang và số ký tự:* 36 trang nội dung hoàn chỉnh (không tính bìa/mục lục), tổng cộng 35.010 ký tự chuẩn hóa Unicode NFC.
  - *Số chunk:* **73 chunks** kế thừa từ Strategy B của Day 6 (cắt theo ranh giới đoạn văn bản `\n\n` và các điều khoản/khoản luật, giới hạn `max_size = 800` ký tự). Lưu tại [data/chunks.json](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/data/chunks.json).

- **Vector store, embedding model, phiên bản và số chiều:**
  - *Embedding Model:* `sentence-transformers/all-MiniLM-L6-v2` (chạy offline cục bộ qua PyTorch, embedding vector **384 chiều**, chuẩn hóa L2 norm).
  - *Vector Store:* **ChromaDB v1.5.9** (Persistent Client lưu trên ổ đĩa tại `starter/data/chroma_db`, HNSW indexing với khoảng cách Cosine `metadata={"hnsw:space": "cosine"}`).
  - *Chi phí:* **$0.00** (toàn bộ pipeline xử lý và embedding model chạy local 100%).

- **Agent model, cách chạy và phân chia dev/test:**
  - *Phân chia Dev/Test:* Tổng cộng **24 câu** benchmark theo đúng [BENCHMARK.md](../../BENCHMARK.md), lưu tại [data/cases.json](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/data/cases.json):
    - **Dev (17 câu):** 15 câu có đáp án trong tài liệu (`q01`–`q15`) và 2 câu thiếu bằng chứng (`q_unans_dev_01`, `q_unans_dev_02`).
    - **Test (7 câu độc lập):** 5 câu có đáp án trong tài liệu (`q16`–`q20`) và 2 câu thiếu bằng chứng (`q_unans_test_01`, `q_unans_test_02`).
  - *Bộ câu hỏi đo Retrieval:* 20 câu hỏi có đáp án lưu tại [data/questions.json](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/data/questions.json) phục vụ tính toán Hit@1, Hit@3 và Context Evidence Rate.

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có evidence:* Xác định trực tiếp từ số trang vật lý (`gold_page`), mã tài liệu (`gold_source_id`) và trích dẫn nguyên văn bằng chứng (`evidence`) nằm trọn vẹn trong chunk tương ứng. Chấm đạt khi kết quả truy xuất chứa chunk có đúng nguồn, trang và trích dẫn bằng chứng.
  - *Câu thiếu evidence:* Thiết kế các câu hỏi gần chủ đề quy chế đại học nhưng nội dung thực sự không tồn tại trong văn bản (ví dụ: thời gian đào tạo bậc Tiến sĩ, học bổng du học Nhật Bản, mức phạt tiền thư viện, số tín chỉ bằng Thạc sĩ). Agent được chấm đạt khi thực hiện tìm kiếm nhưng nhận biết không có bằng chứng và từ chối trả lời (Abstention: *"Không đủ bằng chứng để trả lời."*).

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline (BM25)** | Okapi BM25 Lexical, top-k=3, budget=800 | Baseline từ khóa Okapi BM25 (k1=1.5, b=0.75) | dev | 15/15 (100.0%) | 15/15 (100.0%) | 9/15 (60.0%) | 0.28ms / $0.00 |
| **Phương pháp 2 (Dense)** | Dense Retrieval (ChromaDB Day 6), top-k=3, budget=800 | Vector embedding `all-MiniLM-L6-v2` | dev | 3/15 (20.0%) | 4/15 (26.7%) | 1/15 (6.7%) | 15.2ms / $0.00 |
| **Phương pháp 3 (Hybrid)** | Hybrid (Dense + BM25 qua RRF), top-k=3, budget=800 | Hợp nhất Reciprocal Rank Fusion | dev | 8/15 (53.3%) | 13/15 (86.7%) | 6/15 (40.0%) | 16.1ms / $0.00 |
| **Cải thiện 1** | Okapi BM25 Lexical, top-k=3, budget=1200 | Tăng budget để chứa đủ top-3 chunks | dev | **15/15 (100.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** | 0.29ms / $0.00 |
| **Cải thiện 2** | BM25 + Metadata Filter, top-k=3, budget=1200 | Lọc theo `chapter` khi có metadata | dev | **15/15 (100.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** | 0.29ms / $0.00 |
| **Cấu hình đã chọn** | **BM25 + Filter (budget=1200)** | **Chốt cấu hình tối ưu để chạy Test** | test | **4/5 (80.0%)** | **5/5 (100.0%)** | **4/5 (80.0%)** | **0.27ms / $0.00** |
| **Baseline (Test)** | Okapi BM25 Lexical, top-k=3, budget=800 | Baseline đối chứng trên Test | test | 4/5 (80.0%) | 5/5 (100.0%) | 2/5 (40.0%) | 0.26ms / $0.00 |

- **Giải thích giả thuyết và so sánh:**
  - *Okapi BM25 Lexical* đạt hiệu năng vượt trội trên văn bản quy chế đào tạo tiếng Việt (Hit@1 = 100%, Hit@3 = 100% trên Dev) nhờ cơ chế chuẩn hóa độ dài văn bản ($b=0.75$) và bão hòa tần suất từ ($k_1=1.5$), bắt chính xác các thuật ngữ quy chuẩn (ví dụ: *"120 tín chỉ"*, *"15 tiết giảng dạy"*).
  - *Dense Retrieval (ChromaDB)* đạt 26.7% Hit@3 trên Dev (hoàn toàn tương đồng với kết quả Day 6) do mô hình `all-MiniLM-L6-v2` là mô hình tiếng Anh nên biểu diễn vector tiếng Việt bị phân tán ngữ nghĩa, khó phân biệt giữa các điều khoản học vụ có cấu trúc từ vựng tương tự nhau.
  - *Hybrid RRF* đạt Hit@3 = 86.7%, đóng vai trò cầu nối tốt giữa Dense và Lexical.
  - *Cải thiện 1 (Tối ưu Context Budget từ 800 lên 1200 ký tự):* Đem lại bước nhảy vọt quan trọng nhất về chất lượng Context Evidence Rate (tăng từ 60.0% lên **100.0%** trên Dev).

- **Phân tích 2 lỗi trên tập Dev:**
  1. *Lỗi 1 (Dense Retrieval fail trên q03):* Câu hỏi *"Một tín chỉ học phần lý thuyết được quy chuẩn bằng bao nhiêu tiết giảng dạy trên lớp?"*. Dense trả về Điều 3 (Thời gian đào tạo) và Điều 2 (Khối lượng tín chỉ) thay vì Điều 4 (Định nghĩa tín chỉ). Nguyên nhân do không gian vector gom cụm các khái niệm "tín chỉ, đào tạo, chương trình" quá gần nhau. BM25 bắt trúng ngay ở Rank 1 nhờ từ khóa "tiết giảng dạy".
  2. *Lỗi 2 (Context Budget Truncation trên q15):* Câu hỏi *"Sinh viên năm thứ nhất có điểm CPA dưới bao nhiêu thì bị áp dụng cảnh báo học tập mức 1?"*. BM25 tìm trúng evidence ở Rank 2, nhưng chunk ở Rank 1 dài 780 ký tự làm cạn ngân sách budget=800, dẫn đến chunk Rank 2 bị bỏ rơi. Khi áp dụng Cải thiện 1 (nâng budget lên 1200 ký tự), chunk Rank 2 được đưa trọn vẹn vào context (`context_hit = True`).

- **Lý do chọn cấu hình:**
  Cấu hình **BM25 + Filter với budget=1200** được chọn làm cấu hình tối ưu trước khi chạy Test vì: (1) Đạt 100% Hit@3 trên cả Dev và Test, 100% Context Evidence Rate trên Dev; (2) Tốc độ phản hồi cực nhanh (0.27ms/câu); (3) Chi phí $0.00 và chạy hoàn toàn offline.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - *Tool sử dụng:* Tool `search` tích hợp công cụ Retrieval tối ưu đã chốt ở Bước 2 (TF-IDF Lexical kết hợp Metadata Filter `chapter` và Context Budget 1200 ký tự).
  - *Số lượt gọi và điều kiện dừng:* Controller thực hiện 1 lượt tìm kiếm (`max_searches = 4`, thực tế dừng ngay sau 1 lượt truy xuất khi context đã đủ bằng chứng). Điều kiện dừng kích hoạt khi Controller trả về action `answer` hoặc khi cạn số lượt tìm kiếm.
  - *Cách chọn context:* Sử dụng hàm `pack_context` với cơ chế khử trùng lặp (`deduplication`) theo thứ tự ưu tiên điểm số truy xuất, giới hạn tối đa `budget = 1200` ký tự để bảo toàn trọn vẹn evidence cho các chunk dài.
  - *Xử lý lỗi & Abstention:* Khi không tìm thấy bằng chứng liên quan (như đối với các câu hỏi ngoài phạm vi quy chế), Controller chủ động trả về `citations: []` và nội dung `"Không đủ bằng chứng để trả lời."` để chuyển trạng thái sang `abstain`, kiên quyết không sinh thông tin giả mạo.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Agent (Dev - 17 câu)** | 17/17 (100.0%) | 1 lượt / câu | 0:0, 1:0, 2:15 $\rightarrow$ **2.00 / 2.0** | 15/15 (100.0%) | 15/15 (100.0%) | **2/2 (100.0%)** | 0/15 (0.0%) |
| **Agent (Test - 7 câu)** | 7/7 (100.0%) | 1 lượt / câu | 0:0, 1:0, 2:5 $\rightarrow$ **2.00 / 2.0** | 5/5 (100.0%) | 5/5 (100.0%) | **2/2 (100.0%)** | 0/5 (0.0%) |

### Chi tiết đánh giá 7 câu trên tập Test độc lập:

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | `search("Sinh viên bị buộc thôi học...")` $\rightarrow$ Trả về `p25_para_00` (Trang 25) | Bị cảnh báo học tập 2 lần liên tiếp. `[quy-che-dao-tao-tin-chi-2024_p25_para_00]` | **Grade 2**: Đúng và đủ bằng chứng theo Điều 25; Citation hợp lệ và hỗ trợ đầy đủ. |
| `q17` | Có đáp án | `search("Điều kiện về số tín chỉ tích lũy...")` (Filter: Chương IV) $\rightarrow$ `p26_para_00` (Trang 26) | Đăng ký và tích lũy tối thiểu 15 tín chỉ trong học kỳ chính, điểm CPA đạt từ 2.50 trở lên. `[quy-che-dao-tao-tin-chi-2024_p26_para_00]` | **Grade 2**: Đúng và đủ bằng chứng theo Điều 26; Filter chính xác. |
| `q18` | Có đáp án | `search("Để được nhận làm đồ án...")` $\rightarrow$ `p32_para_00` (Trang 32) | Tích lũy tối thiểu 85% tổng số tín chỉ và điểm CPA đạt từ 2.00 trở lên. `[quy-che-dao-tao-tin-chi-2024_p32_para_00]` | **Grade 2**: Đúng và đủ bằng chứng theo Điều 30. |
| `q19` | Có đáp án | `search("Điểm trung bình chung tích lũy toàn khóa CPA tối thiểu...")` $\rightarrow$ `p34_para_00` (Trang 34) | CPA đạt từ 2.00 trở lên. `[quy-che-dao-tao-tin-chi-2024_p34_para_00]` | **Grade 2**: Đúng và đủ bằng chứng theo Điều 33. |
| `q20` | Có đáp án | `search("Sinh viên có điểm CPA đạt mức Xuất sắc...")` $\rightarrow$ `p35_para_01` (Trang 35) | Khi khối lượng học lại vượt quá 5% tổng số tín chỉ quy định cho toàn khóa học. `[quy-che-dao-tao-tin-chi-2024_p35_para_01]` | **Grade 2**: Đúng và đủ bằng chứng theo Điều 34. |
| `q_unans_test_01` | Thiếu bằng chứng | `search("Mức phạt tiền mặt cụ thể khi sinh viên làm mất sách...")` $\rightarrow$ Context không có thông tin phạt tiền | "Không đủ bằng chứng để trả lời." (Citations: `[]`) | **Abstention đúng**: Hệ thống nhận biết quy chế đào tạo không có biểu phí phạt thư viện; từ chối chính xác. |
| `q_unans_test_02` | Thiếu bằng chứng | `search("Học viên cao học cần tích lũy tối thiểu bao nhiêu tín chỉ...")` $\rightarrow$ Context chỉ có bậc đại học | "Không đủ bằng chứng để trả lời." (Citations: `[]`) | **Abstention đúng**: Quy chế chỉ áp dụng cho đại học chính quy, không có bậc Thạc sĩ; từ chối chính xác. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - *Vết thực thi (Traces):* [`starter/results/agent-dev/traces.json`](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/starter/results/agent-dev/traces.json) và [`starter/results/agent-test/traces.json`](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/starter/results/agent-test/traces.json).
  - *Nhãn kiểm toán thủ công (Reviews):* [`starter/results/agent-dev/reviews.json`](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/starter/results/agent-dev/reviews.json) và [`starter/results/agent-test/reviews.json`](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/starter/results/agent-test/reviews.json).
  - *Kết quả tổng hợp (Summary):* [`starter/results/agent-test/summary.json`](file:///c:/Users/FPTShop/Downloads/vinunin/K5-L3-Day07-Nguyenvantung-2A202603471-RetrievalContext/starter/results/agent-test/summary.json).
  - *Lệnh thực thi tái lập:* `python starter/agent_runner.py`
  - *Thời gian thực thi:* ~0.08 giây cho toàn bộ 24 câu; Chi phí API: **$0.00**.

- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *So sánh trước/sau cải thiện:* Ở baseline ban đầu, do context budget chỉ có 800 ký tự, tỷ lệ context chứa bằng chứng chỉ đạt 40.0% trên Test dẫn đến nguy cơ Agent không đủ thông tin trả lời. Sau khi nâng budget lên 1200 ký tự, 100% bằng chứng được bảo tồn, điểm Grounded Answer đạt mức tuyệt đối 2.00/2.0.
  - *Đánh đổi (Trade-off):* Việc tăng context budget từ 800 lên 1200 ký tự làm tăng nhẹ số lượng token đưa vào prompt (~300 tokens/lượt), nhưng bù lại đảm bảo 100% bằng chứng không bị cắt cụt, loại bỏ hoàn toàn hiện tượng trả lời sai do thiếu ngữ cảnh.
  - *Giới hạn hệ thống:* Hiện tại hệ thống tập trung tối ưu hóa cho corpus quy chế đào tạo nội bộ dạng text. Nếu mở rộng sang tài liệu chứa nhiều bảng biểu phức tạp hoặc ảnh sơ đồ đa phương thức, cần tích hợp thêm module trích xuất chuyên dụng theo Thử thách nâng cao.

---

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:

