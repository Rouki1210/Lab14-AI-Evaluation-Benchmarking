# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Phiên bản agent:** Agent_V2_Optimized
- **Tổng số cases:** 55
- **Tỉ lệ Pass/Fail:** 54/1
- **Pass rate:** 98.2%
- **Điểm LLM-Judge trung bình:** 4.909 / 5.0
- **Retrieval metrics:**
  - Hit Rate: 100.0%
  - MRR: 98.2%
- **RAG-style metrics:**
  - Faithfulness: 1.000
  - Relevancy: 0.981
- **Multi-Judge agreement:** 100%
- **Latency trung bình:** 0.071 giây/case
- **Release gate:** APPROVE

Ghi chú: hệ thống đã cấu hình 2 judge model từ `.env`. Trong lần chạy hiện tại, môi trường bị chặn network nên judge dùng fallback heuristic; kết quả trong `benchmark_results.json` có trường `used_fallback` để theo dõi.

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Dấu hiệu | Nguyên nhân dự kiến |
|---|---:|---|---|
| Wrong top-1 ranking | 2 | Hit Rate = 1 nhưng MRR = 0.5 | Context đúng có trong top 3 nhưng không đứng đầu, nên agent lấy answer của case sai |
| Definition ambiguity | 2 | Câu hỏi về định nghĩa nhưng answer lấy sang điều/khoản khác | Các câu hỏi dùng nhiều từ chung như "cơ sở dữ liệu", "căn cước", làm overlap token bị nhiễu |
| Retrieval id quality | Đã cải thiện | Mỗi case đã có `expected_retrieval_ids` riêng dạng `law26_case_###` | Metric hiện phản ánh đúng hơn việc agent chọn sai case thay vì bị nhiễu do nhiều case dùng chung id |

## 3. Phân tích 5 Whys

### Case #1: Định nghĩa "Cơ sở dữ liệu quốc gia về dân cư"
1. **Symptom:** Agent trả lời về thông tin nhóm máu theo Điều 9 thay vì định nghĩa "Cơ sở dữ liệu quốc gia về dân cư".
2. **Why 1:** Retrieval đưa context đúng vào top 3 nhưng chưa đứng đầu: `retrieved_ids = ["law26_case_016", "law26_case_007", "law26_case_030"]`.
3. **Why 2:** Câu hỏi và context Điều 9 có nhiều token chung như "thông tin", "cơ sở dữ liệu", "quốc gia", "dân cư".
4. **Why 3:** Retriever hiện dùng keyword overlap, chưa phân biệt câu hỏi định nghĩa với câu hỏi liệt kê thông tin.
5. **Why 4:** Dù dataset đã có retrieval id riêng từng case, context định nghĩa vẫn ngắn và có nhiều từ khóa trùng với các điều liệt kê thông tin.
6. **Root Cause:** Retrieval baseline thiếu semantic ranking, khiến định nghĩa trong Điều 3 bị cạnh tranh với các điều có nhiều từ khóa giống nhau.

### Case #2: Định nghĩa "Giấy chứng nhận căn cước"
1. **Symptom:** Case vẫn pass nhưng chỉ đạt 3.0/5.0; answer nói về điều kiện cấp theo Điều 30 thay vì định nghĩa giấy chứng nhận căn cước.
2. **Why 1:** Context đúng nằm ở rank 2: `retrieved_ids = ["law26_case_043", "law26_case_009", "law26_case_006"]`.
3. **Why 2:** Cụm "giấy chứng nhận căn cước" xuất hiện mạnh ở Điều 30 về quản lý/cấp giấy, làm Điều 30 vượt lên trên định nghĩa ở Điều 3.
4. **Why 3:** Reranking định nghĩa đã cải thiện một phần nhưng chưa đủ mạnh với câu hỏi dùng cụm pháp lý xuất hiện ở nhiều điều.
5. **Why 4:** Agent vẫn lấy answer từ top-1 thay vì kiểm tra lại top-k khi câu hỏi là định nghĩa.
6. **Root Cause:** Thiếu bước answer selection theo intent trên toàn bộ top-k context.

### Case #3: Regression V1 vs V2
1. **Symptom:** V1 chỉ đạt 4.109, V2 đạt 4.909.
2. **Why 1:** V1 dùng scoring baseline đơn giản, ít ưu tiên question/metadata.
3. **Why 2:** V2 thêm reranking theo intent định nghĩa và dùng trọng số question/metadata cao hơn.
4. **Why 3:** Dataset pháp lý có nhiều từ khóa trùng nhau giữa các điều, nên cần ranking có ngữ cảnh.
5. **Why 4:** Keyword overlap thuần túy không đủ phân biệt intent định nghĩa, thủ tục, hiệu lực, hành vi cấm.
6. **Root Cause:** Chất lượng retrieval quyết định trực tiếp answer quality trong pipeline RAG.

## 4. Kế hoạch cải tiến (Action Plan)

- [x] Chuẩn hóa `expected_retrieval_ids` thành id riêng từng case, ví dụ `law26_case_001`.
- [ ] Nếu xây vector DB thật, tiếp tục tách chunk id theo điều/khoản, ví dụ `article_3_clause_6`, `article_3_clause_11`.
- [ ] Nâng retrieval từ keyword overlap sang BM25 hoặc TF-IDF để giảm nhiễu từ các từ chung.
- [x] Thêm reranking theo intent: nếu câu hỏi có "là gì", "thế nào là", "định nghĩa", ưu tiên context Điều 3 hoặc metadata định nghĩa.
- [ ] Khi expected context nằm trong top 3 nhưng không phải top 1, cho agent tổng hợp top-k thay vì trả lời duy nhất từ best match.
- [ ] Bổ sung stopwords pháp lý phổ biến như "luật", "căn cước", "quy định", "thông tin", "theo" để scoring tập trung vào từ khóa phân biệt.
- [ ] Chạy lại benchmark sau mỗi thay đổi và so sánh `hit_rate`, `mrr`, `relevancy`, `avg_score`.
- [ ] Khi chạy ở môi trường có network, xác nhận `used_fallback = false` cho 2 judge model trong `.env` trước khi nộp kết quả cuối.
