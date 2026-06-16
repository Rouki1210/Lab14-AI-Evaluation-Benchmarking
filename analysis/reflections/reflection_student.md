# Individual Reflection - Student

## 1. Engineering Contribution
- Xây dựng baseline RAG agent trong `agent/main_agent.py` để đọc golden dataset, retrieve top-k context và trả lời theo case phù hợp nhất.
- Hoàn thiện benchmark runner flow qua `main.py`, tạo `reports/summary.json` và `reports/benchmark_results.json`.
- Triển khai retrieval evaluation trong `engine/retrieval_eval.py` với Hit Rate@K và MRR.
- Triển khai multi-judge consensus trong `engine/llm_judge.py` dùng 2 model cấu hình từ `.env`, có agreement rate và conflict handling.
- Viết báo cáo failure analysis trong `analysis/failure_analysis.md` dựa trên case fail thật.

## 2. Technical Depth
- **Hit Rate@K:** đo xem ít nhất một tài liệu/chunk đúng có nằm trong top-k retrieval hay không.
- **MRR:** đo tài liệu đúng xuất hiện sớm hay muộn trong ranking; tài liệu đúng rank 1 có MRR = 1.0, rank 2 có MRR = 0.5.
- **Agreement Rate:** đo mức đồng thuận giữa 2 judge model; nếu score lệch lớn thì hệ thống dùng logic conflict resolution bảo thủ hơn.
- **Position Bias:** rủi ro judge thiên vị câu trả lời đứng trước/sau; có thể kiểm tra bằng cách đảo vị trí các đáp án khi pairwise judging.
- **Cost vs Quality:** dùng async batch và fallback heuristic để giảm chi phí; khi cần chất lượng cao hơn có thể chỉ gọi LLM judge cho case khó hoặc case gần ngưỡng pass/fail.

## 3. Problem Solving
- Dataset trong repo có thể ở dạng JSON array hoặc JSONL, nên loader được viết để hỗ trợ cả hai format.
- Môi trường hiện tại chặn network, nên multi-judge có fallback heuristic và ghi rõ `used_fallback` trong report.
- Failure analysis chỉ ra lỗi chính nằm ở retrieval/ranking cho câu hỏi định nghĩa pháp lý, không phải ở generation.

## 4. Next Improvements
- Tách chunk id chi tiết theo điều/khoản, ví dụ `article_3_clause_11`.
- Thay keyword overlap bằng BM25, TF-IDF hoặc embedding retrieval.
- Thêm reranking theo intent: định nghĩa, thủ tục, hiệu lực, hành vi bị cấm.
- Chỉ gọi LLM judge cho case fail, case score thấp hoặc case có disagreement để giảm chi phí.
