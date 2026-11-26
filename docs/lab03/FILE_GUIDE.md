# Hướng dẫn cấu trúc tệp Lab 03

| Đường dẫn | Mục đích |
|---|---|
| `config.yaml` | Cấu hình riêng cho Lab 03: dữ liệu, cách chia, kiểm định chéo và siêu tham số mô hình |
| `src/lab03/pipeline.py` | Điều phối toàn bộ quy trình từ đầu đến cuối |
| `scripts/run_lab.py` | Lệnh chạy chung theo Lab |
| `script/lab03/run_lab03.py` | Trình khởi chạy pipeline Lab 03 |
| `src/lab03/data.py` | Đọc, làm sạch dữ liệu và chia theo thời gian |
| `src/lab03/features.py` | Tạo đặc trưng độ trễ, cửa sổ trượt, lịch và suy luận đệ quy |
| `src/lab03/cv.py` | Chia và đánh giá kiểm định chéo cửa sổ mở rộng |
| `src/lab03/metrics.py` | Các chỉ số hồi quy cài đặt bằng NumPy |
| `src/lab03/lab03_models/` | Các mô hình tự cài đặt, mỗi mô hình trong một tệp Python |
| `src/lab03/lab03_utils/report.py` | Tạo báo cáo Markdown/HTML/PDF |
| `src/lab03/plotting.py` | Tạo và lưu biểu đồ của pipeline |
| `data/raw/` | Dữ liệu gốc do người dùng cung cấp; pipeline không ghi đè |
| `data/processed/lab03/` | Chuỗi đã làm sạch, đặc trưng, dự báo và tệp kiểm toán |
| `models/lab03/` | Các mô hình đã khớp và được tuần tự hóa |
| `logs/lab03/` | Nhật ký có dấu thời gian cho từng lần chạy |
| `reports/lab03/` | Báo cáo mới nhất, báo cáo theo lần chạy, biểu đồ và ghi chú thí nghiệm cũ |
| `reports/model_summary/lab03/` | Tóm tắt mô hình CSV/JSON/Markdown và chỉ số theo fold/tập dữ liệu |
| `docs/lab03/MODEL_SUMMARY_SPEC.md` | Định nghĩa trường và cách diễn giải tóm tắt mô hình |
| `progress_results/lab03/` | Trạng thái lần chạy mới nhất dạng JSON |
| `docs/lab03/models/` | Tài liệu chi tiết cho từng thuật toán |
| `docs/lab03/phases/` | Mục đích, đầu vào/đầu ra, cách làm và đánh giá cho từng giai đoạn |
| `notebooks/lab03/Lab03_Visual_Exploration.ipynb` | Notebook trực quan hóa, đánh giá mô hình và dự báo Lab 03 |
| `tests/lab03/` | Kiểm thử đơn vị và trình chạy kiểm thử riêng cho Lab 03 |
