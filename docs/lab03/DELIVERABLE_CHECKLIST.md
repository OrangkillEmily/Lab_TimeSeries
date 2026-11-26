# Danh sách kiểm tra sản phẩm bàn giao Lab 03

## Pipeline chạy bằng một lệnh

- [x] `src/lab03/pipeline.py` điều phối toàn bộ quy trình; `script/lab03/run_lab03.py` và `scripts/run_lab.py --lab 03` gọi chung pipeline. Hỗ trợ `--fast`, `--no-cv` và `--run-id`.
- [x] Cấu hình `lab03` có dữ liệu, đặc trưng, dự báo, mô hình và các ghi đè dùng cho chạy nhanh; `src/lab03/utils.py` cung cấp các hàm cấu hình/đường dẫn/thư mục.
- [x] `src/lab03/logging_utils.py` ghi nhật ký có dấu thời gian và cập nhật trạng thái tiến trình sau từng giai đoạn.
- [x] Pipeline chạy trên `AEP_hourly.csv`, thực hiện chia dữ liệu theo thời gian, chọn mô hình trên tập Xác thực, đánh giá trên tập Kiểm tra và tạo dự báo tương lai.
- [x] Kiểm thử pipeline xác nhận trạng thái hoàn tất, mô hình được chọn, nhật ký/trạng thái, báo cáo, mô hình đã lưu và biểu đồ.

## Notebook khám phá trực quan

- [x] [Notebook Lab03_Visual_Exploration](../../notebooks/lab03/Lab03_Visual_Exploration.ipynb) dùng trực tiếp các mô-đun hiện có trong `src/lab03/`.
- [x] Notebook đọc dữ liệu thật từ `data/raw/AEP_hourly.csv`; không tự chuyển sang dữ liệu tổng hợp nếu thiếu tệp.
- [x] Thực hiện chia 70/15/15 theo thời gian, kiểm định chéo năm fold trong tập Huấn luyện, chọn mô hình bằng tập Xác thực và đánh giá trên tập Kiểm tra.
- [x] Đặc trưng độ trễ/cửa sổ trượt chỉ dùng dữ liệu quá khứ; kết quả kiểm tra cuốn chiếu được tách khỏi dự báo tương lai đệ quy 24 giờ.
- [x] Tạo 19 biểu đồ, lưu dưới `reports/lab03/notebook_runs/notebook_visual_exploration/`.

## Báo cáo và sản phẩm đầu ra

- [x] Báo cáo Markdown, HTML và PDF được tạo bằng tiếng Việt.
- [x] Báo cáo mới nhất trỏ đến biểu đồ trong đúng thư mục của lần chạy tương ứng.
- [x] Tóm tắt mô hình Markdown có tiêu đề/cột tiếng Việt; CSV/JSON giữ nguyên tên trường để tương thích xử lý dữ liệu.
- [x] Kết quả notebook cũ được lưu riêng trong `reports/lab03/legacy/`, không trộn với kết quả pipeline hiện tại.

## Xác minh

- Chạy `python script/lab03/run_lab03.py` hoặc `python scripts/run_lab.py --lab 03` để thực hiện pipeline đầy đủ.
- Chạy `python script/lab03/run_lab03.py --fast` để kiểm tra nhanh.
- Chạy `python -m unittest tests.lab03.test_lab03` để kiểm tra các thành phần Lab 03.
- Chạy notebook bằng **Run All** trong Jupyter để kiểm tra toàn bộ ô và biểu đồ.
