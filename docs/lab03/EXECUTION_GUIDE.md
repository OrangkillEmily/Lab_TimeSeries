# Hướng dẫn chạy Lab 03

## Yêu cầu

- Python 3.10 trở lên.
- Các thư viện trong `requirements.txt` ở thư mục gốc.
- Tệp dữ liệu `data/raw/AEP_hourly.csv`.

## Chạy pipeline bằng một lệnh

Chạy từ thư mục gốc của kho mã:

```bash
python -m pip install -r requirements.txt
python scripts/run_lab.py --lab 03
python scripts/run_lab.py --lab 03 --fast
python scripts/run_lab.py --lab 03 --no-cv
python script/lab03/run_lab03.py [--fast] [--no-cv] [--run-id my_run]
```

Lệnh đầy đủ đọc và làm sạch dữ liệu, tạo đặc trưng độ trễ/cửa sổ trượt/lịch, chia dữ liệu theo thời gian 70/15/15, chạy kiểm định chéo cửa sổ mở rộng 5 fold trong tập Huấn luyện, chọn mô hình theo RMSE xác thực, đánh giá một lần trên tập Kiểm tra và dự báo đệ quy 24 giờ. Pipeline lưu các mô hình, biểu đồ, tóm tắt mô hình và báo cáo Markdown/HTML/PDF. Mỗi lần chạy cũng ghi nhật ký và trạng thái tiến trình.

Tùy chọn `--fast` dùng cấu hình giảm nhẹ để kiểm tra nhanh. Tùy chọn `--no-cv` bỏ qua kiểm định chéo; không dùng chế độ này nếu yêu cầu kết quả đủ năm fold.

## Chạy notebook trực quan hóa

Từ thư mục gốc của kho mã:

```bash
python -m jupyter lab
```

Mở `notebooks/lab03/Lab03_Visual_Exploration.ipynb`, chọn kernel Python phù hợp rồi chạy **Run All**. Notebook tự tìm thư mục gốc từ thư mục làm việc hiện tại và báo lỗi rõ ràng nếu thiếu tệp AEP; notebook không tự thay dữ liệu thật bằng dữ liệu tổng hợp.

Notebook tạo 19 biểu đồ gồm khám phá dữ liệu, chia tập, tương quan đặc trưng, kiểm định chéo, so sánh mô hình, chẩn đoán phần dư, ảnh hưởng đặc trưng và dự báo 24 giờ. Biểu đồ được lưu trong:

```text
reports/lab03/notebook_runs/notebook_visual_exploration/
```

## Kết quả pipeline

Mỗi lần chạy pipeline tạo thư mục riêng:

```text
reports/lab03/notebook_runs/<run_id>/
```

Thư mục gồm dữ liệu đã làm sạch, chỉ số CV/Xác thực/Kiểm tra, dự báo, tóm tắt lần chạy, biểu đồ và mô hình đã khớp. Báo cáo mới nhất được ghi tại `reports/lab03/` dưới dạng `Lab03_Report.md`, `Lab03_Report.html` và `Lab03_Report.pdf`; biểu đồ trong báo cáo dẫn đến đúng thư mục lần chạy. Tóm tắt mô hình được ghi tại `reports/model_summary/lab03/`.

Kết quả trên tập Kiểm tra chỉ được tính sau khi khóa lựa chọn mô hình. Đánh giá trên tập Kiểm tra là dự báo cuốn chiếu trước một bước; dự báo 24 giờ từ mốc cố định là quy trình đệ quy riêng.
