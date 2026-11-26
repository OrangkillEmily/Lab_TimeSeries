# Kiến trúc và luồng dữ liệu Lab 03

```text
AEP_hourly.csv
    |
    v
Nạp dữ liệu -> Làm sạch dữ liệu -> Chuỗi giờ đã chuẩn hóa
    |                                  |
    +----------------------------------+
                       |
                       v
             Đặc trưng độ trễ/cửa sổ trượt/lịch
                       |
                       v
          Chia theo thời gian 70/15/15
              /           |          \
             v            v           v
       Huấn luyện      Xác thực      Kiểm tra
             |
             +--> Kiểm định chéo cửa sổ mở rộng 5 fold
             |
             +--> Khớp các mô hình ứng viên
                          |
                          v
              Chọn theo RMSE xác thực
                          |
                          v
             Khớp lại trên Huấn luyện + Xác thực
                          |
                          v
                 Đánh giá trên tập Kiểm tra
                          |
              +-----------+-----------+
              |                       |
              v                       v
      Dự báo cuốn chiếu trên Test   Dự báo tương lai đệ quy 24 bước
              |                       |
              +-----------+-----------+
                          |
                          v
      Chỉ số + Tóm tắt mô hình + Biểu đồ + Nhật ký + Báo cáo
```

## Phân chia trách nhiệm

- `pipeline.py` điều phối các giai đoạn và quản lý vòng đời mỗi lần chạy.
- `src/lab03/data.py` xử lý đọc dữ liệu thô, chuẩn hóa tần suất và chia dữ liệu.
- `src/lab03/features.py` tạo đặc trưng chỉ dựa trên thông tin quá khứ.
- `src/lab03/cv.py` tạo các fold cửa sổ mở rộng.
- `src/lab03/lab03_models/` chứa các mô hình; mỗi mô hình được tách thành một mô-đun.
- `src/lab03/metrics.py` tính chỉ số độc lập với mô hình.
- `src/lab03/plotting.py` lưu các sản phẩm trực quan.
- `src/lab03/lab03_utils/report.py` tạo báo cáo Markdown, HTML và PDF.

## Biện pháp ngăn rò rỉ dữ liệu

1. Chia theo thứ tự thời gian, không xáo trộn ngẫu nhiên.
2. Kiểm định chéo chỉ dùng tập Huấn luyện; mỗi fold xác thực trên đoạn sau phần dữ liệu dùng để huấn luyện.
3. Đặc trưng độ trễ/cửa sổ trượt chỉ dùng quan sát trước timestamp mục tiêu.
4. Mỗi fold khởi tạo mô hình mới và chỉ khớp trên phần huấn luyện của fold đó.
5. Chọn mô hình theo RMSE trên tập Xác thực; không dùng tập Kiểm tra để chọn hoặc tinh chỉnh mô hình.
6. Dự báo tương lai đệ quy đưa dự báo vào lịch sử và không đọc giá trị thực trong tương lai.

## Nguồn gốc thí nghiệm

Các kết quả notebook cũ được lưu riêng trong `reports/lab03/legacy/`. Quy trình chia dữ liệu, mô hình thư viện và cách triển khai trước đây khác pipeline tự cài đặt hiện tại; vì vậy hai nhóm kết quả được tách biệt trong báo cáo.
