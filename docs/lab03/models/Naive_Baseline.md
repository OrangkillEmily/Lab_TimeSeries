# Mô hình đối chứng giá trị gần nhất

## Mục đích

Mô hình giá trị gần nhất là mốc đối chứng tối thiểu cho bài toán dự báo một bước. Với chuỗi tải điện theo giờ, giá trị dự báo tại thời điểm `t` chính là giá trị quan sát gần nhất tại `t-1`. Mô hình không học tham số; nó kiểm tra xem việc huấn luyện các mô hình phức tạp có cải thiện thực sự so với phương pháp giữ nguyên giá trị gần nhất hay không. Nếu các mô hình phức tạp không vượt mốc đối chứng trên cùng timestamp, độ phức tạp bổ sung chưa được chứng minh là hữu ích.

## Đầu vào - Đầu ra

**Đầu vào:** ma trận đặc trưng `X ∈ R^(n×p)`; cột đầu tiên bắt buộc là `lag_1`, tức `y(t-1)`. Vector `y ∈ R^n` được nhận trong `fit` để tuân theo giao diện thống nhất, nhưng mô hình đối chứng không dùng `y` để ước lượng tham số. Các giá trị trong `X` phải hữu hạn và giữ nguyên thứ tự cột khi tạo đặc trưng.

**Đầu ra:** vector `ŷ ∈ R^m`, trong đó `ŷ_i = X_i,0`. Mô hình chỉ lưu số đặc trưng đầu vào để kiểm tra giao diện; không lưu cây/hệ số.

## Cách thực hiện

1. Feature Engineering tạo các cột theo thứ tự lag tăng dần; vì vậy `lag_1` là cột đầu tiên.
2. `fit(X, y)` kiểm tra số chiều và ghi nhận số feature. Không thực hiện tối ưu hóa.
3. Với một nhóm dự báo, `predict(X)` chuẩn hóa đầu vào thành ma trận 2D nếu chỉ có một hàng.
4. Trả về bản sao của cột `X[:, 0]`; không đọc bất kỳ target tương lai nào.
5. Lưu chỉ số của mô hình đối chứng và so sánh với các ứng viên trên cùng timestamp.

Độ phức tạp dự báo là `O(m)` và bộ nhớ phụ là `O(m)` cho vector kết quả. Baseline gần như không có chi phí training.

## Chỉ số đánh giá

Sử dụng cùng bộ metric với các candidate: **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE** và **R²** trên Validation, Test và mỗi CV fold. Báo cáo RMSE là tiêu chí chọn champion; baseline được dùng để tính mức cải thiện tương đối `(RMSE_baseline - RMSE_model) / RMSE_baseline`. MAPE bỏ qua các mẫu có target bằng 0; khi target gần 0 cần đọc thêm MAE/sMAPE.

## Dữ liệu sử dụng

Dùng `AEP_hourly.csv`, biến mục tiêu `AEP_MW`, tần suất một giờ. Đặc trưng đầu vào chỉ chứa lịch sử đã có tại thời điểm dự báo. Với dữ liệu tổng hợp, tên bộ dữ liệu và mọi báo cáo phải giữ nhãn `synthetic_demo`; không được diễn giải thành kết quả tải điện AEP.

## Cài đặt

- File: `src/lab03/lab03_models/naive_baseline.py`
- Thành phần chính: `NaiveLastValue.fit`, `NaiveLastValue.predict`.
- Phụ thuộc: NumPy; không gọi estimator hay metric của scikit-learn/XGBoost.
