# Naive Last-Value Baseline

## Purpose

Naive Last-Value là mốc đối chứng tối thiểu cho bài toán dự báo một bước. Với chuỗi tải điện hourly, giá trị dự báo tại thời điểm `t` chính là giá trị quan sát gần nhất tại `t-1`. Mô hình không học tham số; nó kiểm tra xem việc huấn luyện các mô hình phức tạp có tạo ra cải thiện thực sự so với persistence hay không. Nếu các mô hình phức tạp không vượt baseline trên cùng các timestamp, độ phức tạp bổ sung chưa được chứng minh là hữu ích.

## Input - Output

**Input:** ma trận đặc trưng `X ∈ R^(n×p)`; cột đầu tiên bắt buộc là `lag_1`, tức `y(t-1)`. Vector `y ∈ R^n` được nhận trong `fit` để tuân theo giao diện đồng nhất, nhưng baseline không dùng `y` để ước lượng tham số. Các giá trị trong `X` phải hữu hạn và cùng thứ tự cột như lúc tạo feature.

**Output:** vector `ŷ ∈ R^m`, trong đó `ŷ_i = X_i,0`. Mô hình chỉ lưu số feature đầu vào để kiểm tra giao diện; không lưu cây/hệ số.

## How to do

1. Feature Engineering tạo các cột theo thứ tự lag tăng dần; vì vậy `lag_1` là cột đầu tiên.
2. `fit(X, y)` kiểm tra số chiều và ghi nhận số feature. Không thực hiện tối ưu hóa.
3. Với một batch dự báo, `predict(X)` chuẩn hóa input thành ma trận 2D nếu chỉ có một hàng.
4. Trả về bản sao của cột `X[:, 0]`; không đọc bất kỳ target tương lai nào.
5. Lưu metric của baseline và so sánh với các candidate trên đúng cùng timestamp.

Độ phức tạp dự báo là `O(m)` và bộ nhớ phụ là `O(m)` cho vector kết quả. Baseline gần như không có chi phí training.

## Evaluation Metrics

Sử dụng cùng bộ metric với các candidate: **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE** và **R²** trên Validation, Test và mỗi CV fold. Báo cáo RMSE là tiêu chí chọn champion; baseline được dùng để tính mức cải thiện tương đối `(RMSE_baseline - RMSE_model) / RMSE_baseline`. MAPE bỏ qua các mẫu có target bằng 0; khi target gần 0 cần đọc thêm MAE/sMAPE.

## Used Data

Dùng `AEP_hourly.csv`, target `AEP_MW`, frequency một giờ. Input feature chỉ chứa lịch sử đã có tại thời điểm dự báo. Với dữ liệu synthetic, tên dataset và mọi báo cáo phải giữ nhãn `synthetic_demo`; không được diễn giải thành kết quả tải điện AEP.

## Implementation

- File: `src/lab03/lab03_models/naive_baseline.py`
- Thành phần chính: `NaiveLastValue.fit`, `NaiveLastValue.predict`.
- Phụ thuộc: NumPy; không gọi estimator hay metric của scikit-learn/XGBoost.
