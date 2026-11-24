# BÀI TẬP THỰC HÀNH 3: DỰ BÁO CHUỖI THỜI GIAN BẰNG MACHINE LEARNING

## Mục tiêu học tập
1. Chuyển một chuỗi thời gian đơn biến thành bảng dữ liệu supervised.
2. Tạo lag features, rolling mean/std và biến lịch nhưng không làm rò rỉ dữ liệu tương lai.
3. Huấn luyện và so sánh mô hình tuyến tính/cây cho dự báo tải điện theo giờ.
4. Thực hiện validation theo thời gian, walk-forward cross-validation và đánh giá trên Test độc lập.
5. Đọc feature importance, residual diagnostics và phân biệt rolling one-step với recursive multi-step forecast.

## Bài thực hành

1. Đọc và làm sạch `AEP_hourly.csv` (`Datetime`, `AEP_MW`).
2. Tạo các lag `1, 2, 3, 6, 12, 24, 48, 168`.
3. Tạo rolling mean/std với cửa sổ `6, 12, 24, 168`, luôn shift một bước trước khi tổng hợp.
4. Mã hóa hour/day-of-week/month bằng sin/cos.
5. Chia Train/Validation/Test theo thời gian 70/15/15, không shuffle.
6. Thực hiện 5 expanding-window walk-forward folds bên trong Train.
7. So sánh Naive Last-Value, Linear Regression, CART, Random Forest và XGBoost-style booster tự cài đặt bằng NumPy.
8. Chọn champion bằng Validation RMSE; sau đó refit trên Train+Validation và đánh giá Test.
9. Thực hiện Test rolling one-step và dự báo tương lai recursive 24 giờ.
10. Xuất log, model summary, hình, report Markdown/HTML/PDF và tài liệu từng model/phase.

## Bài tập nâng cao và câu hỏi
- Phân tích tác động của số lượng lag đến feature count, khả năng dự báo và chi phí huấn luyện.
- So sánh mô hình tuyến tính và tree ensembles trên cùng timestamp.
- Giải thích vì sao split ngẫu nhiên không phù hợp với time series.
- Phân biệt dự báo rolling one-step với recursive fixed-origin multi-step.
- Giải thích regularization, subsample, column sampling, learning rate và depth của booster.

## Dataset
Hourly Energy Consumption — AEP region. Đặt file `AEP_hourly.csv` trong `data/raw/`; xem `data/raw/README.md` để biết schema và link nguồn trong repository gốc.
