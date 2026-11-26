# Giai đoạn kiểm định chéo - Cửa sổ mở rộng / cuốn chiếu

## Mục đích

Ước lượng độ ổn định của các thuật toán qua nhiều giai đoạn lịch sử. Với time series, K-fold ngẫu nhiên có thể huấn luyện trên thời điểm tương lai rồi validate trên thời điểm quá khứ; vì vậy Lab 3 dùng expanding-window walk-forward CV có thứ tự thời gian.

## Đầu vào - Đầu ra

**Đầu vào:** chỉ `X_train/y_train` (70% đầu chuỗi), `n_splits=5`, các hàm tạo mô hình, tùy chọn khoảng cách thời gian.

**Đầu ra:** CSV theo fold gồm mô hình, số fold, kích thước tập huấn luyện/xác thực và các chỉ số; tổng hợp trung bình/độ lệch chuẩn theo mô hình tại `reports/model_summary/lab03/`.

## Cách thực hiện

1. Chia tập Huấn luyện thành một đoạn khởi tạo và năm khối xác thực liên tục.
2. Fold 1 fit trên prefix đầu rồi validate block kế tiếp.
3. Mỗi fold tiếp theo mở rộng phần đầu tập huấn luyện để bao gồm lịch sử của các fold trước, còn khối xác thực dịch về phía trước.
4. Khởi tạo mô hình mới cho từng cặp mô hình/fold; mọi tham số học được chỉ khớp trên phần huấn luyện của fold.
5. Dự báo khối xác thực theo giao thức cuốn chiếu trước một bước: đặc trưng tại mỗi timestamp chỉ chứa lịch sử trước thời điểm đó.
6. Tính chỉ số trên khối xác thực, ghi nhật ký fold và tổng hợp trung bình/độ lệch chuẩn của các chỉ số.
7. Không sử dụng 15% Validation hoặc 15% Test bên ngoài trong CV.

Đây là kiểm định chéo chuỗi thời gian với cửa sổ mở rộng, không phải `KFold(shuffle=True)`. Các fold kiểm tra độ ổn định theo thời gian nhưng không thay thế tập Xác thực dùng để chọn mô hình.

## Chỉ số đánh giá

Từng fold: **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²**. Tổng hợp mean/std, đặc biệt RMSE; std cao thể hiện mô hình nhạy với giai đoạn. Theo dõi train/validation row count và tính tăng dần của training window.

## Dữ liệu sử dụng

Chỉ 70% đầu tiên của supervised rows. Fold validation luôn đứng sau fold train. Không dùng thông tin từ partition Validation/Test cuối cùng hoặc tương lai so với fold đang xét.

## Tham chiếu cài đặt

- `src/lab03/cv.py:expanding_window_splits`
- `src/lab03/cv.py:evaluate_walk_forward`
- `pipeline.py`
