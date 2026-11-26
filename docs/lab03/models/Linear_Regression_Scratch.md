# Hồi quy tuyến tính - Bình phương tối thiểu thông thường (tự cài đặt)

## Mục đích

Linear Regression thiết lập quan hệ tuyến tính giữa các đặc trưng lag/rolling/calendar và tải điện ở giờ kế tiếp. Đây là mô hình có thể diễn giải, tốc độ dự báo nhanh và cung cấp mốc tuyến tính để đánh giá liệu các tương tác phi tuyến mà cây quyết định học được có cải thiện kết quả hay không. Một hệ số dương của `lag_1`, ví dụ, biểu thị mối liên hệ tuyến tính dương giữa tải gần nhất và tải ở giờ kế tiếp khi giữ các biến khác cố định.

## Đầu vào - Đầu ra

**Đầu vào:** ma trận `X ∈ R^(n×p)` và biến mục tiêu `y ∈ R^n`; số hàng phải bằng nhau, `n > 0`, `p > 0`, dữ liệu là số hữu hạn. Hệ số ổn định Ridge `λ ≥ 0` là tham số cấu hình.

**Đầu ra:** vector hệ số `β ∈ R^p`, hệ số chặn `b`, thống kê chuẩn hóa của đặc trưng và dự báo `ŷ = Xβ + b`. Hệ số được chuyển ngược về thang đo gốc trước khi dự báo.

## Cách thực hiện

1. Tính trung bình và độ lệch chuẩn của từng đặc trưng trên đúng tập dữ liệu dùng để khớp. Đặc trưng có độ lệch chuẩn gần 0 được gán hệ số tỷ lệ bằng 1 để tránh chia cho 0.
2. Chuẩn hóa `Z = (X - μ_X) / σ_X`; center target `y_c = y - μ_y` để intercept không bị regularize.
3. Xây dựng ma trận Gram `G = ZᵀZ + λI` và vector phải `r = Zᵀy_c`.
4. Giải hệ phương trình chuẩn `Gβ_z = r` bằng `numpy.linalg.solve`. Nếu hệ suy biến, dùng pseudoinverse để có nghiệm số ổn định.
5. Chuyển về thang gốc: `β = β_z / σ_X`, `b = μ_y - μ_Xᵀβ`.
6. Khi dự báo, chỉ tính phép nhân ma trận `Xβ+b`; không tính lại trung bình/hệ số tỷ lệ từ tập Xác thực hoặc Kiểm tra.

Đây là cài đặt OLS có ridge rất nhỏ để ổn định ma trận, không phải wrapper của `sklearn.linear_model`. Chi phí chính là giải hệ tuyến tính, xấp xỉ `O(p³ + np²)`; với số feature lag nhỏ, phù hợp cho bài toán này.

## Chỉ số đánh giá

Báo cáo **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²** trên Validation/Test và từng fold. RMSE Validation được dùng để chọn champion. Ghi lại fit/predict time, ridge λ và hệ số. Với hệ số tuyến tính, so sánh độ lớn chỉ có ý nghĩa sau khi xét thang đo; pipeline chuẩn hóa khi fit và chuyển hệ số về scale gốc.

## Dữ liệu sử dụng

Dùng bảng có giám sát từ chuỗi `AEP_MW` theo giờ: `lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_168`, trung bình/độ lệch chuẩn trượt của cửa sổ 6/12/24/168 giờ và biến lịch sin/cos. Khi kiểm định chéo, thống kê chuẩn hóa và hệ số được khớp lại từ đầu trong từng fold. Tập Xác thực dùng để đánh giá; chỉ mở tập Kiểm tra sau khi khóa lựa chọn mô hình.

## Cài đặt

- File: `src/lab03/lab03_models/linear_regression.py`
- Thành phần chính: `LinearRegressionScratch.fit`, `predict`, `get_params`.
- Phụ thuộc: NumPy; không gọi estimator LinearRegression/Ridge của scikit-learn.
