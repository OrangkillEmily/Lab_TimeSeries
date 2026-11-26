# Cây quyết định hồi quy CART (tự cài đặt)

## Mục đích

CART Regression Tree học quan hệ phi tuyến bằng cách chia không gian feature thành các vùng. Mỗi lá dự báo trung bình target của các quan sát huấn luyện rơi vào lá đó. Mô hình có thể mô tả ngưỡng tải, tương tác giữa lag và thời điểm trong ngày, đồng thời ít yêu cầu chuẩn hóa feature. CART đơn là nền tảng để xây dựng Random Forest và Gradient Boosted Trees.

## Đầu vào - Đầu ra

**Đầu vào:** `X ∈ R^(n×p)`, `y ∈ R^n`; cấu hình gồm `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `max_thresholds`, `random_state` và giới hạn số dòng huấn luyện tùy chọn.

**Đầu ra:** cây nhị phân. Nút nội bộ lưu `feature_index` và `threshold`; nhánh trái thỏa `x_j ≤ threshold`, nhánh phải thỏa `x_j > threshold`. Nút lá lưu giá trị trung bình của biến mục tiêu. `predict(X)` trả vector dự báo; `feature_importances_` là tổng mức giảm SSE theo từng đặc trưng đã chuẩn hóa.

## Cách thực hiện

1. Bắt đầu tại nút gốc với toàn bộ dòng huấn luyện. Giá trị dự báo tạm thời của nút là trung bình biến mục tiêu; sai số của nút cha là `SSE = Σ(y_i - mean(y))²`.
2. Dừng khi đạt `max_depth`, số hàng nhỏ hơn `min_samples_split`, hoặc SSE gần 0.
3. Chọn số đặc trưng ứng viên theo `max_features` (`all`, `sqrt`, `log2`, tỷ lệ hoặc số nguyên). Rừng ngẫu nhiên chọn ngẫu nhiên đặc trưng tại mỗi nút.
4. Với từng đặc trưng ứng viên, tạo một số ngưỡng theo phân vị của dữ liệu tại nút. Cách cài đặt dùng lưới phân vị để giảm chi phí; đây là CART xấp xỉ, không quét mọi trung điểm duy nhất.
5. Với từng ngưỡng, tạo hai nhóm trái/phải; loại phép chia nếu một bên có ít hơn `min_samples_leaf`.
6. Tính `gain = parent_SSE - left_SSE - right_SSE`. Chọn split có gain dương lớn nhất.
7. Đệ quy xây dựng hai node con. Nếu không có split hợp lệ, giữ node hiện tại là leaf.
8. Cộng mức tăng theo đặc trưng qua các nút, sau đó chuẩn hóa thành `feature_importances_`.
9. Khi suy luận, mỗi hàng đi theo các điều kiện `≤ threshold` cho đến leaf; trả giá trị leaf.

Cây có thể quá khớp nếu độ sâu lớn hoặc lá quá nhỏ. Giới hạn độ sâu và kích thước lá là cách điều chuẩn hóa chính. `max_train_rows` chọn đều các dòng theo thứ tự thời gian để giảm thời gian trên chuỗi dài; số dòng thực tế được ghi vào tóm tắt mô hình.

## Chỉ số đánh giá

Trên Validation, Test và từng walk-forward fold: **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²**. Theo dõi thêm fit time, predict time, số hàng train thực dùng, depth, số lá và feature importance. RMSE Validation dùng để chọn champion; Test không được dùng để điều chỉnh depth/threshold.

## Dữ liệu sử dụng

Bảng đặc trưng theo giờ từ `AEP_MW`; biến dự báo chỉ dùng độ trễ/cửa sổ trượt thuộc quá khứ và đặc trưng lịch sin/cos. Trong mỗi fold CV, cây chỉ được khớp trên phần đầu tập huấn luyện; khối xác thực và các tập bên ngoài không tham gia tìm phép chia. Tập Kiểm tra không tham gia xây cây.

## Cài đặt

- File: `src/lab03/lab03_models/decision_tree.py`
- Thành phần chính: `DecisionTreeRegressorScratch._build`, `_predict_node`, `fit`, `predict`.
- Phụ thuộc: NumPy và Python standard library; không dùng `DecisionTreeRegressor` của scikit-learn.
