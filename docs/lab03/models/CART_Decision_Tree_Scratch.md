# CART Regression Decision Tree (Scratch)

## Purpose

CART Regression Tree học quan hệ phi tuyến bằng cách chia không gian feature thành các vùng. Mỗi lá dự báo trung bình target của các quan sát huấn luyện rơi vào lá đó. Mô hình có thể mô tả ngưỡng tải, tương tác giữa lag và thời điểm trong ngày, đồng thời ít yêu cầu chuẩn hóa feature. CART đơn là nền tảng để xây dựng Random Forest và Gradient Boosted Trees.

## Input - Output

**Input:** `X ∈ R^(n×p)`, `y ∈ R^n`; cấu hình gồm `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `max_thresholds`, `random_state` và giới hạn hàng train tùy chọn.

**Output:** cây nhị phân. Node nội bộ lưu `feature_index` và `threshold`; nhánh trái thỏa `x_j ≤ threshold`, nhánh phải thỏa `x_j > threshold`. Node lá lưu giá trị trung bình target. `predict(X)` trả vector dự báo; `feature_importances_` là tổng mức giảm SSE theo feature đã chuẩn hóa.

## How to do

1. Bắt đầu tại root với toàn bộ hàng train. Giá trị dự báo tạm thời của node là mean(target); parent loss là `SSE = Σ(y_i - mean(y))²`.
2. Dừng khi đạt `max_depth`, số hàng nhỏ hơn `min_samples_split`, hoặc SSE gần 0.
3. Chọn số feature ứng viên theo `max_features` (`all`, `sqrt`, `log2`, tỷ lệ hoặc số nguyên). Random Forest dùng lựa chọn ngẫu nhiên theo node.
4. Với từng feature ứng viên, tạo một số threshold bằng quantile trên dữ liệu của node. Implementation dùng quantile grid để giảm chi phí; đây là CART xấp xỉ, không quét mọi midpoint duy nhất.
5. Với từng threshold, tạo hai nhóm trái/phải; loại split nếu một bên có ít hơn `min_samples_leaf`.
6. Tính `gain = parent_SSE - left_SSE - right_SSE`. Chọn split có gain dương lớn nhất.
7. Đệ quy xây dựng hai node con. Nếu không có split hợp lệ, giữ node hiện tại là leaf.
8. Cộng gain theo feature qua các node, sau đó chuẩn hóa thành `feature_importances_`.
9. Khi suy luận, mỗi hàng đi theo các điều kiện `≤ threshold` cho đến leaf; trả giá trị leaf.

Cây có thể overfit nếu depth lớn hoặc leaf quá nhỏ. Giới hạn depth và kích thước leaf là điều chuẩn hóa chính. Giới hạn `max_train_rows` lấy mẫu đều theo thứ tự để giảm thời gian trên chuỗi dài; số hàng thực tế được ghi vào model summary.

## Evaluation Metrics

Trên Validation, Test và từng walk-forward fold: **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²**. Theo dõi thêm fit time, predict time, số hàng train thực dùng, depth, số lá và feature importance. RMSE Validation dùng để chọn champion; Test không được dùng để điều chỉnh depth/threshold.

## Used Data

Feature table hourly từ `AEP_MW`; predictor chỉ sử dụng lag/rolling đã dịch về quá khứ và calendar sin/cos. Trong mỗi CV fold, cây chỉ được fit trên prefix train của fold; validation block và partition bên ngoài không tham gia tìm split. Dữ liệu Test không tham gia xây cây.

## Implementation

- File: `src/lab03/lab03_models/decision_tree.py`
- Thành phần chính: `DecisionTreeRegressorScratch._build`, `_predict_node`, `fit`, `predict`.
- Phụ thuộc: NumPy và Python standard library; không dùng `DecisionTreeRegressor` của scikit-learn.
