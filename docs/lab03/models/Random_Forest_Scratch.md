# Random Forest Regressor (Scratch)

## Purpose

Random Forest kết hợp nhiều CART được huấn luyện trên các bootstrap sample khác nhau. Việc lấy mẫu hàng có hoàn lại và chỉ xét một tập con feature tại mỗi node làm giảm tương quan giữa cây; trung bình dự báo của nhiều cây thường giảm variance so với một cây riêng lẻ. Với dữ liệu tải điện, mô hình có thể nắm bắt quan hệ phi tuyến và tương tác lag/calendar mà không yêu cầu scale đồng nhất.

## Input - Output

**Input:** `X ∈ R^(n×p)`, `y ∈ R^n` và các tham số: số cây `n_estimators`, depth, giới hạn split/leaf, `max_features`, tỷ lệ bootstrap `max_samples`, số threshold, seed và giới hạn hàng train.

**Output:** danh sách `B` cây CART scratch; dự báo là trung bình `ŷ(x) = (1/B) Σ_b T_b(x)`. Feature importance là trung bình importance từ các cây, sau đó chuẩn hóa.

## How to do

1. Khởi tạo random generator bằng `random_state` để có thể tái lập.
2. Nếu train vượt `max_train_rows`, chọn các vị trí đều trên khoảng thời gian huấn luyện; ghi số hàng được dùng vào `n_training_rows_`.
3. Với mỗi cây `b`, lấy `ceil(n × max_samples)` chỉ số từ tập train bằng sampling có hoàn lại (bootstrap).
4. Khởi tạo một `DecisionTreeRegressorScratch` độc lập. Tại mỗi node của CART, chọn ngẫu nhiên số feature theo `max_features` (thường là `sqrt(p)`).
5. Fit cây trên bootstrap sample; không cây nào được nhìn thấy Validation/Test.
6. Cộng feature importance của cây vào tổng ensemble.
7. Khi predict, lấy dự báo của từng cây trên cùng ma trận X và tính trung bình theo cây.
8. Chuẩn hóa tổng feature importance; lưu toàn bộ cây để tái sử dụng khi inference.

Random Forest có chi phí training xấp xỉ tổng chi phí của B cây; dự báo cần chạy qua B cây. Tăng số cây thường làm kết quả ổn định hơn nhưng làm tăng thời gian và kích thước model. Tính ngẫu nhiên được kiểm soát bởi seed cố định.

## Evaluation Metrics

Báo cáo **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²** trên Validation/Test và 5 CV folds. Báo cáo mean/std giữa fold, số cây, depth, fit/predict time, số mẫu train thực dùng và importance. So sánh trực tiếp với CART trên cùng timestamp để xem ensemble có giảm sai số/variance.

## Used Data

Dữ liệu đầu vào là supervised AEP hourly; tất cả feature chỉ đại diện cho lịch sử tới `t-1`. Bootstrap được lấy riêng trong Train của từng fold hoặc Train+Validation ở bước refit cuối. Không lấy mẫu từ Validation/Test; không shuffle chuỗi để xây dựng tập CV.

## Implementation

- File: `src/lab03/lab03_models/random_forest.py`
- CART nền tảng: `src/lab03/lab03_models/decision_tree.py`
- Thành phần chính: `RandomForestRegressorScratch.fit`, `predict`.
- Phụ thuộc: NumPy; không dùng `RandomForestRegressor` của scikit-learn.
