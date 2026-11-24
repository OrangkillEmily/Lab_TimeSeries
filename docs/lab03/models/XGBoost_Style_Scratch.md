# XGBoost-Style Gradient Boosted Trees (Scratch)

## Purpose

Gradient Boosting xây dựng các cây tuần tự: mỗi cây mới học cách hiệu chỉnh sai số còn lại của ensemble trước đó. Implementation Lab 3 dùng đạo hàm bậc nhất/bậc hai của squared-error, regularization L1/L2, sampling theo hàng/cột và histogram bins. Mục tiêu là cung cấp một phiên bản giáo dục để minh họa ý tưởng XGBoost, không phải bản sao chính thức hoặc tương đương hoàn toàn với thư viện XGBoost.

## Input - Output

**Input:** `X ∈ R^(n×p)`, `y ∈ R^n`; hyperparameters gồm `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`, `gamma`, `max_bins`, `max_train_rows`, `random_state`.

**Output:** `base_score = mean(y)`, danh sách các regression trees, các bin edges học từ tập fit, feature importance dựa trên split gain và vector dự báo tổng hợp `ŷ = base_score + η Σ_t f_t(x)`.

## How to do

1. Xác thực X/y hữu hạn, đúng số chiều. Nếu cần, chọn một subset train đều theo index để giới hạn runtime.
2. Với mỗi feature, tính quantile cut points từ X của tập fit; chuyển giá trị thành integer bin bằng `searchsorted`. Bin edges được lưu lại để dùng đúng cách khi predict. Không fit bin trên Validation/Test.
3. Khởi tạo prediction bằng `mean(y)`. Với squared error `L = 1/2 (ŷ-y)²`, gradient `g = ŷ-y` và Hessian `h=1`.
4. Ở mỗi boosting round, lấy row subsample không hoàn lại và column subsample theo cấu hình.
5. Tại node, cộng gradient/Hessian theo từng bin. Tính gain của split theo tổng gradient/Hessian trái-phải và regularization. Split không thỏa `min_child_weight` hoặc gain không vượt `gamma` bị loại.
6. Trọng số leaf là `w* = -SoftThreshold(G, α)/(H+λ)`, trong đó SoftThreshold áp dụng L1 `reg_alpha`; `reg_lambda` ổn định mẫu số.
7. Đệ quy xây cây tới `max_depth`. Cập nhật ensemble `prediction += learning_rate × tree_prediction`.
8. Lặp tới `n_estimators`. Ghi importance theo tổng split gain.
9. Khi predict, ánh xạ input bằng bin edges đã fit, khởi tạo base score rồi cộng lần lượt learning-rate-scaled prediction của từng cây.

Thuật toán khác CART ở chỗ cây mới không dự báo target gốc độc lập mà dự báo hướng hiệu chỉnh của loss. Histogram bins giảm số ngưỡng cần xét. Implementation dùng squared-error objective và một tập bin theo quantile; chưa bao gồm mọi kỹ thuật tối ưu/hệ sinh thái của XGBoost chính thức.

## Evaluation Metrics

Đánh giá **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²** trên Validation, Test và từng CV fold. Theo dõi mean/std CV RMSE, training/prediction time, số hàng được sử dụng, số boosting rounds, depth, learning rate, regularization và feature importance. Chọn hyperparameters trên Train/CV/Validation; không tune theo Test.

## Used Data

Dùng các lag/rolling/calendar features được xây dựng chỉ từ thông tin trước giờ dự báo. Quantile bin edges, base score và cây đều được fit lại trên phần Train ở từng fold; final fit dùng Train+Validation. Test chỉ đi qua `predict`. Dữ liệu AEP thật phải có cột `Datetime`, `AEP_MW`; synthetic chỉ là smoke test.

## Implementation

- File: `src/lab03/lab03_models/xgboost_scratch.py`
- Thành phần chính: `XGBoostScratch._make_bins`, `_build_tree`, `_leaf_weight`, `fit`, `predict`.
- Phụ thuộc: NumPy; không import hoặc gọi package `xgboost`.
