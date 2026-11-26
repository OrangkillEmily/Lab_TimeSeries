# Cây tăng cường gradient kiểu XGBoost (tự cài đặt)

## Mục đích

Gradient Boosting xây dựng các cây tuần tự: mỗi cây mới học cách hiệu chỉnh sai số còn lại của ensemble trước đó. Implementation Lab 3 dùng đạo hàm bậc nhất/bậc hai của squared-error, regularization L1/L2, sampling theo hàng/cột và histogram bins. Mục tiêu là cung cấp một phiên bản giáo dục để minh họa ý tưởng XGBoost, không phải bản sao chính thức hoặc tương đương hoàn toàn với thư viện XGBoost.

## Đầu vào - Đầu ra

**Đầu vào:** `X ∈ R^(n×p)`, `y ∈ R^n`; siêu tham số gồm `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`, `gamma`, `max_bins`, `max_train_rows`, `random_state`.

**Đầu ra:** `base_score = mean(y)`, danh sách cây hồi quy, các biên bin học từ tập khớp, mức quan trọng đặc trưng dựa trên độ lợi của phép chia và vector dự báo tổng hợp `ŷ = base_score + η Σ_t f_t(x)`.

## Cách thực hiện

1. Xác nhận X/y hữu hạn và đúng số chiều. Nếu cần, chọn một tập con huấn luyện cách đều theo chỉ mục để giới hạn thời gian chạy.
2. Với mỗi đặc trưng, tính các điểm cắt phân vị từ X của tập huấn luyện; chuyển giá trị thành bin nguyên bằng `searchsorted`. Lưu biên bin để dùng nhất quán khi dự báo. Không khớp biên bin trên Xác thực/Kiểm tra.
3. Khởi tạo dự báo bằng `mean(y)`. Với sai số bình phương `L = 1/2 (ŷ-y)²`, gradient `g = ŷ-y` và Hessian `h=1`.
4. Ở mỗi vòng tăng cường, lấy mẫu dòng không hoàn lại và lấy mẫu cột theo cấu hình.
5. Tại mỗi nút, cộng gradient/Hessian theo từng bin. Tính độ lợi phép chia từ tổng gradient/Hessian hai phía và điều chuẩn. Loại phép chia không đạt `min_child_weight` hoặc có độ lợi không vượt `gamma`.
6. Trọng số leaf là `w* = -SoftThreshold(G, α)/(H+λ)`, trong đó SoftThreshold áp dụng L1 `reg_alpha`; `reg_lambda` ổn định mẫu số.
7. Đệ quy xây cây tới `max_depth`. Cập nhật ensemble `prediction += learning_rate × tree_prediction`.
8. Lặp đến `n_estimators`. Ghi mức quan trọng theo tổng độ lợi phép chia.
9. Khi dự báo, ánh xạ đầu vào theo các biên bin đã khớp, khởi tạo điểm cơ sở rồi lần lượt cộng dự báo của từng cây sau khi nhân với tốc độ học.

Thuật toán khác CART ở chỗ cây mới không dự báo độc lập biến mục tiêu gốc mà dự báo hướng hiệu chỉnh hàm mất mát. Bin dạng histogram giảm số ngưỡng cần xét. Cách cài đặt dùng hàm mục tiêu sai số bình phương và các bin theo phân vị; không bao gồm toàn bộ kỹ thuật tối ưu/hệ sinh thái của XGBoost chính thức.

## Chỉ số đánh giá

Đánh giá **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²** trên Validation, Test và từng CV fold. Theo dõi mean/std CV RMSE, training/prediction time, số hàng được sử dụng, số boosting rounds, depth, learning rate, regularization và feature importance. Chọn hyperparameters trên Train/CV/Validation; không tune theo Test.

## Dữ liệu sử dụng

Dùng đặc trưng độ trễ/cửa sổ trượt/lịch chỉ được tạo từ thông tin trước giờ dự báo. Biên bin phân vị, điểm cơ sở và cây đều được khớp lại trên phần Huấn luyện ở từng fold; lần khớp cuối dùng Huấn luyện + Xác thực. Tập Kiểm tra chỉ được đưa qua `predict`. Dữ liệu AEP thật phải có cột `Datetime`, `AEP_MW`; dữ liệu tổng hợp chỉ dùng để kiểm tra nhanh.

## Cài đặt

- File: `src/lab03/lab03_models/xgboost_scratch.py`
- Thành phần chính: `XGBoostScratch._make_bins`, `_build_tree`, `_leaf_weight`, `fit`, `predict`.
- Phụ thuộc: NumPy; không import hoặc gọi package `xgboost`.
