# Rừng ngẫu nhiên hồi quy (tự cài đặt)

## Mục đích

Random Forest kết hợp nhiều CART được huấn luyện trên các bootstrap sample khác nhau. Việc lấy mẫu hàng có hoàn lại và chỉ xét một tập con feature tại mỗi node làm giảm tương quan giữa cây; trung bình dự báo của nhiều cây thường giảm variance so với một cây riêng lẻ. Với dữ liệu tải điện, mô hình có thể nắm bắt quan hệ phi tuyến và tương tác lag/calendar mà không yêu cầu scale đồng nhất.

## Đầu vào - Đầu ra

**Đầu vào:** `X ∈ R^(n×p)`, `y ∈ R^n` và các tham số: số cây `n_estimators`, độ sâu, giới hạn chia/lá, `max_features`, tỷ lệ lấy mẫu bootstrap `max_samples`, số ngưỡng, seed và giới hạn số dòng huấn luyện.

**Đầu ra:** danh sách `B` cây CART tự cài đặt; dự báo là trung bình `ŷ(x) = (1/B) Σ_b T_b(x)`. Mức quan trọng đặc trưng là trung bình mức quan trọng của các cây, sau đó được chuẩn hóa.

## Cách thực hiện

1. Khởi tạo random generator bằng `random_state` để có thể tái lập.
2. Nếu tập huấn luyện vượt `max_train_rows`, chọn các vị trí cách đều trong khoảng thời gian; ghi số dòng đã dùng vào `n_training_rows_`.
3. Với mỗi cây `b`, lấy `ceil(n × max_samples)` chỉ số từ tập huấn luyện bằng phép lấy mẫu có hoàn lại (bootstrap).
4. Khởi tạo một `DecisionTreeRegressorScratch` độc lập. Tại mỗi node của CART, chọn ngẫu nhiên số feature theo `max_features` (thường là `sqrt(p)`).
5. Khớp cây trên mẫu bootstrap; không cây nào được dùng dữ liệu Xác thực/Kiểm tra.
6. Cộng feature importance của cây vào tổng ensemble.
7. Khi dự báo, lấy kết quả của từng cây trên cùng ma trận X và tính trung bình.
8. Chuẩn hóa tổng feature importance; lưu toàn bộ cây để tái sử dụng khi inference.

Rừng ngẫu nhiên có chi phí huấn luyện xấp xỉ tổng chi phí của B cây; khi dự báo phải chạy qua cả B cây. Tăng số cây thường làm kết quả ổn định hơn nhưng cũng tăng thời gian và kích thước mô hình. Tính ngẫu nhiên được kiểm soát bằng seed cố định.

## Chỉ số đánh giá

Báo cáo **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²** trên Validation/Test và 5 CV folds. Báo cáo mean/std giữa fold, số cây, depth, fit/predict time, số mẫu train thực dùng và importance. So sánh trực tiếp với CART trên cùng timestamp để xem ensemble có giảm sai số/variance.

## Dữ liệu sử dụng

Dữ liệu đầu vào là bảng có giám sát AEP theo giờ; mọi đặc trưng chỉ đại diện cho lịch sử đến `t-1`. Mẫu bootstrap được lấy riêng trong tập Huấn luyện của từng fold hoặc Huấn luyện + Xác thực ở bước khớp cuối. Không lấy mẫu từ Xác thực/Kiểm tra và không xáo trộn chuỗi khi tạo CV.

## Cài đặt

- File: `src/lab03/lab03_models/random_forest.py`
- CART nền tảng: `src/lab03/lab03_models/decision_tree.py`
- Thành phần chính: `RandomForestRegressorScratch.fit`, `predict`.
- Phụ thuộc: NumPy; không dùng `RandomForestRegressor` của scikit-learn.
