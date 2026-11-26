# Kết quả thí nghiệm cũ - Lab 03 (notebook lưu trữ)

> **Nguồn gốc:** Bảng này được chép lại từ kết quả đã lưu và phần diễn giải Markdown trong `notebooks/lab03/archive/Lab03_TimeSeries_legacy.ipynb`. Đây là lần chạy trước đây với XGBoost / Linear Regression / Random Forest từ thư viện và cách chia dữ liệu theo thời gian 80/20. Kết quả được lưu riêng với thí nghiệm NumPy tự cài đặt trong pipeline hiện tại. Không xem các số liệu này là phép so sánh đối chứng tương đương vì cách chia, tập đặc trưng, cách cài đặt và quy trình tinh chỉnh khác nhau.

## Kết quả làm sạch dữ liệu đã thực hiện

- Số dòng CSV thô: **121.273**.
- Timestamp trùng: **4** (tệp gốc chưa được sắp xếp).
- Chuỗi giờ liên tục sau làm sạch: **121.296 quan sát**, từ **2004-10-01 01:00** đến **2018-08-03 00:00**.
- Số giờ thiếu đã bổ sung: **27**; số giá trị thiếu còn lại sau làm sạch: **0**.
- Biến mục tiêu: `AEP_MW`, đơn vị MW.

## Mốc đối chứng và kết quả mô hình cũ (dự báo kiểm tra một bước)

| Mô hình | MAE (MW) | RMSE (MW) | MAPE | R² |
|---|---:|---:|---:|---:|
| Naive (lag_1) | 407.6 | 525.5 | 2.81% | 0.954 |
| Hồi quy tuyến tính (24 độ trễ) | 147.9 | 203.4 | 1.02% | 0.993 |
| XGBoost mặc định (24 độ trễ) | 152.5 | 205.9 | 1.05% | 0.993 |
| Rừng ngẫu nhiên (24 độ trễ) | - | 211.9 | - | - |
| XGBoost đã tinh chỉnh (24 độ trễ) | 138.5 | 189.0 | 0.96% | - |

Notebook lưu trữ ghi nhận XGBoost đã tinh chỉnh giảm RMSE tập Kiểm tra khoảng **8,2%** so với XGBoost mặc định. Phần diễn giải cũng lưu ý XGBoost mặc định không vượt hồi quy tuyến tính trên cách chia đó.

## Các kết quả nâng cao đã thực hiện

- Trung bình/độ lệch chuẩn trượt với cửa sổ 24 và 168 giờ, dịch một timestamp: trên tập Kiểm tra chung 24.226 dòng, RMSE XGBoost đổi từ **205,9 → 203,7** và hồi quy tuyến tính từ **203,4 → 203,0**. Bốn đặc trưng trượt đóng góp tổng cộng khoảng **0,8%** mức quan trọng đặc trưng của XGBoost.
- Dự báo nhiều bước 24 giờ: RMSE trung bình qua 24 mốc xấp xỉ **962,7** cho đệ quy, **914,9** cho trực tiếp và **1.220,9** cho dự báo ngây thơ theo mùa. Notebook ghi chú cấu hình mô hình không giống nhau nên đây chỉ là so sánh tham khảo, không phải phép loại bỏ biến được kiểm soát chặt chẽ.
- Tinh chỉnh: lấy mẫu 8 tổ hợp tham số, dùng `TimeSeriesSplit` 3 fold chỉ trên tập Huấn luyện. Cấu hình tốt nhất được ghi nhận là `n_estimators=200`, `max_depth=8`, `learning_rate=0.05`, `subsample=1.0`, `colsample_bytree=1.0`, `min_child_weight=10`, với RMSE CV khoảng **209,1**.
- Mức quan trọng đặc trưng: `lag_1` và `lag_2` đóng góp tổng cộng khoảng **78,9%** trong lần chạy XGBoost cũ; các độ trễ 22-24 đóng góp tổng cộng khoảng **11,7%**.
- Khảo sát số độ trễ: notebook ghi RMSE hồi quy tuyến tính khoảng **149,5** với 168 độ trễ, trong khi XGBoost chưa tinh chỉnh đạt khoảng **189,2** với 168 độ trễ.

## Cách chia dữ liệu cũ

- Huấn luyện: **97.018 dòng**, từ 2004-10-02 01:00 đến 2015-10-27 10:00.
- Kiểm tra: **24.254 dòng**, từ 2015-10-27 11:00 đến 2018-08-03 00:00.
- Không xáo trộn dữ liệu. Đây là cách chia 80/20, khác cách chia Huấn luyện/Xác thực/Kiểm tra 70/15/15 chuẩn hóa hiện tại.

## Sản phẩm được lưu lại

- Notebook gốc có kết quả đã chạy: `notebooks/lab03/archive/Lab03_TimeSeries_legacy.ipynb`.
- Hình được trích xuất trước đây: `images/lab03/legacy/`.
- Mã nguồn trợ giúp gốc được cung cấp riêng trong tệp ZIP tải lên và lưu tại `legacy/source/` để truy xuất nguồn gốc; pipeline mới dùng các mô-đun tự cài đặt trong `src/lab03/`.

## Giới hạn diễn giải

Đây là kết quả lịch sử trong notebook, không phải kết quả chạy lại từ mã hiện tại. Môi trường chính xác, phiên bản thư viện và toàn bộ dữ liệu nguồn không được đóng gói cùng. Gói mã Lab 03 không kèm `AEP_hourly.csv`. Pipeline hiện tại mặc định dừng nếu thiếu CSV thật; chỉ dùng dữ liệu tổng hợp khi bật rõ `--synthetic`, và khi đó kết quả được gắn nhãn bản minh họa.
