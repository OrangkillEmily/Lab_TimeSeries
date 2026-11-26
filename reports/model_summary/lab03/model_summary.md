# Tóm tắt mô hình Lab 03 (lần chạy report_fix_20261127)

Mô hình được chọn theo RMSE xác thực thấp nhất: **Hồi quy tuyến tính**

| Mô hình                         | Được chọn   |   RMSE CV trung bình |   RMSE xác thực |   RMSE kiểm tra |   Thời gian khớp (giây) |   Số dòng huấn luyện |
|:--------------------------------|:------------|---------------------:|----------------:|----------------:|------------------------:|---------------------:|
| Giá trị gần nhất                | Không       |              575.926 |         534.446 |         534.753 |                   0     |               102958 |
| Hồi quy tuyến tính              | Có          |              277.589 |         246.137 |         228.494 |                   0.049 |               102958 |
| Cây hồi quy CART                | Không       |              337.364 |         322.511 |         316.215 |                   4.906 |                15000 |
| Rừng ngẫu nhiên                 | Không       |              498.419 |         493.904 |         460.137 |                   8.043 |                15000 |
| Mô hình tăng cường kiểu XGBoost | Không       |              775.685 |         777.448 |         726.126 |                   0.652 |                18000 |
