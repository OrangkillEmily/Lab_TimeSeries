# Đặc tả tóm tắt mô hình Lab 03

Notebook và pipeline ghi ba dạng đồng bộ để so sánh các mô hình ứng viên:

- `reports/model_summary/lab03/model_summary.csv`
- `reports/model_summary/lab03/model_summary.json`
- `reports/model_summary/lab03/model_summary.md`

Ngoài ra còn có `cv_fold_metrics.csv`, `validation_metrics.json` và `test_metrics.json`.

## Các trường dữ liệu

| Nhóm trường | Ý nghĩa |
|---|---|
| `model` | Tên mô hình ứng viên |
| `cv_*_mean`, `cv_*_std` | Trung bình và độ lệch chuẩn chỉ số qua 5 fold cửa sổ mở rộng, chỉ dùng tập Huấn luyện |
| `validation_*` | Chỉ số khi khớp trên tập Huấn luyện và dự báo tập Xác thực |
| `test_*` | Chỉ số cuối cùng sau khi khớp lại trên Huấn luyện + Xác thực và dự báo tập Kiểm tra được giữ riêng |
| `fit_seconds`, `predict_seconds_test` | Thời gian khớp/dự báo của lần khớp cuối |
| `training_rows_used` | Số dòng thực tế mô hình đã dùng; có thể bị giới hạn để giảm thời gian chạy cây |
| `total_train_validation_rows` | Tổng số dòng của tập dữ liệu dùng để khớp lần cuối |
| `feature_count` | Số biến dự báo độ trễ/cửa sổ trượt/lịch |
| `model_file` | Đường dẫn tương đối đến tệp mô hình đã khớp |
| `model_parameters` | Siêu tham số mô hình được tuần tự hóa dưới dạng JSON |

## Quy tắc chọn mô hình

Chọn mô hình có **RMSE xác thực thấp nhất**. Bảng tóm tắt có thể được sắp xếp theo RMSE kiểm tra để dễ đọc, nhưng việc sắp xếp không thay đổi mô hình đã chọn. Không dùng chỉ số tập Kiểm tra làm đầu vào lựa chọn.

## Cách diễn giải

- MAE/RMSE thấp hơn nghĩa là sai số theo đơn vị MW nhỏ hơn; RMSE phạt sai số lớn mạnh hơn.
- MAPE tính theo phần trăm và bỏ qua giá trị mục tiêu bằng 0; cần thận trọng khi giá trị thực gần 0.
- sMAPE dùng tổng trị tuyệt đối của giá trị thực và dự báo làm mẫu số.
- R² không phải phần trăm độ chính xác và có thể âm.
- Độ lệch chuẩn CV dùng để đánh giá độ ổn định: giá trị cao cho thấy hiệu năng thay đổi đáng kể theo giai đoạn.
- Khi có giới hạn thời gian chạy, cần xem `training_rows_used` cùng `total_train_validation_rows`.

## Nguồn gốc kết quả

Điểm số notebook cũ được ghi riêng trong `reports/lab03/legacy/Legacy_Experiment_Results.md`; không gộp chúng vào bảng tóm tắt hiện tại vì thư viện mô hình, cách chia dữ liệu và tập đặc trưng khác nhau.
