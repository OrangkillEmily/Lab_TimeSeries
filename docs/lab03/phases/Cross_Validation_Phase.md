# Cross-Validation Phase - Expanding Window / Walk-Forward

## Purpose

Ước lượng độ ổn định của các thuật toán qua nhiều giai đoạn lịch sử. Với time series, K-fold ngẫu nhiên có thể huấn luyện trên thời điểm tương lai rồi validate trên thời điểm quá khứ; vì vậy Lab 3 dùng expanding-window walk-forward CV có thứ tự thời gian.

## Input - Output

**Input:** chỉ `X_train/y_train` (70% đầu chuỗi), `n_splits=5`, model factories, tùy chọn temporal gap.

**Output:** fold-level CSV gồm model, fold, train/validation size và metrics; aggregate mean/std theo model tại `reports/model_summary/lab03/`.

## How to do

1. Chia phần Train thành một vùng khởi tạo và năm validation blocks liên tục.
2. Fold 1 fit trên prefix đầu rồi validate block kế tiếp.
3. Mỗi fold sau mở rộng train prefix để bao gồm lịch sử của các fold trước, còn validation block tiến về phía trước.
4. Khởi tạo model mới cho từng cặp model/fold; mọi tham số học được chỉ fit trên train prefix của fold.
5. Dự báo validation block theo giao thức rolling one-step: feature tại mỗi timestamp chỉ chứa lịch sử trước timestamp đó.
6. Tính metric trên validation block, ghi log fold và tổng hợp mean/std của các metric.
7. Không sử dụng 15% Validation hoặc 15% Test bên ngoài trong CV.

Đây là expanding-window time-series CV, không phải `KFold(shuffle=True)`. Các fold kiểm tra sự ổn định theo thời gian nhưng không thay thế tập Validation dùng để chọn champion.

## Evaluation Metrics

Từng fold: **MAE**, **MSE**, **RMSE**, **MAPE**, **sMAPE**, **R²**. Tổng hợp mean/std, đặc biệt RMSE; std cao thể hiện mô hình nhạy với giai đoạn. Theo dõi train/validation row count và tính tăng dần của training window.

## Used Data

Chỉ 70% đầu tiên của supervised rows. Fold validation luôn đứng sau fold train. Không dùng thông tin từ partition Validation/Test cuối cùng hoặc tương lai so với fold đang xét.

## Implementation Reference

- `src/lab03/cv.py:expanding_window_splits`
- `src/lab03/cv.py:evaluate_walk_forward`
- `pipeline.py`
