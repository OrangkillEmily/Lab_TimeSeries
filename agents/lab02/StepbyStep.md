# HƯỚNG DẪN TỪNG BƯỚC & BÀN GIAO DỰ ÁN (STEP-BY-STEP & HANDOFF) - LAB 02

## 1. Rules From Project (Quy tắc làm việc & AI)
- **Quy tắc chia dữ liệu:** Tỷ lệ 80% Train - 20% Test, tuyệt đối KHÔNG ĐƯỢC XÁO TRỘN (`shuffle=False`) để giữ nguyên cấu trúc thời gian.
- **Quy tắc xây dựng mô hình:** Không code tay thuật toán bằng toán học từ đầu. Tận dụng sức mạnh của class `ARIMA` hoặc `SARIMAX` thuộc thư viện `statsmodels`.
- **Bắt buộc triển khai:** Cùng một dataset `AirPassengers`, phải chạy cả 2 mô hình ARIMA (không mùa vụ) và SARIMA (có mùa vụ) để làm bật lên tầm quan trọng của việc xử lý yếu tố chu kỳ.

## 2. Chosen Direction (Định hướng & Kiến trúc)
- **Nguồn dữ liệu:** Dùng file `airline-passengers.csv` (144 quan sát, dữ liệu hàng tháng, chu kỳ $s=12$).
- **Quy trình Phân tích (Modeling Workflow):** Khám phá (EDA) -> Kiểm định ADF gốc -> Sai phân (thường & mùa vụ) -> Kiểm định ADF lại -> Vẽ ACF/PACF để tìm $(p,d,q)$ và $(P,D,Q)$ -> Huấn luyện mô hình -> Chẩn đoán phần dư (Residuals) -> Dự báo (Forecast) -> Đánh giá (MAE, RMSE, MAPE).
- **Thư viện cốt lõi:** `pandas`, `statsmodels` (kiểm định ADF/ACF/PACF và huấn luyện mô hình), `matplotlib`, `sklearn.metrics` (tính sai số).

## 3. Current Files And Roles (Cấu trúc file hiện tại của Lab 02)
- `docs/methodology.md`: Nơi lưu trữ tài liệu giải thích phương pháp chọn tham số và trả lời các câu hỏi lý thuyết của Lab 02.
- `notebooks/lab02/`: Chứa các file Jupyter Notebook thực thi chia nhỏ theo từng bài:
  - `bai1_EDA.ipynb`: Bài 1 - Khám phá dữ liệu và chia tập Train/Test.
  - `bai2_ADF.ipynb`: Bài 2 - Kiểm định ADF chuỗi gốc.
  - `bai3_SaiPhanKiemDinh.ipynb`: Bài 3 - Lấy sai phân và kiểm định lại.
  - `bai4_ACF-PACF.ipynb`: Bài 4 - Vẽ đồ thị tương quan và chốt tham số.
  - `bai5_ARIMA.ipynb`: Bài 5 - Huấn luyện, dự báo và đánh giá mô hình ARIMA.
  - `bai6_SARIMA.ipynb`: Bài Nâng cao - Huấn luyện, dự báo và đánh giá mô hình SARIMA.
  - `Lab02_TimeSeries.ipynb`: File notebook tổng hợp hoàn chỉnh luồng chạy từ Bài 1 đến Bài 6 để nộp.
- `reports/`: Nơi lưu các hình ảnh biểu đồ ACF/PACF, chẩn đoán phần dư và đường dự báo.
- `src/`: Thư mục chứa các code logic dùng chung:
  - `data_loader.py`: Hàm dùng chung để load file CSV và ép kiểu thời gian.
  - `lab02_utils/stats.py`: Module chạy hàm kiểm định ADF và vẽ đồ thị tương quan.
  - `lab02_utils/modeling.py`: Module chứa hàm fit ARIMA/SARIMA và tính các metric sai số.

## 4. Workflow Timeline (Tiến độ thực thi chi tiết 6 Giai đoạn)
**Giai đoạn 1: Khởi tạo không gian làm việc** (✅ Đã xong)
- Đã dựng xong khung thư mục và file rỗng tĩnh (`src`, `data`, `reports`, `docs`, `notebooks`).
- Đã cấu hình `.gitignore` chặn các file CSV trong thư mục `processed/` và ảnh trong `figures/` để kho lưu trữ Git nhẹ gọn.
- Đã chốt yêu cầu tại thư mục `agents/lab02/`.

**Giai đoạn 2: Tiền xử lý & Khám phá dữ liệu (Bài 1)**
*Mục tiêu: Chuẩn bị trục thời gian chuẩn xác và chia tập dữ liệu huấn luyện.*
- **Nơi thực thi:** File `notebooks/lab02/bai1_EDA.ipynb`.
- **Nhiệm vụ 1:** Gọi hàm từ `src/data_loader.py` để load dữ liệu, set cột tháng làm index thời gian. Vẽ đồ thị chuỗi gốc, quan sát bằng mắt và trả lời 3 câu hỏi: Có xu hướng không? Mùa vụ không? Có dừng không?
- **Nhiệm vụ 2:** Cắt tập Train (80%) và Test (20%) bằng `pandas` với luật `shuffle=False`.

**Giai đoạn 3: Kiểm định ADF & Lấy Sai phân (Bài 2 & Bài 3)**
*Mục tiêu: Trị căn bệnh "không dừng" của dữ liệu chuỗi thời gian.*
- **Nơi thực thi:** Lần lượt tại `bai2_ADF.ipynb` và `bai3_SaiPhanKiemDinh.ipynb` (gọi hàm từ `src/lab02_utils/stats.py`).
- **Nhiệm vụ 1 (Chuỗi gốc):** Chạy kiểm định ADF trên chuỗi ban đầu. Ghi lại ADF statistic, p-value và kết luận chuỗi không dừng.
- **Nhiệm vụ 2 (Sai phân):** 
  - Thử nghiệm sai phân bậc 1 ($d=1$) cho mô hình ARIMA.
  - Kết hợp sai phân bậc 1 ($d=1$) và sai phân mùa vụ ($D=1$, chu kỳ $s=12$) chuẩn bị cho SARIMA.
- **Nhiệm vụ 3 (Kiểm định lại):** Chạy ADF lần 2 trên các chuỗi đã sai phân ở trên để khẳng định p-value đã đạt chuẩn (< 0.05).

**Giai đoạn 4: Đọc ACF/PACF và Chốt tham số (Bài 4)**
*Mục tiêu: Tìm ra "công thức" $(p, d, q)$ và $(P, D, Q, s)$ để nạp vào mô hình.*
- **Nơi thực thi:** File `notebooks/lab02/bai4_ACF-PACF.ipynb`.
- **Nhiệm vụ 1:** Vẽ ACF và PACF của chuỗi **đã lấy sai phân** (KHÔNG vẽ trên chuỗi gốc).
- **Nhiệm vụ 2:** Quan sát các lag không theo mùa để đề xuất $p$ (từ PACF) và $q$ (từ ACF). Quan sát các lag chu kỳ (bội số của 12) để xác định $P, Q$ cho Bài 6.

**Giai đoạn 5: Huấn luyện Mô hình & Chẩn đoán Phần dư (Bài 5 & Nâng cao)**
*Mục tiêu: Dạy máy học, sau đó bắt mạch xem mô hình có bị "bệnh" nội tại không.*
- **Nơi thực thi:** File `notebooks/lab02/bai5_ARIMA.ipynb` và `notebooks/lab02/bai6_SARIMA.ipynb` (gọi hàm từ `src/lab02_utils/modeling.py`).
- **Nhiệm vụ 1 (Fit Model):** Nạp tập Train vào huấn luyện cho 2 mô hình ARIMA và SARIMA. Print ra màn hình chỉ số AIC và BIC.
- **Nhiệm vụ 2 (Chẩn đoán Residuals - Bắt buộc):** Vẽ đồ thị phần dư (Residuals) theo thời gian và vẽ thêm ACF của phần dư. Kiểm chứng xem phần dư có rớt về xấp xỉ 0 và không có tự tương quan (White Noise) hay không.

**Giai đoạn 6: Dự báo, So sánh & Báo cáo (Hoàn thiện Bài 5 & 6)**
*Mục tiêu: Đem mô hình đi thi nghiệm thu trên tập Test và tính điểm.*
- **Nơi thực thi:** Tiếp tục ở `bai5_ARIMA.ipynb`, `bai6_SARIMA.ipynb` và tổng hợp ở `Lab02_TimeSeries.ipynb`.
- **Nhiệm vụ 1 (Forecast):** Chạy dự đoán trên tập Test (20% dữ liệu đã giấu). Vẽ biểu đồ 3 đường: Train, Test thực tế và Forecast.
- **Nhiệm vụ 2 (Tính điểm):** Tính các sai số MAE, RMSE, MAPE cho cả 2 mô hình.
- **Nhiệm vụ 3 (Kết luận):** Ghi chép câu trả lời cho các câu hỏi lý thuyết của Bài 5 vào file `docs/methodology.md`. Kết luận mô hình nào hoạt động tốt hơn trên bộ dữ liệu `AirPassengers`.