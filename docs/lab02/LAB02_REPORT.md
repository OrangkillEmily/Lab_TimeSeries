# TỔNG HỢP KẾT QUẢ THỰC NGHIỆM LAB 02 (`LAB02_REPORT.md`)

Tài liệu này tóm tắt các kết quả định lượng mới nhất từ hai phiên chạy chế độ `full`:
- **Dataset 1 (`Monthly Retail Sales`):** `run_id = dataset1_20261123_154923_d2775a` (`27/27` mô hình thành công, `0` lỗi).
- **Dataset 2 (`AirPassengers`):** `run_id = dataset2_20261123_154951_786eea` (`54/54` mô hình thành công, `0` lỗi).

## 1. Bảng Đối chiếu Tổng thể 2 Bộ dữ liệu
| Hạng mục | Dataset 1 — Monthly Retail Sales | Dataset 2 — AirPassengers |
|---|---|---|
| **Giai đoạn & Kích thước** | `07/1991 – 06/2008` (`204` tháng: `163` Train / `41` Test) | `01/1949 – 12/1960` (`144` tháng: `115` Train / `29` Test) |
| **ADF Raw / `Diff(1)` / `Diff(1,12)`** | `p=0.9989` / `p=0.4280` / **`p=2.28e-14`** | `p=0.9171` / `p=0.1061` / **`p=1.38e-22`** |
| **ARIMA tốt nhất (Không mùa vụ)** | `ARIMA(1,1,1)`: `RMSE=4.62`, `MAPE=18.04%`, `R²=-0.2520`, `LB p=0.0000` | `ARIMA(2,1,1)`: `RMSE=82.62`, `MAPE=12.90%`, `R²=-0.1182`, `LB p=0.0000` |
| **SARIMA tốt nhất (Thang gốc)** | **`SARIMA(0,1,1)x(0,1,1)_12`**: `RMSE=1.84`, `MAPE=8.21%`, `R²=0.8012`, `LB p=0.1053` | **`SARIMA(1,1,3)x(0,1,1)_12`**: `RMSE=24.85`, `MAPE=4.31%`, `R²=0.8988`, `LB p=0.4105` |
| **Log-SARIMA RMSE thấp nhất** | `Log-SARIMA(1,1,1)x(0,1,1)_12`: `RMSE=1.49`, `MAE=1.18`, `R²=0.8697` (`LB p=0.0170`) | `Log-SARIMA(0,1,0)x(1,1,1)_12`: `RMSE=14.66`, `R²=0.9648` (`LB p=0.0353`) |
| **Mô hình tối ưu đạt đồng thời WN ($p \ge 0.05$) & Hội tụ** | **`SARIMA(0,1,1)x(0,1,1)_12`** (`RMSE=1.84`, `R²=0.8012`) | **`Log-SARIMA(1,1,1)x(0,1,1)_12`** (`RMSE=15.02`, `R²=0.9630`, `LB p=0.9959`) & bản tối giản **`Log-SARIMA(1,1,0)x(0,1,1)_12`** (`RMSE=15.12`, `AIC=-327.47`, `LB p=0.9970`) |

## 2. Nhận định Chuyên môn Cốt lõi
1. **Bắt buộc sai phân kết hợp $(d=1, D=1, s=12)$:** Trên cả 2 tập, chỉ lấy sai phân bậc 1 ($d=1$) không thể làm chuỗi dừng ($p = 0.4280$ và $p = 0.1061$); toàn bộ họ AR/MA/ARIMA không mùa vụ đều cho $R^2 < 0$ và thất bại kiểm định Ljung-Box ($p = 0.0000$).
2. **Bước nhảy vượt trội từ SARIMA và Log-SARIMA:**
   - Trên **Dataset 1**, `SARIMA(0,1,1)x(0,1,1)_12` giảm RMSE từ `4.62` xuống `1.84` (-60.2%) và thỏa mãn chặt chẽ nhiễu trắng (`p = 0.1053`); `Log-SARIMA(1,1,1)x(0,1,1)_12` tiếp tục kéo RMSE xuống `1.49` ($R^2 = 0.8697$) nhưng đánh đổi bằng tự tương quan nhẹ ở phần dư Log (`p = 0.0170`).
   - Trên **Dataset 2 (`AirPassengers`)**, vì cấu trúc mùa vụ nhân (Multiplicative) rất điển hình, `Log-SARIMA(1,1,1)x(0,1,1)_12` và `Log-SARIMA(1,1,0)x(0,1,1)_12` vừa đạt sai số cực thấp (`MAPE ≈ 2.80%`, `R² > 0.96`), vừa hội tụ ổn định và đạt chuẩn nhiễu trắng gần như tuyệt đối (`LB p > 0.995`).
3. **Báo cáo chi tiết tách riêng từng bộ dữ liệu:**
   - Dataset 1: `reports/lab02/LAB_02_DATASET1_REPORT.{html,pdf,md}` & `reports/model_summary/lab02/dataset1/dataset1_model_summary.md`.
   - Dataset 2: `reports/lab02/LAB_02_DATASET2_REPORT.{html,pdf,md}` & `reports/model_summary/lab02/dataset2/dataset2_model_summary.md`.
