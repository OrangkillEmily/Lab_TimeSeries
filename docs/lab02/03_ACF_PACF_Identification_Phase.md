# 03 — Giai đoạn Nhận diện Tham số qua ACF & PACF (`ACF_PACF_Identification_Phase`)

## 1. Purpose
Sau khi đưa tập Train về trạng thái dừng, giai đoạn này phân tích cấu trúc tự tương quan **ACF** và tự tương quan riêng phần **PACF** (36 lags) để:
- Giải thích vì sao chuỗi `Diff(d=1)` vẫn còn đỉnh mùa vụ lặp lại rất mạnh tại `lag 12, 24, 36`.
- Đọc nhanh bộ ứng viên bậc thường $(p, q)$ và bậc mùa vụ $(P, Q)$ trên chuỗi dừng `Diff(d=1, D=1, s=12)` làm đầu vào cho lưới ứng viên (`candidate_registry.py`) và cơ chế tự động nhận diện hai không gian (`auto_identify_candidates_from_data`).

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi sai phân tập Train `diff_1` và `diff_1_12` của Dataset 1 và Dataset 2 (`nlags = 36`, `seasonal_period = 12`). |
| **Module xử lý** | `src/lab02/stats.py` (`significant_lags`, `plot_acf_pacf`), `src/lab02/acf_pacf_analysis.py` (`analyze_acf_pacf`), `src/lab02/lab02_models/candidate_registry.py` (`auto_identify_candidates_from_data`). |
| **Output (Data/JSON)** | Nhánh `acf_pacf` (`diff_1`, `diff_1_12`, `bound`, `acf_sig`, `pacf_sig`, bảng giá trị lag 1..6, 12, 24, 36) trong `progress_results/lab02/{dataset1,dataset2}/latest_run.json`. |
| **Output (Figures)** | Biểu đồ ACF/PACF 36 lags trước và sau sai phân mùa vụ tại `reports/figures/lab02/{dataset1,dataset2}/report/`. |

## 3. How to do
1. **Tính ngưỡng tin cậy 95%:** $\pm \frac{1.96}{\sqrt{n}}$ (với $n$ là số quan sát hợp lệ sau sai phân).
2. **Quy tắc đọc Box-Jenkins:**
   - **MA($q$):** ACF cắt cụt sau lag $q$, PACF tắt dần theo hàm mũ/hình sin.
   - **AR($p$):** PACF cắt cụt sau lag $p$, ACF tắt dần.
   - **ARMA($p, q$):** Cả ACF và PACF cùng tắt dần.
   - **Thành phần mùa vụ $(P, Q)_{12}$:** Kiểm tra tương tự tại các bội số của chu kỳ mùa vụ (`lag 12, 24, 36`).
3. **Tự động đề xuất lưới ứng viên (`auto_identify_candidates_from_data`):** Chạy đồng thời trên không gian gốc và không gian Log (`include_log=True`) để sinh đủ 5 họ mô hình (`AR`, `MA`, `ARIMA`, `SARIMA`, `Log-SARIMA`).

## 4. Evaluation Metrics
| Chuỗi phân tích | $n$ / Ngưỡng 95% | Các lag ACF có ý nghĩa | Các lag PACF có ý nghĩa | Giá trị nổi bật & Đề xuất bậc |
|---|---|---|---|---|
| **DS1 — `Diff(d=1)`** | `162` / `±0.154` | `1, 11, 12, 13, 23, 24, 25, 35, 36` | `1, 2, 10, 11, 12, 13, 15` | `ACF(12)=0.84`, `ACF(24)=0.68`, `ACF(36)=0.55` → mùa vụ 12 tháng rất mạnh |
| **DS1 — `Diff(d=1, D=1, s=12)`** | `150` / `±0.160` | `1, 3, 4, 7, 23, 24, 26, 27` | `1, 2, 4, 5, 25, 36` | `ACF(1)=-0.57`, `PACF(1)=-0.57`, `PACF(2)=-0.42`, `ACF(12)=-0.11` → thử $p\in\{0,1,2\}, q\in\{0,1,3\}, Q\in\{0,1\}$ |
| **DS2 — `Diff(d=1)`** | `114` / `±0.184` | `1, 4, 8, 9, 11, 12, 13, 16, 20, 23, 24, 32, 36` | `1, 2, 4, 7, 8, 10, 11, 12` | `ACF(12)=0.80`, `ACF(24)=0.64`, `ACF(36)=0.53` → bắt buộc sai phân mùa vụ $D=1$ |
| **DS2 — `Diff(d=1, D=1, s=12)`** | `102` / `±0.194` | `1, 3, 23` | `1, 20, 23` | `ACF(1)=-0.20`, `ACF(3)=-0.21`, `PACF(1)=-0.20`, `ACF(12)=-0.17` → thử $p\in\{0,1\}, q\in\{0,1,3\}, P,Q\in\{0,1\}$ |

## 5. Used data
- Toàn bộ ước lượng ACF/PACF được thực hiện **chỉ trên tập Train** (`163` tháng DS1 và `115` tháng DS2), giữ hoàn toàn độc lập tập Test để đánh giá dự báo ngoài mẫu ở Giai đoạn 4.
