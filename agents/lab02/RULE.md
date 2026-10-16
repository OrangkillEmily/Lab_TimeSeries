# QUY TẮC LẬP TRÌNH & TƯƠNG TÁC (LAB 02 - ARIMA/SARIMA MODELING)

**Quy tắc 1: Quyền quyết định và Tương tác AI - Con người**
- **Vai trò của AI:** Hỗ trợ viết code kiểm định ADF, vẽ đồ thị tương quan, huấn luyện mô hình (fit model) và thiết lập các hàm tính toán sai số (MAE, RMSE, MAPE).
- **Vai trò của Con người:** Là người quan sát đồ thị ACF/PACF sau sai phân để chốt bộ tham số `(p, d, q)` và `(P, D, Q, s)`. AI không được tự ý "bịa" ra tham số nếu con người chưa quyết định.
- **Nguyên tắc "Hiểu mới dùng":** Tuyệt đối KHÔNG chạy code huấn luyện nếu chưa hiểu rõ ý nghĩa của các tham số đầu vào.

**Quy tắc 2: Vibe Coding (Dos & Don'ts)**
- **Được làm (Dos):** Viết code rõ ràng, rành mạch từng bước (Kiểm định -> Sai phân -> Chọn tham số -> Fit mô hình -> Chẩn đoán phần dư -> Dự báo). Biểu đồ so sánh Train, Test và Forecast phải có chú giải (legend) và màu sắc dễ phân biệt.
- **Không được làm (Don'ts):** Không dùng các vòng lặp phức tạp để tự viết lại thuật toán ARIMA bằng toán học từ đầu. Phải tận dụng triệt để class `ARIMA` hoặc `SARIMAX` của thư viện `statsmodels`.

**Quy tắc 3: Bắt buộc triển khai CẢ 2 MÔ HÌNH**
- Bắt buộc phải huấn luyện cả 2 mô hình trên bộ dữ liệu `AirPassengers`: **ARIMA** (ở file `bai5_ARIMA.ipynb`) và **SARIMA** với chu kỳ `s=12` (ở file `bai6_SARIMA.ipynb`).
- Bắt buộc phải có bảng hoặc biểu đồ so sánh hiệu năng trực tiếp giữa ARIMA và SARIMA để rút ra kết luận mô hình nào tốt hơn.

**Quy tắc 4: Nguyên tắc Vàng về Dữ liệu Chuỗi thời gian (Tuyệt đối tuân thủ)**
- **KHÔNG SHUFFLE:** Khi chia tập dữ liệu thành Train (80%) và Test (20%), tuyệt đối phải giữ nguyên thứ tự thời gian bằng tham số `shuffle=False`. Nếu xáo trộn, cấu trúc chuỗi thời gian sẽ bị phá vỡ hoàn toàn.
- **Chống rò rỉ dữ liệu (Data Leakage):** Mô hình chỉ được học (`fit`) trên tập Train. Tập Test chỉ được mang ra ở bước cuối cùng để đối chiếu với hàm dự đoán (`forecast`).

**Quy tắc 5: Đánh giá và Chẩn đoán bắt buộc**
- **Chẩn đoán phần dư (Residual Diagnostics):** Trước khi dùng mô hình để dự báo, BẮT BUỘC phải vẽ đồ thị phần dư theo thời gian và đồ thị ACF của phần dư. Phải kiểm tra xem phần dư có trung bình xấp xỉ 0 (mean ≈ 0) và có đạt chuẩn Nhiễu trắng (White Noise) không.
- **Chỉ số đánh giá:** Bắt buộc tính và báo cáo đầy đủ 3 chỉ số đo lường sai số: **MAE, RMSE, MAPE**.

**Quy tắc 6: Giải thích code (Docstring) & Trả lời bằng Markdown**
- Bất kỳ hàm nào tự định nghĩa trong `src/lab02_utils/` (như `stats.py`, `modeling.py`) đều phải có Docstring ghi rõ 3 ý: (1) Mục đích, (2) Tham số đầu vào, (3) Kết quả đầu ra.
- Trong các file Jupyter Notebook (từ `bai1` đến `bai6`), BẮT BUỘC phải dùng thẻ Markdown để trả lời đầy đủ các câu hỏi lý thuyết của đề bài (Ví dụ: *Ý nghĩa p, d, q? Ho của ADF là gì? RMSE lớn hay nhỏ nói lên điều gì?*). Không được để trống.

**Quy tắc 7: Tôn trọng kiến trúc thư mục**
- **Chia để trị:** Tuân thủ đúng cấu trúc thực thi trên các file notebook đã chia nhỏ (`bai1_EDA.ipynb`, `bai2_ADF.ipynb`...). Chỉ gộp code lại khi làm file tổng `Lab02_TimeSeries.ipynb` ở bước cuối cùng.
- File dữ liệu Train/Test (`train_data.csv`, `test_data.csv`) phải được xuất ra đúng thư mục `data/processed/lab02/`.
- File trả lời tự luận bắt buộc phải ghi chép lại vào `docs/methodology.md`.