# QUY TẮC LẬP TRÌNH & TƯƠNG TÁC (LAB 01 - TIME SERIES EDA)

**Quy tắc 1: Quyền quyết định và Tương tác AI - Con người**
- **Vai trò của AI:** Đóng vai trò trợ lý viết code xử lý dữ liệu và vẽ biểu đồ. Có thể đưa ra gợi ý đọc hiểu đồ thị.
- **Vai trò của Con người:** Là người quan sát biểu đồ bằng mắt thường và ra quyết định cuối cùng (Ví dụ: Nhìn ACF/PACF để chốt xem có yếu tố mùa vụ hay không). Tuyệt đối không phó mặc hoàn toàn kết luận phân tích cho AI.
- **Nguyên tắc "Hiểu mới dùng":** Tuyệt đối KHÔNG copy/paste code nếu chưa hiểu rõ các tham số truyền vào (như `window`, `lags`, `period`). Yêu cầu AI giải thích lại bằng ngôn ngữ bình dân nếu code quá phức tạp.

**Quy tắc 2: Vibe Coding (Dos & Don'ts)**
- **Được làm (Dos):** Ưu tiên tính đơn giản, tận dụng tối đa sức mạnh có sẵn của `pandas` và `statsmodels`. Code phải ngắn gọn, tường minh, đồ thị vẽ ra phải dễ nhìn, có đủ tiêu đề (title) và nhãn trục (x/y label).
- **Không được làm (Don'ts):** Tuyệt đối không "over-engineer". Không dùng vòng lặp `for/while` thủ công để tính toán những thứ mà `pandas` (như `.rolling()`, `.interpolate()`) đã hỗ trợ. Không vẽ đồ thị quá màu mè gây rối mắt.

**Quy tắc 3: Tuân thủ TÀI LIỆU ĐẶC TẢ (Strict Parameters)**
Bắt buộc phải tuân thủ chính xác các con số kỹ thuật được nêu trong đề bài, không được tự ý làm tròn hay thay đổi:
- **Bài 1:** Tính ACF và PACF tối đa đến đúng **40 lags**.
- **Bài 2:** Sinh chính xác **1000 điểm** dữ liệu cho tập White Noise.
- **Bài 3 & 5:** Chu kỳ mùa vụ (period/window) mặc định gán là **12** (vì dữ liệu AirPassengers tính theo tháng).

**Quy tắc 4: Giới hạn thư viện**
- Chỉ được phép sử dụng các thư viện phân tích cơ bản và Time Series: `pandas`, `numpy`, `matplotlib`, `seaborn` (dùng để vẽ Heatmap), và `statsmodels` (dùng cho ACF/PACF, Decomposition).
- **Lưu ý:** Lab 01 chỉ tập trung vào Tiền xử lý & Khám phá dữ liệu (EDA). NGHIÊM CẤM sử dụng các thư viện huấn luyện mô hình (như `scikit-learn` hay `pmdarima`) trong thư mục của Lab 01.

**Quy tắc 5: Giải thích code (Docstring) & Trả lời bằng Markdown**
- **Trong file `.py` (`src/lab01_utils/`):** Mọi hàm phải có Docstring 3 ý chuẩn mực: (1) Hàm làm gì? (2) Đầu vào (Parameters)? (3) Đầu ra (Returns)?
- **Trong file `.ipynb` (`notebooks/`):** Sau mỗi đoạn code sinh ra biểu đồ, BẮT BUỘC phải có một ô text (Markdown) trả lời trực diện các câu hỏi của đề bài (Ví dụ: *Chuỗi có xu hướng không? Lag nào quan trọng nhất? Residual có giống white noise không?*). Không được để trống phần nhận xét.

**Quy tắc 6: Tôn trọng kiến trúc thư mục**
- Tuyệt đối KHÔNG tự ý đổi tên file hay gộp chung file logic với file chạy thực tế.
- File logic chứa hàm (ví dụ: `quality.py`, `plotting.py`) phải nằm gọn trong `src/lab01_utils/`. 
- Thư mục `notebooks/lab01/` chỉ dùng để gọi hàm ra chạy và trình bày kết quả (import từ thư mục `src`).

**Quy tắc 7: Bất khả xâm phạm Dữ liệu gốc (Data Integrity)**
- Tệp `data/raw/airline-passengers.csv` là **Read-Only** (Chỉ đọc). Cấm mọi hành vi ghi đè, xóa hoặc sửa đổi trực tiếp vào file này.
- Khi thực hành Bài 4 (Cố tình tạo dữ liệu thiếu NaN) và làm sạch dữ liệu (Điền khuyết, loại bỏ Outlier), toàn bộ kết quả phải được xuất ra thành các file CSV mới lưu tại `data/processed/lab01/` (Ví dụ: `data_missing_injected.csv` và `data_cleaned.csv`).