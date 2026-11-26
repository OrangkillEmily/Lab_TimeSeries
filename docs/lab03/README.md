# Tài liệu Lab 03

## Bắt đầu tại đây

1. [Notebook khám phá trực quan Lab 03](../../notebooks/lab03/Lab03_Visual_Exploration.ipynb) — notebook tạo và lưu 19 biểu đồ khám phá, đánh giá mô hình và dự báo.
2. [Hướng dẫn chạy](EXECUTION_GUIDE.md) — môi trường, cách chạy pipeline và notebook, cùng các bước kiểm tra kết quả.
3. [Hướng dẫn tệp](FILE_GUIDE.md) — vai trò của từng thư mục và mô-đun chính.
4. [Kiến trúc](ARCHITECTURE.md) — các giai đoạn pipeline và biện pháp ngăn rò rỉ dữ liệu.
5. [Phương pháp](methodology.md) — quy trình dự báo, đặc trưng và định nghĩa chỉ số.
6. [Đặc tả tóm tắt mô hình](MODEL_SUMMARY_SPEC.md) — các trường so sánh mô hình được tạo và cách diễn giải.
7. [Danh sách kiểm tra sản phẩm bàn giao](DELIVERABLE_CHECKLIST.md) — mức độ đáp ứng yêu cầu.

## Tài liệu thuật toán

Mỗi mô hình có một tệp cài đặt trong `src/lab03/lab03_models/` và một tài liệu giải thích chi tiết tương ứng:

- [Mốc đối chứng giá trị gần nhất](models/Naive_Baseline.md)
- [Hồi quy tuyến tính tự cài đặt](models/Linear_Regression_Scratch.md)
- [Cây quyết định CART tự cài đặt](models/CART_Decision_Tree_Scratch.md)
- [Rừng ngẫu nhiên tự cài đặt](models/Random_Forest_Scratch.md)
- [Mô hình kiểu XGBoost tự cài đặt](models/XGBoost_Style_Scratch.md)

Tài liệu của từng mô hình gồm mục đích, đầu vào/đầu ra, cách thực hiện, chỉ số đánh giá và dữ liệu sử dụng.

## Tài liệu các giai đoạn pipeline

- [Nạp dữ liệu](phases/Data_Ingestion_Phase.md)
- [Làm sạch dữ liệu](phases/Data_Cleaning_Phase.md)
- [Tạo đặc trưng](phases/Feature_Engineering_Phase.md)
- [Chia dữ liệu](phases/Data_Splitting_Phase.md)
- [Kiểm định chéo](phases/Cross_Validation_Phase.md)
- [Huấn luyện](phases/Training_Phase.md)
- [Chọn mô hình](phases/Model_Selection_Phase.md)
- [Suy luận](phases/Inference_Phase.md)
- [Đánh giá](phases/Evaluation_Phase.md)
- [Lập báo cáo](phases/Reporting_Phase.md)
- [Theo dõi thí nghiệm và ghi nhật ký](phases/Experiment_Tracking_Phase.md)

## Tài liệu lưu trữ

Kết quả thí nghiệm notebook cũ được lưu riêng tại `reports/lab03/legacy/Legacy_Experiment_Results.md`; không gộp chúng với kết quả của pipeline NumPy hiện tại.
