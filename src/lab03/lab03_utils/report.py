"""Generate synchronized Markdown, HTML and PDF report artifacts."""
from __future__ import annotations
from pathlib import Path
import html
import math
import mistune
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether,
)


def _fmt(value, digits=3):
    if value is None:
        return "N/A"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "N/A"
        return f"{number:,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return out


_FIGURE_TITLES = {
    "series": "Chuỗi tải điện theo giờ",
    "split": "Chia dữ liệu theo thời gian",
    "forecast_test": "Dự báo cuốn chiếu trên tập kiểm tra",
    "residuals": "Chẩn đoán phần dư",
    "model_comparison": "So sánh mô hình",
    "cv": "Kiểm định chéo cuốn chiếu",
    "importance": "Mức quan trọng của đặc trưng",
    "future_forecast": "Dự báo tương lai đệ quy",
    "actual_vs_prediction": "Giá trị thực tế và dự báo",
    "test_mae_weekday_hour": "MAE theo thứ trong tuần và giờ",
    "hourly_series": "Chuỗi tải điện theo giờ",
    "temporal_split": "Chia dữ liệu theo thời gian",
    "test_and_residuals": "Dự báo và phần dư trên tập kiểm tra",
    "test_model_comparison": "So sánh mô hình trên tập kiểm tra",
    "future_24h_forecast": "Dự báo 24 giờ tiếp theo",
    "temporal_profiles": "Hồ sơ theo thời gian",
    "daily_trend": "Xu hướng theo ngày",
    "walk_forward_cv": "Kiểm định chéo cuốn chiếu",
    "test_error_profiles": "Phân bố sai số trên tập kiểm tra",
    "feature_effects": "Ảnh hưởng của đặc trưng",
}
_MODEL_TITLES = {
    "Naive Last-Value": "Giá trị gần nhất",
    "Linear Regression": "Hồi quy tuyến tính",
    "CART": "Cây hồi quy CART",
    "Random Forest": "Rừng ngẫu nhiên",
    "XGBoost-style": "Mô hình tăng cường kiểu XGBoost",
    "Naive Baseline": "Mốc đối chứng giá trị gần nhất",
    "Linear Regression (Scratch)": "Hồi quy tuyến tính (tự cài đặt)",
}
_DATASET_TYPES = {
    "real_source": "dữ liệu thật",
    "synthetic_demo": "dữ liệu mô phỏng",
}
_RUN_STATUSES = {
    "COMPLETED": "Hoàn tất",
    "FAILED": "Thất bại",
    "SUCCESS": "Thành công",
}


def display_model_name(name):
    return _MODEL_TITLES.get(name, name)


def display_dataset_type(dataset_type):
    return _DATASET_TYPES.get(dataset_type, dataset_type)


def display_run_status(status):
    return _RUN_STATUSES.get(status, status)


def _markdown(report):
    ds = report["dataset"]
    split = report["split"]
    forecast_rows = report.get("forecast_training_rows", "toàn bộ số quan sát có nhãn")
    forecast_rows_text = f"{forecast_rows:,}" if isinstance(forecast_rows, (int, np.integer)) else str(forecast_rows)
    lines = [
        "# Báo cáo dự báo chuỗi thời gian — Lab 03",
        "",
        f"**Mã lần chạy:** `{report['run_id']}`  ",
        f"**Bộ dữ liệu:** `{ds['dataset_name']}` (**{display_dataset_type(ds['dataset_type'])}**)  ",
        f"**Mô hình được chọn:** **{display_model_name(report['champion'])}** (theo RMSE trên tập xác thực)  ",
        f"**Trạng thái lần chạy:** **{display_run_status(report['status'])}**",
        "",
        "> **Lưu ý:** Chỉ tính chỉ số trên tập kiểm tra sau khi đã chọn mô hình. Đặc trưng ở tập kiểm tra dùng giao thức dự báo cuốn chiếu trước một bước; giá trị thực tế ở các thời điểm kiểm tra trước đó có thể được dùng để tạo độ trễ cho thời điểm tiếp theo. Giao thức này khác với dự báo 24 giờ từ một mốc cố định.",
        "",
        "## 1. Tóm tắt kết quả",
        "",
        report["summary_text"],
        "",
        "## 2. Dữ liệu và chất lượng dữ liệu",
        "",
        f"- Đường dẫn nguồn: `{ds.get('source_path', 'tạo trong bộ nhớ')}`",
        f"- Loại dữ liệu: **{display_dataset_type(ds['dataset_type'])}**",
        f"- Số quan sát sau làm sạch: **{ds['observations_after_cleaning']:,}**",
        f"- Khoảng thời gian: **{ds['start']}** đến **{ds['end']}**",
        f"- Tần suất: **{ds.get('frequency', 'h')}**",
        f"- Số mốc thời gian trùng đã loại: **{ds.get('duplicate_timestamps_removed', 0)}**",
        f"- Số khoảng thời gian thiếu đã bổ sung: **{ds.get('missing_periods_inserted', 0)}**",
        f"- Số giá trị thiếu còn lại sau làm sạch: **{ds.get('missing_values_remaining', 0)}**",
        "",
        "## 3. Tiền xử lý và chia dữ liệu theo thời gian",
        "",
        "Các dòng đặc trưng được sắp theo mốc thời gian. Tập huấn luyện, tập xác thực và tập kiểm tra là các đoạn thời gian liên tiếp, không xáo trộn.",
    ]
    lines += _md_table(
        ["Tập dữ liệu", "Số dòng", "Bắt đầu", "Kết thúc", "Tỷ lệ mục tiêu"],
        [[{"train": "Huấn luyện", "validation": "Xác thực", "test": "Kiểm tra"}[name],
          f"{split[name]['rows']:,}", split[name]["start"],
          split[name]["end"], f"{split[name]['ratio']:.0%}"]
         for name in ("train", "validation", "test")],
    )
    lines += [
        "",
        "### Tạo đặc trưng",
        "",
        f"- Đặc trưng độ trễ: `{report['feature_config']['lags']}`",
        f"- Cửa sổ trượt: `{report['feature_config']['rolling_windows']}`; phép tính được dịch một mốc thời gian trước khi tổng hợp.",
        f"- Đặc trưng lịch: {'Có' if report['feature_config']['include_calendar_features'] else 'Không'}",
        f"- Tổng số đặc trưng: **{report['feature_count']}**",
        "",
        "## 4. Kiểm định chéo cuốn chiếu theo thời gian",
        "",
        f"Kiểm định chéo cửa sổ mở rộng sử dụng **{report['n_splits']} lượt chia** chỉ trong tập huấn luyện. Mỗi lượt huấn luyện trên đoạn trước và xác thực trên đoạn kế tiếp. Tập xác thực và tập kiểm tra bên ngoài không tham gia kiểm định chéo.",
        "",
    ]
    cv_rows = []
    for row in report["cv_summary"]:
        cv_rows.append([display_model_name(row["model"]), _fmt(row.get("RMSE_mean")), _fmt(row.get("RMSE_std")),
                        _fmt(row.get("MAE_mean")), _fmt(row.get("R2_mean"))])
    if cv_rows:
        lines += _md_table(["Mô hình", "RMSE trung bình", "Độ lệch chuẩn RMSE", "MAE trung bình", "R² trung bình"], cv_rows)
    else:
        lines.append("Lần chạy này đã tắt kiểm định chéo; không diễn giải kết quả như một đánh giá đủ 5 lượt chia.")
    lines += ["", "## 5. So sánh mô hình", ""]
    model_rows = []
    for row in report["model_rows"]:
        model_rows.append([
            display_model_name(row["model"]), _fmt(row.get("cv_RMSE_mean")), _fmt(row.get("validation_RMSE")),
            _fmt(row.get("test_MAE")), _fmt(row.get("test_RMSE")),
            _fmt(row.get("test_MAPE (%)")), _fmt(row.get("test_R2")),
            _fmt(row.get("fit_seconds"), 2),
        ])
    lines += _md_table(
        ["Mô hình", "RMSE CV trung bình", "RMSE xác thực", "MAE kiểm tra", "RMSE kiểm tra", "MAPE kiểm tra (%)", "R² kiểm tra", "Thời gian khớp (giây)"],
        model_rows,
    )
    lines += [
        "",
        "### Quy trình chọn mô hình",
        "",
        "1. Các thuật toán ứng viên được huấn luyện trên tập huấn luyện và so sánh trên tập xác thực.",
        "2. Mô hình tốt nhất được chọn theo RMSE xác thực thấp nhất; không dùng tập kiểm tra để chọn mô hình.",
        "3. Mỗi ứng viên được khớp lại trên tập huấn luyện + tập xác thực, sau đó đánh giá một lần trên tập kiểm tra được giữ riêng.",
        "4. Để giảm thời gian chạy, mô hình dạng cây có thể giới hạn số dòng huấn luyện bằng cách chọn đều các dòng theo thời gian. Số dòng thực tế được ghi trong `model_summary.csv`.",
        "",
        "## 6. Giai đoạn suy luận",
        "",
        f"Sau khi đánh giá tập kiểm tra, mô hình tốt nhất theo tập xác thực được khớp lại trên {forecast_rows_text} quan sát có nhãn"
        f"{' đến ' + str(report['forecast_training_end']) if report.get('forecast_training_end') else ''}, rồi dùng để **dự báo đệ quy {report['forecast_horizon']} giờ**. Mỗi giá trị dự báo được nối vào lịch sử để tạo độ trễ/cửa sổ trượt cho bước tiếp theo. Không sử dụng giá trị thực tương lai. Đây là mô hình riêng với mô hình khớp trên tập huấn luyện + tập xác thực để chấm điểm tập kiểm tra.",
        "",
        "## 7. Kết quả thí nghiệm notebook lưu trữ (thí nghiệm riêng)",
        "",
        "Notebook lưu trữ có một thí nghiệm trước đây dùng các thư viện scikit-learn và XGBoost. Các giá trị được giữ lại để truy xuất nguồn gốc; đây không phải kết quả hiện tại từ mô hình NumPy tự cài đặt và không nên xem là phép so sánh đối chứng tương đương.",
        "",
    ]
    legacy = report.get("legacy_snapshot", [])
    if legacy:
        lines += _md_table(["Mô hình trong notebook cũ", "MAE", "RMSE", "MAPE (%)", "R²"],
                           [[display_model_name(r["model"]), _fmt(r.get("MAE")), _fmt(r.get("RMSE")),
                             _fmt(r.get("MAPE (%)")), _fmt(r.get("R2"))] for r in legacy])
    else:
        lines.append("Kết quả lưu trữ được ghi riêng tại `reports/lab03/legacy/`.")
    lines += [
        "",
        "## 8. Biểu đồ",
        "",
    ]
    for title, path in report["figures"].items():
        p = Path(path)
        fig_ref = report.get('figure_relative_paths', {}).get(title, 'figures/' + p.name)
        figure_title = _FIGURE_TITLES.get(title, title.replace('_', ' ').title())
        lines += [f"### {figure_title}", "", f"![{figure_title}]({fig_ref})", ""]
    lines += [
        "## 9. Các chỉ số đánh giá",
        "",
        "- **MAE:** sai số tuyệt đối trung bình theo đơn vị của biến mục tiêu; càng thấp càng tốt.",
        "- **MSE:** sai số bình phương trung bình; phạt sai số lớn mạnh hơn.",
        "- **RMSE:** căn bậc hai của MSE, cùng đơn vị với biến mục tiêu; càng thấp càng tốt.",
        "- **MAPE:** sai số phần trăm tuyệt đối trung bình trên các giá trị thực khác 0; không phù hợp khi giá trị thực bằng hoặc gần 0.",
        "- **sMAPE:** sai số phần trăm đối xứng, dùng |giá trị thực| + |giá trị dự báo| làm mẫu số.",
        "- **R²:** tỷ lệ biến thiên của biến mục tiêu được giải thích so với mốc dự báo trung bình; có thể âm trên dữ liệu ngoài mẫu.",
        "",
        "## 10. Khả năng tái lập và giới hạn",
        "",
        "- Để chạy lại, mở `notebooks/lab03/Lab03_Visual_Exploration.ipynb` từ thư mục gốc của kho mã và chạy lần lượt tất cả ô lệnh.",
        "- Đặt tệp gốc `AEP_hourly.csv` trong `data/raw/`, gồm các cột `Datetime` và `AEP_MW`, để chạy thí nghiệm trên dữ liệu thật.",
        "- Nếu thiếu tệp AEP, chỉ chạy kiểm tra nhanh với dữ liệu tổng hợp khi được ghi nhãn rõ ràng; điểm số đó không phải kết quả trên AEP.",
        "- Linear Regression, CART, Random Forest và bộ tăng cường kiểu XGBoost được cài đặt bằng NumPy; pipeline mới không gọi estimator/chỉ số của scikit-learn hoặc XGBoost.",
        "- Bộ tăng cường là bản cài đặt phục vụ học tập của cây tăng cường gradient bậc hai với các ngưỡng chia dạng histogram, không phải thư viện XGBoost chính thức.",
        "- CART độc lập dùng xấp xỉ ngưỡng ứng viên theo phân vị để giới hạn thời gian; không tìm kiếm toàn bộ ngưỡng có thể.",
        "- Đánh giá trên tập kiểm tra là dự báo cuốn chiếu trước một bước. Dự báo tự hồi quy nhiều bước được trình bày riêng trong kết quả suy luận tương lai.",
        "",
    ]
    return "\n".join(lines)


def _register_font():
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ]
    for regular_path, bold_path in candidates:
        if not Path(regular_path).exists() or not Path(bold_path).exists():
            continue
        if "LabDV" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("LabDV", regular_path))
        if "LabDV-Bold" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("LabDV-Bold", bold_path))
        return "LabDV", "LabDV-Bold"
    return "Helvetica", "Helvetica-Bold"


def _build_pdf(report, pdf_path):
    regular, bold = _register_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LabTitle", parent=styles["Title"], fontName=bold,
                              fontSize=17, leading=22, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="LabH1", parent=styles["Heading2"], fontName=bold,
                              fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="LabBody", parent=styles["BodyText"], fontName=regular,
                              fontSize=8, leading=11, spaceAfter=4))
    styles.add(ParagraphStyle(name="LabSmall", parent=styles["BodyText"], fontName=regular,
                              fontSize=6.4, leading=8))
    story = [
        Paragraph("LAB 03 - DỰ BÁO CHUỖI THỜI GIAN", styles["LabTitle"]),
        Paragraph(f"Mã lần chạy: {html.escape(str(report['run_id']))} | Bộ dữ liệu: {html.escape(str(report['dataset']['dataset_name']))}", styles["LabBody"]),
        Paragraph(f"Loại dữ liệu: {html.escape(display_dataset_type(report['dataset']['dataset_type']))} | Mô hình được chọn: {html.escape(display_model_name(report['champion']))}", styles["LabBody"]),
        Paragraph("1. Tóm tắt kết quả", styles["LabH1"]),
        Paragraph(html.escape(str(report["summary_text"])), styles["LabBody"]),
        Paragraph("2. Dữ liệu và chia dữ liệu theo thời gian", styles["LabH1"]),
    ]
    ds = report["dataset"]
    story.append(Paragraph(
        f"Số quan sát sau làm sạch: {ds['observations_after_cleaning']:,}; khoảng thời gian: "
        f"{html.escape(str(ds['start']))} đến {html.escape(str(ds['end']))}; "
        f"tần suất: {html.escape(str(ds.get('frequency', 'h')))}; "
        f"mốc thời gian trùng đã loại: {ds.get('duplicate_timestamps_removed', 0)}; "
        f"khoảng thời gian thiếu đã bổ sung: {ds.get('missing_periods_inserted', 0)}.", styles["LabBody"]))
    split_rows = [["Tập dữ liệu", "Số dòng", "Bắt đầu", "Kết thúc", "Tỷ lệ"]]
    for name in ("train", "validation", "test"):
        p = report["split"][name]
        split_rows.append([{"train": "Huấn luyện", "validation": "Xác thực", "test": "Kiểm tra"}[name],
                           f"{p['rows']:,}", str(p["start"]), str(p["end"]), f"{p['ratio']:.0%}"])
    split_table = Table(split_rows, repeatRows=1, colWidths=[22*mm, 16*mm, 49*mm, 49*mm, 17*mm])
    split_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), regular), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTSIZE", (0, 0), (-1, -1), 6.0), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5ECF3")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [split_table, Spacer(1, 6), Paragraph("3. So sánh mô hình trên tập kiểm tra", styles["LabH1"])]
    result_rows = [["Mô hình", "RMSE CV", "RMSE xác thực", "MAE kiểm tra", "RMSE kiểm tra", "MAPE %", "R²"]]
    for row in report["model_rows"]:
        result_rows.append([
            Paragraph(html.escape(display_model_name(row["model"])), styles["LabSmall"]),
            _fmt(row.get("cv_RMSE_mean")), _fmt(row.get("validation_RMSE")),
            _fmt(row.get("test_MAE")), _fmt(row.get("test_RMSE")),
            _fmt(row.get("test_MAPE (%)")), _fmt(row.get("test_R2")),
        ])
    res_table = Table(result_rows, repeatRows=1,
                      colWidths=[42*mm, 22*mm, 22*mm, 21*mm, 22*mm, 20*mm, 14*mm])
    res_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), regular), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTSIZE", (0, 0), (-1, -1), 5.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5ECF3")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [res_table, Spacer(1, 6), Paragraph("4. Kiểm định chéo cuốn chiếu theo thời gian", styles["LabH1"])]
    if report["cv_summary"]:
        cv_rows = [["Mô hình", "RMSE trung bình", "Độ lệch chuẩn RMSE", "MAE trung bình", "R² trung bình"]]
        for row in report["cv_summary"]:
            cv_rows.append([display_model_name(row["model"]), _fmt(row.get("RMSE_mean")), _fmt(row.get("RMSE_std")),
                            _fmt(row.get("MAE_mean")), _fmt(row.get("R2_mean"))])
        cv_table = Table(cv_rows, repeatRows=1, colWidths=[48*mm, 28*mm, 28*mm, 28*mm, 25*mm])
        cv_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), regular), ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 6.0), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5ECF3")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ]))
        story.append(cv_table)
    else:
        story.append(Paragraph("Lần chạy này đã tắt kiểm định chéo.", styles["LabBody"]))
    legacy_rows = report.get("legacy_snapshot", [])
    if legacy_rows:
        story += [Spacer(1, 8), Paragraph("5. Kết quả notebook lưu trữ (thí nghiệm riêng)", styles["LabH1"])]
        story.append(Paragraph(
            "Các giá trị sau được trích từ notebook lưu trữ đã chạy trước đây. "
            "Chúng được tạo bằng estimator từ thư viện và cách chia 80/20; không phải kết quả "
            "từ pipeline NumPy hiện tại và không phải phép so sánh tương đương.",
            styles["LabBody"]))
        legacy_table_rows = [["Mô hình", "MAE", "RMSE", "MAPE %", "R²"]]
        for row in legacy_rows:
            legacy_table_rows.append([
                Paragraph(html.escape(display_model_name(row["model"])), styles["LabSmall"]),
                _fmt(row.get("MAE")), _fmt(row.get("RMSE")),
                _fmt(row.get("MAPE (%)")), _fmt(row.get("R2"))])
        legacy_table = Table(legacy_table_rows, repeatRows=1,
                             colWidths=[70*mm, 23*mm, 23*mm, 24*mm, 18*mm])
        legacy_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), regular), ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 6.2), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5ECF3")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(legacy_table)
    story += [
        Spacer(1, 8), Paragraph("6. Suy luận và ghi chú phương pháp", styles["LabH1"]),
        Paragraph(
            f"Mô hình được chọn theo RMSE trên tập xác thực: {html.escape(display_model_name(report['champion']))}. "
            f"Sau khi đánh giá tập kiểm tra, mô hình dự báo tương lai được khớp lại trên "
            f"{html.escape(str(report.get('forecast_training_rows', 'toàn bộ số quan sát có nhãn')))} quan sát có nhãn"
            f"{' đến ' + html.escape(str(report['forecast_training_end'])) if report.get('forecast_training_end') else ''} và dự báo đệ quy "
            f"{report['forecast_horizon']} giờ tiếp theo. Đánh giá trên tập kiểm tra là cuốn chiếu trước một bước: "
            "đặc trưng độ trễ ở một thời điểm kiểm tra có thể chứa giá trị thực tế của các thời điểm kiểm tra trước đó.",
            styles["LabBody"]),
        Paragraph(
            "Không xáo trộn ngẫu nhiên dữ liệu. Kiểm định chéo cuốn chiếu chỉ dùng tập huấn luyện. "
            "Đặc trưng độ trễ và cửa sổ trượt chỉ dùng thông tin quá khứ. Các thuật toán mới được "
            "cài đặt bằng NumPy; mô hình kiểu XGBoost phục vụ học tập, không phải thư viện XGBoost chính thức. "
            "Kết quả kiểm tra nhanh bằng dữ liệu tổng hợp không phải kết luận trên dữ liệu AEP.", styles["LabBody"]),
    ]
    # Each generated image is embedded in the PDF, not merely referenced by filename.
    for key, path in report["figures"].items():
        if not Path(path).exists():
            continue
        story += [PageBreak(), Paragraph(f"Biểu đồ: {html.escape(_FIGURE_TITLES.get(key, key.replace('_', ' ').title()))}", styles["LabH1"]),
                  Image(str(path), width=170*mm, height=60*mm)]
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=13*mm, bottomMargin=13*mm,
                            title="Báo cáo dự báo chuỗi thời gian Lab 03", author="Lab TimeSeries")
    doc.build(story)


def write_report_bundle(report: dict, output_dir: str | Path):
    """Purpose: create matching Markdown, HTML and PDF reports.

    Input: a report dictionary with dataset/split/model/CV/figure metadata.
    Output: absolute paths to Lab03_Report.md, Lab03_Report.html and Lab03_Report.pdf.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    md = _markdown(report)
    md_path = output / "Lab03_Report.md"
    html_path = output / "Lab03_Report.html"
    pdf_path = output / "Lab03_Report.pdf"
    md_path.write_text(md, encoding="utf-8")
    renderer = mistune.create_markdown(plugins=["table", "strikethrough"])
    body = renderer(md)
    html_doc = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo cáo Lab 03</title><style>
body{font-family:Arial,'DejaVu Sans',sans-serif;max-width:1120px;margin:32px auto;padding:0 28px;line-height:1.58;color:#20242a}
h1,h2,h3{color:#18324f}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0;display:block;overflow-x:auto}
th,td{border:1px solid #c9d2dc;padding:7px 9px;text-align:left}th{background:#e8eef5}code{background:#f2f4f7;padding:2px 4px;border-radius:3px}
li{margin:4px 0}img{max-width:100%;height:auto;border:1px solid #e3e7ec;border-radius:4px}
blockquote{border-left:4px solid #8b9bad;padding:5px 14px;color:#4a5562;background:#f7f9fb}
</style></head><body>""" + body + "</body></html>"
    html_path.write_text(html_doc, encoding="utf-8")
    _build_pdf(report, pdf_path)
    return {"md": str(md_path), "html": str(html_path), "pdf": str(pdf_path)}
