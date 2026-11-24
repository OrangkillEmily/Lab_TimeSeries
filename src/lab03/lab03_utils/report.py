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


def _markdown(report):
    ds = report["dataset"]
    split = report["split"]
    lines = [
        "# Lab 03 - Time Series Forecasting Report",
        "",
        f"**Run ID:** `{report['run_id']}`  ",
        f"**Dataset:** `{ds['dataset_name']}` (`{ds['dataset_type']}`)  ",
        f"**Selected model:** **{report['champion']}** (selected by Validation RMSE)  ",
        f"**Run status:** `{report['status']}`",
        "",
        "> **Important:** Test metrics are reported only after model selection. The Test features are rolling one-step-ahead features, so actual observations from earlier test timestamps are available to form later lag features. This is not the same as a fixed-origin 24-hour forecast.",
        "",
        "## 1. Executive summary",
        "",
        report["summary_text"],
        "",
        "## 2. Dataset and data quality",
        "",
        f"- Source path: `{ds.get('source_path', 'generated in memory')}`",
        f"- Dataset type: **{ds['dataset_type']}**",
        f"- Observations after cleaning: **{ds['observations_after_cleaning']:,}**",
        f"- Time span: **{ds['start']}** to **{ds['end']}**",
        f"- Frequency: **{ds.get('frequency', 'h')}**",
        f"- Duplicate timestamps removed: **{ds.get('duplicate_timestamps_removed', 0)}**",
        f"- Missing periods inserted: **{ds.get('missing_periods_inserted', 0)}**",
        f"- Missing values remaining after cleaning: **{ds.get('missing_values_remaining', 0)}**",
        "",
        "## 3. Preprocessing and chronological split",
        "",
        "Feature rows are ordered by timestamp. Train/Validation/Test are contiguous chronological partitions with no shuffle.",
    ]
    lines += _md_table(
        ["Partition", "Rows", "Start", "End", "Target ratio"],
        [[name.title(), f"{split[name]['rows']:,}", split[name]["start"],
          split[name]["end"], f"{split[name]['ratio']:.0%}"]
         for name in ("train", "validation", "test")],
    )
    lines += [
        "",
        "### Feature engineering",
        "",
        f"- Lag features: `{report['feature_config']['lags']}`",
        f"- Rolling windows: `{report['feature_config']['rolling_windows']}`; rolling calculations are shifted by one timestamp before aggregation.",
        f"- Calendar features: `{report['feature_config']['include_calendar_features']}`",
        f"- Total feature count: **{report['feature_count']}**",
        "",
        "## 4. Walk-forward cross-validation",
        "",
        f"Expanding-window CV uses **{report['n_splits']} folds** inside the Train partition only. Each fold trains on an earlier block and validates on the following block. Validation and Test partitions are not included in CV.",
        "",
    ]
    cv_rows = []
    for row in report["cv_summary"]:
        cv_rows.append([row["model"], _fmt(row.get("RMSE_mean")), _fmt(row.get("RMSE_std")),
                        _fmt(row.get("MAE_mean")), _fmt(row.get("R2_mean"))])
    if cv_rows:
        lines += _md_table(["Model", "Mean RMSE", "Std RMSE", "Mean MAE", "Mean R²"], cv_rows)
    else:
        lines.append("CV was disabled for this run; do not interpret this run as a 5-fold evaluation.")
    lines += ["", "## 5. Model comparison", ""]
    model_rows = []
    for row in report["model_rows"]:
        model_rows.append([
            row["model"], _fmt(row.get("cv_RMSE_mean")), _fmt(row.get("validation_RMSE")),
            _fmt(row.get("test_MAE")), _fmt(row.get("test_RMSE")),
            _fmt(row.get("test_MAPE (%)")), _fmt(row.get("test_R2")),
            _fmt(row.get("fit_seconds"), 2),
        ])
    lines += _md_table(
        ["Model", "CV RMSE mean", "Validation RMSE", "Test MAE", "Test RMSE", "Test MAPE (%)", "Test R²", "Fit s"],
        model_rows,
    )
    lines += [
        "",
        "### Model selection protocol",
        "",
        "1. Candidate algorithms are trained on Train and compared on Validation.",
        "2. The champion is selected by the lowest Validation RMSE; Test is not used to select the champion.",
        "3. Each candidate is refitted on Train + Validation, then evaluated once on the held-out Test partition.",
        "4. Tree-based models may cap their internal training rows to a deterministic evenly spaced subset for runtime. The actual number of rows used is recorded in `model_summary.csv`.",
        "",
        "## 6. Inference phase",
        "",
        f"The selected model was refitted on Train + Validation and used for **{report['forecast_horizon']} recursive hourly steps**. Every predicted value is appended to history and is used to build the next lag/rolling row. No future ground truth is used for this future forecast.",
        "",
        "## 7. Legacy notebook experiment snapshot (separate experiment)",
        "",
        "The archived notebook contains a prior experiment using the library implementations of scikit-learn and XGBoost. Its values are preserved for provenance only; they are not current results from the NumPy scratch models and should not be compared as a controlled, apples-to-apples benchmark.",
        "",
    ]
    legacy = report.get("legacy_snapshot", [])
    if legacy:
        lines += _md_table(["Prior notebook model", "MAE", "RMSE", "MAPE (%)", "R²"],
                           [[r["model"], _fmt(r.get("MAE")), _fmt(r.get("RMSE")),
                             _fmt(r.get("MAPE (%)")), _fmt(r.get("R2"))] for r in legacy])
    else:
        lines.append("Legacy snapshot file is archived separately in `reports/lab03/legacy/`.")
    lines += [
        "",
        "## 8. Figures",
        "",
    ]
    for title, path in report["figures"].items():
        p = Path(path)
        fig_ref = report.get('figure_relative_paths', {}).get(title, 'figures/' + p.name)
        lines += [f"### {title.replace('_', ' ').title()}", "", f"![{title}]({fig_ref})", ""]
    lines += [
        "## 9. Evaluation metrics",
        "",
        "- **MAE:** average absolute error, in target units; lower is better.",
        "- **MSE:** average squared error; penalizes large errors more strongly.",
        "- **RMSE:** square root of MSE, in target units; lower is better.",
        "- **MAPE:** average absolute percentage error over non-zero actual values; unsuitable when actual values are zero or close to zero.",
        "- **sMAPE:** symmetric percentage error using |actual| + |prediction| in the denominator.",
        "- **R²:** proportion of target variance explained relative to a mean baseline; can be negative on out-of-sample data.",
        "",
        "## 10. Reproducibility and limitations",
        "",
        "- Re-run from the repository root by opening `notebooks/lab03/Lab03_TimeSeries.ipynb` and choosing Run All.",
        "- Place the original `AEP_hourly.csv` in `data/raw/` with columns `Datetime` and `AEP_MW` for the real-data experiment.",
        "- If the AEP file is absent, the pipeline can run a clearly labelled synthetic smoke test; its scores are not AEP findings.",
        "- Linear Regression, CART, Random Forest and the XGBoost-style booster are implemented in this project with NumPy; no scikit-learn or XGBoost estimators/metrics are called by the new pipeline.",
        "- The booster is an educational second-order gradient-boosted-tree implementation with histogram split candidates, not the official XGBoost library.",
        "- The standalone CART uses a quantile candidate-threshold approximation to limit runtime; it is not exhaustive threshold search.",
        "- The Test evaluation is rolling one-step ahead. Multi-step free-running prediction is represented separately by the recursive future inference output.",
        "",
    ]
    return "\n".join(lines)


def _register_font():
    regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if Path(regular_path).exists() and Path(bold_path).exists():
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
        Paragraph("LAB 03 - TIME SERIES FORECASTING", styles["LabTitle"]),
        Paragraph(f"Run ID: {html.escape(str(report['run_id']))} | Dataset: {html.escape(str(report['dataset']['dataset_name']))}", styles["LabBody"]),
        Paragraph(f"Dataset type: {html.escape(str(report['dataset']['dataset_type']))} | Selected model: {html.escape(str(report['champion']))}", styles["LabBody"]),
        Paragraph("1. Executive summary", styles["LabH1"]),
        Paragraph(html.escape(str(report["summary_text"])), styles["LabBody"]),
        Paragraph("2. Dataset and chronological split", styles["LabH1"]),
    ]
    ds = report["dataset"]
    story.append(Paragraph(
        f"Observations after cleaning: {ds['observations_after_cleaning']:,}; time span: "
        f"{html.escape(str(ds['start']))} to {html.escape(str(ds['end']))}; "
        f"frequency: {html.escape(str(ds.get('frequency', 'h')))}; "
        f"duplicate timestamps removed: {ds.get('duplicate_timestamps_removed', 0)}; "
        f"missing periods inserted: {ds.get('missing_periods_inserted', 0)}.", styles["LabBody"]))
    split_rows = [["Partition", "Rows", "Start", "End", "Ratio"]]
    for name in ("train", "validation", "test"):
        p = report["split"][name]
        split_rows.append([name.title(), f"{p['rows']:,}", str(p["start"]), str(p["end"]), f"{p['ratio']:.0%}"])
    split_table = Table(split_rows, repeatRows=1, colWidths=[22*mm, 16*mm, 49*mm, 49*mm, 17*mm])
    split_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), regular), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTSIZE", (0, 0), (-1, -1), 6.0), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5ECF3")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [split_table, Spacer(1, 6), Paragraph("3. Model comparison on held-out Test", styles["LabH1"])]
    result_rows = [["Model", "CV RMSE", "Val RMSE", "Test MAE", "Test RMSE", "MAPE %", "R²"]]
    for row in report["model_rows"]:
        result_rows.append([
            Paragraph(html.escape(str(row["model"])), styles["LabSmall"]),
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
    story += [res_table, Spacer(1, 6), Paragraph("4. Walk-forward cross-validation", styles["LabH1"])]
    if report["cv_summary"]:
        cv_rows = [["Model", "Mean RMSE", "Std RMSE", "Mean MAE", "Mean R²"]]
        for row in report["cv_summary"]:
            cv_rows.append([row["model"], _fmt(row.get("RMSE_mean")), _fmt(row.get("RMSE_std")),
                            _fmt(row.get("MAE_mean")), _fmt(row.get("R2_mean"))])
        cv_table = Table(cv_rows, repeatRows=1, colWidths=[48*mm, 28*mm, 28*mm, 28*mm, 25*mm])
        cv_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), regular), ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 6.0), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5ECF3")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ]))
        story.append(cv_table)
    else:
        story.append(Paragraph("Cross-validation was disabled for this run.", styles["LabBody"]))
    legacy_rows = report.get("legacy_snapshot", [])
    if legacy_rows:
        story += [Spacer(1, 8), Paragraph("5. Legacy notebook snapshot (separate experiment)", styles["LabH1"])]
        story.append(Paragraph(
            "The following values were transcribed from the archived, previously executed notebook. "
            "They were produced by library-based estimators and an 80/20 split; they are not results "
            "from the current NumPy scratch pipeline and are not an apples-to-apples comparison.",
            styles["LabBody"]))
        legacy_table_rows = [["Model", "MAE", "RMSE", "MAPE %", "R²"]]
        for row in legacy_rows:
            legacy_table_rows.append([
                Paragraph(html.escape(str(row["model"])), styles["LabSmall"]),
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
        Spacer(1, 8), Paragraph("6. Inference and methodology notes", styles["LabH1"]),
        Paragraph(
            f"Champion selected on Validation RMSE: {html.escape(str(report['champion']))}. "
            f"The final model is refitted on Train + Validation and recursively predicts "
            f"{report['forecast_horizon']} future hourly steps. The held-out Test evaluation is "
            "rolling one-step ahead: lag features for a test timestamp may include earlier actual test observations.",
            styles["LabBody"]),
        Paragraph(
            "No random shuffle is used. Walk-forward CV is confined to Train. Lag and rolling "
            "features use past information only. The new algorithms are NumPy implementations; "
            "the XGBoost-style model is educational and not the official XGBoost library. "
            "Explicit synthetic smoke-test data are not AEP findings.", styles["LabBody"]),
    ]
    # Each generated image is embedded in the PDF, not merely referenced by filename.
    for key, path in report["figures"].items():
        if not Path(path).exists():
            continue
        story += [PageBreak(), Paragraph(f"Figure: {html.escape(key.replace('_', ' ').title())}", styles["LabH1"]),
                  Image(str(path), width=170*mm, height=60*mm)]
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=13*mm, bottomMargin=13*mm,
                            title="Lab 03 Time Series Forecasting Report", author="Lab TimeSeries")
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
<title>Lab 03 Report</title><style>
body{font-family:Arial,'DejaVu Sans',sans-serif;max-width:1120px;margin:32px auto;padding:0 28px;line-height:1.58;color:#20242a}
h1,h2,h3{color:#18324f}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0;display:block;overflow-x:auto}
th,td{border:1px solid #c9d2dc;padding:7px 9px;text-align:left}th{background:#e8eef5}code{background:#f2f4f7;padding:2px 4px;border-radius:3px}
li{margin:4px 0}img{max-width:100%;height:auto;border:1px solid #e3e7ec;border-radius:4px}
blockquote{border-left:4px solid #8b9bad;padding:5px 14px;color:#4a5562;background:#f7f9fb}
</style></head><body>""" + body + "</body></html>"
    html_path.write_text(html_doc, encoding="utf-8")
    _build_pdf(report, pdf_path)
    return {"md": str(md_path), "html": str(html_path), "pdf": str(pdf_path)}
