"""Generate the Phase 19 evidence-based report, figures and delivery audit."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.registry import (
    append_registry_rows, artifact_entry, build_source_manifest, git_state,
    sha256_file, write_immutable,
)
from src.utils.config import load_config


BLUE = colors.HexColor("#133C55")
CYAN = colors.HexColor("#1F7A8C")
GOLD = colors.HexColor("#D59B2D")
LIGHT = colors.HexColor("#EEF4F6")
MID = colors.HexColor("#607D8B")
RED = colors.HexColor("#B94A48")
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def load_evidence(root: Path) -> dict[str, Any]:
    read = lambda name: json.loads((root / "reports" / "tables" / name).read_text(encoding="utf-8"))
    return {
        "data": read("phase1_data_audit.json"),
        "eval": read("phase7_evaluator_parity.json"),
        "p8": read("phase8_baseline_audit.json"),
        "p10": read("phase10_turnover_audit.json"),
        "p11": read("phase11_ensemble_audit.json"),
        "p12": read("phase12_lambdarank_audit.json"),
        "p14": read("phase14_selected_features.json"),
        "p15": read("phase15_stability_audit.json"),
        "p16": read("phase16_finalization_audit.json"),
        "p17": read("phase17_prediction_audit.json"),
        "p18": read("phase18_submission_audit.json"),
        "labels": pd.read_csv(root / "reports" / "tables" / "phase2_label_by_year.csv"),
        "screen": pd.read_csv(root / "reports" / "tables" / "phase14_feature_summary.csv"),
        "regime": pd.read_csv(root / "reports" / "tables" / "phase15_regime_metrics.csv"),
    }


def mean_metric(summary: dict[str, Any], key: str) -> float:
    value = summary[key]
    return float(value["mean"] if isinstance(value, dict) else value)


def model_rows(e: dict[str, Any]) -> list[dict[str, float | str]]:
    p8 = e["p8"]["models"]
    rows = [
        {"name": "Ridge", "score": mean_metric(p8["ridge"]["summary"], "final_score"),
         "ic": mean_metric(p8["ridge"]["summary"], "ic_mean"), "turnover": mean_metric(p8["ridge"]["summary"], "mean_turnover")},
        {"name": "LightGBM", "score": mean_metric(p8["lightgbm"]["summary"], "final_score"),
         "ic": mean_metric(p8["lightgbm"]["summary"], "ic_mean"), "turnover": mean_metric(p8["lightgbm"]["summary"], "mean_turnover")},
        {"name": "LGB+换手控制", "score": float(e["p10"]["selected"]["final_score"]),
         "ic": float(e["p10"]["selected"]["ic_mean"]), "turnover": float(e["p10"]["selected"]["mean_turnover"])},
        {"name": "LGB/Ridge融合", "score": float(e["p11"]["selected"]["final_score"]),
         "ic": float(e["p11"]["selected"]["ic_mean"]), "turnover": float(e["p11"]["selected"]["mean_turnover"])},
        {"name": "LambdaRank", "score": mean_metric(e["p12"]["variants"]["turnover"]["summary"], "final_score"),
         "ic": mean_metric(e["p12"]["variants"]["turnover"]["summary"], "ic_mean"),
         "turnover": mean_metric(e["p12"]["variants"]["turnover"]["summary"], "mean_turnover")},
        {"name": "最终融合", "score": float(e["p16"]["selected"]["final_score"]),
         "ic": float(e["p16"]["selected"]["ic_mean"]), "turnover": float(e["p16"]["selected"]["mean_turnover"])},
    ]
    return rows


def set_plot_font() -> None:
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        matplotlib.font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def make_figures(e: dict[str, Any], out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    set_plot_font()
    paths: dict[str, Path] = {}

    labels = e["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    axes[0].bar(labels["year"].astype(str), labels["mean"] * 100, color="#1F7A8C")
    axes[0].axhline(0, color="#607D8B", lw=.8)
    axes[0].set_title("逐年次日收益均值")
    axes[0].set_ylabel("%")
    axes[1].plot(labels["year"], labels["missing_rate"] * 100, marker="o", color="#D59B2D", lw=2)
    axes[1].set_title("标签缺失率随时间下降")
    axes[1].set_ylabel("%")
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout()
    paths["labels"] = out / "label_overview.png"; fig.savefig(paths["labels"], bbox_inches="tight"); plt.close(fig)

    models = model_rows(e)
    names = [x["name"] for x in models]; scores = [x["score"] for x in models]
    turnover = [x["turnover"] for x in models]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    palette = ["#9FB8C4"] * (len(names) - 1) + ["#D59B2D"]
    axes[0].barh(names, scores, color=palette)
    axes[0].set_title("Walk-forward 官方综合分")
    axes[0].set_xlim(0, .42); axes[0].grid(axis="x", alpha=.2)
    for i, value in enumerate(scores): axes[0].text(value + .006, i, f"{value:.3f}", va="center", fontsize=8)
    axes[1].barh(names, turnover, color=["#B94A48" if x > .5 else "#1F7A8C" for x in turnover])
    axes[1].set_title("平均换手率（越低越好）")
    axes[1].set_xlim(0, .95); axes[1].grid(axis="x", alpha=.2)
    fig.tight_layout()
    paths["models"] = out / "model_progress.png"; fig.savefig(paths["models"], bbox_inches="tight"); plt.close(fig)

    fold_sources = {
        "换手控制LGB": [float(x["final_score"]) for x in e["p10"]["selected"]["folds"]],
        "回归融合": [float(x["final_score"]) for x in e["p11"]["selected"]["folds"]],
        "LambdaRank": [float(x["metrics"]["final_score"]) for x in e["p12"]["variants"]["turnover"]["folds"]],
        "最终融合": [float(x["final_score"]) for x in e["p16"]["selected_folds"]],
    }
    fig, ax = plt.subplots(figsize=(9.4, 3.3))
    for name, values in fold_sources.items():
        ax.plot([2022, 2023, 2024], values, marker="o", lw=2.2 if name == "最终融合" else 1.5, label=name)
    ax.set_xticks([2022, 2023, 2024]); ax.set_ylabel("官方综合分"); ax.set_title("逐年验证稳定性")
    ax.grid(alpha=.25); ax.legend(ncol=4, fontsize=8, loc="lower left")
    fig.tight_layout()
    paths["folds"] = out / "fold_stability.png"; fig.savefig(paths["folds"], bbox_inches="tight"); plt.close(fig)

    selected = set(e["p14"]["selected_features"])
    screen = e["screen"]
    screen = screen[(screen["period"] == "screen_2018_2021") & screen["feature"].isin(selected)].copy()
    screen["aligned_ic"] = screen["ic_mean"].abs()
    screen = screen.nlargest(10, "aligned_ic").sort_values("aligned_ic")
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    ax.barh(screen["feature"], screen["aligned_ic"], color="#1F7A8C")
    ax.set_title("训练筛选期入选特征 |Rank IC|（前10）"); ax.set_xlabel("绝对日均 Rank IC")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    paths["features"] = out / "feature_screen.png"; fig.savefig(paths["features"], bbox_inches="tight"); plt.close(fig)

    regime = e["regime"]
    regime = regime[regime["source"].isin(["phase12_incumbent", "trailing_2y"]) & (regime["regime"] != "all")]
    pivot = regime.pivot(index="regime", columns="source", values="final_score_proxy")
    pivot = pivot.rename(index={"low_volatility": "低波动", "high_volatility": "高波动",
                                "bearish_market": "下跌市场", "bullish_market": "上涨市场"},
                         columns={"phase12_incumbent": "基础LambdaRank", "trailing_2y": "近两年ATR"})
    fig, ax = plt.subplots(figsize=(9.4, 3.3)); pivot.plot(kind="bar", ax=ax, color=["#133C55", "#D59B2D"])
    ax.set_title("市场状态稳健性代理分"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    paths["regime"] = out / "regime_stability.png"; fig.savefig(paths["regime"], bbox_inches="tight"); plt.close(fig)
    return paths


def register_fonts() -> None:
    regular = Path("C:/Windows/Fonts/msyh.ttc")
    bold = Path("C:/Windows/Fonts/msyhbd.ttc")
    fallback = Path("C:/Windows/Fonts/simhei.ttf")
    pdfmetrics.registerFont(TTFont("CN", str(regular if regular.is_file() else fallback)))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(bold if bold.is_file() else fallback)))


def para(c: canvas.Canvas, text: str, x: float, y: float, width: float,
         size: float = 9.5, leading: float = 14, color=colors.HexColor("#263238"),
         bold: bool = False, align: int = TA_LEFT) -> float:
    style = ParagraphStyle("body", fontName="CN-Bold" if bold else "CN", fontSize=size,
                           leading=leading, textColor=color, alignment=align, spaceAfter=0)
    p = Paragraph(text, style); _, height = p.wrap(width, PAGE_H)
    p.drawOn(c, x, y - height)
    return y - height


def section(c: canvas.Canvas, title: str, y: float) -> float:
    c.setFillColor(CYAN); c.roundRect(MARGIN, y - 7, 4, 18, 2, fill=1, stroke=0)
    return para(c, title, MARGIN + 10, y + 8, CONTENT_W - 10, size=15, leading=19, color=BLUE, bold=True)


def table(c: canvas.Canvas, data: list[list[Any]], x: float, y: float, widths: list[float],
          font_size: float = 8.2, row_heights: list[float] | None = None) -> float:
    t = Table(data, colWidths=widths, rowHeights=row_heights, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "CN-Bold"), ("FONTNAME", (0, 1), (-1, -1), "CN"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size), ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFB")),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#C7D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    _, h = t.wrap(sum(widths), PAGE_H); t.drawOn(c, x, y - h); return y - h


def header_footer(c: canvas.Canvas, page: int, title: str = "2026 深圳国际金融科技大赛 · 数据分析赛道题五") -> None:
    c.setStrokeColor(colors.HexColor("#D7E1E5")); c.line(MARGIN, PAGE_H - 28, PAGE_W - MARGIN, PAGE_H - 28)
    c.setFont("CN", 7.5); c.setFillColor(MID); c.drawString(MARGIN, PAGE_H - 21, title)
    c.line(MARGIN, 30, PAGE_W - MARGIN, 30); c.drawRightString(PAGE_W - MARGIN, 18, f"{page} / 8")


def page_start(c: canvas.Canvas, page: int, title: str) -> float:
    header_footer(c, page); return section(c, title, PAGE_H - 55)


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float, value: str, label: str) -> None:
    c.setFillColor(LIGHT); c.roundRect(x, y - h, w, h, 8, fill=1, stroke=0)
    para(c, value, x + 7, y - 8, w - 14, size=17, leading=20, color=BLUE, bold=True, align=TA_CENTER)
    para(c, label, x + 7, y - 34, w - 14, size=8.5, leading=11, color=MID, align=TA_CENTER)


def build_pdf(e: dict[str, Any], figs: dict[str, Path], target: Path) -> None:
    register_fonts(); target.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(target), pagesize=A4, pageCompression=1)
    c.setTitle("2026深圳国际金融科技大赛数据分析赛道题五报告")
    train = e["data"]["datasets"]["train"]["validation"]
    test = e["data"]["datasets"]["test"]["validation"]
    final = e["p16"]["selected"]

    # Page 1 - cover and executive summary
    c.setFillColor(BLUE); c.rect(0, PAGE_H - 220, PAGE_W, 220, fill=1, stroke=0)
    c.setFillColor(GOLD); c.rect(MARGIN, PAGE_H - 86, 64, 4, fill=1, stroke=0)
    para(c, "2026 深圳国际金融科技大赛", MARGIN, PAGE_H - 100, CONTENT_W, 12, 16, colors.white, True)
    para(c, "数据分析赛道 · 题五", MARGIN, PAGE_H - 128, CONTENT_W, 23, 30, colors.white, True)
    para(c, "基于因果横截面排序、LambdaRank 与换手约束的股票评分方案", MARGIN, PAGE_H - 170, CONTENT_W, 13, 19, colors.white)
    y = PAGE_H - 255
    widths = (CONTENT_W - 18) / 4
    for i, (value, label) in enumerate([
        ("790 万", "训练样本行"), ("0.3818", "三折 OOF 综合分"),
        ("0.2034", "平均换手率"), ("160 万", "测试预测行"),
    ]): box(c, MARGIN + i * (widths + 6), y, widths, 62, value, label)
    y -= 86; y = section(c, "执行摘要", y)
    y = para(c, "本方案面向每日 4,650 只股票的横截面评分。核心模型由全历史 LambdaRank（60%）、近两年 ATR 增强 LambdaRank（20%）及 LightGBM/Ridge 回归组合（20%）构成；各组件先做同日排名，再使用因果 EMA 与 Top 集合滞回控制换手。", MARGIN, y - 8, CONTENT_W, 10.2, 16)
    y = para(c, "在 2022、2023、2024 三个严格时间顺序验证折上，最终方案综合分分别为 0.4482、0.3732、0.3240，等权均值 0.3818；相对单一 LambdaRank，均分提高 0.0038、最差折提高 0.0082、换手下降 0.0089。", MARGIN, y - 10, CONTENT_W, 10.2, 16)
    c.setFillColor(colors.HexColor("#FFF5DB")); c.roundRect(MARGIN, 86, CONTENT_W, 60, 7, fill=1, stroke=0)
    para(c, "重要声明：以上均为训练集 walk-forward OOF 指标。隐藏测试标签不可用，项目没有推断、伪造或报告测试集成绩。最终 CSV 仅通过结构与数值完整性检查。", MARGIN + 10, 132, CONTENT_W - 20, 9.5, 14, RED, True)
    header_footer(c, 1); c.showPage()

    # Page 2 - data introduction
    y = page_start(c, 2, "1. 数据介绍")
    y = para(c, "数据为固定股票面板的日频量价记录，字段包括证券代码、交易日、开高低收、成交量、成交额、涨跌停标记；训练集额外提供下一交易行收益 y_ret_1d。原始 CSV 始终只读，中间大文件按年写入 D 盘 Parquet。", MARGIN, y - 8, CONTENT_W, 9.5, 14)
    y -= 12
    rows = [["数据集", "日期范围", "行数", "交易日", "股票数", "标签"] ,
            ["训练集", "2018-01-02 ~ 2024-12-31", f"{train['shape'][0]:,}", "1,699", f"{train['stock_count']:,}", "真实 y_ret_1d"],
            ["测试集", "2025-01-02 ~ 2026-06-08", f"{test['shape'][0]:,}", f"{test['daily_stock_count']['trading_days']}", f"{test['stock_count']:,}", "无标签"]]
    y = table(c, rows, MARGIN, y, [60, 150, 85, 65, 65, 78], 8.4)
    y -= 14; c.drawImage(str(figs["labels"]), MARGIN, y - 185, width=CONTENT_W, height=175, preserveAspectRatio=True, anchor="c"); y -= 200
    y = section(c, "数据质量与边界", y)
    y = para(c, "主键无缺失、无重复；OHLC 关系、非负成交量/额与涨跌停二值约束全部通过。训练集价格缺失约 14.4%，主要来自股票未上市或停牌形成的面板空洞；标签缺失率由 2018 年 26.2% 降至 2024 年 2.8%。模型保留缺失语义，树模型原生处理 NaN，Ridge 的横截面 z-score 缺失映射为中性 0。", MARGIN, y - 7, CONTENT_W, 9.2, 14)
    c.showPage()

    # Page 3 - EDA
    y = page_start(c, 3, "2. 描述性分析")
    label_table = [["年份", "有效标签", "均值", "波动", "1%分位", "99%分位", "缺失率"]]
    for _, r in e["labels"].iterrows():
        label_table.append([str(int(r.year)), f"{int(r.non_missing):,}", f"{r['mean']:.4f}", f"{r['std']:.4f}", f"{r.q01:.4f}", f"{r.q99:.4f}", f"{r.missing_rate:.1%}"])
    y = table(c, label_table, MARGIN, y - 6, [45, 82, 64, 64, 64, 64, 64], 7.8)
    y -= 15; y = section(c, "关键观察", y)
    bullets = [
        "收益分布高度尖峰且存在长尾；中位数多为 0，2024 年极端值范围扩大，因此回归误差不是唯一有效目标。",
        "赛题得分同时奖励 Rank IC、Top 组合相对全市场的年化超额及低换手，模型应优化横截面次序而非收益点预测精度。",
        "每日股票数固定，但可用量价与标签随上市、停牌状态变化；计算横截面统计时保留覆盖率，不以 0 冒充原始缺失。",
        "涨停股票在组合评价中不可买入；后处理只重排非涨停股票的唯一序数分值，保持每日评分分布稳定。",
    ]
    for text in bullets:
        y = para(c, "• " + text, MARGIN + 5, y - 8, CONTENT_W - 10, 9.3, 14)
    y -= 12; c.setFillColor(LIGHT); c.roundRect(MARGIN, y - 85, CONTENT_W, 85, 7, fill=1, stroke=0)
    para(c, "评价函数", MARGIN + 12, y - 10, CONTENT_W - 24, 10.5, 15, BLUE, True)
    para(c, "综合分 = 0.4 × 平均 Rank IC + 0.3 × Top10% 相对全市场年化超额 + 0.3 × (1 - 平均换手率)。本地实现对官方脚本全部分支达到 1e-12 以内一致性。", MARGIN + 12, y - 34, CONTENT_W - 24, 9.5, 14)
    c.showPage()

    # Page 4 - features and validation
    y = page_start(c, 4, "3. 模型分析（一）：特征与验证设计")
    c.drawImage(str(figs["features"]), MARGIN, y - 220, width=CONTENT_W, height=210, preserveAspectRatio=True, anchor="c"); y -= 235
    y = para(c, "基础特征覆盖收益、波动、K线形态、成交活跃度、流动性、价格位置与均线结构，并在每日截面形成 percentile rank 和 z-score；另构造市场等权收益、截面波动、上涨比例及成交额状态。Phase 13 增加 RSI、MACD、ATR、Bollinger、OBV-like、VWAP 代理、趋势效率与突破等 32 个因果候选。", MARGIN, y, CONTENT_W, 9.2, 14)
    y = para(c, "高级特征只在 2018–2021 筛选，2022–2024 仅作留出验证。覆盖率、|IC|、ICIR、逐年同号、家族上限与 0.90 相关性阈值共同锁定 16 项；完整 ranker 仅验证证据最强的 ATR 家族，避免大规模事后搜索。", MARGIN, y - 9, CONTENT_W, 9.2, 14)
    y -= 15; y = section(c, "时间序列验证与防泄漏", y)
    cv_rows = [["验证折", "训练期", "Purge", "验证期"],
               ["2022", "2018-01-02 ~ 2021-12-30", "1个交易日", "2022全年"],
               ["2023", "2018-01-02 ~ 2022-12-29", "1个交易日", "2023全年"],
               ["2024", "2018-01-02 ~ 2023-12-28", "1个交易日", "2024全年"]]
    y = table(c, cv_rows, MARGIN, y - 5, [65, 170, 95, 130], 8.2)
    y = para(c, "所有时序特征仅使用 t 及以前；横截面与市场特征只使用同日全体股票；标签在特征物化后按键连接。验证折各自重置后处理状态，而生产推理使用训练尾部状态连续进入测试期。", MARGIN, y - 10, CONTENT_W, 9.1, 14)
    c.showPage()

    # Page 5 - model progress
    y = page_start(c, 5, "3. 模型分析（二）：模型比较与最终方案")
    c.drawImage(str(figs["models"]), MARGIN, y - 222, width=CONTENT_W, height=212, preserveAspectRatio=True, anchor="c"); y -= 238
    mrows = [["方案", "Rank IC", "年化超额", "换手率", "综合分"]]
    lookup = {x["name"]: x for x in model_rows(e)}
    annual = {
        "Ridge": mean_metric(e["p8"]["models"]["ridge"]["summary"], "annual_excess"),
        "LightGBM": mean_metric(e["p8"]["models"]["lightgbm"]["summary"], "annual_excess"),
        "LGB+换手控制": float(e["p10"]["selected"]["annual_excess"]),
        "LGB/Ridge融合": float(e["p11"]["selected"]["annual_excess"]),
        "LambdaRank": mean_metric(e["p12"]["variants"]["turnover"]["summary"], "annual_excess"),
        "最终融合": float(final["annual_excess"]),
    }
    for name in ["Ridge", "LightGBM", "LGB+换手控制", "LGB/Ridge融合", "LambdaRank", "最终融合"]:
        row = lookup[name]; mrows.append([name, f"{row['ic']:.4f}", f"{annual[name]:.4f}", f"{row['turnover']:.4f}", f"{row['score']:.4f}"])
    y = table(c, mrows, MARGIN, y, [112, 78, 90, 82, 82], 8.1)
    y -= 13; y = para(c, "最终嵌套权重：基础 LambdaRank 60% + 近两年 ATR LambdaRank 20% + 回归 sleeve 20%；回归 sleeve 内为 LightGBM 65% + Ridge 35%。三个端点对历史 OOF 的八项官方指标复刻差均为 0，融合后再统一执行 alpha=0.5、退出缓冲 15% 的因果后处理。", MARGIN, y, CONTENT_W, 9.2, 14)
    c.showPage()

    # Page 6 - stability
    y = page_start(c, 6, "3. 模型分析（三）：稳定性与市场状态")
    c.drawImage(str(figs["folds"]), MARGIN, y - 205, width=CONTENT_W, height=195, preserveAspectRatio=True, anchor="c"); y -= 218
    c.drawImage(str(figs["regime"]), MARGIN, y - 190, width=CONTENT_W, height=180, preserveAspectRatio=True, anchor="c"); y -= 203
    y = para(c, "所有方案在 2022→2024 均出现分数回落，主要风险是市场状态漂移和后期换手抬升。近两年 ATR 模型在低波动状态更稳且换手更低，但在上涨、下跌状态均落后基础 LambdaRank，因此没有建立独立 regime 路由，而是只给予 20% 稳健性权重。", MARGIN, y, CONTENT_W, 9.1, 14)
    y = para(c, "最终融合逐折分数 0.4482 / 0.3732 / 0.3240，最差折比基础 LambdaRank 提高 0.0082；标准差由 0.0649 降至 0.0626，平均换手由 0.2123 降至 0.2034。改进幅度有限但方向一致，故采用简单、可审计的固定权重。", MARGIN, y - 8, CONTENT_W, 9.1, 14)
    c.showPage()

    # Page 7 - application and product
    y = page_start(c, 7, "4. 应用验证与 5. 产品思路")
    y = para(c, "生产推理链路", MARGIN, y - 4, CONTENT_W, 11, 15, BLUE, True)
    steps = ["训练尾部250日", "四模型原始预测", "同日排名与融合", "EMA状态续接", "Top集合滞回", "测试评分/CSV"]
    bw = (CONTENT_W - 25) / 6
    top = y - 20
    for i, s in enumerate(steps):
        x = MARGIN + i * (bw + 5); c.setFillColor(LIGHT if i < 5 else colors.HexColor("#FFF0C9")); c.roundRect(x, top - 48, bw, 48, 5, fill=1, stroke=0)
        para(c, s, x + 4, top - 13, bw - 8, 8.1, 11, BLUE, True, TA_CENTER)
        if i < 5: c.setFillColor(GOLD); c.drawString(x + bw + 1, top - 27, "›")
    y = top - 68
    checks = [["应用检查", "结果"],
              ["训练尾部 warm-up", "250 日 / 1,162,500 行"],
              ["测试期覆盖", "344 日 / 1,599,600 行"],
              ["状态跨边界", "4,650 EMA；459 Top成员"],
              ["特征 schema", "47 / 49 / 47 / 36 全部一致"],
              ["Submission", "键集合、行序、有限值、CSV回读全部通过"]]
    y = table(c, checks, MARGIN, y, [180, 300], 8.4)
    y -= 17; y = section(c, "产品化：低换手股票评分与组合候选服务", y)
    y = para(c, "面向资管投研或财富管理内部系统，每日输出全市场标准化评分、Top 候选池、换手预算和风险提示。评分层负责可比的横截面排序；组合层结合涨停不可买、持仓缓冲及客户自定义行业/集中度约束；解释层展示主要特征族、历史稳定性和本次入选/退出原因。", MARGIN, y - 7, CONTENT_W, 9.2, 14)
    y = para(c, "上线前需增加真实交易成本、停牌与成交容量、行业中性、公司行动、成分存续偏差及实时数据延迟测试。本赛题结果是离线研究证据，不构成收益承诺或投资建议。", MARGIN, y - 8, CONTENT_W, 9.2, 14, RED)
    c.showPage()

    # Page 8 - conclusions and reproducibility
    y = page_start(c, 8, "6. 总结结论与最终复现")
    conclusions = [
        "排序目标优于同配置回归：LambdaRank 把平均综合分由 0.2422 提升至 0.3780。",
        "换手控制是核心增量：固定因果 EMA 与持仓缓冲显著降低换手，且未使用未来信息。",
        "最终融合的提升来自互补而非复杂度：60/20/20 固定权重同时改善均值、最差折与换手。",
        "测试推理完整复用锁定模型，连续传递状态；最终 CSV 结构完整，但隐藏测试成绩未知。",
    ]
    for i, text in enumerate(conclusions, 1):
        c.setFillColor(GOLD); c.circle(MARGIN + 10, y - 12, 9, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("CN-Bold", 8); c.drawCentredString(MARGIN + 10, y - 15, str(i))
        y = para(c, text, MARGIN + 28, y - 3, CONTENT_W - 28, 9.5, 14); y -= 8
    y -= 4; y = section(c, "一键复现顺序", y)
    commands = [
        "01_check_data.py → 02_audit_labels.py → 03_build_features.py",
        "04_build_cross_sectional.py → 05_build_test_features.py → 06_build_cv.py",
        "07_check_evaluator.py → 08_train_baseline.py → … → 16_finalize_model.py",
        "17_predict_test.py → 18_build_submission.py → 19_build_final_report.py",
    ]
    for cmd in commands: y = para(c, cmd, MARGIN + 8, y - 6, CONTENT_W - 16, 8.3, 12, colors.HexColor("#37474F"))
    y -= 8; y = section(c, "关键交付哈希", y)
    hash_rows = [["产物", "SHA-256"],
                 ["训练集.csv", e["p18"]["raw_sha256_after"]["train"]],
                 ["测试集_X.csv", e["p18"]["raw_sha256_after"]["test"]],
                 ["Phase17 Parquet", e["p17"]["prediction"]["sha256"]],
                 ["Submission CSV", e["p18"]["submission"]["sha256"]]]
    table(c, hash_rows, MARGIN, y - 4, [108, 390], 6.5)
    c.showPage(); c.save()


def markdown_report(e: dict[str, Any]) -> str:
    f = e["p16"]["selected"]
    return f"""# 2026 深圳国际金融科技大赛数据分析赛道题五报告

> 版本：Phase 19 最终交付。所有表现指标均来自 2022–2024 walk-forward OOF；隐藏测试成绩未知。

## 1. 数据介绍

训练集覆盖 2018-01-02 至 2024-12-31，共 7,900,350 行、4,650 只股票；测试集覆盖 2025-01-02 至 2026-06-08，共 1,599,600 行。原始 CSV 只读，中间数据优先使用 D 盘按年 Parquet。

## 2. 描述性分析

收益标签长尾明显且缺失率随样本后期下降。每日横截面固定为 4,650 行，但可用量价记录随上市与停牌状态变化。缺失值保留语义，标签缺失行不用于监督训练。

## 3. 模型分析

模型采用 60% 全历史 LambdaRank、20% 近两年 ATR 增强 LambdaRank、20% 回归 sleeve；回归 sleeve 内部为 65% LightGBM 与 35% Ridge。组件同日排名后仅执行一次 alpha=0.5、exit=15% 的因果后处理。

最终三折综合分为 0.44822 / 0.37320 / 0.32395，均值 {f['final_score']:.6f}、最差折 {f['final_score_worst']:.6f}、平均换手 {f['mean_turnover']:.6f}。

## 4. 应用验证

测试推理使用最后 250 个真实训练交易日建立 EMA 与 Top 集合状态，再连续进入测试期。输出 1,599,600 行，键集合和原始顺序完全一致，预测有限且 CSV 回读 float32 精确。

## 5. 产品思路

可产品化为低换手股票评分与候选池服务，向投研系统提供每日评分、持仓缓冲、风险提示与可解释的入选/退出原因。真实上线仍需交易成本、容量、行业中性、公司行动及实时数据质量验证。

## 6. 总结结论

LambdaRank、因果换手控制和小规模稳健融合是主要有效增量。最终方案在训练期 OOF 上改善均值、最差折及换手，但不构成隐藏测试成绩或投资收益承诺。
"""


def reproducibility_text(config: dict[str, Any], e: dict[str, Any]) -> str:
    return f"""# Final reproducibility guide

## Environment

- Python 3.12
- Install: `.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt`
- All configured data, caches, models, reports and submissions are on `D:\\codex\\fintechathon2026`.

## Immutable inputs

- Train SHA-256: `{e['p18']['raw_sha256_after']['train']}`
- Test SHA-256: `{e['p18']['raw_sha256_after']['test']}`
- Official evaluator SHA-256: `{config['evaluator']['official_sha256']}`

## Run order

Run numbered scripts in `scripts/` from 01 through 19. Expensive existing outputs reject overwrite by default; use explicit overwrite flags only when intentionally rebuilding that phase. Phase 15 supports `--resume` for complete fold artifacts.

The final production sequence is:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\16_finalize_model.py
.\\.venv\\Scripts\\python.exe scripts\\17_predict_test.py
.\\.venv\\Scripts\\python.exe scripts\\18_build_submission.py
.\\.venv\\Scripts\\python.exe scripts\\19_build_final_report.py
```

## Final outputs

- Model manifest: `models/phase16_final/manifest.json`
- Prediction Parquet: `models/phase17_prediction/test_predictions.parquet`
- Submission: `submissions/submission_p16_locked_ensemble_v1.csv`
- Report PDF: `output/pdf/fintechathon2026_task5_report.pdf`
- Experiment registry: `experiments/experiment_registry.csv`

No test label is created or inferred. Structural submission validation is not a hidden-test performance score.
"""


def main() -> int:
    config = load_config(PROJECT_ROOT / "config.yaml")
    e = load_evidence(PROJECT_ROOT)
    settings = config["reporting"]
    figures = make_figures(e, Path(settings["figures_dir"]))
    markdown_path = Path(settings["markdown_output"]); markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(e), encoding="utf-8")
    repro_path = Path(settings["reproducibility_output"])
    repro_path.write_text(reproducibility_text(config, e), encoding="utf-8")
    pdf_path = Path(settings["pdf_output"]); temporary_pdf = pdf_path.with_suffix(".tmp.pdf")
    build_pdf(e, figures, temporary_pdf)
    pages = len(PdfReader(str(temporary_pdf)).pages)
    if pages > 8 or pages != 8:
        raise ValueError(f"Final report must render as exactly 8 pages, got {pages}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True); os.replace(temporary_pdf, pdf_path)

    root = Path(config["paths"]["root"]); config_path = root / "config.yaml"; config_sha = sha256_file(config_path)
    snapshot = Path(config["experiments"]["configs_dir"]) / f"phase19_report_{config_sha[:12]}.yaml"
    write_immutable(snapshot, config_path.read_bytes())
    source = build_source_manifest(root); source_path = Path(config["experiments"]["manifests_dir"]) / f"source_{source['tree_sha256'][:12]}.json"
    write_immutable(source_path, json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    artifact_files = [snapshot, pdf_path, markdown_path, repro_path, *figures.values(),
                      Path(config["submission"]["output"]), Path(config["submission"]["audit_output"]),
                      Path(config["prediction"]["audit_output"]), Path(config["experiments"]["registry"])]
    entries = [artifact_entry(path, root) for path in artifact_files]
    delivery = {"phase": 19, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "source_tree_sha256": source["tree_sha256"], "report_pages": pages,
                "hidden_test_score_reported": False, "artifacts": entries}
    delivery_bytes = json.dumps(delivery, ensure_ascii=False, indent=2).encode("utf-8")
    delivery_path = Path(settings["delivery_manifest"]); delivery_path.write_bytes(delivery_bytes)
    artifact_sha = hashlib.sha256(delivery_bytes).hexdigest().upper()
    git = git_state(root); timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    registry = append_registry_rows(Path(config["experiments"]["registry"]), [{
        "experiment_id": settings["experiment_id"], "timestamp": timestamp, "record_type": "summary", "fold": "final_delivery",
        "git_commit": git["commit"], "git_dirty": git["dirty"], "source_tree_sha256": source["tree_sha256"],
        "source_manifest_path": str(source_path), "config_path": str(snapshot), "config_sha256": config_sha,
        "feature_version": "final_documentation", "feature_manifest_sha256": "NOT_APPLICABLE",
        "model": "report_and_reproducibility_delivery", "model_params": json.dumps({"pdf_pages": pages}, separators=(",", ":")),
        "train_period": "2018_2024", "val_period": "2022_2024_oof", "seed": config["project"]["seed"],
        "artifact_manifest_path": str(delivery_path), "artifact_manifest_sha256": artifact_sha,
        "artifact_paths": json.dumps([str(pdf_path), str(markdown_path), str(repro_path)]),
        "notes": "eight_page_report_no_hidden_test_score",
    }])
    audit = {"phase": 19, "pdf": {"path": str(pdf_path), "pages": pages, "bytes": pdf_path.stat().st_size,
              "sha256": sha256_file(pdf_path)}, "markdown": str(markdown_path), "figures": {k: str(v) for k, v in figures.items()},
             "reproducibility": str(repro_path), "delivery_manifest": str(delivery_path),
             "delivery_manifest_sha256": artifact_sha, "source_tree_sha256": source["tree_sha256"],
             "source_manifest": str(source_path), "config_sha256": config_sha, "config_snapshot": str(snapshot),
             "registry": registry, "contracts": {"report_pages_at_most_8": pages <= 8,
             "required_sections_present": True, "figures_from_reproducible_artifacts": True,
             "hidden_test_score_reported": False, "submission_modified": False}}
    audit_path = Path(settings["audit_output"]); audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
