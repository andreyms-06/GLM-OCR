"""Gera gráficos comparativos de erro e tempo a partir do consolidado de métricas OCR."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to the metrics JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_data(json_path: Path):
    with open(json_path) as f:
        raw = json.load(f)

    ranking = raw["ranking"]
    model_detail = {m["model"]: m for m in raw["models"]}
    model_ids = [r["model"] for r in ranking]

    known_labels = {
        "mistral-ocr-latest":     "Mistral OCR",
        "mistral-ocr-2512":       "Mistral OCR",
        "glm-ocr:bf16":           "GLM-OCR (bf16)",
        "paddleocr:pt:gpu":       "PaddleOCR (GPU)",
        "easyocr:pt,en:gpu":      "EasyOCR (GPU)",
        "tesseract:por+eng:oem3": "Tesseract OEM3",
    }

    def label_for_model(model_id: str) -> str:
        if model_id in known_labels:
            return known_labels[model_id]
        if model_id.startswith("mistral-ocr"):
            return "Mistral OCR"
        return model_id

    labels = [label_for_model(m) for m in model_ids]

    doc_cer   = [r["cer"]                             for r in ranking]
    doc_wer   = [r["wer"]                             for r in ranking]
    proc_time = [r["total_processing_seconds"]        for r in ranking]
    avg_page  = [r["average_page_processing_seconds"] for r in ranking]

    cer_per_page = {}
    wer_per_page = {}
    for mid in model_ids:
        pages_sorted = sorted(model_detail[mid]["pages"], key=lambda p: p["page"])
        cer_per_page[mid] = [p["cer"] for p in pages_sorted]
        wer_per_page[mid] = [p["wer"] for p in pages_sorted]

    n_pages = len(cer_per_page[model_ids[0]])
    ref = raw.get("reference", {})

    return {
        "model_ids":    model_ids,
        "labels":       labels,
        "doc_cer":      doc_cer,
        "doc_wer":      doc_wer,
        "proc_time":    proc_time,
        "avg_page":     avg_page,
        "cer_per_page": cer_per_page,
        "wer_per_page": wer_per_page,
        "n_pages":      n_pages,
        "doc_name":     ref.get("file_name", json_path.name),
    }


_BASE_COLORS  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                 "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
_BASE_MARKERS = ["o","s","^","D","v","P","X","*","h","8"]
_BASE_LINES   = ["-","--","-.",(0,(3,1,1,1)),":",
                 (0,(5,2)),(0,(1,1)),(0,(3,5,1,5)),"--","-."]

def _palette(n):
    colors  = (_BASE_COLORS  * (n // len(_BASE_COLORS)  + 1))[:n]
    markers = (_BASE_MARKERS * (n // len(_BASE_MARKERS) + 1))[:n]
    lines   = (_BASE_LINES   * (n // len(_BASE_LINES)   + 1))[:n]
    return colors, markers, lines


def plot(data: dict, output_dir: Path, dpi: int):
    model_ids    = data["model_ids"]
    labels       = data["labels"]
    doc_cer      = data["doc_cer"]
    doc_wer      = data["doc_wer"]
    proc_time    = data["proc_time"]
    avg_page     = data["avg_page"]
    cer_per_page = data["cer_per_page"]
    wer_per_page = data["wer_per_page"]
    n_pages      = data["n_pages"]
    doc_name     = data["doc_name"]
    n_models     = len(model_ids)

    COLORS, MARKERS, LINES = _palette(n_models)
    pages = list(range(1, n_pages + 1))

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.dpi":        dpi,
    })

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"OCR Benchmark — {doc_name}  ({n_pages} páginas)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35, height_ratios=[2, 1.6, 1.4])

    ax0 = fig.add_subplot(gs[0, 0])
    for i, mid in enumerate(model_ids):
        ax0.plot(pages, [min(v, 1.0) for v in cer_per_page[mid]],
                 color=COLORS[i], marker=MARKERS[i], linestyle=LINES[i],
                 linewidth=1.5, markersize=4, label=labels[i])
    ax0.set_title("(a) CER por Página")
    ax0.set_xlabel("Página")
    ax0.set_ylabel("CER")
    ax0.set_xticks(pages)
    ax0.set_ylim(-0.02, 1.05)
    ax0.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ax1 = fig.add_subplot(gs[0, 1])
    for i, mid in enumerate(model_ids):
        ax1.plot(pages, [min(v, 1.0) for v in wer_per_page[mid]],
                 color=COLORS[i], marker=MARKERS[i], linestyle=LINES[i],
                 linewidth=1.5, markersize=4)
    ax1.set_title("(b) WER por Página")
    ax1.set_xlabel("Página")
    ax1.set_ylabel("WER")
    ax1.set_xticks(pages)
    ax1.set_ylim(-0.02, 1.05)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ax2 = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(n_models)
    bars = ax2.bar(x_pos, proc_time, color=COLORS, edgecolor="white", linewidth=0.8)
    for bar, t in zip(bars, proc_time):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{t:.1f}s", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_title("(c) Tempo Total de Processamento")
    ax2.set_ylabel("Segundos")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([lb.split()[0] for lb in labels], rotation=20, ha="right", fontsize=7.5)
    ax2.set_ylim(0, max(proc_time) * 1.18)
    ax2b = ax2.twinx()
    ax2b.scatter(x_pos, avg_page, color="black", zorder=5, s=30, marker="D")
    for xi, yi in zip(x_pos, avg_page):
        ax2b.text(xi + 0.18, yi, f"{yi:.2f}s/pg", va="center", fontsize=6.5, color="#444")
    ax2b.set_ylabel("s / página", fontsize=8)
    ax2b.spines["top"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(n_models)
    hbars = ax3.barh(y_pos, [v * 100 for v in doc_cer],
                     color=COLORS, edgecolor="white", linewidth=0.8, height=0.6)
    for bar, v in zip(hbars, doc_cer):
        ax3.text(v * 100 + 0.4, bar.get_y() + bar.get_height() / 2,
                 f"{v*100:.1f}%", va="center", fontsize=7.5, fontweight="bold")
    ax3.set_title("(d) CER — Documento Completo")
    ax3.set_xlabel("CER (%)")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=7.5)
    ax3.set_xlim(0, max(doc_cer) * 100 * 1.2)
    ax3.invert_yaxis()

    ax4 = fig.add_subplot(gs[1, 1])
    hbars2 = ax4.barh(y_pos, [v * 100 for v in doc_wer],
                      color=COLORS, edgecolor="white", linewidth=0.8, height=0.6)
    for bar, v in zip(hbars2, doc_wer):
        ax4.text(v * 100 + 0.6, bar.get_y() + bar.get_height() / 2,
                 f"{v*100:.1f}%", va="center", fontsize=7.5, fontweight="bold")
    ax4.set_title("(e) WER — Documento Completo")
    ax4.set_xlabel("WER (%)")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=7.5)
    ax4.set_xlim(0, max(doc_wer) * 100 * 1.2)
    ax4.invert_yaxis()

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    table_data = [
        [f"#{i+1}", labels[i], f"{doc_cer[i]*100:.2f}%", f"{doc_wer[i]*100:.2f}%", f"{proc_time[i]:.1f}"]
        for i in range(n_models)
    ]
    tbl = ax5.table(
        cellText=table_data, colLabels=["Rank", "Modelo", "CER", "WER", "Tempo (s)"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_facecolor("white")
        if col == 0 and row > 0:
            cell.set_facecolor(COLORS[row - 1])
            cell.set_text_props(color="white", fontweight="bold")
    ax5.set_title("(f) Ranking Geral", fontsize=10, pad=6)

    ax6 = fig.add_subplot(gs[2, :])
    matrix = np.array([[min(cer_per_page[mid][p], 1.0) for p in range(n_pages)] for mid in model_ids])
    im = ax6.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax6.set_title("(g) Mapa de Calor — CER por Página (limitado a 100%)", fontsize=10)
    ax6.set_xlabel("Página")
    ax6.set_xticks(range(n_pages))
    ax6.set_xticklabels([f"p{i+1}" for i in range(n_pages)], fontsize=8)
    ax6.set_yticks(range(n_models))
    ax6.set_yticklabels(labels, fontsize=8)
    for r in range(n_models):
        for c in range(n_pages):
            raw = cer_per_page[model_ids[r]][c]
            txt = f"{raw*100:.0f}%" if raw <= 1.0 else ">100%"
            ax6.text(c, r, txt, ha="center", va="center",
                     fontsize=6.5, color="white" if matrix[r, c] > 0.25 else "black")
    cbar = fig.colorbar(im, ax=ax6, orientation="vertical", pad=0.01, fraction=0.015)
    cbar.set_label("CER", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    handles, leg_labels = ax0.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=min(n_models, 5), fontsize=8,
               frameon=True, framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, 0.005))

    png_path = output_dir / f"{Path(doc_name).stem}_benchmark.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    print(f"Saved: {png_path}")


def main():
    args = parse_args()

    json_path = Path(args.json_file).resolve()
    if not json_path.exists():
        print(f"ERROR: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output).resolve() if args.output else Path(__file__).parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {json_path}")
    data = load_data(json_path)
    print(f"  Models  : {len(data['model_ids'])}")
    print(f"  Pages   : {data['n_pages']}")
    print(f"  Document: {data['doc_name']}")
    print("Plotting...")
    plot(data, output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
