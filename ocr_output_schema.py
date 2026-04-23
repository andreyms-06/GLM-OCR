"""Normaliza textos e padroniza o schema JSON salvo pelos benchmarks de OCR."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.0"
EXTRACTION_PIPELINE = (
    "pagina_inteira",
    "pagina_inteira_pre",
    "faixas",
    "faixas_pre",
)

_PAGE_METADATA_EXCLUDED_KEYS = {
    "page",
    "image_file",
    "status",
    "text",
    "text_for_cer_wer",
    "char_count",
    "word_count",
    "processing_seconds",
    "rotation_applied_degrees",
    "extraction_mode",
    "ocr_method",
    "warnings",
    "error",
    "metadata",
    "meta",
}


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 3)


def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", str(text))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")

    lines: list[str] = []
    blank_pending = False
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if not line:
            if lines and not blank_pending:
                lines.append("")
            blank_pending = True
            continue
        lines.append(line)
        blank_pending = False

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def cleanup_llm_markup(text: str) -> str:
    if not text:
        return ""

    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue

        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"</?(?:p|br|div|tr|td|th|table|thead|tbody|ul|ol|li|h[1-6])[^>]*>", " ", line)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
        line = re.sub(r"!\S+\.(?:png|jpe?g|gif|webp)\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)

        if "|" in line:
            if re.fullmatch(r"\|?[\s:\-]+\|[\s|:\-]*", line):
                continue
            cells = [re.sub(r"\s+", " ", cell.strip()) for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell and not re.fullmatch(r"[\s:\-]+", cell)]
            if cells:
                line = " ".join(cells)

        line = line.replace("**", "").replace("__", "")
        line = line.strip()
        if re.fullmatch(r"\[\d+\]", line):
            continue
        if re.fullmatch(r"\[(?:handwritten|illustration|image|photo|figure)[^]]*\]", line, flags=re.IGNORECASE):
            continue
        if not line:
            continue
        lines.append(line)

    return "\n".join(lines)


def normalize_text_for_cer_wer(text: str) -> str:
    canonical = normalize_ocr_text(text)
    if not canonical:
        return ""
    return re.sub(r"\s+", " ", canonical).strip()


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def _merge_metadata(base: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(extra, dict):
        for key, value in extra.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
    return merged


def standardize_page(raw_page: dict[str, Any]) -> dict[str, Any]:
    text = normalize_ocr_text(str(raw_page.get("text", "") or ""))
    text_for_cer_wer = normalize_text_for_cer_wer(text)

    warnings_raw = raw_page.get("warnings", [])
    warnings = []
    if isinstance(warnings_raw, list):
        warnings = [str(item).strip() for item in warnings_raw if str(item).strip()]

    error = str(raw_page.get("error", "") or "").strip() or None
    status = "error" if error else ("ok" if text else "empty")

    metadata = {}
    if isinstance(raw_page.get("metadata"), dict):
        metadata.update(raw_page["metadata"])
    if isinstance(raw_page.get("meta"), dict):
        metadata.update(raw_page["meta"])
    for key, value in raw_page.items():
        if key in _PAGE_METADATA_EXCLUDED_KEYS:
            continue
        metadata[key] = value

    try:
        page_number = int(raw_page.get("page", 0))
    except (TypeError, ValueError):
        page_number = 0

    try:
        processing_seconds = _round_seconds(raw_page.get("processing_seconds", 0.0))
    except (TypeError, ValueError):
        processing_seconds = 0.0

    try:
        rotation_applied_degrees = int(raw_page.get("rotation_applied_degrees", 0) or 0)
    except (TypeError, ValueError):
        rotation_applied_degrees = 0

    extraction_mode = str(raw_page.get("extraction_mode") or raw_page.get("ocr_method") or "").strip() or None

    return {
        "page": page_number,
        "image_file": str(raw_page.get("image_file", "") or ""),
        "status": status,
        "text": text,
        "text_for_cer_wer": text_for_cer_wer,
        "char_count": len(text),
        "word_count": count_words(text_for_cer_wer),
        "processing_seconds": processing_seconds,
        "rotation_applied_degrees": rotation_applied_degrees,
        "extraction_mode": extraction_mode,
        "warnings": warnings,
        "error": error,
        "metadata": metadata,
    }


def build_document_output(
    pdf_path: Path,
    model: str,
    pages: list[dict[str, Any]],
    page_count: int,
    run_elapsed_seconds: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sorted((standardize_page(page) for page in pages), key=lambda page: int(page["page"]))
    document_text = "\n\n".join(page["text"] for page in ordered if page["text"])
    document_text_for_cer_wer = "\n".join(page["text_for_cer_wer"] for page in ordered if page["text_for_cer_wer"])

    timed_pages = [
        float(page["processing_seconds"])
        for page in ordered
        if isinstance(page.get("processing_seconds"), (int, float))
    ]
    total_processing_seconds = _round_seconds(sum(timed_pages))
    average_page_processing_seconds = _round_seconds(
        total_processing_seconds / len(timed_pages) if timed_pages else 0.0
    )

    document_metadata = _merge_metadata(
        {
            "text_normalization": {
                "text": "multiline_canonical",
                "text_for_cer_wer": "single_line_whitespace_normalized_per_page",
            }
        },
        metadata,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": str(pdf_path.resolve()),
        "file_name": pdf_path.name,
        "model": model,
        "page_count": page_count,
        "timed_pages_count": len(timed_pages),
        "successful_pages_count": sum(1 for page in ordered if page["status"] == "ok"),
        "failed_pages_count": sum(1 for page in ordered if page["status"] == "error"),
        "extraction_pipeline": list(EXTRACTION_PIPELINE),
        "text": document_text,
        "text_for_cer_wer": document_text_for_cer_wer,
        "total_char_count": len(document_text),
        "total_word_count": count_words(document_text_for_cer_wer),
        "total_processing_seconds": total_processing_seconds,
        "average_page_processing_seconds": average_page_processing_seconds,
        "run_elapsed_seconds": _round_seconds(run_elapsed_seconds),
        "metadata": document_metadata,
        "pages": ordered,
    }
