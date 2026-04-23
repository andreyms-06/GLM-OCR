"""Centraliza a construção dos caminhos de saída usados pelos resultados de OCR."""

from __future__ import annotations

import re
from pathlib import Path


def _sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "arquivo"


def build_extracted_output_path(root_dir: Path, ocr_model_name: str, source_file: Path) -> Path:
    extracted_dir = root_dir / "data" / "extracted"
    model_name = _sanitize_filename_component(ocr_model_name)
    source_name = _sanitize_filename_component(source_file.stem)
    return extracted_dir / f"{model_name}_output_{source_name}.json"
