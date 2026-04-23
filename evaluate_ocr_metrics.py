"""Calcula CER/WER dos JSONs de OCR do projeto e gera um consolidado comparativo."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ocr_output_schema import build_document_output


def _page_count_from_pages(raw_pages: list[dict[str, Any]]) -> int:
    numbered_pages = []
    for page in raw_pages:
        try:
            numbered_pages.append(int(page.get("page", 0)))
        except (TypeError, ValueError):
            continue
    max_number = max((page for page in numbered_pages if page > 0), default=0)
    return max_number or len(raw_pages)


def _load_document(path: Path, *, default_model: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        raw_pages = [page for page in data if isinstance(page, dict)]
        page_count = _page_count_from_pages(raw_pages)
        return build_document_output(
            pdf_path=path,
            model=default_model,
            pages=raw_pages,
            page_count=page_count,
            metadata={"evaluation_input": {"source_type": "page_list_json"}},
        )

    if isinstance(data, dict) and isinstance(data.get("pages"), list):
        raw_pages = [page for page in data["pages"] if isinstance(page, dict)]
        page_count = int(data.get("page_count") or _page_count_from_pages(raw_pages))
        source_pdf = Path(str(data.get("source_pdf") or path))
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else None
        return build_document_output(
            pdf_path=source_pdf,
            model=str(data.get("model") or default_model),
            pages=raw_pages,
            page_count=page_count,
            run_elapsed_seconds=float(data.get("run_elapsed_seconds") or 0.0),
            metadata=metadata,
        )

    raise ValueError(f"Formato JSON não suportado em {path}")


def _iter_output_files(explicit_paths: list[str]) -> list[Path]:
    if explicit_paths:
        files = [Path(item).expanduser() for item in explicit_paths]
    else:
        extracted_files = sorted(Path("data/extracted").glob("*_output_*.json"))
        files = extracted_files or sorted(Path("ocr_models").glob("*/*_output.json"))

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(file_path)
    return unique_files


def _bitparallel_levenshtein(source: Iterable[Any], target: Iterable[Any]) -> int:
    source_seq = tuple(source)
    target_seq = tuple(target)

    if source_seq == target_seq:
        return 0
    if not source_seq:
        return len(target_seq)
    if not target_seq:
        return len(source_seq)

    pattern = source_seq
    text = target_seq
    if len(pattern) > len(text):
        pattern, text = text, pattern

    pattern_len = len(pattern)
    bitmask = (1 << pattern_len) - 1
    highest_bit = 1 << (pattern_len - 1)

    positions: dict[Any, int] = {}
    for index, token in enumerate(pattern):
        positions[token] = positions.get(token, 0) | (1 << index)

    positive = bitmask
    negative = 0
    score = pattern_len

    for token in text:
        eq = positions.get(token, 0)
        xv = eq | negative
        xh = ((((eq & positive) + positive) ^ positive) | eq) & bitmask
        ph = (negative | ~(xh | positive)) & bitmask
        mh = positive & xh

        if ph & highest_bit:
            score += 1
        elif mh & highest_bit:
            score -= 1

        ph = ((ph << 1) | 1) & bitmask
        mh = (mh << 1) & bitmask
        positive = (mh | ~(xv | ph)) & bitmask
        negative = ph & xv

    return score


def _safe_rate(distance: int, reference_length: int) -> float:
    if reference_length <= 0:
        return 0.0 if distance == 0 else 1.0
    return distance / reference_length


def _compute_metrics(reference_text: str, hypothesis_text: str) -> dict[str, Any]:
    char_distance = _bitparallel_levenshtein(reference_text, hypothesis_text)
    reference_words = tuple(reference_text.split())
    hypothesis_words = tuple(hypothesis_text.split())
    word_distance = _bitparallel_levenshtein(reference_words, hypothesis_words)

    return {
        "reference_char_count": len(reference_text),
        "hypothesis_char_count": len(hypothesis_text),
        "reference_word_count": len(reference_words),
        "hypothesis_word_count": len(hypothesis_words),
        "char_distance": char_distance,
        "word_distance": word_distance,
        "cer": _safe_rate(char_distance, len(reference_text)),
        "wer": _safe_rate(word_distance, len(reference_words)),
    }


def _page_map(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for page in document.get("pages", []):
        try:
            page_number = int(page.get("page", 0))
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            mapping[page_number] = page
    return mapping


def _evaluate_model(reference: dict[str, Any], candidate: dict[str, Any], source_path: Path) -> dict[str, Any]:
    reference_pages = _page_map(reference)
    candidate_pages = _page_map(candidate)

    page_results: list[dict[str, Any]] = []
    for page_number in sorted(reference_pages):
        reference_page = reference_pages[page_number]
        candidate_page = candidate_pages.get(page_number, {})
        metrics = _compute_metrics(
            str(reference_page.get("text_for_cer_wer") or ""),
            str(candidate_page.get("text_for_cer_wer") or ""),
        )
        page_results.append(
            {
                "page": page_number,
                "status": candidate_page.get("status", "missing"),
                "missing_page": page_number not in candidate_pages,
                "reference_excerpt": str(reference_page.get("text") or "")[:120],
                "hypothesis_excerpt": str(candidate_page.get("text") or "")[:120],
                "error": candidate_page.get("error"),
                **metrics,
            }
        )

    extra_pages = sorted(page for page in candidate_pages if page not in reference_pages)
    missing_pages = sorted(page for page in reference_pages if page not in candidate_pages)

    document_metrics = _compute_metrics(
        str(reference.get("text_for_cer_wer") or ""),
        str(candidate.get("text_for_cer_wer") or ""),
    )
    timing = {
        "total_processing_seconds": float(candidate.get("total_processing_seconds") or 0.0),
        "average_page_processing_seconds": float(candidate.get("average_page_processing_seconds") or 0.0),
        "run_elapsed_seconds": float(candidate.get("run_elapsed_seconds") or 0.0),
    }

    return {
        "model": str(candidate.get("model") or source_path.parent.name),
        "output_file": str(source_path.resolve()),
        "successful_pages_count": int(candidate.get("successful_pages_count") or 0),
        "failed_pages_count": int(candidate.get("failed_pages_count") or 0),
        "page_count": int(candidate.get("page_count") or 0),
        "missing_pages": missing_pages,
        "extra_pages": extra_pages,
        "document": document_metrics,
        "timing": timing,
        "pages": page_results,
        "metadata": candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {},
    }


def _format_rate(value: float) -> str:
    return f"{value:.4f}"


def _print_summary(reference: dict[str, Any], results: list[dict[str, Any]], *, show_pages: bool) -> None:
    reference_chars = len(str(reference.get("text_for_cer_wer") or ""))
    reference_words = len(str(reference.get("text_for_cer_wer") or "").split())

    print(f"Referencia: {reference.get('file_name')} | paginas={reference.get('page_count')} | chars={reference_chars} | words={reference_words}")
    print()
    print(
        f"{'Modelo':<20} {'CER':>8} {'WER':>8} {'DistC':>8} {'DistW':>8} "
        f"{'Tempo(s)':>10} {'Falhas':>8} {'Faltantes':>10} {'Extras':>8}"
    )
    print("-" * 97)
    for result in results:
        doc = result["document"]
        timing = result["timing"]
        print(
            f"{result['model'][:20]:<20} "
            f"{_format_rate(doc['cer']):>8} "
            f"{_format_rate(doc['wer']):>8} "
            f"{doc['char_distance']:>8} "
            f"{doc['word_distance']:>8} "
            f"{timing['total_processing_seconds']:>10.3f} "
            f"{result['failed_pages_count']:>8} "
            f"{len(result['missing_pages']):>10} "
            f"{len(result['extra_pages']):>8}"
        )

    if not show_pages:
        return

    for result in results:
        print()
        print(f"[{result['model']}]")
        print(f"{'Pag':>4} {'CER':>8} {'WER':>8} {'DistC':>8} {'DistW':>8} {'Status':>10}")
        print("-" * 56)
        for page in result["pages"]:
            print(
                f"{page['page']:>4} "
                f"{_format_rate(page['cer']):>8} "
                f"{_format_rate(page['wer']):>8} "
                f"{page['char_distance']:>8} "
                f"{page['word_distance']:>8} "
                f"{str(page['status'])[:10]:>10}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula CER/WER das saídas de OCR contra um gabarito JSON."
    )
    parser.add_argument(
        "reference",
        nargs="?",
        default="data/pdf2_gabarito.json",
        help="JSON de gabarito. Aceita lista de páginas ou schema consolidado.",
    )
    parser.add_argument(
        "outputs",
        nargs="*",
        help="Arquivos *_output.json a comparar. Se omitido, usa todos em ocr_models/*/*_output.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/pdf2_metrics.json",
        help="Caminho do JSON consolidado de métricas.",
    )
    parser.add_argument(
        "--show-pages",
        action="store_true",
        help="Exibe também o detalhe por página no terminal.",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    if not reference_path.exists():
        raise SystemExit(f"Gabarito não encontrado: {reference_path}")

    output_files = _iter_output_files(args.outputs)
    if not output_files:
        raise SystemExit("Nenhum arquivo de saída de OCR encontrado para comparar.")

    reference = _load_document(reference_path, default_model="reference")

    evaluated_models: list[dict[str, Any]] = []
    for output_file in output_files:
        if not output_file.exists():
            raise SystemExit(f"Arquivo de saída não encontrado: {output_file}")
        candidate = _load_document(output_file, default_model=output_file.parent.name)
        evaluated_models.append(_evaluate_model(reference, candidate, output_file))

    evaluated_models.sort(key=lambda item: (item["document"]["cer"], item["document"]["wer"], item["model"]))

    ranking = []
    for index, result in enumerate(evaluated_models, 1):
        ranking.append(
            {
                "rank": index,
                "model": result["model"],
                "output_file": result["output_file"],
                "cer": result["document"]["cer"],
                "wer": result["document"]["wer"],
                "char_distance": result["document"]["char_distance"],
                "word_distance": result["document"]["word_distance"],
                "total_processing_seconds": result["timing"]["total_processing_seconds"],
                "average_page_processing_seconds": result["timing"]["average_page_processing_seconds"],
                "run_elapsed_seconds": result["timing"]["run_elapsed_seconds"],
                "failed_pages_count": result["failed_pages_count"],
                "missing_pages_count": len(result["missing_pages"]),
                "extra_pages_count": len(result["extra_pages"]),
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "source_file": str(reference_path.resolve()),
            "file_name": reference.get("file_name"),
            "page_count": reference.get("page_count"),
            "total_char_count": len(str(reference.get("text_for_cer_wer") or "")),
            "total_word_count": len(str(reference.get("text_for_cer_wer") or "").split()),
        },
        "ranking": ranking,
        "models": evaluated_models,
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(reference, evaluated_models, show_pages=args.show_pages)
    print()
    print(f"Métricas salvas em: {output_path.resolve()}")


if __name__ == "__main__":
    main()
