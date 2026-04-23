"""Renderizacao de paginas PDF com fallback que evita dependencia obrigatoria de Poppler."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 220,
    output_file: str = "page",
    fmt: str = "jpeg",
) -> list[Path]:
    pdf_path = Path(pdf_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = os.getenv("PDF_RENDERER", "auto").strip().lower()
    errors: list[str] = []

    if renderer in {"auto", "pypdfium2", "pdfium"}:
        try:
            return _render_with_pypdfium2(pdf_path, output_dir, dpi=dpi, fmt=fmt)
        except Exception as e:
            errors.append(f"pypdfium2: {e}")
            if renderer in {"pypdfium2", "pdfium"}:
                raise RuntimeError(_build_render_error_message(errors)) from e

    if renderer in {"auto", "pdf2image", "poppler"}:
        try:
            return _render_with_pdf2image(pdf_path, output_dir, dpi=dpi, output_file=output_file, fmt=fmt)
        except Exception as e:
            errors.append(f"pdf2image: {e}")
            raise RuntimeError(_build_render_error_message(errors)) from e

    raise RuntimeError(
        "Valor invalido para PDF_RENDERER. Use `auto`, `pypdfium2` ou `pdf2image`."
    )


def _render_with_pypdfium2(pdf_path: Path, output_dir: Path, dpi: int, fmt: str) -> list[Path]:
    try:
        import pypdfium2 as pdfium  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise RuntimeError(
            "pacote `pypdfium2` nao esta instalado. Instale o `requirements.txt` atualizado desta pasta."
        ) from e

    fmt_normalized = _normalize_format(fmt)
    scale = max(float(dpi), 1.0) / 72.0
    output_paths: list[Path] = []

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            bitmap = None
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                destination = output_dir / f"page_{index + 1:04d}.{fmt_normalized.extension}"
                image.convert("RGB").save(destination, format=fmt_normalized.pil_format, quality=92)
                output_paths.append(destination)
            finally:
                if bitmap is not None:
                    bitmap.close()
                page.close()
    finally:
        pdf.close()

    if not output_paths:
        raise RuntimeError("nenhuma pagina foi renderizada.")
    return output_paths


def _render_with_pdf2image(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    output_file: str,
    fmt: str,
) -> list[Path]:
    try:
        from pdf2image import convert_from_path  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise RuntimeError(
            "pacote `pdf2image` nao esta instalado. Instale o `requirements.txt` desta pasta."
        ) from e

    poppler_path = _resolve_poppler_path()
    kwargs: dict[str, object] = {
        "dpi": dpi,
        "fmt": fmt,
        "output_folder": str(output_dir),
        "output_file": output_file,
        "paths_only": True,
        "thread_count": max(1, (os.cpu_count() or 2) // 2),
    }
    if poppler_path:
        kwargs["poppler_path"] = str(poppler_path)

    paths = convert_from_path(str(pdf_path), **kwargs)
    normalized = _normalize_format(fmt)

    renamed: list[Path] = []
    for idx, raw_path in enumerate(paths, 1):
        src = Path(raw_path)
        dst = output_dir / f"page_{idx:04d}.{normalized.extension}"
        if src != dst:
            src.replace(dst)
        renamed.append(dst)

    if not renamed:
        raise RuntimeError("nenhuma pagina foi renderizada.")
    return renamed


def _resolve_poppler_path() -> Path | None:
    for env_name in ("POPPLER_PATH", "PDF2IMAGE_POPPLER_PATH", "POPPLER_BIN"):
        candidate = os.getenv(env_name, "").strip()
        if not candidate:
            continue
        resolved = _validate_poppler_dir(Path(candidate).expanduser())
        if resolved:
            return resolved

    if shutil.which("pdftoppm"):
        return None

    project_root = Path(__file__).resolve().parent
    candidates: list[Path] = [
        project_root / "tools" / "poppler" / "Library" / "bin",
        project_root / "tools" / "poppler" / "bin",
        project_root / "poppler" / "Library" / "bin",
        project_root / "poppler" / "bin",
        Path.home() / "scoop" / "apps" / "poppler" / "current" / "Library" / "bin",
    ]

    for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
        base = os.getenv(env_name, "").strip()
        if not base:
            continue
        base_path = Path(base)
        candidates.extend(
            [
                base_path / "poppler" / "Library" / "bin",
                base_path / "poppler" / "bin",
                base_path / "Programs" / "poppler" / "Library" / "bin",
                base_path / "Programs" / "poppler" / "bin",
            ]
        )

    for candidate in candidates:
        resolved = _validate_poppler_dir(candidate)
        if resolved:
            return resolved
    return None


def _validate_poppler_dir(candidate: Path) -> Path | None:
    if not candidate.exists():
        return None
    if candidate.is_file():
        candidate = candidate.parent
    exe_name = "pdftoppm.exe" if os.name == "nt" else "pdftoppm"
    return candidate if (candidate / exe_name).exists() else None


def _build_render_error_message(errors: list[str]) -> str:
    details = " | ".join(errors)
    return (
        "Falha ao converter o PDF em imagens. O projeto agora tenta primeiro `pypdfium2` "
        "(sem Poppler) e depois `pdf2image`/Poppler. "
        "Reinstale o ambiente com o `requirements.txt` atualizado ou configure `POPPLER_PATH` "
        "se quiser continuar usando Poppler. "
        f"Detalhes: {details}"
    )


class _NormalizedFormat:
    def __init__(self, extension: str, pil_format: str):
        self.extension = extension
        self.pil_format = pil_format


def _normalize_format(fmt: str) -> _NormalizedFormat:
    normalized = fmt.strip().lower() or "jpeg"
    if normalized in {"jpg", "jpeg"}:
        return _NormalizedFormat(extension="jpg", pil_format="JPEG")
    if normalized == "png":
        return _NormalizedFormat(extension="png", pil_format="PNG")
    raise RuntimeError(f"Formato de saida nao suportado: {fmt}")
