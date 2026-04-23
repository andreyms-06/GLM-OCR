"""Benchmark de OCR para PDFs usando EasyOCR.

Renderiza as páginas do PDF, executa a extração e salva a saída no JSON padronizado do projeto.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import easyocr as easyocr_lib
except ImportError:
    easyocr_lib = None

try:
    import torch
except ImportError:
    torch = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ocr_output_paths import build_extracted_output_path
from ocr_output_schema import build_document_output, normalize_ocr_text


_FRAGMENTOS_GENERICOS = (
    "transforme uma imagem",
    "no texto extração",
    "the text in the image",
    "the text content",
    "cannot be extracted",
    "not visible in the image",
    "texto não está visível",
    "não é possível extrair",
)

PROJECT_EASYOCR_VERSION = "1.7.2"


def _env_flag(nome: str, padrao: bool = False) -> bool:
    raw = os.getenv(nome)
    return raw.strip().lower() in {"1", "true", "yes", "on", "sim"} if raw else padrao


def _env_int_tuple(nome: str, padrao: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(nome, "").strip()
    if not raw:
        return padrao
    valores = [int(p) for p in raw.split(",") if p.strip().lstrip("-").isdigit() and int(p) > 0]
    return tuple(valores) if valores else padrao


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 3)


def _env_optional_flag(nome: str) -> bool | None:
    raw = os.getenv(nome)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"", "auto", "default"}:
        return None
    if value in {"1", "true", "yes", "on", "sim"}:
        return True
    if value in {"0", "false", "no", "off", "nao", "não"}:
        return False
    return None


def _cuda_available() -> bool:
    try:
        return bool(torch and torch.cuda.is_available())
    except Exception:
        return False


class EasyOCRBenchmarkExtractor:
    def __init__(
        self,
        pdf_path: Path,
        output_path: Path,
        languages: list[str],
        gpu: bool | None,
        model_storage_dir: str | None = None,
        user_network_dir: str | None = None,
        download_enabled: bool = True,
    ):
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.languages = languages
        self.gpu = self._resolve_gpu_mode(gpu)
        self.easyocr_version = getattr(easyocr_lib, "__version__", PROJECT_EASYOCR_VERSION)
        self.download_enabled = download_enabled
        self.model_storage_dir = Path(model_storage_dir).expanduser() if model_storage_dir else None
        self.user_network_dir = Path(user_network_dir).expanduser() if user_network_dir else None
        self.model = f"easyocr:{','.join(self.languages)}:{'gpu' if self.gpu else 'cpu'}"

        self.pages_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)

        self.render_dpi = int(os.getenv("PDF_RENDER_DPI", "220"))
        self.ocr_max_dim = int(os.getenv("OCR_IMAGE_MAX_DIM", "1600"))
        self.ocr_fallback_dims = _env_int_tuple(
            "OCR_FALLBACK_MAX_DIMS",
            (self.ocr_max_dim, max(1200, int(self.ocr_max_dim * 0.75))),
        )
        self.band_heights = _env_int_tuple("OCR_BAND_HEIGHTS", (1200, 900))
        self.save_preprocessed = _env_flag("SAVE_PREPROCESSED_IMAGES", False)
        self.resume_progress = _env_flag("RESUME_OCR_PROGRESS", True)
        self.detect_orientation = _env_flag("ENABLE_OCR_ORIENTATION_DETECTION", False)
        self.paragraph_mode = _env_flag("EASYOCR_PARAGRAPH_MODE", False)
        self.min_confidence = float(os.getenv("EASYOCR_MIN_CONFIDENCE", "0.0"))

        self.reader = self._build_reader()

    def _resolve_gpu_mode(self, requested_gpu: bool | None) -> bool:
        cuda_available = _cuda_available()

        if requested_gpu is None:
            return cuda_available

        if requested_gpu and not cuda_available:
            print("   ⚠️ EasyOCR: GPU solicitada, mas CUDA não está disponível. Usando CPU.")
            return False

        return requested_gpu

    def _build_reader(self):
        if easyocr_lib is None:
            raise RuntimeError(
                "A biblioteca `easyocr` não está instalada. Crie sua venv e instale o `requirements.txt` desta pasta."
            )

        kwargs: dict[str, Any] = {
            "gpu": self.gpu,
            "download_enabled": self.download_enabled,
            "verbose": False,
        }
        if self.model_storage_dir:
            self.model_storage_dir.mkdir(parents=True, exist_ok=True)
            kwargs["model_storage_directory"] = str(self.model_storage_dir)
        if self.user_network_dir:
            kwargs["user_network_directory"] = str(self.user_network_dir)
        return easyocr_lib.Reader(self.languages, **kwargs)

    @staticmethod
    def _resize_if_needed(image: Image.Image, max_dim: int = 1800) -> Image.Image:
        if max(image.size) <= max_dim:
            return image
        ratio = max_dim / max(image.size)
        new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        return image.resize(new_size, Image.BILINEAR)

    def _ocr_candidates(self, image: Image.Image) -> list[Image.Image]:
        rgb = image.convert("RGB")
        dims = []
        if max(rgb.size) <= self.ocr_max_dim:
            dims.append(max(rgb.size))
        dims.extend(self.ocr_fallback_dims)

        seen: set[tuple[int, int]] = set()
        candidates = []
        for dim in dims:
            if dim <= 0:
                continue
            candidate = self._resize_if_needed(rgb, max_dim=dim)
            if candidate.size in seen:
                continue
            seen.add(candidate.size)
            candidates.append(candidate)
        return candidates or [rgb]

    def _preprocess(self, image: Image.Image) -> Image.Image:
        try:
            img = ImageOps.autocontrast(image.convert("L"), cutoff=1)
            img = ImageEnhance.Contrast(img).enhance(1.4)
            img = ImageEnhance.Sharpness(img).enhance(1.1)
            img = img.filter(ImageFilter.MedianFilter(size=3))

            arr = np.asarray(img, dtype=np.uint8)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            arr = clahe.apply(arr)
            if arr.mean() < 127:
                arr = 255 - arr

            out = Image.fromarray(arr, mode="L")
            return out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
        except Exception:
            return image.convert("RGB")

    def _band_heights(self, image: Image.Image) -> tuple[int, ...]:
        heights: list[int] = []
        min_height = min(self.band_heights) if self.band_heights else 0

        if min_height and image.height >= min_height * 2:
            heights.append(math.ceil(image.height / 2))

        for height in self.band_heights:
            if height > 0 and height not in heights:
                heights.append(height)

        return tuple(heights) or (max(1, image.height),)

    def _split_into_bands(self, image: Image.Image, max_height: int = 1200) -> list[Image.Image]:
        w, h = image.size
        if h <= max_height:
            return [image]

        n = max(2, math.ceil(h / max_height))
        height = math.ceil(h / n)
        overlap = min(96, max(24, height // 12))
        bands: list[Image.Image] = []

        for i in range(n):
            top = 0 if i == 0 else max(0, i * height - overlap)
            bottom = h if i == n - 1 else min(h, (i + 1) * height + overlap)
            bands.append(image.crop((0, top, w, bottom)))

        return bands

    @staticmethod
    def _score_text(text: str) -> float:
        if not text:
            return 0.0
        alnum = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", text))
        words = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
        lines = len([l for l in text.splitlines() if l.strip()])
        diversity = len(set(text)) / max(len(text), 1)
        return float(alnum + words * 0.7 + lines * 0.3 + diversity * 8.0)

    @staticmethod
    def _is_generic_response(text: str) -> bool:
        if not text:
            return True
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        if any(fragment in norm for fragment in _FRAGMENTOS_GENERICOS):
            return True
        alnum = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", text))
        lines = len([l for l in text.splitlines() if l.strip()])
        return alnum < 25 and lines <= 1

    def _acceptable_text(self, text: str) -> bool:
        return bool(text) and not self._is_generic_response(text) and self._score_text(text) >= 90

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        return normalize_ocr_text(text)

    @staticmethod
    def _normalize_line(line: str) -> str:
        line = re.sub(r"\s+", " ", line.strip().lower())
        return re.sub(r"[^a-zà-öø-ÿ0-9 ]+", "", line)

    def _dedupe_lines(self, text: str) -> str:
        deduped: list[str] = []
        last_norm = None
        for line in text.splitlines():
            raw = line.rstrip()
            if not raw.strip():
                if deduped and deduped[-1] != "":
                    deduped.append("")
                continue
            norm = self._normalize_line(raw)
            if not norm or norm == last_norm:
                continue
            deduped.append(raw)
            last_norm = norm
        return normalize_ocr_text("\n".join(deduped))

    @staticmethod
    def _image_to_array(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    def _normalize_result_item(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None

        box = item[0]
        text = self._clean_ocr_text(str(item[1] or ""))
        confidence = float(item[2]) if len(item) > 2 and item[2] is not None else 0.0

        if not text:
            return None
        if confidence < self.min_confidence and len(text) <= 2:
            return None

        xs: list[float] = []
        ys: list[float] = []
        if isinstance(box, (list, tuple)):
            for point in box:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
        if not xs or not ys:
            return None

        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        height = max(1.0, bottom - top)

        return {
            "text": text,
            "confidence": confidence,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "center_y": (top + bottom) / 2.0,
            "height": height,
        }

    @staticmethod
    def _new_line(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "items": [item],
            "top": item["top"],
            "bottom": item["bottom"],
            "center_y": item["center_y"],
            "avg_height": item["height"],
        }

    @staticmethod
    def _update_line(line: dict[str, Any]) -> None:
        items = line["items"]
        line["top"] = min(i["top"] for i in items)
        line["bottom"] = max(i["bottom"] for i in items)
        line["center_y"] = sum(i["center_y"] for i in items) / len(items)
        line["avg_height"] = sum(i["height"] for i in items) / len(items)

    def _results_to_text(self, results: list[Any]) -> str:
        items = [normalized for item in results if (normalized := self._normalize_result_item(item))]
        if not items:
            return ""

        items.sort(key=lambda i: (i["top"], i["left"]))

        lines: list[dict[str, Any]] = []
        for item in items:
            if not lines:
                lines.append(self._new_line(item))
                continue

            current = lines[-1]
            tolerance = max(12.0, min(current["avg_height"], item["height"]) * 0.65)
            if abs(item["center_y"] - current["center_y"]) <= tolerance:
                current["items"].append(item)
                self._update_line(current)
            else:
                lines.append(self._new_line(item))

        rendered_lines: list[str] = []
        prev_bottom = None
        prev_height = None
        for line in lines:
            line_items = sorted(line["items"], key=lambda i: (i["left"], i["top"]))
            line_text = " ".join(i["text"] for i in line_items).strip()
            if not line_text:
                continue

            if prev_bottom is not None and prev_height is not None:
                gap = line["top"] - prev_bottom
                paragraph_gap = max(18.0, prev_height * 1.4, line["avg_height"] * 1.2)
                if gap > paragraph_gap:
                    rendered_lines.append("")

            rendered_lines.append(line_text)
            prev_bottom = line["bottom"]
            prev_height = line["avg_height"]

        return self._dedupe_lines("\n".join(rendered_lines))

    def _ocr_easyocr(self, image: Image.Image) -> str:
        last_error = None
        for candidate in self._ocr_candidates(image):
            candidate_label = f"{candidate.width}x{candidate.height}"
            try:
                results = self.reader.readtext(
                    self._image_to_array(candidate),
                    detail=1,
                    paragraph=self.paragraph_mode,
                )
                text = self._results_to_text(results)
                if text and not self._is_generic_response(text):
                    return text
                last_error = RuntimeError(f"EasyOCR não retornou texto utilizável em {candidate_label}.")
            except Exception as e:
                last_error = RuntimeError(f"Falha no EasyOCR em {candidate_label}: {e}")

        if last_error:
            raise RuntimeError(f"Falha ao executar OCR no EasyOCR: {last_error}") from last_error
        return ""

    def _try_full_page_ocr(self, image: Image.Image, mode: str) -> dict[str, Any]:
        try:
            text = self._clean_ocr_text(self._ocr_easyocr(image))
            return {"mode": mode, "text": text, "score": self._score_text(text), "warnings": [], "error": None}
        except Exception as e:
            return {"mode": mode, "text": "", "score": 0.0, "warnings": [], "error": str(e)}

    def _try_band_ocr(self, image: Image.Image, mode: str) -> dict[str, Any]:
        best_text, best_score, warnings, last_error = "", 0.0, [], None

        for height in self._band_heights(image):
            band_texts = []
            bands = self._split_into_bands(image, max_height=height)
            for idx, band in enumerate(bands, 1):
                try:
                    text = self._clean_ocr_text(self._ocr_easyocr(band))
                    if text:
                        band_texts.append(text)
                except Exception as e:
                    last_error = e
                    warnings.append(f"faixa {idx}/{len(bands)} altura {height}: {e}")

            if not band_texts:
                continue

            merged = self._dedupe_lines("\n".join(band_texts))
            score = self._score_text(merged)
            if score > best_score:
                best_score, best_text = score, merged
            if merged and not self._is_generic_response(merged):
                break

        if best_text:
            return {"mode": mode, "text": best_text, "score": best_score, "warnings": warnings, "error": None}
        return {"mode": mode, "text": "", "score": 0.0, "warnings": warnings, "error": str(last_error) if last_error else None}

    @staticmethod
    def _best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [a for a in attempts if a.get("text")]
        if valid:
            return max(valid, key=lambda a: a.get("score", 0.0))
        return attempts[-1] if attempts else {"mode": "undefined", "text": "", "score": 0.0, "warnings": [], "error": "Nenhuma tentativa executada."}

    def _orientation_score(self, image: Image.Image) -> float:
        try:
            text = self._ocr_easyocr(image)
            return 0.0 if not text or self._is_generic_response(text) else self._score_text(text)
        except Exception:
            return 0.0

    def _detect_rotation(self, image: Image.Image) -> int:
        try:
            sample = self._resize_if_needed(image, 1400)
            pre = self._preprocess(sample)
            normal = self._orientation_score(pre.convert("RGB"))
            inverted = self._orientation_score(pre.rotate(180, expand=True).convert("RGB"))
            if inverted > normal * 1.2 and (inverted - normal) > 25:
                return 180
            if normal < 15 and inverted > 45:
                return 180
        except Exception:
            pass
        return 0

    def _fix_orientation(self, image: Image.Image, page_num: int) -> tuple[Image.Image, int]:
        if not self.detect_orientation:
            return image, 0
        rotation = self._detect_rotation(image)
        if rotation:
            print(f"   ↻ Página {page_num}: rotação de {rotation}° aplicada.")
            return image.rotate(rotation, expand=True), rotation
        return image, 0

    def extract_text_from_image(self, image: Image.Image, page_num: int, save_pre_to: Path | None = None) -> dict[str, Any]:
        print(f"   🔍 Página {page_num}: executando OCR...")

        corrected, rotation = self._fix_orientation(image, page_num)
        pre = self._preprocess(corrected)

        if self.save_preprocessed and save_pre_to:
            try:
                save_pre_to.parent.mkdir(parents=True, exist_ok=True)
                pre.convert("L").save(save_pre_to.with_suffix(".jpg"), format="JPEG", quality=92)
            except Exception:
                pass

        attempts = [self._try_full_page_ocr(corrected.convert("RGB"), "pagina_inteira")]
        best = self._best_attempt(attempts)

        if not self._acceptable_text(best["text"]):
            attempts.append(self._try_full_page_ocr(pre.convert("RGB"), "pagina_inteira_pre"))
            best = self._best_attempt(attempts)

        if not self._acceptable_text(best["text"]):
            attempts.append(self._try_band_ocr(corrected.convert("RGB"), "faixas"))
            best = self._best_attempt(attempts)

        if not self._acceptable_text(best["text"]):
            attempts.append(self._try_band_ocr(pre.convert("RGB"), "faixas_pre"))
            best = self._best_attempt(attempts)

        text = best.get("text", "")
        warnings = [w for a in attempts for w in a.get("warnings", [])]
        errors = [f"{a['mode']}: {a['error']}" for a in attempts if a.get("error")]

        if not text:
            raise RuntimeError(" | ".join(errors) or "OCR não retornou texto utilizável.")

        preview = text[:100].replace("\n", " ") + ("..." if len(text) > 100 else "")
        print(f"   ✓ Texto extraído ({len(text)} caracteres): {preview}\n")

        return {
            "text": text,
            "rotation_applied_degrees": rotation,
            "ocr_method": best.get("mode"),
            "warnings": warnings,
        }

    def _load_existing_result(self) -> dict[str, Any] | None:
        if not self.resume_progress or not self.output_path.exists():
            return None
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict) or data.get("file_name") != self.pdf_path.name:
            return None
        source_pdf = str(self.pdf_path.resolve())
        if data.get("source_pdf") and data.get("source_pdf") != source_pdf:
            return None
        if data.get("model") and data.get("model") != self.model:
            return None
        if not isinstance(data.get("pages"), list):
            return None
        return data

    def _is_reusable_cached_page(self, page: dict[str, Any]) -> bool:
        if not isinstance(page, dict) or page.get("error"):
            return False
        text = str(page.get("text", "") or "").strip()
        return bool(text) and not self._is_generic_response(text)

    def _render_or_reuse_pages(self) -> list[Path]:
        existing = sorted(self.pages_dir.glob("page_*.jpg")) or sorted(self.pages_dir.glob("page_*.png"))
        if existing:
            print(f"♻️ Reutilizando {len(existing)} imagem(ns) em cache: {self.pages_dir}")
            return existing

        print(f"🖼 Convertendo PDF em imagens: {self.pdf_path.name}")
        paths = convert_from_path(
            self.pdf_path,
            dpi=self.render_dpi,
            fmt="jpeg",
            output_folder=str(self.pages_dir),
            output_file="page",
            paths_only=True,
            thread_count=max(1, (os.cpu_count() or 2) // 2),
        )

        renamed = []
        for idx, p in enumerate(paths, 1):
            src = Path(p)
            dst = self.pages_dir / f"page_{idx:04d}.jpg"
            if src != dst:
                src.rename(dst)
            renamed.append(dst)

        print(f"   ✓ {len(renamed)} página(s) salvas em: {self.pages_dir}\n")
        return renamed

    def save_result(self, result: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def process(self) -> dict[str, Any]:
        run_started_at = time.perf_counter()
        print(f"\n{'='*70}")
        print(f"🚀 PROCESSANDO: {self.pdf_path.name}")
        print(f"{'='*70}\n")
        print(
            f"⚙️ EasyOCR versão do projeto: {PROJECT_EASYOCR_VERSION} | instalada: {self.easyocr_version}"
        )
        print(f"⚙️ EasyOCR usando {'GPU' if self.gpu else 'CPU'}.\n")

        images = self._render_or_reuse_pages()

        cached_pages: dict[int, dict] = {}
        skipped_cached_pages = 0
        previous = self._load_existing_result()
        if previous:
            for page in previous.get("pages", []):
                if isinstance(page, dict) and isinstance(page.get("page"), int):
                    num = page["page"]
                    if 1 <= num <= len(images):
                        if self._is_reusable_cached_page(page):
                            cached_pages[num] = page
                        elif page.get("text") and not page.get("error"):
                            skipped_cached_pages += 1
            if cached_pages:
                print(f"   ♻️ Retomando: {len(cached_pages)} página(s) reutilizadas do JSON existente.")
            if skipped_cached_pages:
                print(f"   ↺ Ignorando {skipped_cached_pages} página(s) com OCR salvo inválido/genérico.")

        pages = []
        for i, img_path in enumerate(images, 1):
            if i in cached_pages:
                cached = dict(cached_pages[i])
                cached["image_file"] = img_path.name
                cached["char_count"] = len(cached.get("text", ""))
                print(f"   ↪ Página {i}: reutilizando OCR salvo ({cached.get('char_count', len(cached.get('text', '')))} caracteres).")
                pages.append(cached)
                continue

            page_started_at = time.perf_counter()
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    pre_path = img_path.with_name(f"{img_path.stem}-pre.jpg")
                    data = self.extract_text_from_image(img, i, save_pre_to=pre_path)
                processing_seconds = _round_seconds(time.perf_counter() - page_started_at)
                page = {
                    "page": i,
                    "image_file": img_path.name,
                    "text": data["text"],
                    "char_count": len(data["text"]),
                    "processing_seconds": processing_seconds,
                    "rotation_applied_degrees": data["rotation_applied_degrees"],
                    "ocr_method": data.get("ocr_method"),
                }
                if data.get("warnings"):
                    page["warnings"] = data["warnings"]
            except Exception as e:
                processing_seconds = _round_seconds(time.perf_counter() - page_started_at)
                print(f"   ⚠️ Página {i}: erro no OCR. {e}\n")
                page = {
                    "page": i,
                    "image_file": img_path.name,
                    "text": "",
                    "char_count": 0,
                    "processing_seconds": processing_seconds,
                    "rotation_applied_degrees": 0,
                    "error": str(e),
                }

            pages.append(page)
            partial = self._build_output(pages, len(images), run_elapsed_seconds=time.perf_counter() - run_started_at)
            self.save_result(partial)

        result = self._build_output(pages, len(images), run_elapsed_seconds=time.perf_counter() - run_started_at)
        self.save_result(result)
        return result

    def _build_output(self, pages: list[dict[str, Any]], page_count: int, run_elapsed_seconds: float = 0.0) -> dict[str, Any]:
        return build_document_output(
            pdf_path=self.pdf_path,
            model=self.model,
            pages=pages,
            page_count=page_count,
            run_elapsed_seconds=run_elapsed_seconds,
            metadata={
                "runtime": {
                    "backend": "easyocr",
                    "device": "gpu" if self.gpu else "cpu",
                    "languages": list(self.languages),
                    "project_easyocr_version": PROJECT_EASYOCR_VERSION,
                    "easyocr_version": self.easyocr_version,
                }
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai texto de PDF usando EasyOCR")
    parser.add_argument("pdf", help="Caminho do PDF de entrada")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Caminho opcional para salvar uma cópia adicional do JSON final",
    )
    parser.add_argument(
        "--languages",
        default=os.getenv("EASYOCR_LANGUAGES", "pt,en"),
        help="Lista CSV de idiomas do EasyOCR, por exemplo: pt,en",
    )
    parser.add_argument(
        "--model-storage-dir",
        default=os.getenv("EASYOCR_MODEL_STORAGE_DIR", ""),
        help="Diretório opcional para armazenar os modelos do EasyOCR",
    )
    parser.add_argument(
        "--user-network-dir",
        default=os.getenv("EASYOCR_USER_NETWORK_DIR", ""),
        help="Diretório opcional para redes customizadas do EasyOCR",
    )
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Usa GPU no EasyOCR")
    parser.add_argument("--cpu", dest="gpu", action="store_false", help="Força execução em CPU")
    parser.add_argument(
        "--download-models",
        dest="download_enabled",
        action="store_true",
        help="Permite baixar modelos automaticamente",
    )
    parser.add_argument(
        "--no-download-models",
        dest="download_enabled",
        action="store_false",
        help="Desabilita download automático de modelos",
    )
    parser.set_defaults(
        gpu=_env_optional_flag("EASYOCR_GPU"),
        download_enabled=_env_flag("EASYOCR_DOWNLOAD_ENABLED", True),
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists():
        print(f"PDF não encontrado: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    languages = [part.strip() for part in args.languages.split(",") if part.strip()]
    if not languages:
        print("Nenhum idioma válido informado em `--languages`.", file=sys.stderr)
        sys.exit(1)

    output_path = build_extracted_output_path(ROOT_DIR, "easyocr", pdf_path)
    extractor = EasyOCRBenchmarkExtractor(
        pdf_path=pdf_path,
        output_path=output_path,
        languages=languages,
        gpu=args.gpu,
        model_storage_dir=args.model_storage_dir or None,
        user_network_dir=args.user_network_dir or None,
        download_enabled=args.download_enabled,
    )
    result = extractor.process()

    additional_output = Path(args.output).expanduser() if args.output else None
    if additional_output and additional_output.resolve() != output_path.resolve():
        additional_output.parent.mkdir(parents=True, exist_ok=True)
        additional_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ Processamento concluído!")
    print(f"📄 Páginas: {result['page_count']}")
    print(f"📝 Caracteres extraídos: {len(result['text'])}")
    print(f"⏱ Tempo total do documento: {result.get('total_processing_seconds', 0.0):.3f}s")
    print(f"⏱ Tempo médio por página: {result.get('average_page_processing_seconds', 0.0):.3f}s")
    print(f"⏱ Tempo desta execução: {result.get('run_elapsed_seconds', 0.0):.3f}s")
    print(f"💾 JSON salvo em: {output_path.resolve()}")
    if additional_output and additional_output.resolve() != output_path.resolve():
        print(f"💾 Cópia adicional salva em: {additional_output.resolve()}")


if __name__ == "__main__":
    main()
