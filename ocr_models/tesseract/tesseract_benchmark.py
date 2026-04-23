"""Benchmark de OCR para PDFs usando Tesseract.

Renderiza as páginas do PDF, executa a extração e salva a saída no JSON padronizado do projeto.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None
    Output = None

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

PROJECT_TESSERACT_VERSION = "5.3.4"
PROJECT_PYTESSERACT_VERSION = "0.3.13"


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


def _parse_languages(raw: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,+]", raw) if part.strip()]


class TesseractBenchmarkExtractor:
    def __init__(
        self,
        pdf_path: Path,
        output_path: Path,
        languages: list[str],
        tesseract_cmd: str | None = None,
        oem: int = 3,
        full_page_psm: int = 3,
        band_psm: int = 6,
        tessdata_dir: str | None = None,
    ):
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.languages = languages
        self.language_string = "+".join(self.languages)
        self.oem = oem
        self.full_page_psm = full_page_psm
        self.band_psm = band_psm
        self.tesseract_cmd = tesseract_cmd.strip() if tesseract_cmd else ""
        self.tessdata_dir = Path(tessdata_dir).expanduser() if tessdata_dir else None
        self.model = f"tesseract:{self.language_string}:oem{self.oem}"
        self.pytesseract_version = getattr(pytesseract, "__version__", PROJECT_PYTESSERACT_VERSION)

        self.pages_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)

        self.render_dpi = int(os.getenv("PDF_RENDER_DPI", "220"))
        self.tesseract_timeout = float(os.getenv("TESSERACT_TIMEOUT_SECONDS", "60"))
        self.ocr_max_dim = int(os.getenv("OCR_IMAGE_MAX_DIM", "1600"))
        self.ocr_fallback_dims = _env_int_tuple(
            "OCR_FALLBACK_MAX_DIMS",
            (self.ocr_max_dim, max(1200, int(self.ocr_max_dim * 0.75))),
        )
        self.band_heights = _env_int_tuple("OCR_BAND_HEIGHTS", (1200, 900))
        self.save_preprocessed = _env_flag("SAVE_PREPROCESSED_IMAGES", False)
        self.resume_progress = _env_flag("RESUME_OCR_PROGRESS", True)
        self.detect_orientation = _env_flag("ENABLE_OCR_ORIENTATION_DETECTION", False)
        self.min_confidence = float(os.getenv("TESSERACT_MIN_CONFIDENCE", "0.0"))
        self.preserve_interword_spaces = _env_flag("TESSERACT_PRESERVE_INTERWORD_SPACES", True)

        self._validate_tesseract()

    def _resolve_tesseract_cmd(self) -> str:
        requested = self.tesseract_cmd or os.getenv("TESSERACT_CMD", "").strip()
        if requested:
            return requested

        local_wrapper = Path(__file__).resolve().parent / "local" / "bin" / "tesseract"
        if local_wrapper.exists():
            return str(local_wrapper)

        found = shutil.which("tesseract")
        if found:
            return found

        return "tesseract"

    def _validate_tesseract(self) -> None:
        if pytesseract is None or Output is None:
            raise RuntimeError(
                "A biblioteca `pytesseract` não está instalada. Crie sua venv e instale o `requirements.txt` desta pasta."
            )

        resolved_cmd = self._resolve_tesseract_cmd()
        pytesseract.pytesseract.tesseract_cmd = resolved_cmd
        self.tesseract_cmd = resolved_cmd

        try:
            self.tesseract_version = str(pytesseract.get_tesseract_version()).strip().splitlines()[0]
        except Exception as e:
            raise RuntimeError(
                "Não foi possível executar o binário do Tesseract. "
                "Instale o Tesseract OCR no sistema e/ou ajuste `--tesseract-cmd` / `TESSERACT_CMD`."
            ) from e

        try:
            available_languages = set(pytesseract.get_languages(config=""))
        except Exception as e:
            raise RuntimeError(
                "Não foi possível listar os idiomas do Tesseract instalado. "
                "Verifique a instalação do binário e dos dados de idioma."
            ) from e

        missing_languages = [lang for lang in self.languages if lang not in available_languages]
        if missing_languages:
            available = ", ".join(sorted(available_languages)) or "nenhum"
            raise RuntimeError(
                f"Idiomas do Tesseract não encontrados: {', '.join(missing_languages)}. Disponíveis: {available}"
            )

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

    def _tesseract_config(self, psm: int) -> str:
        parts = [f"--oem {self.oem}", f"--psm {psm}"]
        if self.preserve_interword_spaces:
            parts.append("-c preserve_interword_spaces=1")
        if self.tessdata_dir:
            parts.append(f'--tessdata-dir "{self.tessdata_dir}"')
        return " ".join(parts)

    def _data_to_text(self, data: dict[str, list[Any]]) -> str:
        texts = data.get("text", [])
        if not texts:
            return ""

        items: list[dict[str, Any]] = []
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])
        block_nums = data.get("block_num", [])
        par_nums = data.get("par_num", [])
        line_nums = data.get("line_num", [])
        word_nums = data.get("word_num", [])
        confs = data.get("conf", [])

        for idx, raw_text in enumerate(texts):
            text = self._clean_ocr_text(str(raw_text or ""))
            if not text:
                continue

            conf_raw = confs[idx] if idx < len(confs) else "-1"
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = -1.0

            if confidence < self.min_confidence and len(text) <= 2:
                continue

            try:
                left = int(float(lefts[idx] if idx < len(lefts) else 0))
                top = int(float(tops[idx] if idx < len(tops) else 0))
                width = int(float(widths[idx] if idx < len(widths) else 0))
                height = int(float(heights[idx] if idx < len(heights) else 0))
                block_num = int(float(block_nums[idx] if idx < len(block_nums) else 0))
                par_num = int(float(par_nums[idx] if idx < len(par_nums) else 0))
                line_num = int(float(line_nums[idx] if idx < len(line_nums) else 0))
                word_num = int(float(word_nums[idx] if idx < len(word_nums) else idx))
            except (TypeError, ValueError, IndexError):
                continue

            items.append(
                {
                    "index": idx,
                    "text": text,
                    "confidence": confidence,
                    "left": left,
                    "top": top,
                    "bottom": top + max(1, height),
                    "height": max(1, height),
                    "block_num": block_num,
                    "par_num": par_num,
                    "line_num": line_num,
                    "word_num": word_num,
                }
            )

        if not items:
            return ""

        rendered_lines: list[str] = []
        current_key = None
        current_words: list[str] = []
        current_meta: dict[str, Any] | None = None
        prev_bottom = None
        prev_height = None
        prev_block_par = None

        def flush_current() -> None:
            nonlocal current_words, current_meta, prev_bottom, prev_height, prev_block_par
            if not current_words or current_meta is None:
                return

            line_text = " ".join(current_words).strip()
            if not line_text:
                current_words = []
                current_meta = None
                return

            block_par = (current_meta["block_num"], current_meta["par_num"])
            if rendered_lines and prev_bottom is not None and prev_height is not None:
                gap = current_meta["top"] - prev_bottom
                paragraph_gap = max(18.0, prev_height * 1.4, current_meta["avg_height"] * 1.2)
                if block_par != prev_block_par or gap > paragraph_gap:
                    rendered_lines.append("")

            rendered_lines.append(line_text)
            prev_bottom = current_meta["bottom"]
            prev_height = current_meta["avg_height"]
            prev_block_par = block_par
            current_words = []
            current_meta = None

        for item in items:
            key = (item["block_num"], item["par_num"], item["line_num"])
            if current_key is None:
                current_key = key
                current_meta = {
                    "top": item["top"],
                    "bottom": item["bottom"],
                    "avg_height": float(item["height"]),
                    "block_num": item["block_num"],
                    "par_num": item["par_num"],
                }
            elif key != current_key:
                flush_current()
                current_key = key
                current_meta = {
                    "top": item["top"],
                    "bottom": item["bottom"],
                    "avg_height": float(item["height"]),
                    "block_num": item["block_num"],
                    "par_num": item["par_num"],
                }

            current_words.append(item["text"])
            if current_meta is not None:
                current_meta["top"] = min(current_meta["top"], item["top"])
                current_meta["bottom"] = max(current_meta["bottom"], item["bottom"])
                current_meta["avg_height"] = (current_meta["avg_height"] + item["height"]) / 2.0

        flush_current()
        return self._dedupe_lines("\n".join(rendered_lines))

    def _ocr_tesseract(self, image: Image.Image, psm: int) -> str:
        last_error = None
        config = self._tesseract_config(psm)

        for candidate in self._ocr_candidates(image):
            candidate_label = f"{candidate.width}x{candidate.height}"
            try:
                data = pytesseract.image_to_data(
                    candidate,
                    lang=self.language_string,
                    config=config,
                    output_type=Output.DICT,
                    timeout=self.tesseract_timeout,
                )
                text = self._data_to_text(data)
                if text and not self._is_generic_response(text):
                    return text
                last_error = RuntimeError(f"Tesseract não retornou texto utilizável em {candidate_label}.")
            except RuntimeError as e:
                last_error = RuntimeError(f"Falha no Tesseract em {candidate_label}: {e}")
            except Exception as e:
                last_error = RuntimeError(f"Falha no Tesseract em {candidate_label}: {e}")

        if last_error:
            raise RuntimeError(f"Falha ao executar OCR no Tesseract: {last_error}") from last_error
        return ""

    def _try_full_page_ocr(self, image: Image.Image, mode: str, psm: int) -> dict[str, Any]:
        try:
            text = self._clean_ocr_text(self._ocr_tesseract(image, psm=psm))
            return {"mode": mode, "text": text, "score": self._score_text(text), "warnings": [], "error": None}
        except Exception as e:
            return {"mode": mode, "text": "", "score": 0.0, "warnings": [], "error": str(e)}

    def _try_band_ocr(self, image: Image.Image, mode: str, psm: int) -> dict[str, Any]:
        best_text, best_score, warnings, last_error = "", 0.0, [], None

        for height in self._band_heights(image):
            band_texts = []
            bands = self._split_into_bands(image, max_height=height)
            for idx, band in enumerate(bands, 1):
                try:
                    text = self._clean_ocr_text(self._ocr_tesseract(band, psm=psm))
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
            text = self._ocr_tesseract(image, psm=self.full_page_psm)
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

        attempts = [self._try_full_page_ocr(corrected.convert("RGB"), "pagina_inteira", psm=self.full_page_psm)]
        best = self._best_attempt(attempts)

        if not self._acceptable_text(best["text"]):
            attempts.append(self._try_full_page_ocr(pre.convert("RGB"), "pagina_inteira_pre", psm=self.full_page_psm))
            best = self._best_attempt(attempts)

        if not self._acceptable_text(best["text"]):
            attempts.append(self._try_band_ocr(corrected.convert("RGB"), "faixas", psm=self.band_psm))
            best = self._best_attempt(attempts)

        if not self._acceptable_text(best["text"]):
            attempts.append(self._try_band_ocr(pre.convert("RGB"), "faixas_pre", psm=self.band_psm))
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
        print(f"⚙️ Tesseract usando binário: {self.tesseract_cmd}")
        print(f"⚙️ Tesseract versão do projeto: {PROJECT_TESSERACT_VERSION} | instalada: {self.tesseract_version}")
        print(f"⚙️ pytesseract versão Python: {self.pytesseract_version}\n")

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
                    "backend": "tesseract",
                    "tesseract_cmd": self.tesseract_cmd,
                    "project_tesseract_version": PROJECT_TESSERACT_VERSION,
                    "tesseract_version": self.tesseract_version,
                    "project_pytesseract_version": PROJECT_PYTESSERACT_VERSION,
                    "pytesseract_version": self.pytesseract_version,
                    "languages": list(self.languages),
                    "oem": self.oem,
                    "full_page_psm": self.full_page_psm,
                    "band_psm": self.band_psm,
                }
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai texto de PDF usando Tesseract OCR")
    parser.add_argument("pdf", help="Caminho do PDF de entrada")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Caminho opcional para salvar uma cópia adicional do JSON final",
    )
    parser.add_argument(
        "--languages",
        default=os.getenv("TESSERACT_LANGUAGES", "por+eng"),
        help="Idiomas do Tesseract. Aceita `por+eng` ou `por,eng`.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=os.getenv("TESSERACT_CMD", ""),
        help="Caminho opcional para o executável do Tesseract",
    )
    parser.add_argument(
        "--tessdata-dir",
        default=os.getenv("TESSDATA_PREFIX", ""),
        help="Diretório opcional do tessdata",
    )
    parser.add_argument("--oem", type=int, default=int(os.getenv("TESSERACT_OEM", "3")), help="OEM do Tesseract")
    parser.add_argument(
        "--full-page-psm",
        type=int,
        default=int(os.getenv("TESSERACT_FULL_PAGE_PSM", "3")),
        help="PSM usado na tentativa de página inteira",
    )
    parser.add_argument(
        "--band-psm",
        type=int,
        default=int(os.getenv("TESSERACT_BAND_PSM", "6")),
        help="PSM usado nas tentativas por faixas",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists():
        print(f"PDF não encontrado: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    languages = _parse_languages(args.languages)
    if not languages:
        print("Nenhum idioma válido informado em `--languages`.", file=sys.stderr)
        sys.exit(1)

    output_path = build_extracted_output_path(ROOT_DIR, "tesseract", pdf_path)
    extractor = TesseractBenchmarkExtractor(
        pdf_path=pdf_path,
        output_path=output_path,
        languages=languages,
        tesseract_cmd=args.tesseract_cmd or None,
        oem=args.oem,
        full_page_psm=args.full_page_psm,
        band_psm=args.band_psm,
        tessdata_dir=args.tessdata_dir or None,
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
