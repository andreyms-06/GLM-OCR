"""Benchmark de OCR para PDFs usando a API OCR da Mistral.

Renderiza as páginas do PDF, executa a extração e salva a saída no JSON padronizado do projeto.
"""

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


import cv2 as cv
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

load_dotenv()

API_KEY = os.getenv("API_KEY")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ocr_output_paths import build_extracted_output_path
from ocr_output_schema import build_document_output, cleanup_llm_markup, normalize_ocr_text


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


class MistralOCRBenchmarkExtractor:
    def __init__(
        self,
        pdf_path: Path,
        output_path: Path,
        api_key: str,
        model: str,
        api_base: str,
        table_format: str | None = None,
        extract_header: bool = False,
        extract_footer: bool = False,
    ):
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.api_base = api_base.rstrip("/")
        self.table_format = table_format.strip() if table_format else None
        self.extract_header = extract_header
        self.extract_footer = extract_footer

        self.pages_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)

        self.render_dpi = int(os.getenv("PDF_RENDER_DPI", "220"))
        self.api_timeout = float(os.getenv("MISTRAL_TIMEOUT_SECONDS", "120"))
        self.max_retries = max(0, int(os.getenv("MISTRAL_MAX_RETRIES", "2")))
        self.retry_backoff = max(0.0, float(os.getenv("MISTRAL_RETRY_BACKOFF_SECONDS", "2.0")))
        self.ocr_max_dim = int(os.getenv("OCR_IMAGE_MAX_DIM", "1600"))
        self.ocr_fallback_dims = _env_int_tuple(
            "OCR_FALLBACK_MAX_DIMS",
            (self.ocr_max_dim, max(1200, int(self.ocr_max_dim * 0.75))),
        )
        self.band_heights = _env_int_tuple("OCR_BAND_HEIGHTS", (1200, 900))
        self.save_preprocessed = _env_flag("SAVE_PREPROCESSED_IMAGES", False)
        self.resume_progress = _env_flag("RESUME_OCR_PROGRESS", True)
        self.detect_orientation = _env_flag("ENABLE_OCR_ORIENTATION_DETECTION", False)

        self._validate_config()

    def _validate_config(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "A chave da API da Mistral não foi informada. Use `--api-key` ou defina `MISTRAL_API_KEY`."
            )
        if not self.model:
            raise RuntimeError("O modelo da Mistral não foi informado.")

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
        return normalize_ocr_text(cleanup_llm_markup(text))

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
    def _image_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _extract_error_message(payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("message", "detail", "error", "title"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            if isinstance(payload.get("error"), dict):
                nested = payload["error"]
                for key in ("message", "detail", "type"):
                    value = nested.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        return "Erro desconhecido."

    def _post_ocr_request(self, image: Image.Image) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "document": {
                "type": "image_url",
                "image_url": self._image_to_data_url(image),
            },
            "include_image_base64": False,
        }
        if self.table_format:
            payload["table_format"] = self.table_format
        if self.extract_header:
            payload["extract_header"] = True
        if self.extract_footer:
            payload["extract_footer"] = True

        request = urllib.request.Request(
            f"{self.api_base}/ocr",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ocr-benchmark-mistral/1.0",
            },
            method="POST",
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.api_timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise RuntimeError("A API da Mistral retornou um JSON inválido.")
                return parsed
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                payload_error: Any = body
                try:
                    payload_error = json.loads(body) if body else body
                except Exception:
                    pass
                message = self._extract_error_message(payload_error)
                last_error = RuntimeError(f"HTTP {e.code} na API da Mistral: {message}")
                if e.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    break
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"Falha de conexão com a API da Mistral: {e.reason}")
                if attempt >= self.max_retries:
                    break
            except Exception as e:
                last_error = RuntimeError(f"Falha ao chamar a API da Mistral: {e}")
                if attempt >= self.max_retries:
                    break

            sleep_seconds = self.retry_backoff * (2**attempt)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if last_error:
            raise last_error
        raise RuntimeError("Falha desconhecida ao chamar a API da Mistral.")

    def _response_to_text(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        pages = payload.get("pages")
        if not isinstance(pages, list) or not pages:
            raise RuntimeError("A API da Mistral não retornou páginas OCR.")

        parts: list[str] = []
        page_indices: list[int] = []
        tables_count = 0
        images_count = 0
        hyperlinks_count = 0

        for page in pages:
            if not isinstance(page, dict):
                continue

            markdown = str(page.get("markdown", "") or "").strip()
            header = str(page.get("header", "") or "").strip() if self.extract_header else ""
            footer = str(page.get("footer", "") or "").strip() if self.extract_footer else ""

            page_text_parts = []
            if header:
                page_text_parts.append(header)
            if markdown:
                page_text_parts.append(markdown)
            if footer:
                page_text_parts.append(footer)
            page_text = "\n\n".join(part for part in page_text_parts if part).strip()

            if page_text:
                parts.append(page_text)
            if isinstance(page.get("index"), int):
                page_indices.append(int(page["index"]))
            if isinstance(page.get("tables"), list):
                tables_count += len(page["tables"])
            if isinstance(page.get("images"), list):
                images_count += len(page["images"])
            if isinstance(page.get("hyperlinks"), list):
                hyperlinks_count += len(page["hyperlinks"])

        text = self._clean_ocr_text("\n\n".join(parts))
        meta: dict[str, Any] = {
            "response_pages_count": len(pages),
            "response_page_indices": page_indices,
            "tables_count": tables_count,
            "images_count": images_count,
            "hyperlinks_count": hyperlinks_count,
        }
        if isinstance(payload.get("usage_info"), dict):
            meta["usage_info"] = payload["usage_info"]
        return text, meta

    def _ocr_mistral(self, image: Image.Image) -> tuple[str, dict[str, Any]]:
        last_error = None
        for candidate in self._ocr_candidates(image):
            candidate_label = f"{candidate.width}x{candidate.height}"
            try:
                payload = self._post_ocr_request(candidate)
                text, meta = self._response_to_text(payload)
                if text and not self._is_generic_response(text):
                    meta["candidate_size"] = candidate_label
                    return text, meta
                last_error = RuntimeError(f"Mistral OCR não retornou texto utilizável em {candidate_label}.")
            except Exception as e:
                last_error = RuntimeError(f"Falha no Mistral OCR em {candidate_label}: {e}")

        if last_error:
            raise RuntimeError(f"Falha ao executar OCR na Mistral: {last_error}") from last_error
        return "", {}

    def _try_full_page_ocr(self, image: Image.Image, mode: str) -> dict[str, Any]:
        try:
            text, meta = self._ocr_mistral(image)
            text = self._clean_ocr_text(text)
            return {
                "mode": mode,
                "text": text,
                "score": self._score_text(text),
                "warnings": [],
                "error": None,
                "meta": meta,
            }
        except Exception as e:
            return {"mode": mode, "text": "", "score": 0.0, "warnings": [], "error": str(e), "meta": {}}

    def _try_band_ocr(self, image: Image.Image, mode: str) -> dict[str, Any]:
        best_text, best_score, best_meta, warnings, last_error = "", 0.0, {}, [], None

        for height in self._band_heights(image):
            band_texts = []
            meta_batches = []
            bands = self._split_into_bands(image, max_height=height)
            for idx, band in enumerate(bands, 1):
                try:
                    text, meta = self._ocr_mistral(band)
                    text = self._clean_ocr_text(text)
                    if text:
                        band_texts.append(text)
                        meta_batches.append(meta)
                except Exception as e:
                    last_error = e
                    warnings.append(f"faixa {idx}/{len(bands)} altura {height}: {e}")

            if not band_texts:
                continue

            merged = self._dedupe_lines("\n".join(band_texts))
            score = self._score_text(merged)
            if score > best_score:
                best_score = score
                best_text = merged
                best_meta = {
                    "bands_count": len(band_texts),
                    "band_height": height,
                    "tables_count": sum(int(meta.get("tables_count", 0)) for meta in meta_batches),
                    "images_count": sum(int(meta.get("images_count", 0)) for meta in meta_batches),
                    "hyperlinks_count": sum(int(meta.get("hyperlinks_count", 0)) for meta in meta_batches),
                }
            if merged and not self._is_generic_response(merged):
                break

        if best_text:
            return {
                "mode": mode,
                "text": best_text,
                "score": best_score,
                "warnings": warnings,
                "error": None,
                "meta": best_meta,
            }
        return {
            "mode": mode,
            "text": "",
            "score": 0.0,
            "warnings": warnings,
            "error": str(last_error) if last_error else None,
            "meta": {},
        }

    @staticmethod
    def _best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [a for a in attempts if a.get("text")]
        if valid:
            return max(valid, key=lambda a: a.get("score", 0.0))
        return attempts[-1] if attempts else {"mode": "undefined", "text": "", "score": 0.0, "warnings": [], "error": "Nenhuma tentativa executada.", "meta": {}}

    def _orientation_score(self, image: Image.Image) -> float:
        try:
            text, _meta = self._ocr_mistral(image)
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
            "meta": best.get("meta", {}),
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
        print(f"⚙️ Mistral OCR usando o modelo: {self.model}")
        print(f"⚙️ Endpoint OCR: {self.api_base}/ocr\n")

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
                if isinstance(data.get("meta"), dict):
                    for key in ("response_pages_count", "response_page_indices", "tables_count", "images_count", "hyperlinks_count", "candidate_size", "usage_info", "band_height", "bands_count"):
                        if key in data["meta"]:
                            page[key] = data["meta"][key]
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
                    "backend": "mistral_ocr",
                    "api_base": self.api_base,
                    "ocr_endpoint": f"{self.api_base}/ocr",
                    "table_format": self.table_format,
                    "extract_header": self.extract_header,
                    "extract_footer": self.extract_footer,
                }
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai texto de PDF usando Mistral OCR")
    parser.add_argument("pdf", help="Caminho do PDF de entrada")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Caminho opcional para salvar uma cópia adicional do JSON final",
    )
    parser.add_argument("--api-key", default=os.getenv("MISTRAL_API_KEY", API_KEY), help="Chave da API da Mistral")
    parser.add_argument(
        "--model",
        default=os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-2512"),
        help="Modelo OCR da Mistral",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
        help="Base URL da API da Mistral",
    )
    parser.add_argument(
        "--table-format",
        default=os.getenv("MISTRAL_OCR_TABLE_FORMAT", ""),
        choices=["", "markdown", "html"],
        help="Formato opcional para tabelas extraídas",
    )
    parser.add_argument(
        "--extract-header",
        action="store_true",
        default=_env_flag("MISTRAL_OCR_EXTRACT_HEADER", False),
        help="Solicita extração de cabeçalho no OCR",
    )
    parser.add_argument(
        "--extract-footer",
        action="store_true",
        default=_env_flag("MISTRAL_OCR_EXTRACT_FOOTER", False),
        help="Solicita extração de rodapé no OCR",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists():
        print(f"PDF não encontrado: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_path = build_extracted_output_path(ROOT_DIR, "mistral", pdf_path)
    extractor = MistralOCRBenchmarkExtractor(
        pdf_path=pdf_path,
        output_path=output_path,
        api_key=args.api_key,
        model=args.model,
        api_base=args.api_base,
        table_format=args.table_format or None,
        extract_header=args.extract_header,
        extract_footer=args.extract_footer,
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
