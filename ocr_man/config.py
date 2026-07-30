"""Configuration loading.

Defaults live here in Python so the pipeline always has a complete, valid
configuration; `config.yaml` only carries the user's overrides and may be
partial or absent.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "paths": {
        "input": "INPUT",
        "output": "OUTPUT",
        "work": "WORK",
    },
    "render": {
        # Target long-edge size of the page image fed to the OCR model.
        # ~1800-2200px is the sweet spot for current document VLMs: below that
        # small print degrades, above it you pay for tokens without gaining
        # accuracy.  Never upsampled beyond the scan's own resolution.
        "target_long_edge_px": 2000,
        "min_dpi": 150,
        "max_dpi": 400,
        # Re-OCR from images even when the PDF already has a text layer.
        # Internet Archive scans routinely ship a garbage text layer; trusting
        # it produces nonsense, so this defaults to on.
        "force_ocr": True,
        # If a PDF text layer looks genuinely good (born-digital), skip OCR.
        # Only consulted when force_ocr is False.
        "text_layer_quality_threshold": 0.80,
    },
    "preprocess": {
        "enabled": True,
        # Auto = only clean up pages whose quality score says they need it.
        "mode": "auto",              # auto | always | never
        "deskew": True,
        "max_deskew_deg": 8.0,
        "illumination_correction": True,
        "clahe": True,
        "despeckle": True,
        "crop_scan_border": True,
        # Grayscale is what document VLMs are happiest with; hard binarisation
        # destroys anti-aliased glyph edges and hurts more than it helps.
        "binarize": False,
    },
    "ocr": {
        # Ordered preference list. The first engine that reports itself
        # installed and working is used; the rest become fallbacks.
        "engines": ["mineru", "dots", "paddle_vl", "rapidocr"],
        "engine": "auto",            # "auto" honours the list above
        "language": "en",
        "batch_size": 4,
        "device": "cuda",
        # Re-run a page with the next engine when the first returns
        # suspiciously little / low-confidence text.
        "fallback_on_low_yield": True,
        "min_chars_per_page": 60,
        "formula": True,
        "table": True,
    },
    "layout": {
        # Trust the engine's own reading order when it supplies one.
        "prefer_engine_order": True,
        # Column-aware XY-cut used when it does not (or when its order looks wrong).
        "max_columns": 4,
        # Minimum gutter width as a fraction of page width. Real book gutters
        # are often only 2-3% wide, and line detectors pad their boxes, so
        # anything stricter than this misses genuine two-column pages.
        "column_gap_ratio": 0.02,
        "validate_engine_order": True,
    },
    "assemble": {
        "strip_running_heads": True,
        "dehyphenate": True,
        "stitch_paragraphs": True,   # across column and page breaks
        "detect_headings": True,
        "keep_figures": True,
        "min_figure_area_ratio": 0.01,
    },
    "llm": {
        "enabled": True,
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "qwen3.6:35b",
        "fallback_models": ["mistral-small3.2:24b", "qwen2.5:14b"],
        "chunk_chars": 4000,
        "overlap_chars": 250,
        "temperature": 0.0,
        "num_ctx": 8192,
        "timeout_s": 600,
        # Guard rails: a corrected chunk is rejected (original kept) when the
        # model strayed too far.  This is what stops a chatty model from
        # silently rewriting or inventing content.
        "min_similarity": 0.72,
        "min_length_ratio": 0.80,
        "max_length_ratio": 1.25,
        "max_retries": 2,
    },
    "output": {
        "markdown": True,
        "epub": True,
        "searchable_pdf": True,
        "page_markers": True,        # emit <!-- page N --> anchors in the .md
        "front_matter": True,        # YAML front matter in the .md
        "epub_include_figures": True,
        "searchable_pdf_jpeg_quality": 80,
        # Font used for the invisible text layer. Empty = pick the widest
        # Unicode coverage available on this machine (Arial / DejaVu / helv).
        "searchable_pdf_font": "",
    },
    "run": {
        "resume": True,              # reuse cached per-page results in WORK/
        "keep_work_dir": True,
        "max_pages": 0,              # 0 = all; useful for a quick test run
        "workers": 4,                # CPU workers for render/preprocess
        "verbose": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Dotted-path access over the merged config dict."""

    def __init__(self, data: dict[str, Any], root: Path = ROOT):
        self.data = data
        self.root = root

    def __getitem__(self, path: str) -> Any:
        cur: Any = self.data
        for part in path.split("."):
            cur = cur[part]
        return cur

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        cur = self.data
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    # -- resolved directories ------------------------------------------------
    def _dir(self, key: str) -> Path:
        p = Path(self[f"paths.{key}"])
        if not p.is_absolute():
            p = self.root / p
        return p

    @property
    def input_dir(self) -> Path:
        return self._dir("input")

    @property
    def output_dir(self) -> Path:
        return self._dir("output")

    @property
    def work_dir(self) -> Path:
        return self._dir("work")


def load_config(path: Path | None = None, root: Path = ROOT) -> Config:
    """Load config.yaml (if present) on top of DEFAULTS."""
    path = path or (root / "config.yaml")
    override: dict[str, Any] = {}
    if path.exists():
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - guarded at setup time
            raise SystemExit(
                f"config.yaml exists but PyYAML is not installed ({exc}).\n"
                "Run SETUP.py, or delete config.yaml to use built-in defaults."
            ) from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded:
            if not isinstance(loaded, dict):
                raise SystemExit(f"{path} must contain a YAML mapping at the top level.")
            override = loaded

    cfg = Config(_deep_merge(DEFAULTS, override), root=root)
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: Config) -> None:
    """OCRMAN_<DOTTED_PATH> env vars win over the file, for scripted runs."""
    prefix = "OCRMAN_"
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().replace("__", ".")
        if cfg.get(path, "\0__missing__") == "\0__missing__":
            continue
        cur = cfg[path]
        try:
            if isinstance(cur, bool):
                val: Any = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            elif isinstance(cur, list):
                val = [x.strip() for x in raw.split(",") if x.strip()]
            else:
                val = raw
        except ValueError:
            continue
        cfg.set(path, val)
