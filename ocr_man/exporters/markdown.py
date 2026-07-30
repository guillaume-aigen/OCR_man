"""Markdown writer, aimed squarely at later LLM consumption.

Priorities differ from the EPUB: nothing decorative, no HTML wrappers that
waste tokens, headings that make the structure explicit, and page anchors so
a passage found in the Markdown can be traced back to the scan it came from.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

from ..config import Config
from ..doctypes import (
    CAPTION, DocumentMeta, Element, FIGURE, FOOTNOTE, FORMULA, HEADING, LIST,
    TABLE, TEXT,
)


def _yaml_escape(value: str) -> str:
    value = (value or "").replace("\\", "\\\\").replace('"', '\\"')
    value = value.replace("\n", " ").strip()
    return f'"{value}"'


def _front_matter(meta: DocumentMeta, stats: dict) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d")
    lines = [
        "---",
        f"title: {_yaml_escape(meta.title)}",
    ]
    if meta.author:
        lines.append(f"author: {_yaml_escape(meta.author)}")
    lines += [
        f"language: {_yaml_escape(meta.language or 'en')}",
        f"source_file: {_yaml_escape(meta.source_file)}",
        f"ocr_date: {now}",
    ]
    for key in ("pages", "engine", "ocr_engine", "llm_cleanup", "words"):
        if key in stats and stats[key] not in (None, ""):
            lines.append(f"{key}: {_yaml_escape(str(stats[key]))}")
    lines.append("---")
    return "\n".join(lines)


def _html_table_to_markdown(html: str) -> str | None:
    """Convert a simple HTML table to a pipe table; give up on complex ones."""
    if not html or "<table" not in html.lower():
        return None
    if re.search(r"\b(rowspan|colspan)\s*=", html, re.I):
        return None
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    if not rows:
        return None
    table: list[list[str]] = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S | re.I)
        cleaned = []
        for c in cells:
            text = re.sub(r"<[^>]+>", " ", c)
            text = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
            cleaned.append(text)
        if cleaned:
            table.append(cleaned)
    if not table or len(table) < 2:
        return None
    width = max(len(r) for r in table)
    if width < 2:
        return None
    table = [r + [""] * (width - len(r)) for r in table]
    out = ["| " + " | ".join(table[0]) + " |",
           "|" + "|".join([" --- "] * width) + "|"]
    for r in table[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_markdown(
    elements: list[Element],
    meta: DocumentMeta,
    out_path: Path,
    cfg: Config,
    page_numbers: dict[int, str] | None = None,
    stats: dict | None = None,
) -> Path:
    page_numbers = page_numbers or {}
    stats = dict(stats or {})
    parts: list[str] = []

    if cfg.get("output.front_matter", True):
        words = sum(len(e.text.split()) for e in elements if e.text)
        stats.setdefault("words", words)
        parts.append(_front_matter(meta, stats))

    if meta.title:
        parts.append(f"# {meta.title}")
        if meta.author:
            parts.append(f"*{meta.author}*")

    emit_pages = cfg.get("output.page_markers", True)

    for el in elements:
        if el.type == "page_break":
            if emit_pages and el.pages:
                idx = el.pages[0]
                folio = page_numbers.get(idx)
                label = f"page {folio}" if folio else f"scan page {idx + 1}"
                parts.append(f"<!-- {label} -->")
            continue

        text = (el.text or "").strip()

        if el.type == HEADING:
            if not text:
                continue
            level = max(1, min(6, el.level or 2))
            # The document title already occupies H1.
            if meta.title and level == 1:
                level = 2
            parts.append(f"{'#' * level} {text}")
        elif el.type == TABLE:
            md = _html_table_to_markdown(el.html or "")
            if md:
                parts.append(md)
            elif el.html:
                parts.append(el.html.strip())
            elif text:
                parts.append(text)
        elif el.type == FIGURE:
            alt = re.sub(r"\s+", " ", text)[:200] if text else "figure"
            if el.image_path:
                parts.append(f"![{alt}]({el.image_path})")
            elif text:
                parts.append(f"> **Figure.** {text}")
        elif el.type == FORMULA:
            body = (el.html or text).strip()
            if body:
                if body.startswith("$$"):
                    parts.append(body)
                else:
                    parts.append(f"$$\n{body}\n$$")
        elif el.type == CAPTION:
            if text:
                parts.append(f"*{text}*")
        elif el.type == FOOTNOTE:
            if text:
                parts.append(f"> {text}")
        elif el.type == LIST:
            if text:
                parts.append(_format_list(text))
        elif el.type == TEXT:
            if text:
                parts.append(text)

    body = "\n\n".join(p for p in parts if p is not None and p != "")
    body = re.sub(r"\n{4,}", "\n\n\n", body).strip() + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path


_BULLET = re.compile(r"^\s*(?:[•·–—*]|\d+[.)])\s+")


def _format_list(text: str) -> str:
    """Normalise an OCR'd list block into Markdown list syntax."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        single = lines[0] if lines else text.strip()
        return f"- {_BULLET.sub('', single)}" if _BULLET.match(single) else single
    out = []
    for ln in lines:
        m = re.match(r"^\s*(\d+)[.)]\s+(.*)$", ln)
        if m:
            out.append(f"{m.group(1)}. {m.group(2)}")
        else:
            out.append(f"- {_BULLET.sub('', ln)}")
    return "\n".join(out)
