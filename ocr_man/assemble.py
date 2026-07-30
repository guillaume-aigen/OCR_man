"""Turn per-page OCR results into one continuous document.

This is where a pile of correctly-recognised page fragments becomes readable
prose.  Four things matter and all four are page-crossing, which is why none
of them can be done inside an OCR engine:

  * running heads and folios repeat on every page and must not land in the text
  * words broken across a line with a hyphen must be rejoined
  * a paragraph that runs off the bottom of the left column continues at the
    top of the right one, and off the bottom of the page onto the next
  * headings need a level so the EPUB gets a usable table of contents
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .doctypes import (
    CAPTION, FIGURE, FOOTER, FOOTNOTE, FORMULA, HEADER,
    HEADING, LIST, PAGE_NUMBER, TABLE, TEXT, Block, Element, Page,
)
from .util import LOG

# Characters that end a sentence (so the next block starts a new paragraph).
SENTENCE_END = tuple(".!?:…”’\"')]}")
HYPHENS = "-‐‑­–"

ROMAN = re.compile(r"^[ivxlcdm]{1,7}$", re.I)


# ---------------------------------------------------------------------------
# Running heads / feet / folios
# ---------------------------------------------------------------------------

def _normalise_for_repeat(text: str) -> str:
    """Strip digits and punctuation so 'CHAPTER 4 | 61' matches 'CHAPTER 4 | 62'."""
    t = re.sub(r"\d+", "#", text.lower())
    t = re.sub(r"[^\w#]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def looks_like_folio(text: str) -> bool:
    t = text.strip().strip("[](){}.,-—– ")
    if not t or len(t) > 12:
        return False
    if t.isdigit():
        return True
    if ROMAN.match(t):
        return True
    # 'Page 61', '- 61 -'
    return bool(re.fullmatch(r"(page\s*)?[\divxlcdm]+", t, re.I))


def mark_furniture(pages: list[Page], band: float = 0.09, min_repeat: float = 0.30) -> dict:
    """Label repeating top/bottom blocks as header / footer / page number.

    A line only counts as furniture if the *same* normalised text shows up in
    the same band on a good fraction of pages. That keeps a genuine one-off
    heading at the top of a chapter from being deleted.
    """
    if not pages:
        return {}

    top_counter: Counter[str] = Counter()
    bot_counter: Counter[str] = Counter()
    candidates: list[tuple[Page, Block, str, str]] = []

    for page in pages:
        h = page.height or 1
        for b in page.blocks:
            if b.type in {FIGURE, TABLE}:
                continue
            text = b.text.strip()
            if not text or len(text) > 120 or b.height > h * 0.06:
                continue
            key = _normalise_for_repeat(text)
            if not key:
                continue
            if b.y1 <= h * band:
                top_counter[key] += 1
                candidates.append((page, b, key, "top"))
            elif b.y0 >= h * (1 - band):
                bot_counter[key] += 1
                candidates.append((page, b, key, "bottom"))

    n = len(pages)
    need = max(3, int(n * min_repeat))
    stats = {"header": 0, "footer": 0, "page_number": 0}

    for page, b, key, where in candidates:
        counter = top_counter if where == "top" else bot_counter
        text = b.text.strip()
        if looks_like_folio(text):
            b.type = PAGE_NUMBER
            stats["page_number"] += 1
        elif counter[key] >= need:
            b.type = HEADER if where == "top" else FOOTER
            stats["header" if where == "top" else "footer"] += 1

    if any(stats.values()):
        LOG.debug(
            f"    page furniture: {stats['header']} heads, {stats['footer']} feet, "
            f"{stats['page_number']} folios"
        )
    return stats


def extract_page_numbers(pages: list[Page]) -> dict[int, str]:
    """Map page index -> printed folio, for page anchors in the Markdown."""
    out: dict[int, str] = {}
    for page in pages:
        for b in page.blocks:
            if b.type == PAGE_NUMBER:
                t = b.text.strip().strip("[](){}.,-—– ")
                m = re.search(r"[\divxlcdmIVXLCDM]+", t)
                if m:
                    out[page.index] = m.group(0)
                    break
    return out


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}


def normalise_text(text: str) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    # Smart quotes and dashes are fine to keep, but normalise the weird ones.
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by a hyphen.

    Only joins when the next line starts lowercase: an uppercase continuation
    is far more likely to be a real compound ('Anglo-\\nAmerican') than a
    typesetting break.
    """
    def repl(m: re.Match) -> str:
        return m.group(1) + m.group(2)

    # word- \n lowercase-continuation
    text = re.sub(rf"([A-Za-zÀ-ɏ]{{2,}})[{HYPHENS}]\s*\n\s*([a-zà-ɏ])", repl, text)
    # Same, but the engine already flattened the line break to a space.
    text = re.sub(rf"([A-Za-zÀ-ɏ]{{2,}})­\s*([a-zà-ɏ])", repl, text)
    return text


def join_lines(text: str) -> str:
    """Collapse the hard line breaks inside a paragraph into spaces."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

def _heading_level(block: Block, page: Page, seen: list[int]) -> int:
    if block.level:
        return max(1, min(6, block.level))
    text = block.text.strip()
    words = len(text.split())
    rel_h = block.height / max(1.0, page.height)
    upper = sum(1 for ch in text if ch.isupper())
    letters = sum(1 for ch in text if ch.isalpha()) or 1
    shouty = upper / letters > 0.6

    if rel_h > 0.045 or (shouty and words <= 8):
        return 1
    if words <= 12:
        return 2
    return 3


def detect_headings(pages: list[Page]) -> int:
    """Promote short, isolated, emphatic blocks to headings."""
    n = 0
    for page in pages:
        blocks = page.content_blocks()
        for i, b in enumerate(blocks):
            if b.type != TEXT:
                continue
            text = b.text.strip()
            if not text or len(text) > 90 or text.endswith((".", ",", ";")):
                continue
            words = text.split()
            if len(words) > 12:
                continue
            letters = [ch for ch in text if ch.isalpha()]
            if not letters:
                continue
            upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
            title_case = all(w[0].isupper() for w in words if w[0].isalpha())
            if upper_ratio > 0.75 or (title_case and len(words) <= 8):
                b.type = HEADING
                b.level = _heading_level(b, page, [])
                n += 1
    return n


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def export_figures(
    pages: list[Page],
    out_dir: Path,
    min_area_ratio: float = 0.01,
) -> int:
    """Crop figure regions out of the page images into `out_dir`."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    cache: dict[str, Image.Image] = {}
    for page in pages:
        for b in page.blocks:
            if b.type != FIGURE or b.image_path:
                continue
            area = (b.width * b.height) / max(1.0, page.width * page.height)
            if area < min_area_ratio:
                continue
            if not page.image_path or not Path(page.image_path).exists():
                continue
            try:
                img = cache.get(page.image_path)
                if img is None:
                    img = Image.open(page.image_path)
                    img.load()
                    cache[page.image_path] = img
                box = (
                    max(0, int(b.x0)), max(0, int(b.y0)),
                    min(img.width, int(b.x1)), min(img.height, int(b.y1)),
                )
                if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                    continue
                crop = img.crop(box)
                name = f"p{page.index:04d}_{count:02d}.jpg"
                crop.convert("RGB").save(out_dir / name, "JPEG", quality=88, optimize=True)
                b.image_path = f"images/{name}"
                count += 1
            except Exception as exc:
                LOG.debug(f"    could not export figure on page {page.index}: {exc}")
        cache.clear()
    return count


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class AssemblyStats:
    paragraphs: int = 0
    headings: int = 0
    tables: int = 0
    figures: int = 0
    formulas: int = 0
    stitched: int = 0
    dehyphenated: int = 0
    furniture_removed: int = 0


#: A block starting like "3." or "- " or "(b)" begins a list item, not a
#: continuation, even if the previous block ended without punctuation.
_LIST_START = re.compile(r"^\s*(?:[-*•·–—]|\(?[0-9ivxlcIVXLC]{1,4}[.)]|[A-Za-z][.)])\s+")


def _is_continuation(prev_text: str, next_text: str) -> bool:
    """Does `next_text` continue the paragraph that `prev_text` started?

    The governing signal is the *end* of the previous block, not the start of
    the next one: a genuine new paragraph almost always follows a completed
    sentence.  Requiring a lowercase continuation as well would miss the case
    that shows up constantly in reference works -- a small-caps cross-
    reference straddling the break, as in "...set in an undated NEAR" /
    "FUTURE, we have been highly selective".
    """
    tail = (prev_text or "").rstrip()
    head = (next_text or "").lstrip()
    if not tail or not head:
        return False
    if tail.endswith(SENTENCE_END):
        return False
    # Very short trailing blocks are usually a misfiled heading or caption;
    # merging them into the next paragraph does more harm than good.
    if len(tail) < 40:
        return False
    if not (tail[-1].isalnum() or tail[-1] in HYPHENS + ",;"):
        return False
    if _LIST_START.match(head):
        return False
    return True


def assemble(
    pages: list[Page],
    cfg: Config,
    out_dir: Path,
) -> tuple[list[Element], AssemblyStats]:
    """Flatten ordered pages into stitched `Element`s."""
    st = AssemblyStats()
    asm = cfg.get("assemble", {})

    if asm.get("strip_running_heads", True):
        stats = mark_furniture(pages)
        st.furniture_removed = sum(stats.values())

    if asm.get("detect_headings", True):
        st.headings += detect_headings(pages)

    if asm.get("keep_figures", True) and cfg.get("output.epub_include_figures", True):
        st.figures = export_figures(
            pages, out_dir / "images", float(asm.get("min_figure_area_ratio", 0.01))
        )

    elements: list[Element] = []
    dehyph = asm.get("dehyphenate", True)
    stitch = asm.get("stitch_paragraphs", True)

    for page in pages:
        elements.append(Element(type="page_break", pages=[page.index]))
        for b in page.content_blocks():
            text = normalise_text(b.text)

            if b.type == TABLE:
                elements.append(Element(type=TABLE, text=text, html=b.html, pages=[page.index]))
                st.tables += 1
                continue
            if b.type == FIGURE:
                if b.image_path:
                    elements.append(
                        Element(type=FIGURE, text=text, image_path=b.image_path, pages=[page.index])
                    )
                continue
            if b.type == FORMULA:
                elements.append(Element(type=FORMULA, text=text, html=b.html, pages=[page.index]))
                st.formulas += 1
                continue
            if not text:
                continue
            if b.type == HEADING:
                elements.append(
                    Element(type=HEADING, text=join_lines(text),
                            level=b.level or 2, pages=[page.index])
                )
                st.headings += 1
                continue
            if b.type == CAPTION:
                elements.append(Element(type=CAPTION, text=join_lines(text), pages=[page.index]))
                continue
            if b.type == FOOTNOTE:
                elements.append(Element(type=FOOTNOTE, text=join_lines(text), pages=[page.index]))
                continue

            # Flowing prose.
            if dehyph:
                before = text
                text = dehyphenate(text)
                if text != before:
                    st.dehyphenated += 1
            body = join_lines(text)
            if not body:
                continue
            elements.append(
                Element(type=LIST if b.type == LIST else TEXT, text=body, pages=[page.index])
            )
            st.paragraphs += 1

    if stitch:
        elements, joined = stitch_elements(elements, dehyph)
        st.stitched = joined

    return elements, st


def stitch_elements(elements: list[Element], dehyph: bool) -> tuple[list[Element], int]:
    """Merge paragraphs split across a column or page boundary."""
    out: list[Element] = []
    joined = 0
    for el in elements:
        if el.type != TEXT or not out:
            out.append(el)
            continue

        # Look back past page markers to the previous prose element.
        j = len(out) - 1
        crossed_break = False
        while j >= 0 and out[j].type == "page_break":
            crossed_break = True
            j -= 1
        if j < 0 or out[j].type != TEXT:
            out.append(el)
            continue

        prev = out[j]
        if not _is_continuation(prev.text, el.text):
            out.append(el)
            continue

        merged = prev.text.rstrip()
        head = el.text.lstrip()
        if dehyph and merged and merged[-1] in HYPHENS and head[:1].islower():
            merged = merged[:-1] + head
        else:
            merged = merged + " " + head
        prev.text = merged
        prev.pages = sorted(set(prev.pages + el.pages))
        joined += 1
        # Any page markers between the two halves stay where they are, so the
        # page anchors in the Markdown keep pointing at the right scan.
    return out, joined


def assemble_native(
    sources: list[tuple[int, str, str | None]],
    cfg: Config,
) -> tuple[list[Element], AssemblyStats]:
    """Assembly path for born-digital sources we did not OCR.

    `sources` is (page index, plain text, chapter HTML or None).  When HTML is
    available it is parsed for structure; a PDF text layer has none, so its
    paragraphs are recovered from blank lines instead.
    """
    from .htmltext import html_to_elements

    st = AssemblyStats()
    elements: list[Element] = []
    for idx, raw, html in sources:
        elements.append(Element(type="page_break", pages=[idx]))

        if html:
            for el in html_to_elements(html):
                el.text = join_lines(normalise_text(el.text))
                if not el.text and el.type != FIGURE:
                    continue
                el.pages = [idx]
                elements.append(el)
                if el.type == HEADING:
                    st.headings += 1
                elif el.type == FIGURE:
                    st.figures += 1
                else:
                    st.paragraphs += 1
            continue

        for para in re.split(r"\n\s*\n", raw or ""):
            body = join_lines(normalise_text(dehyphenate(para)))
            if not body:
                continue
            elements.append(Element(type=TEXT, text=body, pages=[idx]))
            st.paragraphs += 1

    if cfg.get("assemble.stitch_paragraphs", True):
        elements, st.stitched = stitch_elements(elements, cfg.get("assemble.dehyphenate", True))
    return elements, st
