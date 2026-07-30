"""Build image-only PDFs from page images, and overlay invisible OCR text.

Two jobs live here:

1. `build_normalized_pdf` wraps the rendered (and possibly restored) page
   images into a plain PDF.  Feeding that to the OCR engine instead of the
   original file guarantees two things: the engine sees our cleaned-up
   pixels, and it cannot be misled by a junk text layer, because the
   normalised PDF has no text layer at all.

2. `write_searchable_pdf` puts the recognised text back onto those page
   images as invisible glyphs, giving a PDF that looks exactly like the scan
   but is selectable, searchable and copy-pasteable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .doctypes import FURNITURE, Page
from .util import LOG

#: PDF user-space units per inch.
POINTS_PER_INCH = 72.0


@dataclass
class PageGeometry:
    """Mapping between a page image's pixels and its PDF points."""

    width_px: int
    height_px: int
    width_pt: float
    height_pt: float

    @property
    def scale_px_per_pt(self) -> float:
        return self.width_px / self.width_pt if self.width_pt else 1.0

    def to_points(self, bbox_px: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        s = 1.0 / self.scale_px_per_pt
        return (bbox_px[0] * s, bbox_px[1] * s, bbox_px[2] * s, bbox_px[3] * s)

    def to_pixels(self, bbox_pt: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        s = self.scale_px_per_pt
        return (bbox_pt[0] * s, bbox_pt[1] * s, bbox_pt[2] * s, bbox_pt[3] * s)


def build_normalized_pdf(
    image_paths: list[Path],
    out_path: Path,
    dpis: list[float] | None = None,
) -> list[PageGeometry]:
    """Wrap page images into a PDF, one image per page, no text layer.

    Page size is derived from the image's own DPI so the result has sane
    physical dimensions (a 2000px-wide page at 300 DPI becomes 480pt),
    which keeps downstream layout models in their trained regime.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    geoms: list[PageGeometry] = []

    for i, img_path in enumerate(image_paths):
        dpi = (dpis[i] if dpis and i < len(dpis) else 300.0) or 300.0
        data = Path(img_path).read_bytes()
        pix_doc = pymupdf.open(stream=data, filetype="image")
        rect = pix_doc[0].rect
        w_px, h_px = int(rect.width), int(rect.height)

        w_pt = w_px * POINTS_PER_INCH / dpi
        h_pt = h_px * POINTS_PER_INCH / dpi
        # PDF hard limit is 14400pt a side; clamp absurd cases.
        if max(w_pt, h_pt) > 14000:
            k = 14000 / max(w_pt, h_pt)
            w_pt, h_pt = w_pt * k, h_pt * k

        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(pymupdf.Rect(0, 0, w_pt, h_pt), stream=data)
        geoms.append(PageGeometry(w_px, h_px, w_pt, h_pt))
        pix_doc.close()

    doc.save(out_path, garbage=3, deflate=True)
    doc.close()
    return geoms


# ---------------------------------------------------------------------------
# Searchable PDF
# ---------------------------------------------------------------------------

#: Candidate fonts for the invisible layer, widest Unicode coverage first.
#: The base-14 "helv" only covers Latin-1, which would silently drop Greek,
#: Cyrillic and the typographic marks that turn up constantly in scanned books.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

#: Cheap substitutions for marks a narrow font lacks, chosen so that searching
#: for the plain-ASCII form still finds the passage.
_FALLBACK_CHARS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2010": "-", "\u2011": "-",
    "\u2026": "...", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    "\u2032": "'", "\u2033": '"', "\u00ad": "",
}


def _load_font(preferred: str | None = None) -> pymupdf.Font:
    """Load the widest-coverage font available, or an explicitly chosen one.

    Only the glyphs actually used are embedded, so the cost is a few KB.
    """
    for path in ([preferred] if preferred else []) + _FONT_CANDIDATES:
        if path and Path(path).exists():
            try:
                return pymupdf.Font(fontfile=path)
            except Exception as exc:
                LOG.debug(f"    could not load font {path}: {exc}")
    return pymupdf.Font("helv")


def _encodable(text: str, font: pymupdf.Font) -> str:
    """Map `text` onto glyphs the font actually has.

    Anything with no glyph and no sensible substitute becomes a space rather
    than a box: the invisible layer exists to be searched, and a stray
    replacement character would only pollute copy-paste.
    """
    out = []
    for ch in text:
        if ch in ("\n", "\r", "\t"):
            out.append(" ")
            continue
        try:
            if font.has_glyph(ord(ch)):
                out.append(ch)
                continue
        except Exception:
            pass
        sub = _FALLBACK_CHARS.get(ch)
        if sub is None:
            out.append(" ")
            continue
        out.append(sub if all(_safe_glyph(font, c) for c in sub) else " ")
    return "".join(out)


def _safe_glyph(font: pymupdf.Font, ch: str) -> bool:
    try:
        return bool(font.has_glyph(ord(ch)))
    except Exception:
        return False


#: Rough ratio of line advance to font size for the fonts we use.
_LINE_SPACING = 1.20


def _estimate_fontsize(text: str, width_pt: float, height_pt: float, font: pymupdf.Font) -> float:
    """Font size at which `text` roughly fills a box of the given shape.

    Wrapping `n` lines of a text whose total width at 1pt is `W` into a box
    `w` wide needs `W*s/w` lines, each `s*_LINE_SPACING` tall.  Setting that
    equal to the box height and solving gives the size below; it lands close
    enough that at most one shrink step is needed.
    """
    total_at_1pt = font.text_length(text, fontsize=1.0)
    if total_at_1pt <= 0 or width_pt <= 0 or height_pt <= 0:
        return 1.0
    size = (height_pt * width_pt / (_LINE_SPACING * total_at_1pt)) ** 0.5
    # A single short line must not be blown up to the full box height.
    single_line = width_pt / total_at_1pt
    return max(0.5, min(size, single_line, height_pt / _LINE_SPACING, 300.0))


def _place_text(
    writer: pymupdf.TextWriter,
    page_rect: pymupdf.Rect,
    rect: pymupdf.Rect,
    text: str,
    font: pymupdf.Font,
) -> bool:
    """Lay `text` out inside `rect`, shrinking until all of it fits.

    Fitting is probed on a scratch writer first, because `fill_textbox`
    commits whatever fitted and returns the rest -- writing straight to the
    real writer would drop the overflow from the searchable layer.
    """
    size = _estimate_fontsize(text, rect.width, rect.height, font)
    for _ in range(7):
        if size < 0.5:
            break
        probe = pymupdf.TextWriter(page_rect)
        try:
            leftover = probe.fill_textbox(rect, text, font=font, fontsize=size)
        except Exception:
            leftover = ["x"]
        if not leftover:
            writer.fill_textbox(rect, text, font=font, fontsize=size)
            return True
        size *= 0.82

    # Nothing fits (a box far too small for its text). Keep the words in the
    # file anyway at a minimum size so search still finds the page.
    try:
        writer.fill_textbox(rect, text, font=font, fontsize=0.5)
        return True
    except Exception:
        return False


def write_searchable_pdf(
    pages: list[Page],
    image_paths: list[Path],
    geoms: list[PageGeometry],
    out_path: Path,
    include_furniture: bool = True,
    jpeg_quality: int = 80,
    font_file: str | None = None,
) -> Path:
    """Rebuild the scan as a PDF with an invisible, positioned text layer.

    Text is drawn in render mode 3 (invisible) using Helvetica, sized per
    line so that a text selection lands roughly on the ink it belongs to.
    Word-level alignment is only as good as the boxes the engine gave us:
    with line boxes, selection tracks the page closely; with block boxes it
    is approximate, but search and copy-paste are exact either way.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    font = _load_font(font_file)
    n_lines = 0

    for page in pages:
        i = page.index
        if i >= len(image_paths) or i >= len(geoms):
            continue
        geom = geoms[i]
        data = Path(image_paths[i]).read_bytes()
        pdf_page = doc.new_page(width=geom.width_pt, height=geom.height_pt)
        pdf_page.insert_image(pymupdf.Rect(0, 0, geom.width_pt, geom.height_pt), stream=data)

        writer = pymupdf.TextWriter(pdf_page.rect)
        wrote = False

        blocks = page.blocks if include_furniture else [
            b for b in page.blocks if b.type not in FURNITURE
        ]
        for block in blocks:
            units = block.lines if block.lines else (
                [type("L", (), {"text": block.text, "bbox": block.bbox})()] if block.text.strip() else []
            )
            for unit in units:
                text = _encodable(" ".join((unit.text or "").split()), font)
                if not text.strip():
                    continue
                x0, y0, x1, y1 = geom.to_points(tuple(unit.bbox))
                if x1 - x0 < 2 or y1 - y0 < 1:
                    continue
                rect = pymupdf.Rect(x0, y0, x1, y1) & pdf_page.rect
                if rect.is_empty:
                    continue
                # The engine's "lines" are often whole column segments rather
                # than visual lines, so the text is wrapped into the box
                # instead of stretched across it as one run.
                if _place_text(writer, pdf_page.rect, rect, text, font):
                    wrote = True
                    n_lines += 1

        if wrote:
            # render_mode=3 -> "neither fill nor stroke": present but invisible.
            writer.write_text(pdf_page, render_mode=3)

    doc.set_metadata({"producer": "OCR_man", "creator": "OCR_man searchable PDF"})
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()
    LOG.debug(f"    searchable PDF: {n_lines} text runs placed")
    return out_path
