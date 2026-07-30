"""Turn whatever is in INPUT/ into page images the OCR engine can read.

Handles:
  * image-only PDFs (the main case) -- rendered at a DPI matched to the
    embedded scan's own resolution so we never upsample noise
  * born-digital PDFs -- detected and passed through without OCR
  * EPUBs, both image-scan EPUBs and real-text EPUBs
  * loose page images and multi-page TIFFs

A PDF that already carries a text layer is *not* trusted by default.  Scans
from mass-digitisation projects routinely ship an OCR layer that is one word
per line, mis-hyphenated and full of substitution errors; re-reading the
pixels beats trying to repair it.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pymupdf

from .config import Config
from .doctypes import DocumentMeta
from .util import LOG, Progress, human_size, slugify

PDF_EXT = {".pdf"}
EPUB_EXT = {".epub"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jp2", ".pnm", ".ppm"}
SUPPORTED_EXT = PDF_EXT | EPUB_EXT | IMAGE_EXT

# An image covering at least this fraction of the page means "this page is a scan".
SCAN_COVERAGE = 0.55


@dataclass
class SourcePage:
    """One page of the source, rendered and ready for OCR."""

    index: int
    image_path: Path
    width: int
    height: int
    source_page: int
    #: Populated only for born-digital pages we chose not to OCR.
    native_text: str | None = None
    #: Raw chapter XHTML, when the source was a text EPUB. Preferred over
    #: `native_text` because it still carries headings and paragraph breaks.
    native_html: str | None = None
    #: DPI the image was rendered at, for pixel <-> PDF-point mapping.
    dpi: float = 300.0


@dataclass
class IngestResult:
    source: Path
    stem: str
    meta: DocumentMeta
    pages: list[SourcePage] = field(default_factory=list)
    #: "ocr" (run the OCR engine) or "native" (text layer was already good)
    mode: str = "ocr"
    #: For the searchable-PDF exporter: the original PDF, if there was one.
    source_pdf: Path | None = None
    notes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    found = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT and not p.name.startswith(("~", "."))
    ]
    # Loose images in the same folder are treated as one document per folder,
    # so drop the individual files and keep the folders (handled by caller).
    return found


def group_inputs(paths: list[Path], input_dir: Path) -> list[list[Path]]:
    """Group loose page images sitting in the same subfolder into one document."""
    docs: list[list[Path]] = []
    image_groups: dict[Path, list[Path]] = {}
    for p in paths:
        if p.suffix.lower() in IMAGE_EXT:
            image_groups.setdefault(p.parent, []).append(p)
        else:
            docs.append([p])
    for folder, imgs in image_groups.items():
        imgs.sort(key=_natural_key)
        if folder == input_dir and len(imgs) == 1:
            docs.append(imgs)          # a single loose image = its own document
        elif folder == input_dir:
            # Several loose images directly in INPUT/: treat each separately,
            # since we cannot know they belong together.
            docs.extend([[i] for i in imgs])
        else:
            docs.append(imgs)          # a subfolder of images = one document
    return docs


def _natural_key(p: Path) -> tuple:
    parts = re.split(r"(\d+)", p.name.lower())
    return tuple(int(x) if x.isdigit() else x for x in parts)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _page_native_dpi(doc: pymupdf.Document, pno: int) -> tuple[float, float]:
    """Effective DPI of the largest embedded image, and its page coverage."""
    page = doc[pno]
    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        return 300.0, 0.0
    best_dpi, best_cov = 0.0, 0.0
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []
    for info in infos:
        w, h = info.get("width", 0), info.get("height", 0)
        bbox = info.get("bbox")
        if not w or not h or not bbox:
            continue
        bw = max(1e-6, bbox[2] - bbox[0])
        bh = max(1e-6, bbox[3] - bbox[1])
        dpi = max(w / (bw / 72.0), h / (bh / 72.0))
        cov = (bw * bh) / (rect.width * rect.height)
        if cov > best_cov:
            best_cov, best_dpi = cov, dpi
    return (best_dpi or 300.0), best_cov


def _text_layer_quality(page: pymupdf.Page) -> float:
    """0..1 estimate of whether an existing text layer is worth keeping.

    Penalises the signature of a bad mass-digitisation OCR layer: very short
    lines (one word per line), few multi-letter words, high symbol ratio.
    """
    text = page.get_text().strip()
    if len(text) < 40:
        return 0.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    words = re.findall(r"[A-Za-zÀ-ɏ]{2,}", text)
    if not words:
        return 0.0

    avg_line_words = sum(len(ln.split()) for ln in lines) / len(lines)
    # Real prose lines carry ~8-14 words; the broken layers carry 1-2.
    line_score = min(1.0, avg_line_words / 6.0)

    alpha = sum(ch.isalpha() or ch.isspace() for ch in text)
    alpha_score = alpha / len(text)

    long_words = sum(1 for w in words if len(w) >= 4)
    word_score = min(1.0, (long_words / len(words)) / 0.45)

    return round(0.45 * line_score + 0.30 * alpha_score + 0.25 * word_score, 3)


def probe_pdf(path: Path, sample: int = 12) -> dict[str, Any]:
    """Decide whether a PDF is a scan or born-digital, and at what DPI to render."""
    doc = pymupdf.open(path)
    n = doc.page_count
    idxs = sorted({int(i * (n - 1) / max(1, min(sample, n) - 1)) for i in range(min(sample, n))})
    dpis, covs, quals = [], [], []
    for i in idxs:
        d, cov = _page_native_dpi(doc, i)
        dpis.append(d)
        covs.append(cov)
        quals.append(_text_layer_quality(doc[i]))
    meta = doc.metadata or {}
    doc.close()

    covs_sorted = sorted(covs)
    median_cov = covs_sorted[len(covs_sorted) // 2] if covs_sorted else 0.0
    dpis_sorted = sorted(d for d in dpis if d > 0)
    median_dpi = dpis_sorted[len(dpis_sorted) // 2] if dpis_sorted else 300.0
    quals_sorted = sorted(quals)
    median_qual = quals_sorted[len(quals_sorted) // 2] if quals_sorted else 0.0

    return {
        "page_count": n,
        "is_scan": median_cov >= SCAN_COVERAGE,
        "image_coverage": round(median_cov, 3),
        "native_dpi": round(median_dpi, 1),
        "text_layer_quality": round(median_qual, 3),
        "metadata": meta,
    }


def _choose_dpi(cfg: Config, page_rect: pymupdf.Rect, native_dpi: float) -> float:
    long_pt = max(page_rect.width, page_rect.height)
    if long_pt <= 0:
        return 300.0
    want = cfg["render.target_long_edge_px"] / (long_pt / 72.0)
    # Never render above the scan's own resolution: upsampling adds no
    # information and costs the VLM tokens.
    dpi = min(want, native_dpi if native_dpi > 0 else want)
    return float(max(cfg["render.min_dpi"], min(cfg["render.max_dpi"], dpi)))


def _render_pdf_chunk(args: tuple) -> list[dict]:
    """Process-pool worker: render a contiguous range of PDF pages to disk."""
    (src, out_dir, page_indices, target_long, min_dpi, max_dpi, jpeg_quality) = args
    import pymupdf as _pm
    from PIL import Image

    doc = _pm.open(src)
    results = []
    for pno in page_indices:
        page = doc[pno]
        rect = page.rect
        native_dpi, _ = _page_native_dpi(doc, pno)
        long_pt = max(rect.width, rect.height) or 612.0
        want = target_long / (long_pt / 72.0)
        # PyMuPDF wants an integer DPI.
        dpi = int(round(max(min_dpi, min(max_dpi, min(want, native_dpi or want)))))
        pix = page.get_pixmap(dpi=dpi, colorspace=_pm.csRGB)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        # Scans are effectively greyscale; storing them that way cuts the work
        # cache by ~3x with no effect on recognition.
        if _is_greyscale(img):
            img = img.convert("L")
        out = Path(out_dir) / f"page_{pno:05d}.jpg"
        img.save(out, "JPEG", quality=jpeg_quality, optimize=True, subsampling=0)
        results.append(
            {"source_page": pno, "path": str(out), "width": img.width, "height": img.height, "dpi": dpi}
        )
    doc.close()
    return results


def _is_greyscale(img, sample_px: int = 4000) -> bool:
    """True when R==G==B (within tolerance) across a random-ish sample."""
    from PIL import Image

    small = img.resize((min(64, img.width), min(64, img.height)), Image.BILINEAR)
    px = list(small.getdata())[:sample_px]
    for r, g, b in px:
        if abs(r - g) > 12 or abs(g - b) > 12 or abs(r - b) > 12:
            return False
    return True


def ingest_pdf(cfg: Config, src: Path, work: Path) -> IngestResult:
    info = probe_pdf(src)
    meta_raw = info["metadata"]
    meta = DocumentMeta(
        title=(meta_raw.get("title") or src.stem).strip(),
        author=(meta_raw.get("author") or "").strip(),
        language=cfg.get("ocr.language", "en"),
        source_file=src.name,
    )
    force = bool(cfg["render.force_ocr"])
    quality = info["text_layer_quality"]
    threshold = float(cfg["render.text_layer_quality_threshold"])

    LOG.info(
        f"  {src.name}: {info['page_count']} pages, "
        f"{'image scan' if info['is_scan'] else 'born-digital'}, "
        f"native {info['native_dpi']:.0f} DPI, text-layer quality {quality:.2f}"
    )

    if not force and not info["is_scan"] and quality >= threshold:
        LOG.info("    text layer looks good -> extracting directly, no OCR")
        return _ingest_pdf_native(cfg, src, meta, info)

    if info["is_scan"] and quality > 0.0:
        LOG.info(
            f"    ignoring the embedded text layer (quality {quality:.2f}) and re-reading the pixels"
        )

    img_dir = work / "pages"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(src)
    n = doc.page_count
    doc.close()
    max_pages = int(cfg.get("run.max_pages", 0) or 0)
    page_list = list(range(n if max_pages <= 0 else min(n, max_pages)))

    workers = max(1, int(cfg.get("run.workers", 4)))
    chunks = _chunk(page_list, max(1, len(page_list) // (workers * 2) or 1))
    args = [
        (
            str(src), str(img_dir), ch,
            int(cfg["render.target_long_edge_px"]),
            int(cfg["render.min_dpi"]), int(cfg["render.max_dpi"]),
            92,
        )
        for ch in chunks
    ]

    rendered: list[dict] = []
    prog = Progress(len(page_list), label="render", enabled=cfg.get("run.verbose", True))
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_render_pdf_chunk, a) for a in args]
            for f in as_completed(futs):
                res = f.result()
                rendered.extend(res)
                prog.update(len(res))
    else:
        for a in args:
            res = _render_pdf_chunk(a)
            rendered.extend(res)
            prog.update(len(res))
    prog.close()

    rendered.sort(key=lambda r: r["source_page"])
    pages = [
        SourcePage(
            index=i,
            image_path=Path(r["path"]),
            width=r["width"],
            height=r["height"],
            source_page=r["source_page"],
            dpi=r["dpi"],
        )
        for i, r in enumerate(rendered)
    ]
    return IngestResult(
        source=src, stem=slugify(src.stem), meta=meta, pages=pages,
        mode="ocr", source_pdf=src, notes=info,
    )


def _ingest_pdf_native(cfg: Config, src: Path, meta: DocumentMeta, info: dict) -> IngestResult:
    doc = pymupdf.open(src)
    max_pages = int(cfg.get("run.max_pages", 0) or 0)
    limit = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
    pages = []
    for i in range(limit):
        page = doc[i]
        pages.append(
            SourcePage(
                index=i,
                image_path=Path(),
                width=int(page.rect.width),
                height=int(page.rect.height),
                source_page=i,
                native_text=page.get_text("text"),
            )
        )
    doc.close()
    return IngestResult(
        source=src, stem=slugify(src.stem), meta=meta, pages=pages,
        mode="native", source_pdf=src, notes=info,
    )


def _chunk(seq: list, size: int) -> list[list]:
    size = max(1, size)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

_NS = {
    "cnt": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


def _epub_opf_path(z: zipfile.ZipFile) -> str:
    with z.open("META-INF/container.xml") as f:
        root = ET.parse(f).getroot()
    rootfile = root.find(".//cnt:rootfile", _NS)
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("EPUB container.xml has no rootfile")
    return rootfile.get("full-path")


def probe_epub(path: Path) -> dict[str, Any]:
    """Read the OPF: metadata, spine order, and whether it is a scan or real text."""
    with zipfile.ZipFile(path) as z:
        opf_path = _epub_opf_path(z)
        base = posixpath.dirname(opf_path)
        with z.open(opf_path) as f:
            opf = ET.parse(f).getroot()

        md = opf.find("opf:metadata", _NS)
        def _dc(tag: str) -> str:
            if md is None:
                return ""
            el = md.find(f"dc:{tag}", _NS)
            return (el.text or "").strip() if el is not None and el.text else ""

        manifest = {}
        mani = opf.find("opf:manifest", _NS)
        for item in (mani.findall("opf:item", _NS) if mani is not None else []):
            iid = item.get("id")
            href = item.get("href")
            if iid and href:
                manifest[iid] = {
                    "href": posixpath.normpath(posixpath.join(base, href)) if base else href,
                    "type": item.get("media-type", ""),
                }

        spine_ids = []
        spine = opf.find("opf:spine", _NS)
        for ref in (spine.findall("opf:itemref", _NS) if spine is not None else []):
            idref = ref.get("idref")
            if idref in manifest:
                spine_ids.append(idref)

        names = set(z.namelist())
        text_chars, img_refs = 0, 0
        docs = []
        for iid in spine_ids:
            entry = manifest[iid]
            href = entry["href"]
            if href not in names:
                continue
            raw = z.read(href)
            body = raw.decode("utf-8", errors="replace")
            # Strip <head> before measuring: a <title> counted as body text
            # makes an image-only page look like it has real content.
            plain = re.sub(r"(?is)<(head|script|style)\b.*?</\1>", " ", body)
            plain = re.sub(r"<[^>]+>", " ", plain)
            plain = re.sub(r"\s+", " ", plain).strip()
            imgs = re.findall(r'<(?:img|image)[^>]*?(?:src|xlink:href)\s*=\s*["\']([^"\']+)["\']',
                              body, flags=re.I)
            resolved = []
            for s in imgs:
                cand = posixpath.normpath(posixpath.join(posixpath.dirname(href), s))
                if cand in names:
                    resolved.append(cand)
            text_chars += len(plain)
            img_refs += len(resolved)
            docs.append({"href": href, "text": plain, "images": resolved, "html": body})

    per_doc_text = text_chars / max(1, len(docs))
    # An image-scan EPUB is one image per spine document and almost no text.
    is_scan = img_refs >= max(1, len(docs) * 0.8) and per_doc_text < 200

    return {
        "title": _dc("title"), "author": _dc("creator"), "language": _dc("language") or "en",
        "docs": docs, "is_scan": is_scan, "page_count": len(docs),
        "avg_text_per_doc": round(per_doc_text, 1), "image_refs": img_refs,
    }


def ingest_epub(cfg: Config, src: Path, work: Path) -> IngestResult:
    info = probe_epub(src)
    meta = DocumentMeta(
        title=info["title"] or src.stem,
        author=info["author"],
        language=info["language"] or cfg.get("ocr.language", "en"),
        source_file=src.name,
    )
    LOG.info(
        f"  {src.name}: {info['page_count']} spine documents, "
        f"{'image scan' if info['is_scan'] else 'real text'}, "
        f"{info['avg_text_per_doc']:.0f} chars/doc"
    )

    max_pages = int(cfg.get("run.max_pages", 0) or 0)
    docs = info["docs"] if max_pages <= 0 else info["docs"][:max_pages]

    # `force_ocr` exists to distrust an OCR layer sitting over page images.
    # A text EPUB has no page images to re-read, so honouring the flag here
    # would just produce an empty document -- it deliberately does not apply.
    if not info["is_scan"]:
        LOG.info("    real text, no page images -> converting directly, no OCR")
        pages = [
            SourcePage(index=i, image_path=Path(), width=0, height=0,
                       source_page=i, native_text=d["text"], native_html=d["html"])
            for i, d in enumerate(docs)
        ]
        return IngestResult(source=src, stem=slugify(src.stem), meta=meta,
                            pages=pages, mode="native", notes=info)

    img_dir = work / "pages"
    img_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    pages: list[SourcePage] = []
    with zipfile.ZipFile(src) as z:
        prog = Progress(len(docs), label="extract", enabled=cfg.get("run.verbose", True))
        for i, d in enumerate(docs):
            if not d["images"]:
                prog.update(1)
                continue
            # Largest image in the spine document is the page scan.
            best, best_area = None, -1
            for href in d["images"]:
                try:
                    with Image.open(io.BytesIO(z.read(href))) as im:
                        area = im.width * im.height
                        if area > best_area:
                            best, best_area = (href, im.copy()), area
                except Exception as exc:
                    LOG.debug(f"    skipping unreadable image {href}: {exc}")
            if best is None:
                prog.update(1)
                continue
            _, im = best
            im = im.convert("RGB")
            if _is_greyscale(im):
                im = im.convert("L")
            out = img_dir / f"page_{i:05d}.jpg"
            im.save(out, "JPEG", quality=92, optimize=True, subsampling=0)
            pages.append(SourcePage(index=len(pages), image_path=out, width=im.width,
                                    height=im.height, source_page=i, dpi=300.0))
            prog.update(1)
        prog.close()

    return IngestResult(source=src, stem=slugify(src.stem), meta=meta, pages=pages,
                        mode="ocr", notes=info)


# ---------------------------------------------------------------------------
# Loose images
# ---------------------------------------------------------------------------

def ingest_images(cfg: Config, srcs: list[Path], work: Path) -> IngestResult:
    from PIL import Image, ImageSequence

    img_dir = work / "pages"
    img_dir.mkdir(parents=True, exist_ok=True)
    first = srcs[0]
    name = first.parent.name if len(srcs) > 1 else first.stem
    meta = DocumentMeta(title=name, language=cfg.get("ocr.language", "en"),
                        source_file=first.name)

    pages: list[SourcePage] = []
    max_pages = int(cfg.get("run.max_pages", 0) or 0)
    for src in srcs:
        with Image.open(src) as im:
            frames = ImageSequence.Iterator(im) if getattr(im, "n_frames", 1) > 1 else [im]
            for frame in frames:
                if max_pages > 0 and len(pages) >= max_pages:
                    break
                f = frame.convert("RGB")
                if _is_greyscale(f):
                    f = f.convert("L")
                out = img_dir / f"page_{len(pages):05d}.jpg"
                f.save(out, "JPEG", quality=92, optimize=True, subsampling=0)
                pages.append(SourcePage(index=len(pages), image_path=out, width=f.width,
                                        height=f.height, source_page=len(pages), dpi=300.0))
    return IngestResult(source=first, stem=slugify(name), meta=meta, pages=pages,
                        mode="ocr", notes={"page_count": len(pages)})


# ---------------------------------------------------------------------------

def ingest(cfg: Config, srcs: list[Path], work: Path) -> IngestResult:
    """Dispatch on file type. `srcs` is a group from `group_inputs`."""
    first = srcs[0]
    ext = first.suffix.lower()
    total = sum(p.stat().st_size for p in srcs)
    LOG.debug(f"  ingesting {len(srcs)} file(s), {human_size(total)}")
    if ext in PDF_EXT:
        return ingest_pdf(cfg, first, work)
    if ext in EPUB_EXT:
        return ingest_epub(cfg, first, work)
    if ext in IMAGE_EXT:
        return ingest_images(cfg, srcs, work)
    raise ValueError(f"Unsupported input type: {first}")
