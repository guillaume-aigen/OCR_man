"""End-to-end orchestration for one document.

    ingest -> restore -> OCR -> reading order -> assemble -> LLM -> export

Every expensive stage writes its result into WORK/ keyed by a fingerprint of
the source file and the settings that affect it, so an interrupted run over a
thousand-page book resumes instead of starting again.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import assemble as asm
from .config import Config
from .doctypes import DocumentMeta, Element, Page, load_elements, save_elements
from .engines.base import Engine, PageJob, select_engine, select_fallback_engine
from .exporters import write_epub, write_markdown
from .ingest import IngestResult, ingest
from .llm_clean import clean_elements
from .pdfbuild import PageGeometry, write_searchable_pdf
from .preprocess import preprocess_worker
from .reading_order import resolve_reading_order
from .util import LOG, Progress, file_fingerprint, human_size


@dataclass
class DocResult:
    source: Path
    stem: str
    ok: bool = True
    error: str = ""
    pages: int = 0
    outputs: dict[str, Path] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def _settings_fingerprint(cfg: Config) -> str:
    """Only the settings that change OCR results take part in the cache key."""
    keys = ["render", "preprocess", "ocr"]
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)
    import hashlib

    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def process_document(
    cfg: Config,
    srcs: list[Path],
    engine: Engine | None = None,
) -> DocResult:
    t0 = time.perf_counter()
    src = srcs[0]
    result = DocResult(source=src, stem=src.stem)

    try:
        # -- 1. ingest -------------------------------------------------------
        fingerprint = file_fingerprint(src, _settings_fingerprint(cfg))
        work = cfg.work_dir / f"{src.stem}_{fingerprint}"
        work.mkdir(parents=True, exist_ok=True)
        out_dir = cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ing = _cached_ingest(cfg, srcs, work)
        result.stem = ing.stem
        result.pages = len(ing.pages)
        if not ing.pages:
            raise RuntimeError("no pages found in this file")

        # -- born-digital shortcut ------------------------------------------
        if ing.mode == "native":
            elements, stats = asm.assemble_native(
                [(p.index, p.native_text or "", p.native_html) for p in ing.pages], cfg
            )
            page_numbers: dict[int, str] = {}
            ocr_pages: list[Page] = []
            geoms: list[PageGeometry] = []
            image_paths: list[Path] = []
        else:
            # -- 2. restore --------------------------------------------------
            image_paths, quality = _preprocess_pages(cfg, ing, work)
            result.stats["preprocessed"] = quality["changed"]
            result.stats["median_scan_quality"] = quality["median_score"]

            # -- 3. OCR ------------------------------------------------------
            engine = engine or select_engine(cfg)
            ocr_pages = _run_ocr(cfg, ing, image_paths, work, engine)
            result.stats["engine"] = engine.name

            # -- 4. reading order --------------------------------------------
            _apply_reading_order(cfg, ocr_pages)

            # -- 5. assemble --------------------------------------------------
            page_numbers = asm.extract_page_numbers(ocr_pages)
            elements, stats = asm.assemble(ocr_pages, cfg, out_dir)
            geoms = _geometries(image_paths, ing)

        _infer_title(ing.meta, elements, src)

        result.stats.update({
            "paragraphs": stats.paragraphs,
            "headings": stats.headings,
            "tables": stats.tables,
            "figures": stats.figures,
            "stitched_paragraphs": stats.stitched,
            "furniture_removed": stats.furniture_removed,
        })

        # -- 6. LLM cleanup ---------------------------------------------------
        clean_path = work / "elements_clean.json"
        raw_path = work / "elements_raw.json"
        save_elements(elements, raw_path)
        if cfg.get("llm.enabled", True):
            cached_clean = None
            if cfg.get("run.resume", True) and clean_path.exists():
                try:
                    candidate = load_elements(clean_path)
                    # The correction pass is 1:1 on elements, so a different
                    # count means this cache came from a run over a different
                    # slice of the book (a --pages trial) and must not be used.
                    if len(candidate) == len(elements):
                        cached_clean = candidate
                    else:
                        LOG.info(
                            f"  LLM cache covers {len(candidate)} elements, this run has "
                            f"{len(elements)}; redoing the correction pass"
                        )
                except Exception as exc:
                    LOG.debug(f"  LLM cache unusable ({exc}); redoing")

            if cached_clean is not None:
                LOG.info("  LLM cleanup: reusing cached result")
                elements = cached_clean
            else:
                elements, cstats = clean_elements(elements, cfg)
                save_elements(elements, clean_path)
                result.stats["llm_cleanup"] = cstats.summary()

        # -- 7. export --------------------------------------------------------
        stem = ing.stem
        if cfg.get("output.markdown", True):
            md = write_markdown(
                elements, ing.meta, out_dir / f"{stem}.md", cfg,
                page_numbers=page_numbers, stats=dict(result.stats, pages=result.pages),
            )
            result.outputs["markdown"] = md
            LOG.info(f"  wrote {md.name} ({human_size(md.stat().st_size)})")

        if cfg.get("output.epub", True):
            ep = write_epub(elements, ing.meta, out_dir / f"{stem}.epub", cfg,
                            image_dir=out_dir / "images")
            result.outputs["epub"] = ep
            LOG.info(f"  wrote {ep.name} ({human_size(ep.stat().st_size)})")

        if cfg.get("output.searchable_pdf", True) and ocr_pages and image_paths:
            pdf = write_searchable_pdf(
                ocr_pages, image_paths, geoms, out_dir / f"{stem}_searchable.pdf",
                jpeg_quality=int(cfg.get("output.searchable_pdf_jpeg_quality", 80)),
                font_file=cfg.get("output.searchable_pdf_font") or None,
            )
            result.outputs["searchable_pdf"] = pdf
            LOG.info(f"  wrote {pdf.name} ({human_size(pdf.stat().st_size)})")

        if not cfg.get("run.keep_work_dir", True):
            shutil.rmtree(work, ignore_errors=True)

    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        LOG.error(f"  FAILED: {result.error}")
        LOG.debug(traceback.format_exc())

    result.seconds = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _cached_ingest(cfg: Config, srcs: list[Path], work: Path) -> IngestResult:
    manifest = work / "ingest.json"
    pages_dir = work / "pages"
    # Only the OCR path is worth caching: it costs a render pass per page.
    # Re-reading a text EPUB or a born-digital PDF takes milliseconds and
    # caching it would mean writing the whole book into the manifest.
    if cfg.get("run.resume", True) and manifest.exists() and pages_dir.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            from .ingest import SourcePage

            pages = [
                SourcePage(
                    index=p["index"], image_path=Path(p["image_path"]),
                    width=p["width"], height=p["height"], source_page=p["source_page"],
                    native_text=p.get("native_text"), dpi=p.get("dpi", 300.0),
                )
                for p in data["pages"]
            ]
            # A cache written by a `--pages N` trial run holds only part of the
            # book. Reusing it for the full run would silently produce an
            # N-page document, so the cache is only good when it covers at
            # least as many pages as this run asks for.
            total = int(data.get("total_pages", 0) or 0)
            want = int(cfg.get("run.max_pages", 0) or 0)
            need = min(want, total) if (want > 0 and total > 0) else (want or total)
            if need and len(pages) < need:
                LOG.info(
                    f"  cached ingest covers only {len(pages)} of {need} pages "
                    "needed; re-reading the source"
                )
            elif data["mode"] == "native" or all(p.image_path.exists() for p in pages):
                if want > 0 and len(pages) > want:
                    pages = pages[:want]
                LOG.info(f"  reusing {len(pages)} cached page images")
                return IngestResult(
                    source=Path(data["source"]), stem=data["stem"],
                    meta=DocumentMeta(**data["meta"]), pages=pages,
                    mode=data["mode"],
                    source_pdf=Path(data["source_pdf"]) if data.get("source_pdf") else None,
                    notes=data.get("notes", {}),
                )
        except Exception as exc:
            LOG.debug(f"  ingest cache unusable ({exc}); re-ingesting")

    ing = ingest(cfg, srcs, work)
    if ing.mode == "native":
        # Nothing cacheable and nothing worth caching; see above.
        return ing
    manifest.write_text(json.dumps({
        "source": str(ing.source), "stem": ing.stem, "mode": ing.mode,
        "meta": ing.meta.to_dict(),
        "source_pdf": str(ing.source_pdf) if ing.source_pdf else None,
        "notes": {k: v for k, v in ing.notes.items() if k != "docs"},
        # How many pages the *source* has, so a later run can tell whether this
        # cache is complete or was truncated by --pages.
        "total_pages": int(ing.notes.get("page_count") or len(ing.pages)),
        "max_pages": int(cfg.get("run.max_pages", 0) or 0),
        "pages": [
            {"index": p.index, "image_path": str(p.image_path), "width": p.width,
             "height": p.height, "source_page": p.source_page,
             "native_text": p.native_text, "dpi": p.dpi}
            for p in ing.pages
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return ing


def _preprocess_pages(cfg: Config, ing: IngestResult, work: Path) -> tuple[list[Path], dict]:
    """Restore each page image; returns the paths to actually OCR."""
    if not cfg.get("preprocess.enabled", True) or cfg.get("preprocess.mode") == "never":
        return [p.image_path for p in ing.pages], {"changed": 0, "median_score": None}

    clean_dir = work / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    cache = work / "preprocess.json"
    if cfg.get("run.resume", True) and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            paths = [Path(p) for p in data["paths"]]
            # Same partial-cache trap as the ingest step: a cache from a
            # `--pages N` run covers fewer pages than a full run needs.
            if len(paths) >= len(ing.pages) and all(p.exists() for p in paths):
                paths = paths[:len(ing.pages)]
                LOG.info(f"  reusing cached page restoration ({data['changed']} pages cleaned)")
                return paths, data
        except Exception:
            pass

    args = [
        (str(p.image_path), str(clean_dir / f"{p.image_path.stem}_clean.jpg"), cfg.data)
        for p in ing.pages
    ]
    workers = max(1, int(cfg.get("run.workers", 4)))
    results: list[dict | None] = [None] * len(args)

    prog = Progress(len(args), label="restore", enabled=cfg.get("run.verbose", True))
    if workers > 1 and len(args) > 2:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(preprocess_worker, a): i for i, a in enumerate(args)}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    results[i] = f.result()
                except Exception as exc:
                    LOG.debug(f"    restore failed on page {i}: {exc}")
                    results[i] = {"path": args[i][0], "quality": {}, "actions": {}}
                prog.update(1)
    else:
        for i, a in enumerate(args):
            try:
                results[i] = preprocess_worker(a)
            except Exception as exc:
                LOG.debug(f"    restore failed on page {i}: {exc}")
                results[i] = {"path": a[0], "quality": {}, "actions": {}}
            prog.update(1)

    paths = [Path(r["path"]) for r in results if r]
    changed = sum(1 for r in results if r and r.get("actions"))
    scores = sorted(r["quality"].get("score", 1.0) for r in results if r and r.get("quality"))
    median = round(scores[len(scores) // 2], 3) if scores else None
    prog.close(f"{changed} page(s) restored, median scan quality {median}")

    data = {"paths": [str(p) for p in paths], "changed": changed, "median_score": median}
    cache.write_text(json.dumps(data), encoding="utf-8")
    return paths, data


def _run_ocr(
    cfg: Config,
    ing: IngestResult,
    image_paths: list[Path],
    work: Path,
    engine: Engine,
) -> list[Page]:
    ocr_dir = work / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    resume = cfg.get("run.resume", True)

    jobs: list[PageJob] = []
    pages: dict[int, Page] = {}
    for sp, img in zip(ing.pages, image_paths):
        cached = ocr_dir / f"page_{sp.index:05d}.json"
        if resume and cached.exists():
            try:
                pages[sp.index] = Page.load(cached)
                continue
            except Exception:
                pass
        jobs.append(PageJob(index=sp.index, image_path=img, width=sp.width,
                            height=sp.height, source_page=sp.source_page, dpi=sp.dpi))

    if pages:
        LOG.info(f"  reusing {len(pages)} cached OCR page(s)")

    if jobs:
        engine.warmup()
        prog = Progress(len(jobs), label=f"OCR[{engine.name}]",
                        enabled=cfg.get("run.verbose", True))
        for page in engine.run(jobs):
            pages[page.index] = page
            page.save(ocr_dir / f"page_{page.index:05d}.json")
            prog.update(1)
        prog.close()

    ordered = [pages[i] for i in sorted(pages)]
    _retry_weak_pages(cfg, ordered, ing, image_paths, ocr_dir, engine)
    return [pages[i] for i in sorted(pages)]


def _retry_weak_pages(
    cfg: Config,
    pages: list[Page],
    ing: IngestResult,
    image_paths: list[Path],
    ocr_dir: Path,
    engine: Engine,
) -> None:
    """Re-run near-empty pages on the next engine, keeping whichever is better."""
    if not cfg.get("ocr.fallback_on_low_yield", True):
        return
    threshold = int(cfg.get("ocr.min_chars_per_page", 60))
    weak = [p for p in pages if len(p.plain_text().strip()) < threshold]
    if not weak:
        return

    # A scan can legitimately be a blank leaf or a full-page plate; only bother
    # when enough pages look wrong that it smells like an engine problem.
    if len(weak) < max(2, len(pages) * 0.02):
        LOG.debug(f"    {len(weak)} low-yield page(s); too few to be systematic, leaving as-is")
        return

    fallback = select_fallback_engine(cfg, exclude={engine.name})
    if fallback is None:
        LOG.debug(f"    {len(weak)} low-yield page(s) but no fallback engine installed")
        return

    LOG.info(f"  retrying {len(weak)} low-yield page(s) with {fallback.name}")
    by_index = {p.index: p for p in pages}
    jobs = []
    for p in weak:
        sp = ing.pages[p.index]
        jobs.append(PageJob(index=p.index, image_path=image_paths[p.index], width=sp.width,
                            height=sp.height, source_page=sp.source_page, dpi=sp.dpi))
    try:
        fallback.warmup()
        for alt in fallback.run(jobs):
            if len(alt.plain_text().strip()) > len(by_index[alt.index].plain_text().strip()):
                by_index[alt.index].blocks = alt.blocks
                by_index[alt.index].engine = f"{engine.name}+{fallback.name}"
                by_index[alt.index].save(ocr_dir / f"page_{alt.index:05d}.json")
    except Exception as exc:
        LOG.warning(f"    fallback engine failed: {exc}")
    finally:
        fallback.close()


def _apply_reading_order(cfg: Config, pages: list[Page]) -> None:
    methods: dict[str, int] = {}
    for page in pages:
        if not page.blocks:
            continue
        _, layout, method = resolve_reading_order(
            page.blocks, page.width, page.height,
            prefer_engine_order=cfg.get("layout.prefer_engine_order", True),
            validate=cfg.get("layout.validate_engine_order", True),
            max_columns=int(cfg.get("layout.max_columns", 4)),
            gap_ratio=float(cfg.get("layout.column_gap_ratio", 0.035)),
        )
        page.notes["columns"] = layout.n_columns
        page.notes["order_method"] = method
        methods[method] = methods.get(method, 0) + 1
    if methods:
        LOG.debug("    reading order: " + ", ".join(f"{k} x{v}" for k, v in methods.items()))


def _infer_title(meta: DocumentMeta, elements: list[Element], src: Path) -> None:
    """Fall back to the book's own title page when the file has no metadata.

    A PDF produced by a scanner usually has either no title or the filename,
    neither of which is much use on an e-reader shelf.
    """
    from .doctypes import HEADING

    placeholder = (not meta.title) or meta.title.strip().lower() in {
        src.stem.lower(), src.name.lower(), "untitled", "unknown",
    }
    if not placeholder:
        return
    for el in elements[:40]:
        if el.type == HEADING and el.text.strip():
            candidate = " ".join(el.text.split())
            if 3 <= len(candidate) <= 120:
                meta.title = candidate
                LOG.debug(f"    title taken from the first heading: {candidate!r}")
                return
    meta.title = meta.title or src.stem


def _geometries(image_paths: list[Path], ing: IngestResult) -> list[PageGeometry]:
    from .pdfbuild import POINTS_PER_INCH

    geoms: list[PageGeometry] = []
    for sp, img in zip(ing.pages, image_paths):
        try:
            from PIL import Image

            with Image.open(img) as im:
                w_px, h_px = im.width, im.height
        except Exception:
            w_px, h_px = sp.width, sp.height
        dpi = sp.dpi or 300.0
        geoms.append(
            PageGeometry(w_px, h_px, w_px * POINTS_PER_INCH / dpi, h_px * POINTS_PER_INCH / dpi)
        )
    return geoms
