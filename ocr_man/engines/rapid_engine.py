"""RapidOCR fallback.

A pure-ONNX detector + recogniser with no PyTorch dependency.  It is a large
step down from MinerU in quality -- it recognises text lines and nothing
else, so layout, reading order and block types are all left to us -- but it
installs in seconds, runs on CPU, and gives the pipeline something to fall
back on when the VLM engine is unavailable or chokes on a page.

Because it returns line boxes and no reading order, `reading_order.py` does
the column analysis and XY-cut here, and lines are grouped into paragraph
blocks by vertical spacing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from ..config import Config
from ..doctypes import TEXT, Block, Line, Page
from ..reading_order import detect_columns, xy_cut
from ..util import LOG
from .base import Engine, PageJob, register


@register
class RapidOCREngine(Engine):
    name = "rapidocr"
    description = "RapidOCR ONNX line recogniser (fallback, CPU-capable)"
    provides_reading_order = False
    provides_line_boxes = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._engine = None

    @classmethod
    def check_available(cls, cfg: Config) -> tuple[bool, str]:
        try:
            import rapidocr  # noqa: F401
            return True, ""
        except ImportError:
            try:
                import rapidocr_onnxruntime  # noqa: F401
                return True, ""
            except ImportError:
                return False, "not installed (pip install rapidocr onnxruntime)"

    def warmup(self) -> None:
        if self._engine is not None:
            return
        try:
            from rapidocr import RapidOCR  # type: ignore
            self._engine = RapidOCR()
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self._engine = RapidOCR()

    def run(
        self,
        jobs: list[PageJob],
        on_page: Callable[[Page], None] | None = None,
    ) -> Iterator[Page]:
        self.warmup()
        for job in jobs:
            page = self._run_page(job)
            if on_page:
                on_page(page)
            yield page

    def _run_page(self, job: PageJob) -> Page:
        lines: list[Line] = []
        try:
            result = self._engine(str(job.image_path))  # type: ignore[misc]
            for box, text, score in _iter_results(result):
                if not text or not text.strip():
                    continue
                # RapidOCR hands back numpy scalars; cast now so the rest of
                # the pipeline only ever sees plain Python floats.
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                lines.append(
                    Line(text=text.strip(),
                         bbox=(min(xs), min(ys), max(xs), max(ys)),
                         confidence=float(score or 0.0))
                )
        except Exception as exc:
            LOG.warning(f"    RapidOCR failed on page {job.index}: {exc}")

        blocks = _group_lines(lines, job.width, job.height)
        ordered = xy_cut(blocks, job.width, job.height)
        for i, b in enumerate(ordered):
            b.order = i
        return Page(
            index=job.index, width=job.width, height=job.height, blocks=ordered,
            engine=self.name, image_path=str(job.image_path), source_page=job.source_page,
        )


def _iter_results(result) -> Iterator[tuple[list, str, float]]:
    """RapidOCR's return shape has changed between versions; handle both."""
    if result is None:
        return
    # v1.x: (list_of_[box, text, score], elapse)
    if isinstance(result, tuple):
        result = result[0]
    if result is None:
        return
    # v2.x: an object with .boxes / .txts / .scores
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        txts = getattr(result, "txts", []) or []
        scores = getattr(result, "scores", []) or []
        for i, box in enumerate(boxes):
            yield (
                list(box),
                txts[i] if i < len(txts) else "",
                scores[i] if i < len(scores) else 0.0,
            )
        return
    for item in result:
        if len(item) >= 3:
            yield list(item[0]), str(item[1]), float(item[2])


def _group_lines(lines: list[Line], width: int, height: int) -> list[Block]:
    """Cluster recognised lines into paragraph-ish blocks.

    Lines join the current block while they stay in the same column and the
    vertical gap stays close to the running line pitch; a gap noticeably
    larger than the pitch starts a new paragraph.
    """
    if not lines:
        return []

    probe = [Block(type=TEXT, bbox=ln.bbox, text=ln.text, lines=[ln]) for ln in lines]
    layout = detect_columns(probe, width, height)

    buckets: dict[int, list[Line]] = {}
    for ln in lines:
        b = Block(type=TEXT, bbox=ln.bbox, text=ln.text)
        buckets.setdefault(layout.column_of(b), []).append(ln)

    heights = sorted(ln.bbox[3] - ln.bbox[1] for ln in lines)
    median_h = heights[len(heights) // 2] or 12.0

    blocks: list[Block] = []
    for col in sorted(buckets):
        col_lines = sorted(buckets[col], key=lambda l: (l.bbox[1], l.bbox[0]))
        cur: list[Line] = []
        for ln in col_lines:
            if cur:
                gap = ln.bbox[1] - cur[-1].bbox[3]
                indented = ln.bbox[0] - min(x.bbox[0] for x in cur) > median_h * 1.2
                if gap > median_h * 0.9 or indented:
                    blocks.append(_block_from(cur))
                    cur = []
            cur.append(ln)
        if cur:
            blocks.append(_block_from(cur))
    return blocks


def _block_from(group: list[Line]) -> Block:
    x0 = float(min(l.bbox[0] for l in group))
    y0 = float(min(l.bbox[1] for l in group))
    x1 = float(max(l.bbox[2] for l in group))
    y1 = float(max(l.bbox[3] for l in group))
    conf = float(sum(l.confidence for l in group) / len(group))
    return Block(
        type=TEXT, bbox=(x0, y0, x1, y1),
        text="\n".join(l.text for l in group),
        lines=list(group), confidence=conf,
    )
