"""MinerU back-end (default).

MinerU 3.x pairs a document layout model with a VLM recogniser and already
solves the two hardest parts of this job: it reads multi-column pages in the
right order, and it merges a paragraph that runs from the foot of one column
to the head of the next into a single block.

We drive it through its CLI rather than its Python API.  The CLI is the
stable, documented surface, it keeps MinerU's heavy dependency tree in its
own process, and a crash there cannot take the pipeline down with it.

Input is always a *normalised* PDF we build from our own page images, which
means MinerU sees the deskewed, contrast-corrected pixels, and can never be
misled by a bad text layer in the original file -- there isn't one.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config import Config
from ..doctypes import (
    CAPTION, FIGURE, FOOTER, FOOTNOTE, FORMULA, HEADER, HEADING, LIST,
    PAGE_NUMBER, TABLE, TEXT, Block, Line, Page,
)
from ..pdfbuild import PageGeometry, build_normalized_pdf
from ..util import LOG, fmt_duration
from .base import Engine, PageJob, register

#: MinerU block type -> our normalised category.
TYPE_MAP = {
    "title": HEADING,
    "text": TEXT,
    "list": LIST,
    "index": LIST,
    "image": FIGURE,
    "image_body": FIGURE,
    "image_caption": CAPTION,
    "image_footnote": CAPTION,
    "table": TABLE,
    "table_body": TABLE,
    "table_caption": CAPTION,
    "table_footnote": CAPTION,
    "equation": FORMULA,
    "interline_equation": FORMULA,
    "header": HEADER,
    "footer": FOOTER,
    "page_number": PAGE_NUMBER,
    "page_footnote": FOOTNOTE,
    "footnote": FOOTNOTE,
    "aside_text": TEXT,
    "discarded": HEADER,
    "algorithm": TEXT,
    "code": TEXT,
}


@register
class MinerUEngine(Engine):
    name = "mineru"
    description = "MinerU 3.x layout+VLM, best multi-column reading order"
    provides_reading_order = True
    provides_line_boxes = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._exe = find_mineru_exe()
        self._geoms: list[PageGeometry] = []

    # -- availability --------------------------------------------------------
    @classmethod
    def check_available(cls, cfg: Config) -> tuple[bool, str]:
        exe = find_mineru_exe()
        if not exe:
            return False, "mineru executable not found (pip install 'mineru[core]')"
        return True, ""

    # -- run -----------------------------------------------------------------
    def run(
        self,
        jobs: list[PageJob],
        on_page: Callable[[Page], None] | None = None,
    ) -> Iterator[Page]:
        if not jobs:
            return
        batch = max(1, int(self.cfg.get("ocr.mineru_batch_pages", 200) or 200))
        for start in range(0, len(jobs), batch):
            chunk = jobs[start:start + batch]
            LOG.debug(
                f"    MinerU batch {start // batch + 1}"
                f"/{(len(jobs) + batch - 1) // batch} ({len(chunk)} pages)"
            )
            for page in self._run_batch(chunk):
                if on_page:
                    on_page(page)
                yield page

    def _run_batch(self, jobs: list[PageJob]) -> list[Page]:
        tmpdir = Path(tempfile.mkdtemp(prefix="ocrman_mineru_"))
        try:
            pdf_path = tmpdir / "batch.pdf"
            geoms = build_normalized_pdf(
                [j.image_path for j in jobs], pdf_path, dpis=[j.dpi for j in jobs],
            )
            out_dir = tmpdir / "out"
            self._invoke(pdf_path, out_dir, n_pages=len(jobs))
            middle = _find_output(out_dir, "_middle.json")
            if middle is None:
                raise RuntimeError(f"MinerU produced no *_middle.json under {out_dir}")
            data = json.loads(middle.read_text(encoding="utf-8"))
            return self._parse(data, jobs, geoms, middle.parent)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _invoke(self, pdf_path: Path, out_dir: Path, n_pages: int = 0) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg
        backend = cfg.get("ocr.mineru_backend", "hybrid-engine")
        effort = cfg.get("ocr.mineru_effort", "high")
        cmd = [
            str(self._exe),
            "-p", str(pdf_path),
            "-o", str(out_dir),
            "-b", backend,
            "-m", "ocr",
            "-f", "true" if cfg.get("ocr.formula", True) else "false",
            "-t", "true" if cfg.get("ocr.table", True) else "false",
        ]
        if backend.startswith("hybrid"):
            cmd += ["--effort", effort]
        if not cfg.get("ocr.mineru_image_analysis", True):
            cmd += ["--image-analysis", "false"]

        env = os.environ.copy()
        env.setdefault("MINERU_MODEL_SOURCE", cfg.get("ocr.mineru_model_source", "huggingface"))
        # See ocr_man/hf_windows.py -- works around a symlink race on Windows.
        env["OCRMAN_HF_NO_SYMLINK"] = "1"
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")

        LOG.debug(f"    $ {' '.join(cmd[1:])}")
        _run_with_heartbeat(cmd, env, n_pages)

    # -- parsing -------------------------------------------------------------
    def _parse(
        self,
        data: dict[str, Any],
        jobs: list[PageJob],
        geoms: list[PageGeometry],
        asset_dir: Path,
    ) -> list[Page]:
        infos = data.get("pdf_info") or []
        pages: list[Page] = []
        for local_idx, info in enumerate(infos):
            if local_idx >= len(jobs):
                break
            job = jobs[local_idx]
            geom = geoms[local_idx]
            size = info.get("page_size") or [geom.width_pt, geom.height_pt]
            # MinerU reports boxes in the PDF's own point space; convert to the
            # page image's pixels so every engine speaks the same coordinates.
            sx = job.width / max(1e-6, float(size[0]))
            sy = job.height / max(1e-6, float(size[1]))

            blocks: list[Block] = []
            for raw in (info.get("para_blocks") or []):
                blocks.extend(_convert_block(raw, sx, sy))
            for raw in (info.get("discarded_blocks") or []):
                blocks.extend(_convert_block(raw, sx, sy, discarded=True))

            blocks.sort(key=lambda b: b.order)
            for i, b in enumerate(blocks):
                b.order = i

            pages.append(
                Page(
                    index=job.index,
                    width=job.width,
                    height=job.height,
                    blocks=blocks,
                    engine=self.name,
                    image_path=str(job.image_path),
                    source_page=job.source_page,
                    notes={"mineru_backend": self.cfg.get("ocr.mineru_backend", "hybrid-engine")},
                )
            )
        return pages


# ---------------------------------------------------------------------------
# Subprocess supervision
# ---------------------------------------------------------------------------

#: MinerU's phase bars look like "Layout Predict:  45%|####  | 23/52".
_MINERU_PHASE = re.compile(r"([A-Za-z][A-Za-z \-]{2,30}?):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")

#: How often to tell the user the run is still alive, in seconds.
_HEARTBEAT_S = 20.0


def _run_with_heartbeat(cmd: list[str], env: dict, n_pages: int) -> None:
    """Run MinerU, reporting progress instead of going silent.

    MinerU draws its own progress bars with carriage returns on a pipe we
    capture, so nothing reaches the console and the pipeline looks hung for
    minutes at a time -- on a 52-page batch, five of them.  Its output is read
    on a background thread and distilled into a periodic one-line status, so
    there is always visible evidence that work is happening.
    """
    est = f", about {fmt_duration(n_pages * 5.0 + 60)} expected" if n_pages else ""
    LOG.info(f"    recognising {n_pages or '?'} page(s){est}")

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0,
    )

    tail: deque[str] = deque(maxlen=60)
    state = {"phase": "", "done": 0, "total": 0}
    lock = threading.Lock()

    def pump() -> None:
        buf = b""
        stream = proc.stdout
        assert stream is not None
        while True:
            chunk = stream.read(1024)
            if not chunk:
                break
            buf += chunk
            # tqdm separates updates with \r, not \n, so split on both.
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                tail.append(line)
                m = _MINERU_PHASE.search(line)
                if m:
                    with lock:
                        state["phase"] = m.group(1).strip()
                        state["done"] = int(m.group(2))
                        state["total"] = int(m.group(3))
        if buf:
            tail.append(buf.decode("utf-8", errors="replace").strip())

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    started = time.monotonic()
    last_beat = started
    while proc.poll() is None:
        time.sleep(0.4)
        now = time.monotonic()
        if now - last_beat < _HEARTBEAT_S:
            continue
        last_beat = now
        with lock:
            phase, done, total = state["phase"], state["done"], state["total"]
        elapsed = fmt_duration(now - started)
        if total:
            LOG.info(f"      still working - {phase} {done}/{total} ({elapsed} elapsed)")
        else:
            LOG.info(f"      still working - loading models ({elapsed} elapsed)")

    reader.join(timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MinerU exited {proc.returncode}:\n" + "\n".join(list(tail)[-25:])
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale(bbox: list[float] | None, sx: float, sy: float) -> tuple[float, float, float, float]:
    if not bbox or len(bbox) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    return (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)


def _span_text(span: dict[str, Any]) -> str:
    stype = span.get("type", "text")
    content = span.get("content") or ""
    if stype == "inline_equation":
        return f"${content}$" if content else ""
    if stype == "interline_equation":
        return f"$$\n{content}\n$$" if content else ""
    return content


def _lines_of(raw: dict[str, Any], sx: float, sy: float) -> tuple[list[Line], str]:
    lines: list[Line] = []
    for ln in (raw.get("lines") or []):
        parts = [_span_text(s) for s in (ln.get("spans") or [])]
        text = " ".join(p for p in parts if p).strip()
        if not text:
            continue
        lines.append(Line(text=text, bbox=_scale(ln.get("bbox"), sx, sy)))
    return lines, "\n".join(ln.text for ln in lines)


def _table_html(raw: dict[str, Any]) -> str | None:
    for ln in (raw.get("lines") or []):
        for s in (ln.get("spans") or []):
            if s.get("type") == "table" and s.get("html"):
                return s["html"]
    for sub in (raw.get("blocks") or []):
        html = _table_html(sub)
        if html:
            return html
    return None


def _convert_block(
    raw: dict[str, Any],
    sx: float,
    sy: float,
    discarded: bool = False,
) -> list[Block]:
    """One MinerU block -> zero or more of ours (composites expand)."""
    mtype = raw.get("type", "text")
    order = int(raw.get("index", 0) or 0)
    bbox = _scale(raw.get("bbox"), sx, sy)

    # Composite blocks (image/table) carry their body and caption as children.
    children = raw.get("blocks") or []
    if children:
        out: list[Block] = []
        for sub in children:
            out.extend(_convert_block(sub, sx, sy, discarded))
        if out:
            # Children inherit the parent's reading-order slot when they lack one.
            for b in out:
                if not b.order:
                    b.order = order
            return out

    our_type = TYPE_MAP.get(mtype, TEXT)
    if discarded and our_type not in {HEADER, FOOTER, PAGE_NUMBER, FOOTNOTE}:
        our_type = HEADER

    lines, text = _lines_of(raw, sx, sy)
    block = Block(
        type=our_type,
        bbox=bbox,
        text=text,
        lines=lines,
        order=order,
        level=int(raw.get("level", 0) or 0) if our_type == HEADING else 0,
    )

    if our_type == TABLE:
        block.html = _table_html(raw)
        if not block.text and block.html:
            block.text = ""
    elif our_type == FIGURE:
        block.text = text
        sub_type = raw.get("sub_type")
        if sub_type:
            block.html = None
    return [block]


def _find_output(root: Path, suffix: str) -> Path | None:
    hits = sorted(root.rglob(f"*{suffix}"))
    return hits[0] if hits else None


def find_mineru_exe() -> Path | None:
    """Locate the mineru CLI, preferring the venv this script runs in."""
    scripts = Path(sys.executable).parent
    for name in ("mineru.exe", "mineru"):
        cand = scripts / name
        if cand.exists():
            return cand
        cand = scripts / "Scripts" / name
        if cand.exists():
            return cand
    found = shutil.which("mineru")
    return Path(found) if found else None
