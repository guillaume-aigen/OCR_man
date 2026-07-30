"""Small shared helpers: logging, hashing, filenames, timing."""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

LOG = logging.getLogger("ocr_man")

_ANSI = {
    "grey": "\033[90m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _supports_colour() -> bool:
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # Windows Terminal / VS Code / modern conhost all handle VT sequences.
        try:
            import colorama  # type: ignore  # noqa: F401

            return True
        except ImportError:
            import os

            return os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")
    return True


_COLOUR = bool(_supports_colour())


def c(text: str, colour: str) -> str:
    if not _COLOUR:
        return text
    return f"{_ANSI.get(colour, '')}{text}{_ANSI['reset']}"


class _Formatter(logging.Formatter):
    LEVEL_COLOURS = {
        logging.DEBUG: "grey",
        logging.INFO: "cyan",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red",
    }

    def format(self, record: logging.LogRecord) -> str:
        tag = {
            logging.DEBUG: "  ",
            logging.INFO: "> ",
            logging.WARNING: "! ",
            logging.ERROR: "X ",
            logging.CRITICAL: "X ",
        }.get(record.levelno, "  ")
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return c(tag, self.LEVEL_COLOURS.get(record.levelno, "grey")) + msg


def setup_logging(verbose: bool = True, logfile: Path | None = None) -> None:
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(_Formatter())
    LOG.addHandler(h)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        LOG.addHandler(fh)
    LOG.propagate = False


def banner(text: str) -> None:
    line = "=" * max(12, min(78, len(text) + 4))
    LOG.info(c(line, "blue"))
    LOG.info(c("  " + text, "bold"))
    LOG.info(c(line, "blue"))


@contextmanager
def timed(label: str, level: int = logging.INFO):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    LOG.log(level, f"{label} {c(f'({fmt_duration(dt)})', 'grey')}")


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def slugify(name: str, max_len: int = 90) -> str:
    """Filesystem-safe stem preserving readability."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"[^\w\s.-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_{2,}", "_", name).strip("._-")
    return (name or "document")[:max_len]


def file_fingerprint(path: Path, extra: str = "") -> str:
    """Cheap but reliable cache key: size + mtime + head/tail bytes + extra."""
    st = path.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    with path.open("rb") as f:
        h.update(f.read(65536))
        if st.st_size > 131072:
            f.seek(-65536, 2)
            h.update(f.read(65536))
    h.update(extra.encode())
    return h.hexdigest()[:16]


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


class Progress:
    """Minimal single-line progress bar (no external dependency)."""

    def __init__(self, total: int, label: str = "", width: int = 28, enabled: bool = True):
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.n = 0
        self.t0 = time.perf_counter()
        self.enabled = enabled and sys.stdout.isatty()
        self._last_len = 0

    def update(self, n: int = 1, suffix: str = "") -> None:
        self.n += n
        if not self.enabled:
            return
        frac = min(1.0, self.n / self.total)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.t0
        rate = self.n / elapsed if elapsed > 0 else 0
        eta = (self.total - self.n) / rate if rate > 0 else 0
        line = (
            f"\r  {self.label} [{bar}] {self.n}/{self.total} "
            f"{fmt_duration(elapsed)} eta {fmt_duration(eta)} {suffix}"
        )
        pad = " " * max(0, self._last_len - len(line))
        sys.stdout.write(line + pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def close(self, suffix: str = "") -> None:
        if self.enabled:
            sys.stdout.write("\r" + " " * (self._last_len + 2) + "\r")
            sys.stdout.flush()
        elapsed = time.perf_counter() - self.t0
        LOG.info(f"{self.label} done: {self.n}/{self.total} in {fmt_duration(elapsed)} {suffix}")
