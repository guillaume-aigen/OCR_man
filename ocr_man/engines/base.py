"""OCR engine interface and registry.

Every engine takes rendered page images and returns `Page` objects with
normalised block types and pixel-space bounding boxes.  Engines are ranked
by preference in config; the pipeline picks the first one that reports
itself installed, and can fall back to the next one for individual pages
that come back suspiciously empty.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Type

from ..config import Config
from ..doctypes import Page
from ..util import LOG


@dataclass
class PageJob:
    """One page handed to an engine."""

    index: int
    image_path: Path
    width: int
    height: int
    source_page: int
    #: DPI the image represents, used to give the engine a sanely-sized page.
    dpi: float = 300.0


class Engine(abc.ABC):
    """Base class for OCR back-ends."""

    #: Short identifier used in config (`ocr.engines`).
    name: str = "base"
    #: Human-readable description shown by `RUN_ME.py --list-engines`.
    description: str = ""
    #: Whether this engine returns its own reading order.
    provides_reading_order: bool = False
    #: Whether it returns per-line boxes (better searchable-PDF alignment).
    provides_line_boxes: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @classmethod
    @abc.abstractmethod
    def check_available(cls, cfg: Config) -> tuple[bool, str]:
        """(usable, reason). Must not raise, and must be cheap."""

    def warmup(self) -> None:
        """Load models. Called once before the first batch."""

    @abc.abstractmethod
    def run(
        self,
        jobs: list[PageJob],
        on_page: Callable[[Page], None] | None = None,
    ) -> Iterator[Page]:
        """Recognise a batch of pages, yielding `Page` objects as they finish."""

    def close(self) -> None:
        """Free models / subprocesses."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Type[Engine]] = {}


def register(cls: Type[Engine]) -> Type[Engine]:
    _REGISTRY[cls.name] = cls
    return cls


def get_engine_class(name: str) -> Type[Engine] | None:
    _load_builtin()
    return _REGISTRY.get(name)


def all_engine_classes() -> dict[str, Type[Engine]]:
    _load_builtin()
    return dict(_REGISTRY)


_loaded = False


def _load_builtin() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Imported for their @register side effect. Any engine whose optional
    # dependencies are missing simply fails its own check_available().
    from . import mineru_engine, rapid_engine  # noqa: F401


def select_engine(cfg: Config) -> Engine:
    """Instantiate the best available engine according to config."""
    _load_builtin()
    requested = cfg.get("ocr.engine", "auto")
    order: Iterable[str]
    if requested and requested != "auto":
        order = [requested]
    else:
        order = cfg.get("ocr.engines", list(_REGISTRY))

    problems: list[str] = []
    for name in order:
        cls = _REGISTRY.get(name)
        if cls is None:
            problems.append(f"{name}: unknown engine")
            continue
        try:
            ok, why = cls.check_available(cfg)
        except Exception as exc:  # an engine probe must never kill the run
            ok, why = False, f"probe raised {exc!r}"
        if ok:
            LOG.info(f"  OCR engine: {name} ({cls.description})")
            return cls(cfg)
        problems.append(f"{name}: {why}")
        LOG.debug(f"  engine '{name}' unavailable -- {why}")

    detail = "\n    ".join(problems)
    raise SystemExit(
        "No OCR engine is available.\n    " + detail +
        "\n\nRun SETUP.py to install one."
    )


def select_fallback_engine(cfg: Config, exclude: set[str]) -> Engine | None:
    """Next-best engine, used to retry pages the primary engine fumbled."""
    _load_builtin()
    for name in cfg.get("ocr.engines", []):
        if name in exclude:
            continue
        cls = _REGISTRY.get(name)
        if cls is None:
            continue
        try:
            ok, _ = cls.check_available(cfg)
        except Exception:
            ok = False
        if ok:
            return cls(cfg)
    return None
