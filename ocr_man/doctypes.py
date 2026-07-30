"""Data model shared by every stage of the pipeline.

Everything an engine produces is normalised into `Page` objects made of
`Block`s.  Coordinates are always pixel coordinates in the *rendered page
image* that was handed to the engine, origin top-left.  Keeping one
coordinate space everywhere is what lets the searchable-PDF exporter map
text back onto the original scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Block categories.  Engines report wildly different vocabularies; every
# engine adapter maps its own labels onto exactly these.
# ---------------------------------------------------------------------------
TEXT = "text"
HEADING = "heading"
LIST = "list"
TABLE = "table"
FIGURE = "figure"
CAPTION = "caption"
FORMULA = "formula"
HEADER = "header"          # running head, stripped from prose
FOOTER = "footer"          # running foot, stripped from prose
PAGE_NUMBER = "page_number"
FOOTNOTE = "footnote"

#: Categories that are page furniture rather than document content.
FURNITURE = frozenset({HEADER, FOOTER, PAGE_NUMBER})

#: Categories whose text flows into running prose (and may need de-hyphenating).
FLOWING = frozenset({TEXT, LIST})


def _json_default(obj: Any) -> Any:
    """Last-resort encoder for engine output.

    OCR back-ends hand back numpy scalars and arrays for coordinates and
    confidences.  Rather than trust every adapter to cast them, anything the
    encoder does not recognise is coerced here, so one stray float32 cannot
    abort a run part-way through a book.
    """
    for attr in ("item", "tolist"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


@dataclass
class Line:
    """A single recognised text line, with its box in page-image pixels."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "bbox": list(self.bbox), "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Line":
        return cls(text=d["text"], bbox=tuple(d["bbox"]), confidence=d.get("confidence", 1.0))


@dataclass
class Block:
    """A layout region: a paragraph, a heading, a table, a figure..."""

    type: str
    bbox: tuple[float, float, float, float]
    text: str = ""
    lines: list[Line] = field(default_factory=list)
    order: int = 0
    #: Heading depth (1 = chapter title). Only meaningful when type == HEADING.
    level: int = 0
    #: Column index assigned by reading-order analysis; -1 = page-spanning.
    column: int = -1
    #: Relative path (from the output dir) of an extracted figure image.
    image_path: str | None = None
    #: HTML source for tables, LaTeX for formulas.
    html: str | None = None
    confidence: float = 1.0

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        d["lines"] = [ln.to_dict() for ln in self.lines]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Block":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        d["lines"] = [Line.from_dict(x) for x in d.get("lines", [])]
        return cls(**d)


@dataclass
class Page:
    """One rendered page plus everything the engine found on it."""

    index: int                     # 0-based position in the output document
    width: int                     # page image width in pixels
    height: int                    # page image height in pixels
    blocks: list[Block] = field(default_factory=list)
    engine: str = ""
    #: Path to the rendered page image on disk (inside WORK/).
    image_path: str = ""
    #: Index of the page in the *source* file, for the searchable-PDF overlay.
    source_page: int = 0
    #: Rotation in degrees applied by preprocessing (for coordinate mapping).
    deskew_angle: float = 0.0
    #: Free-form engine diagnostics.
    notes: dict[str, Any] = field(default_factory=dict)

    def content_blocks(self) -> list[Block]:
        """Blocks in reading order, excluding running heads/feet/page numbers."""
        return [b for b in sorted(self.blocks, key=lambda b: b.order) if b.type not in FURNITURE]

    def furniture(self) -> list[Block]:
        return [b for b in self.blocks if b.type in FURNITURE]

    def plain_text(self) -> str:
        return "\n".join(b.text for b in self.content_blocks() if b.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
            "engine": self.engine,
            "image_path": self.image_path,
            "source_page": self.source_page,
            "deskew_angle": self.deskew_angle,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Page":
        return cls(
            index=d["index"],
            width=d["width"],
            height=d["height"],
            blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
            engine=d.get("engine", ""),
            image_path=d.get("image_path", ""),
            source_page=d.get("source_page", 0),
            deskew_angle=d.get("deskew_angle", 0.0),
            notes=d.get("notes", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Page":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class DocumentMeta:
    """Bibliographic metadata carried into the EPUB."""

    title: str = ""
    author: str = ""
    language: str = "en"
    source_file: str = ""
    identifier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Element:
    """A finished piece of the assembled document, ready for export.

    This is the flattened, cross-page-stitched form: by the time we build
    `Element`s, paragraphs that were split across a column break or a page
    break have already been rejoined.
    """

    type: str
    text: str = ""
    level: int = 0
    image_path: str | None = None
    html: str | None = None
    #: Source page indices this element came from (may span two after stitching).
    pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Element":
        return cls(**d)


def save_elements(elements: Iterable[Element], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.to_dict() for e in elements], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def load_elements(path: Path) -> list[Element]:
    return [Element.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
