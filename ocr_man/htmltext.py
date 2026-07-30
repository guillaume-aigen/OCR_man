"""Convert EPUB chapter XHTML into document elements.

Used for EPUBs that already contain real text, where there are no pixels to
re-read and running them through OCR would only lose information.

Flattening the markup to a plain string and splitting on blank lines does not
work: XHTML carries paragraph structure in tags, not whitespace, so every
chapter comes out as one giant run-on paragraph and the headings are lost.
This walks the markup instead and keeps the block structure.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from .doctypes import CAPTION, FIGURE, HEADING, LIST, TEXT, Element

#: Tags whose end implies a paragraph boundary.
_BLOCK = {
    "p", "div", "section", "article", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "dd", "dt",
    "td", "th", "tr", "figcaption", "caption",
}
#: Tags whose entire contents are metadata or code, never body text.
_SKIP = {"head", "script", "style", "title", "meta", "link"}

_HEADINGS = {f"h{i}": i for i in range(1, 7)}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self._buf: list[str] = []
        self._skip_depth = 0
        #: Stack of open block tags, so we know what the current text is.
        self._stack: list[str] = []

    # -- buffer handling -----------------------------------------------------
    def _flush(self, tag: str | None = None) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf.clear()
        if not text:
            return
        if tag in _HEADINGS:
            self.elements.append(Element(type=HEADING, text=text, level=_HEADINGS[tag]))
        elif tag == "li":
            self.elements.append(Element(type=LIST, text=text))
        elif tag in {"figcaption", "caption"}:
            self.elements.append(Element(type=CAPTION, text=text))
        else:
            self.elements.append(Element(type=TEXT, text=text))

    # -- parser callbacks ----------------------------------------------------
    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._buf.append(" ")
            return
        if tag == "img":
            src = dict(attrs).get("src") or ""
            alt = dict(attrs).get("alt") or ""
            if src:
                self._flush(self._stack[-1] if self._stack else None)
                self.elements.append(Element(type=FIGURE, text=alt, image_path=src))
            return
        if tag in _BLOCK:
            # A block opening inside another block ends the outer one's text.
            self._flush(self._stack[-1] if self._stack else None)
            self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in _BLOCK:
            self._flush(tag)
            # Unwind to and including the matching open tag.
            if tag in self._stack:
                while self._stack and self._stack.pop() != tag:
                    pass

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._buf.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._flush(self._stack[-1] if self._stack else None)


def html_to_elements(html: str) -> list[Element]:
    """Parse one XHTML chapter into headings, paragraphs, lists and figures."""
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # A malformed chapter should degrade, not abort the book.
        text = re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(head|script|style).*?</\1>", " ", html))
        text = re.sub(r"\s+", " ", text).strip()
        return [Element(type=TEXT, text=text)] if text else []
    return parser.elements


def html_to_text(html: str) -> str:
    """Plain text with blank lines between blocks, for quality heuristics."""
    return "\n\n".join(e.text for e in html_to_elements(html) if e.text)
