"""EPUB3 writer, hand-rolled on zipfile.

Written directly rather than through a library so the output is fully under
our control and the pipeline keeps one less dependency: the format is a zip
with a fixed layout, and the parts we need (nav, spine, metadata) are small.

Reflowable text, one chapter per top-level heading, with an EPUB3 nav
document and an EPUB2 NCX so older readers cope too.
"""

from __future__ import annotations

import datetime as _dt
import html
import re
import uuid
import zipfile
from pathlib import Path

from ..config import Config
from ..doctypes import (
    CAPTION, DocumentMeta, Element, FIGURE, FOOTNOTE, FORMULA, HEADING, LIST,
    TABLE, TEXT,
)
from ..util import LOG

CSS = """\
html, body { margin: 0; padding: 0; }
body { font-family: Georgia, "Times New Roman", serif; line-height: 1.5;
       padding: 0.6em 1em; text-align: justify; hyphens: auto; }
h1, h2, h3, h4 { font-family: Helvetica, Arial, sans-serif; line-height: 1.25;
                 text-align: left; page-break-after: avoid; }
h1 { font-size: 1.6em; margin: 1.2em 0 0.7em; }
h2 { font-size: 1.3em; margin: 1.1em 0 0.6em; }
h3 { font-size: 1.1em; margin: 1em 0 0.5em; }
p { margin: 0 0 0.35em; text-indent: 1.2em; }
p.first, h1 + p, h2 + p, h3 + p { text-indent: 0; }
figure { margin: 1em 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 0.85em; font-style: italic; text-align: center;
             margin-top: 0.3em; }
blockquote.footnote { font-size: 0.85em; margin: 0.6em 1.4em; text-indent: 0;
                      color: #444; }
table { border-collapse: collapse; margin: 1em auto; font-size: 0.9em; }
td, th { border: 1px solid #999; padding: 0.25em 0.5em; text-align: left; }
.formula { text-align: center; margin: 0.8em 0; font-family: monospace; }
ul, ol { margin: 0.4em 0 0.6em 1.4em; }
li { margin: 0.15em 0; text-align: left; }
.pagebreak { display: none; }
"""

XHTML_HEAD = """\
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" \
xml:lang="{lang}" lang="{lang}">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
"""

XHTML_TAIL = "</body>\n</html>\n"


def esc(text: str) -> str:
    return html.escape(text or "", quote=False)


# ---------------------------------------------------------------------------
# Chapter splitting
# ---------------------------------------------------------------------------

def split_chapters(
    elements: list[Element],
    max_elements: int = 400,
) -> list[tuple[str, list[Element]]]:
    """Break the element stream into (title, elements) chapters.

    Splits at the shallowest heading level actually present, and force-splits
    an over-long run so a single chapter never becomes a multi-megabyte XHTML
    file that chokes e-readers.
    """
    levels = [e.level for e in elements if e.type == HEADING and e.level]
    split_level = min(levels) if levels else 0

    chapters: list[tuple[str, list[Element]]] = []
    title = ""
    current: list[Element] = []
    body_count = 0

    for el in elements:
        is_split = el.type == HEADING and el.level == split_level and split_level > 0
        too_long = body_count >= max_elements and el.type == HEADING

        if (is_split or too_long) and current:
            chapters.append((title or _fallback_title(current, len(chapters)), current))
            current, body_count = [], 0
            title = el.text.strip() if el.type == HEADING else ""
        elif is_split and not current:
            title = el.text.strip()

        current.append(el)
        if el.type in {TEXT, LIST, TABLE, FIGURE}:
            body_count += 1

    if current:
        chapters.append((title or _fallback_title(current, len(chapters)), current))

    return [c for c in chapters if any(e.type != "page_break" for e in c[1])]


def _fallback_title(elements: list[Element], idx: int) -> str:
    for el in elements:
        if el.type == HEADING and el.text.strip():
            return el.text.strip()
    return f"Part {idx + 1}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_chapter(title: str, elements: list[Element], lang: str) -> str:
    out = [XHTML_HEAD.format(lang=esc(lang or "en"), title=esc(title or "Chapter"))]
    first_para = True
    open_list: str | None = None

    def close_list() -> None:
        nonlocal open_list
        if open_list:
            out.append(f"</{open_list}>")
            open_list = None

    for el in elements:
        text = (el.text or "").strip()

        if el.type == "page_break":
            if el.pages:
                close_list()
                out.append(f'<span class="pagebreak" epub:type="pagebreak" '
                           f'id="page{el.pages[0] + 1}"></span>')
            continue

        if el.type == HEADING:
            close_list()
            level = max(1, min(6, el.level or 2))
            out.append(f"<h{level}>{esc(text)}</h{level}>")
            first_para = True
            continue

        if el.type == LIST:
            items = _list_items(text)
            if items:
                tag = "ol" if items[0][0] else "ul"
                if open_list != tag:
                    close_list()
                    out.append(f"<{tag}>")
                    open_list = tag
                for _, item in items:
                    out.append(f"<li>{esc(item)}</li>")
                continue
            close_list()

        close_list()

        if el.type == TEXT:
            if not text:
                continue
            cls = ' class="first"' if first_para else ""
            out.append(f"<p{cls}>{esc(text)}</p>")
            first_para = False
        elif el.type == FIGURE:
            src = el.image_path
            cap = f"<figcaption>{esc(text)}</figcaption>" if text else ""
            if src:
                out.append(f'<figure><img src="{esc(src)}" alt="{esc(text[:180])}"/>{cap}</figure>')
            elif text:
                out.append(f"<figure>{cap}</figure>")
            first_para = True
        elif el.type == TABLE:
            if el.html and "<table" in el.html.lower():
                out.append(_sanitise_table(el.html))
            elif text:
                out.append(f"<p>{esc(text)}</p>")
            first_para = True
        elif el.type == FORMULA:
            body = (el.html or text).strip().strip("$").strip()
            if body:
                out.append(f'<p class="formula">{esc(body)}</p>')
            first_para = True
        elif el.type == CAPTION:
            if text:
                out.append(f"<p><em>{esc(text)}</em></p>")
            first_para = True
        elif el.type == FOOTNOTE:
            if text:
                out.append(f'<blockquote class="footnote">{esc(text)}</blockquote>')
            first_para = True

    close_list()
    out.append(XHTML_TAIL)
    return "\n".join(out)


_ALLOWED_TABLE_TAGS = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|td|th)(?:\s+(?:colspan|rowspan)=\"?\d+\"?)*\s*/?>",
    re.I,
)


def _sanitise_table(raw: str) -> str:
    """Keep only table structure tags; XHTML will not tolerate stray markup."""
    pieces = []
    pos = 0
    for m in _ALLOWED_TABLE_TAGS.finditer(raw):
        pieces.append(esc(raw[pos:m.start()]))
        pieces.append(m.group(0))
        pos = m.end()
    pieces.append(esc(raw[pos:]))
    out = "".join(pieces)
    return out if "<table" in out.lower() else f"<p>{esc(raw)}</p>"


def _list_items(text: str) -> list[tuple[bool, str]]:
    items: list[tuple[bool, str]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^(?:\d+[.)]|[-*•·–—])\s+(.*)$", ln)
        ordered = bool(re.match(r"^\d+[.)]\s+", ln))
        items.append((ordered, m.group(1) if m else ln))
    return items


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------

CONTAINER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _mime_for(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def write_epub(
    elements: list[Element],
    meta: DocumentMeta,
    out_path: Path,
    cfg: Config,
    image_dir: Path | None = None,
) -> Path:
    lang = meta.language or "en"
    title = meta.title or out_path.stem
    author = meta.author or "Unknown"
    book_id = meta.identifier or f"urn:uuid:{uuid.uuid4()}"
    modified = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    chapters = split_chapters(elements)
    if not chapters:
        chapters = [(title, elements)]

    # Collect the images actually referenced, so unused crops are not shipped.
    used_images: list[str] = []
    if cfg.get("output.epub_include_figures", True) and image_dir:
        seen = set()
        for el in elements:
            if el.type == FIGURE and el.image_path and el.image_path not in seen:
                seen.add(el.image_path)
                if (image_dir.parent / el.image_path).exists():
                    used_images.append(el.image_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        # The mimetype entry must come first and be stored uncompressed.
        z.writestr(
            zipfile.ZipInfo("mimetype"), "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        z.writestr("META-INF/container.xml", CONTAINER_XML)
        z.writestr("OEBPS/style.css", CSS)

        chapter_files: list[tuple[str, str]] = []
        for i, (ch_title, ch_elements) in enumerate(chapters):
            name = f"ch{i + 1:04d}.xhtml"
            z.writestr(f"OEBPS/{name}", render_chapter(ch_title, ch_elements, lang))
            chapter_files.append((name, ch_title or f"Part {i + 1}"))

        for rel in used_images:
            src = image_dir.parent / rel  # type: ignore[union-attr]
            try:
                z.write(src, f"OEBPS/{rel}")
            except OSError as exc:
                LOG.debug(f"    could not embed {rel}: {exc}")

        z.writestr("OEBPS/nav.xhtml", _nav_xhtml(title, lang, chapter_files))
        z.writestr("OEBPS/toc.ncx", _toc_ncx(title, book_id, chapter_files))
        z.writestr(
            "OEBPS/content.opf",
            _content_opf(title, author, lang, book_id, modified, chapter_files, used_images),
        )

    return out_path


def _nav_xhtml(title: str, lang: str, chapters: list[tuple[str, str]]) -> str:
    items = "\n".join(
        f'      <li><a href="{esc(name)}">{esc(label)}</a></li>'
        for name, label in chapters
    )
    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" \
xml:lang="{esc(lang)}" lang="{esc(lang)}">
<head><meta charset="utf-8"/><title>{esc(title)}</title></head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Contents</h1>
    <ol>
{items}
    </ol>
  </nav>
</body>
</html>
"""


def _toc_ncx(title: str, book_id: str, chapters: list[tuple[str, str]]) -> str:
    points = "\n".join(
        f'    <navPoint id="np{i + 1}" playOrder="{i + 1}">\n'
        f'      <navLabel><text>{esc(label)}</text></navLabel>\n'
        f'      <content src="{esc(name)}"/>\n'
        f'    </navPoint>'
        for i, (name, label) in enumerate(chapters)
    )
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="{esc(book_id)}"/></head>
  <docTitle><text>{esc(title)}</text></docTitle>
  <navMap>
{points}
  </navMap>
</ncx>
"""


def _content_opf(
    title: str, author: str, lang: str, book_id: str, modified: str,
    chapters: list[tuple[str, str]], images: list[str],
) -> str:
    manifest = [
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '    <item id="css" href="style.css" media-type="text/css"/>',
    ]
    spine = []
    for i, (name, _) in enumerate(chapters):
        iid = f"ch{i + 1}"
        manifest.append(
            f'    <item id="{iid}" href="{esc(name)}" media-type="application/xhtml+xml"/>'
        )
        spine.append(f'    <itemref idref="{iid}"/>')
    for i, rel in enumerate(images):
        manifest.append(
            f'    <item id="img{i + 1}" href="{esc(rel)}" media-type="{_mime_for(rel)}"/>'
        )
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">{esc(book_id)}</dc:identifier>
    <dc:title>{esc(title)}</dc:title>
    <dc:creator>{esc(author)}</dc:creator>
    <dc:language>{esc(lang)}</dc:language>
    <meta property="dcterms:modified">{modified}</meta>
  </metadata>
  <manifest>
{chr(10).join(manifest)}
  </manifest>
  <spine toc="ncx">
{chr(10).join(spine)}
  </spine>
</package>
"""
