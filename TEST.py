#!/usr/bin/env python
"""Fast self-checks for the tricky logic. No GPU, no models, ~1 second.

    .venv\\Scripts\\python.exe TEST.py

Covers the parts that are easy to break and hard to notice: column detection,
reading order, paragraph stitching, de-hyphenation, EPUB HTML parsing, and the
LLM guard rails. Run it after changing anything in ocr_man/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ocr_man.assemble import (  # noqa: E402
    _is_continuation, dehyphenate, looks_like_folio, mark_furniture, stitch_elements,
)
from ocr_man.config import load_config  # noqa: E402
from ocr_man.doctypes import Block, Element, Page, TEXT, HEADING, LIST  # noqa: E402
from ocr_man.htmltext import html_to_elements  # noqa: E402
from ocr_man.llm_clean import check_result, strip_wrapper  # noqa: E402
from ocr_man.reading_order import detect_columns, resolve_reading_order, xy_cut  # noqa: E402

PASS, FAIL = 0, 0
FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def section(title: str) -> None:
    print(f"\n{title}")


def _two_column_lines(page_w=1000, page_h=1400, gutter=True):
    """Synthesise a two-column page of line boxes."""
    blocks = []
    left_x, right_x = 60, 540
    width = 400 if gutter else 470   # no gutter => columns touch
    for i in range(25):
        y = 100 + i * 40
        blocks.append(Block(type=TEXT, bbox=(left_x, y, left_x + width, y + 28), text=f"L{i}"))
        blocks.append(Block(type=TEXT, bbox=(right_x, y, right_x + width, y + 28), text=f"R{i}"))
    return blocks, page_w, page_h


# ---------------------------------------------------------------------------
section("column detection")

blocks, w, h = _two_column_lines()
check("clean gutter -> 2 columns", detect_columns(blocks, w, h).n_columns, 2)

# The regression that mattered: adjacent columns whose boxes leave no
# whitespace valley at all. Whitespace analysis cannot see this; the
# left-edge clustering must.
blocks_ng, w, h = _two_column_lines(gutter=False)
check("no whitespace gutter -> still 2 columns", detect_columns(blocks_ng, w, h).n_columns, 2)

# A few wide lines (headings, merged detections) must not sink the split.
blocks_wide = list(blocks)
blocks_wide.append(Block(type=TEXT, bbox=(60, 60, 940, 90), text="RUNNING HEAD ACROSS PAGE"))
blocks_wide.append(Block(type=TEXT, bbox=(60, 700, 830, 730), text="a merged line"))
check("wide lines present -> still 2 columns", detect_columns(blocks_wide, w, h).n_columns, 2)

single = [Block(type=TEXT, bbox=(60, 100 + i * 40, 940, 128 + i * 40), text=f"x{i}")
          for i in range(25)]
check("single column stays 1", detect_columns(single, w, h).n_columns, 1)

# A hanging indent is one column, not two.
hanging = []
for i in range(24):
    x0 = 60 if i % 4 else 110
    hanging.append(Block(type=TEXT, bbox=(x0, 100 + i * 40, 900, 128 + i * 40), text=f"h{i}"))
check("hanging indent stays 1 column", detect_columns(hanging, w, h).n_columns, 1)

# ---------------------------------------------------------------------------
section("reading order")

blocks, w, h = _two_column_lines()
ordered = xy_cut(blocks, w, h)
texts = [b.text for b in ordered]
check("xy-cut reads left column first", texts[:3], ["L0", "L1", "L2"])
check("xy-cut reaches right column after left", texts[25:28], ["R0", "R1", "R2"])

# An engine order that zig-zags across columns must be rejected and rebuilt.
zigzag, w, h = _two_column_lines()
for i, b in enumerate(zigzag):
    b.order = i                      # L0,R0,L1,R1,... == read straight across
fixed, layout, method = resolve_reading_order(
    zigzag, w, h, prefer_engine_order=True, validate=True)
check("zig-zag engine order is rejected", method, "xy_cut(corrected)")
check("zig-zag order is repaired", [b.text for b in fixed][:3], ["L0", "L1", "L2"])

good, w, h = _two_column_lines()
left = [b for b in good if b.text.startswith("L")]
right = [b for b in good if b.text.startswith("R")]
for i, b in enumerate(left + right):
    b.order = i
_, _, method = resolve_reading_order(good, w, h, prefer_engine_order=True, validate=True)
check("correct engine order is kept", method, "engine")

# ---------------------------------------------------------------------------
section("de-hyphenation")

check("joins a broken word", dehyphenate("automobile-\ndevouring"), "automobiledevouring")
check("keeps a real compound", dehyphenate("Anglo-\nAmerican"), "Anglo-\nAmerican")
check("leaves clean text alone", dehyphenate("no hyphens here"), "no hyphens here")

# ---------------------------------------------------------------------------
section("paragraph stitching")

long_tail = "x" * 60
check("continues when no sentence end",
      _is_continuation(long_tail + " set in an undated NEAR", "FUTURE, we have been"), True)
check("stops at a full stop",
      _is_continuation(long_tail + " end of the thought.", "A new paragraph begins"), False)
check("stops before a list item",
      _is_continuation(long_tail + " as follows", "1. First item"), False)
check("ignores a short trailing fragment",
      _is_continuation("Short bit", "continues here"), False)

els = [
    Element(type=TEXT, text="a" * 60 + " set in an undated NEAR", pages=[0]),
    Element(type="page_break", pages=[1]),
    Element(type=TEXT, text="FUTURE, we have been highly selective.", pages=[1]),
]
stitched, n = stitch_elements(els, dehyph=True)
check("stitches across a page break", n, 1)
check("stitched text is joined",
      stitched[0].text.endswith("undated NEAR FUTURE, we have been highly selective."), True)

# ---------------------------------------------------------------------------
section("page furniture")

check("plain folio", looks_like_folio("61"), True)
check("roman folio", looks_like_folio("xiv"), True)
check("prose is not a folio", looks_like_folio("Chapter One begins"), False)

pages = []
for i in range(8):
    pg = Page(index=i, width=1000, height=1400)
    pg.blocks = [
        Block(type=TEXT, bbox=(60, 20, 500, 45), text="A RUNNING HEAD ON EVERY PAGE"),
        Block(type=TEXT, bbox=(60, 200, 900, 600), text="Body text that differs on every page " * 3),
        Block(type=TEXT, bbox=(480, 1360, 520, 1385), text=str(40 + i)),
    ]
    pages.append(pg)
stats = mark_furniture(pages)
check("running head detected on all pages", stats["header"], 8)
check("folios detected on all pages", stats["page_number"], 8)
check("body text is untouched", pages[0].blocks[1].type, TEXT)

# ---------------------------------------------------------------------------
section("EPUB html parsing")

html = """<html><head><title>Chapter 1</title></head><body>
<h1>Chapter 1</h1><p>First paragraph.</p><p>Second paragraph.</p>
<ul><li>alpha</li><li>beta</li></ul></body></html>"""
els = html_to_elements(html)
kinds = [(e.type, e.text) for e in els]
check("title is not duplicated into the body",
      sum(1 for t, x in kinds if x == "Chapter 1"), 1)
check("heading is recognised", kinds[0], (HEADING, "Chapter 1"))
check("paragraphs stay separate",
      [x for t, x in kinds if t == TEXT], ["First paragraph.", "Second paragraph."])
check("list items are recognised",
      [x for t, x in kinds if t == LIST], ["alpha", "beta"])

# ---------------------------------------------------------------------------
section("LLM guard rails")

cfg = load_config(root=ROOT)
orig = ("The biggest difficulty in writing about any creative activity is that in most "
        "cases the articles seem unable to rise above plain technical information.")
check("accepts an identical return", check_result(orig, orig, cfg)[0], True)
check("accepts a small correction",
      check_result(orig, orig.replace("rise", "risc"), cfg)[0], True)
check("rejects a summary",
      check_result(orig, "The author says technical detail is unhelpful.", cfg)[0], False)
check("rejects truncation", check_result(orig, orig[:50], cfg)[0], False)
check("rejects an empty answer", check_result(orig, "", cfg)[0], False)
check("rejects a meta response",
      check_result(orig, "Note: the text appears garbled. " + orig, cfg)[0], False)

check("unwraps preamble plus fence",
      strip_wrapper("Here is the corrected text:\n\n```\n" + orig + "\n```"), orig)
check("unwraps think block",
      strip_wrapper("<think>reasoning</think>\n" + orig), orig)
check("leaves plain text alone", strip_wrapper(orig), orig)

# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
if FAIL:
    print(f"  {PASS} passed, {FAIL} FAILED")
    for f in FAILURES:
        print(f"    - {f}")
    raise SystemExit(1)
print(f"  all {PASS} checks passed")
raise SystemExit(0)
