"""Column detection and reading order.

The failure this module exists to prevent: a two-column page read straight
across, so the first line of the left column is followed by the first line of
the right column.  The text is all there and every word is correct, and the
result is unreadable.

Good VLM engines already emit blocks in reading order.  We keep their order
when it looks sane and fall back to a recursive XY-cut when it does not, or
when the engine gives us no order at all (classic line-level OCR).
"""

from __future__ import annotations

from dataclasses import dataclass

from .doctypes import Block
from .util import LOG


@dataclass
class ColumnLayout:
    n_columns: int
    #: (x0, x1) span of each column in page-image pixels, left to right.
    spans: list[tuple[float, float]]
    confidence: float

    def column_of(self, block: Block) -> int:
        """Index of the column a block sits in; -1 if it spans several."""
        if self.n_columns <= 1:
            return 0
        hits = [
            i for i, (a, b) in enumerate(self.spans)
            if min(block.x1, b) - max(block.x0, a) > 0.5 * min(block.width, b - a)
        ]
        return hits[0] if len(hits) == 1 else -1


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_columns(
    blocks: list[Block],
    page_width: int,
    page_height: int,
    max_columns: int = 4,
    gap_ratio: float = 0.035,
) -> ColumnLayout:
    """Find column gutters from the horizontal distribution of text blocks."""
    text_blocks = [
        b for b in blocks
        if b.type in {"text", "list", "heading", "footnote"} and b.width > 0 and b.height > 0
    ]
    if len(text_blocks) < 3:
        return ColumnLayout(1, [(0.0, float(page_width))], 0.0)

    # Ignore blocks that clearly span the whole text area (titles, wide figures)
    # so they do not bridge the gutter and hide it.
    widths = sorted(b.width for b in text_blocks)
    median_w = widths[len(widths) // 2]
    body = [b for b in text_blocks if b.width <= median_w * 1.6] or text_blocks

    bins = 500
    scale = bins / float(page_width)
    mass = [0.0] * bins
    for b in body:
        lo = max(0, int(b.x0 * scale))
        hi = min(bins - 1, int(b.x1 * scale))
        weight = b.height
        for i in range(lo, hi + 1):
            mass[i] += weight

    total = sum(mass)
    if total <= 0:
        return ColumnLayout(1, [(0.0, float(page_width))], 0.0)

    # A gutter is "low", not "empty". Line-level OCR routinely emits a box or
    # two that straddles the gutter, and with a near-zero threshold a single
    # such box hides the gutter completely and the page reads as one column.
    peak = max(mass)
    threshold = peak * 0.15
    min_gap_bins = max(2, int(gap_ratio * bins))

    # Trim leading/trailing margins before looking for interior gutters.
    first = next((i for i, v in enumerate(mass) if v > threshold), 0)
    last = next((i for i in range(bins - 1, -1, -1) if mass[i] > threshold), bins - 1)

    gutters: list[tuple[int, int]] = []
    run_start = None
    for i in range(first, last + 1):
        if mass[i] <= threshold:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_gap_bins:
                gutters.append((run_start, i))
            run_start = None
    if run_start is not None and last - run_start >= min_gap_bins:
        gutters.append((run_start, last))

    # Widest gutters first, capped at max_columns - 1 of them.
    gutters.sort(key=lambda g: g[1] - g[0], reverse=True)
    gutters = sorted(gutters[: max_columns - 1])

    if not gutters:
        # Whitespace analysis found nothing. Before accepting "one column",
        # try the independent signal: in a multi-column layout the left edges
        # of the lines form one tight cluster per column.
        alt = _columns_from_left_edges(body, page_width, max_columns, gap_ratio)
        if alt is not None:
            return alt
        return ColumnLayout(1, [(0.0, float(page_width))], 0.6)

    spans: list[tuple[float, float]] = []
    cursor = first
    for g0, g1 in gutters:
        spans.append((cursor / scale, g0 / scale))
        cursor = g1
    spans.append((cursor / scale, (last + 1) / scale))

    # A column narrower than 12% of the page is almost certainly a mis-split
    # (a hanging indent, a marginal note). Before giving up on multi-column,
    # let the left-edge method have a go.
    if any((b - a) < 0.12 * page_width for a, b in spans):
        alt = _columns_from_left_edges(body, page_width, max_columns, gap_ratio)
        return alt or ColumnLayout(1, [(0.0, float(page_width))], 0.3)

    gutter_width = max(g[1] - g[0] for g in gutters) / scale
    confidence = min(1.0, gutter_width / (gap_ratio * page_width * 1.5))
    return ColumnLayout(len(spans), spans, round(confidence, 2))


def _kmeans_1d(values: list[float], k: int, iters: int = 25) -> list[list[float]]:
    """Deterministic 1-D k-means, seeded at evenly spaced quantiles."""
    vals = sorted(values)
    if len(vals) < k:
        return []
    centres = [vals[min(len(vals) - 1, int((i + 0.5) * len(vals) / k))] for i in range(k)]
    clusters: list[list[float]] = [[] for _ in range(k)]
    for _ in range(iters):
        clusters = [[] for _ in range(k)]
        for v in vals:
            best = min(range(k), key=lambda i: abs(v - centres[i]))
            clusters[best].append(v)
        moved = False
        for i, c in enumerate(clusters):
            if not c:
                continue
            new = sum(c) / len(c)
            if abs(new - centres[i]) > 1e-6:
                moved = True
            centres[i] = new
        if not moved:
            break
    return clusters


def _columns_from_left_edges(
    blocks: list[Block],
    page_width: int,
    max_columns: int,
    gap_ratio: float,
) -> ColumnLayout | None:
    """Recover columns by clustering line left edges.

    Independent of whitespace, so it survives the boxes that bridge a gutter
    and defeat the projection method.  Deliberately conservative: it only
    reports columns when the clusters are well populated, clearly separated,
    and the resulting spans do not overlap.
    """
    if len(blocks) < 8:
        return None

    lefts = [b.x0 for b in blocks]

    for k in range(max_columns, 1, -1):
        clusters = _kmeans_1d(lefts, k)
        if not clusters or any(len(c) < max(3, len(lefts) * 0.12) for c in clusters):
            continue
        centres = sorted(sum(c) / len(c) for c in clusters)
        # Columns have to be meaningfully far apart, or this is one column
        # with a hanging indent.
        if any(centres[i + 1] - centres[i] < 0.15 * page_width
               for i in range(len(centres) - 1)):
            continue

        assigned: list[list[Block]] = [[] for _ in centres]
        for b in blocks:
            idx = min(range(len(centres)), key=lambda i: abs(b.x0 - centres[i]))
            assigned[idx].append(b)
        if any(not g for g in assigned):
            continue

        # Percentiles, not min/max: a handful of lines legitimately run wide
        # (a heading, a table rule, a line the detector merged across the
        # gutter) and using the extremes would let one of them swallow the
        # neighbouring column and sink an otherwise correct split.
        spans: list[tuple[float, float]] = []
        for g in assigned:
            xs0 = sorted(b.x0 for b in g)
            xs1 = sorted(b.x1 for b in g)
            spans.append((_quantile(xs0, 0.05), _quantile(xs1, 0.90)))

        # Clip each column so it stops before the next one begins.
        for i in range(k - 1):
            boundary = (spans[i + 1][0] + spans[i][1]) / 2.0
            if spans[i][1] > spans[i + 1][0]:
                spans[i] = (spans[i][0], boundary)
                spans[i + 1] = (boundary, spans[i + 1][1])

        if any((hi - lo) < 0.12 * page_width for lo, hi in spans):
            continue
        if any(spans[i][1] > spans[i + 1][0] + 1e-6 for i in range(k - 1)):
            continue

        # Sanity check the split actually explains the page: most lines should
        # sit inside exactly one column.
        singles = 0
        for b in blocks:
            hits = sum(1 for lo, hi in spans
                       if min(b.x1, hi) - max(b.x0, lo) > 0.5 * min(b.width, hi - lo))
            if hits == 1:
                singles += 1
        if singles < 0.7 * len(blocks):
            continue

        LOG.debug(f"    columns recovered from line left edges: {k}")
        return ColumnLayout(k, spans, 0.55)

    return None


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, idx))]


# ---------------------------------------------------------------------------
# Recursive XY-cut
# ---------------------------------------------------------------------------

def xy_cut(
    blocks: list[Block],
    page_width: int,
    page_height: int,
    min_gap_x: float | None = None,
    min_gap_y: float | None = None,
    depth: int = 0,
) -> list[Block]:
    """Classic recursive XY-cut, biased towards cutting columns before rows.

    At each level we look for the widest whitespace valley in both directions.
    A vertical valley (a gutter) is preferred when it is genuinely wide,
    because splitting columns first is what preserves reading order on a
    multi-column page.
    """
    if len(blocks) <= 1 or depth > 12:
        return list(blocks)

    min_gap_x = min_gap_x if min_gap_x is not None else page_width * 0.025
    min_gap_y = min_gap_y if min_gap_y is not None else page_height * 0.012

    v_groups = _split_axis(blocks, axis="x", min_gap=min_gap_x)
    h_groups = _split_axis(blocks, axis="y", min_gap=min_gap_y)

    v_gap = _largest_gap(blocks, "x")
    h_gap = _largest_gap(blocks, "y")

    # Prefer the vertical (column) cut when a real gutter exists; a gutter is
    # a much stronger structural signal than a paragraph gap.
    use_vertical = len(v_groups) > 1 and v_gap >= min_gap_x and v_gap >= h_gap * 0.6

    if use_vertical:
        groups = sorted(v_groups, key=lambda g: min(b.x0 for b in g))
    elif len(h_groups) > 1:
        groups = sorted(h_groups, key=lambda g: min(b.y0 for b in g))
    else:
        # Nothing separates them: fall back to top-to-bottom, left-to-right.
        return sorted(blocks, key=lambda b: (round(b.y0 / max(1.0, page_height * 0.01)), b.x0))

    out: list[Block] = []
    for g in groups:
        out.extend(xy_cut(g, page_width, page_height, min_gap_x, min_gap_y, depth + 1))
    return out


def _split_axis(blocks: list[Block], axis: str, min_gap: float) -> list[list[Block]]:
    """Split blocks into groups separated by a whitespace band on one axis."""
    if axis == "x":
        key_lo, key_hi = (lambda b: b.x0), (lambda b: b.x1)
    else:
        key_lo, key_hi = (lambda b: b.y0), (lambda b: b.y1)

    ordered = sorted(blocks, key=key_lo)
    groups: list[list[Block]] = [[ordered[0]]]
    reach = key_hi(ordered[0])
    for b in ordered[1:]:
        if key_lo(b) - reach > min_gap:
            groups.append([b])
            reach = key_hi(b)
        else:
            groups[-1].append(b)
            reach = max(reach, key_hi(b))
    return groups


def _largest_gap(blocks: list[Block], axis: str) -> float:
    if axis == "x":
        key_lo, key_hi = (lambda b: b.x0), (lambda b: b.x1)
    else:
        key_lo, key_hi = (lambda b: b.y0), (lambda b: b.y1)
    ordered = sorted(blocks, key=key_lo)
    reach = key_hi(ordered[0])
    best = 0.0
    for b in ordered[1:]:
        best = max(best, key_lo(b) - reach)
        reach = max(reach, key_hi(b))
    return best


# ---------------------------------------------------------------------------
# Validation of an engine-supplied order
# ---------------------------------------------------------------------------

def order_looks_wrong(blocks: list[Block], layout: ColumnLayout) -> tuple[bool, str]:
    """Detect the 'read straight across the columns' pathology.

    On a correctly ordered multi-column page you traverse one column fully,
    then jump back to the top of the next.  That produces exactly one large
    backwards jump in y per gutter.  Reading across instead produces a
    backwards jump on nearly every block.
    """
    if layout.n_columns < 2:
        return False, ""

    seq = [b for b in sorted(blocks, key=lambda b: b.order)
           if b.type in {"text", "list", "heading"} and b.height > 0]
    if len(seq) < 4:
        return False, ""

    cols = [layout.column_of(b) for b in seq]
    single = [(b, c) for b, c in zip(seq, cols) if c >= 0]
    if len(single) < 4:
        return False, ""

    # Count column switches: correct order switches once per gutter.
    switches = sum(1 for i in range(1, len(single)) if single[i][1] != single[i - 1][1])
    max_expected = layout.n_columns  # a little slack for interleaved figures
    if switches > max_expected + 1 and switches > len(single) * 0.4:
        return True, f"{switches} column switches across {len(single)} blocks"

    return False, ""


def resolve_reading_order(
    blocks: list[Block],
    page_width: int,
    page_height: int,
    prefer_engine_order: bool,
    validate: bool,
    max_columns: int = 4,
    gap_ratio: float = 0.035,
) -> tuple[list[Block], ColumnLayout, str]:
    """Return blocks renumbered into reading order, plus the layout found."""
    layout = detect_columns(blocks, page_width, page_height, max_columns, gap_ratio)

    engine_supplied = any(b.order for b in blocks)
    method = "engine"

    if prefer_engine_order and engine_supplied:
        ordered = sorted(blocks, key=lambda b: b.order)
        if validate:
            wrong, why = order_looks_wrong(ordered, layout)
            if wrong:
                LOG.debug(f"    engine reading order rejected ({why}); using XY-cut")
                ordered = xy_cut(blocks, page_width, page_height)
                method = "xy_cut(corrected)"
    else:
        ordered = xy_cut(blocks, page_width, page_height)
        method = "xy_cut"

    for i, b in enumerate(ordered):
        b.order = i
        b.column = layout.column_of(b)
    return ordered, layout, method
