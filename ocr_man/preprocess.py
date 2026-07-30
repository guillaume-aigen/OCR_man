"""Scan restoration.

Badly scanned book pages fail OCR for a handful of recurring reasons: the
page is rotated a degree or two, the gutter is a black wedge, the paper is
lit unevenly so one side washes out, and the JPEG is speckled.  Each of
those is cheap to fix and each one costs real accuracy if left alone.

Everything here is deliberately conservative.  Document VLMs are trained on
photographs of pages, not on hard-binarised bitmaps, so the goal is a clean
*greyscale* page, not a black-and-white one.  In `auto` mode a page is only
touched when its quality score says it needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .util import LOG


@dataclass
class PageQuality:
    """Per-page diagnostics used to decide what (if anything) to fix."""

    sharpness: float        # variance of Laplacian; < ~80 is soft/blurry
    contrast: float         # std-dev of intensity, 0..127ish
    skew_deg: float         # estimated rotation
    noise: float            # speckle estimate
    ink_ratio: float        # fraction of dark pixels
    illumination_range: float   # spread of local background level, 0..255
    border_ratio: float     # fraction of the frame that is scanner border

    @property
    def score(self) -> float:
        """0..1, higher is cleaner. Used only to decide whether to intervene."""
        s = 1.0
        s -= 0.30 * (1.0 - min(1.0, self.sharpness / 120.0))
        s -= 0.20 * (1.0 - min(1.0, self.contrast / 55.0))
        s -= 0.20 * min(1.0, abs(self.skew_deg) / 3.0)
        s -= 0.15 * min(1.0, self.noise / 12.0)
        s -= 0.15 * min(1.0, self.illumination_range / 70.0)
        return max(0.0, min(1.0, s))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 3)
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def _downscale(gray: np.ndarray, target_w: int = 1000) -> np.ndarray:
    if gray.shape[1] <= target_w:
        return gray
    scale = target_w / gray.shape[1]
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def estimate_skew(gray: np.ndarray, max_deg: float = 8.0, coarse: float = 0.5) -> float:
    """Projection-profile skew estimation.

    Rotating text back to horizontal maximises the variance of the horizontal
    ink-per-row profile: rows become either "all text" or "all gutter".  This
    beats Hough lines on book scans, which have few long straight edges.
    """
    small = _downscale(gray, 900)
    # Ink mask: Otsu on the inverted image so text is 1.
    _, binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary.astype(np.float32)
    if binary.sum() < 50:
        return 0.0

    h, w = binary.shape
    centre = (w / 2.0, h / 2.0)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1)
        return float(np.var(profile))

    best_angle, best = 0.0, score(0.0)
    a = -max_deg
    while a <= max_deg + 1e-9:
        s = score(a)
        if s > best:
            best, best_angle = s, a
        a += coarse

    # Refine around the coarse winner.
    fine = coarse / 5.0
    a = best_angle - coarse
    while a <= best_angle + coarse + 1e-9:
        s = score(a)
        if s > best:
            best, best_angle = s, a
        a += fine

    return round(float(best_angle), 2)


def _illumination_range(gray: np.ndarray) -> float:
    small = _downscale(gray, 400)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    lo, hi = np.percentile(bg, [5, 95])
    return float(hi - lo)


def _border_ratio(gray: np.ndarray) -> float:
    """Fraction of the frame taken up by near-black scanner border."""
    small = _downscale(gray, 500)
    dark = (small < 45).astype(np.uint8)
    h, w = dark.shape
    edge = np.zeros_like(dark)
    band = max(3, int(min(h, w) * 0.06))
    edge[:band, :] = 1
    edge[-band:, :] = 1
    edge[:, :band] = 1
    edge[:, -band:] = 1
    return float((dark & edge).sum() / max(1, edge.sum()))


def assess(gray: np.ndarray, max_deskew: float = 8.0, want_skew: bool = True) -> PageQuality:
    small = _downscale(gray, 1200)
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    contrast = float(small.std())
    noise = float(np.abs(small.astype(np.int16) - cv2.medianBlur(small, 3).astype(np.int16)).mean())
    _, binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float(binary.mean())
    return PageQuality(
        sharpness=sharpness,
        contrast=contrast,
        skew_deg=estimate_skew(gray, max_deskew) if want_skew else 0.0,
        noise=noise,
        ink_ratio=ink,
        illumination_range=_illumination_range(gray),
        border_ratio=_border_ratio(gray),
    )


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def crop_border(img: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Trim the black scanner surround / facing-page sliver around the paper."""
    h, w = gray.shape
    small = _downscale(gray, 700)
    sh, sw = small.shape
    fx, fy = w / sw, h / sh

    # Paper is bright; find the largest bright connected region.
    _, paper = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, 8)
    if n <= 1:
        return img, (0, 0, w, h)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, cw, ch, area = stats[idx]
    if area < 0.35 * sh * sw:
        return img, (0, 0, w, h)

    pad = 4
    x0 = max(0, int(x * fx) - pad)
    y0 = max(0, int(y * fy) - pad)
    x1 = min(w, int((x + cw) * fx) + pad)
    y1 = min(h, int((y + ch) * fy) + pad)
    # Refuse to crop away more than a third of the page; that means we
    # mis-detected the paper and would be eating text.
    if (x1 - x0) * (y1 - y0) < 0.6 * w * h:
        return img, (0, 0, w, h)
    if x0 == 0 and y0 == 0 and x1 == w and y1 == h:
        return img, (0, 0, w, h)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the page back to horizontal.

    Edges are replicated rather than filled, so straightening never
    introduces a black wedge that a later border-crop would chase.
    """
    if abs(angle) < 0.12:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def correct_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven page lighting by dividing out an estimated background."""
    # Background = heavily blurred morphological closing (removes the text).
    k = max(15, (min(gray.shape) // 40) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0)
    bg = np.maximum(bg, 1).astype(np.float32)
    flat = (gray.astype(np.float32) / bg) * 220.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip: float = 2.0) -> np.ndarray:
    tile = max(4, min(gray.shape) // 200)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def despeckle(gray: np.ndarray) -> np.ndarray:
    """Remove isolated specks without softening glyph edges."""
    return cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)


# ---------------------------------------------------------------------------

def enhance_page(
    image_path: Path,
    cfg: Config,
    out_path: Path | None = None,
) -> tuple[Path, PageQuality, dict]:
    """Assess a page image and, if it needs it, write a cleaned-up copy.

    Returns (path_to_use, quality, actions_taken).
    """
    pre = cfg.get("preprocess", {})
    mode = pre.get("mode", "auto")
    if not pre.get("enabled", True) or mode == "never":
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        q = assess(img, want_skew=False) if img is not None else _null_quality()
        return image_path, q, {}

    raw = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        LOG.warning(f"    could not read {image_path.name}; using it as-is")
        return image_path, _null_quality(), {}

    q = assess(img, max_deskew=float(pre.get("max_deskew_deg", 8.0)),
               want_skew=bool(pre.get("deskew", True)))

    actions: dict = {}
    always = mode == "always"
    work = img

    if pre.get("crop_scan_border", True) and (always or q.border_ratio > 0.12):
        work, box = crop_border(work, work)
        if box != (0, 0, img.shape[1], img.shape[0]):
            actions["crop"] = list(box)

    if pre.get("deskew", True) and abs(q.skew_deg) >= (0.12 if always else 0.30):
        work = deskew(work, q.skew_deg)
        actions["deskew_deg"] = q.skew_deg

    if pre.get("illumination_correction", True) and (always or q.illumination_range > 38):
        work = correct_illumination(work)
        actions["illumination"] = round(q.illumination_range, 1)

    if pre.get("clahe", True) and (always or q.contrast < 48):
        work = apply_clahe(work)
        actions["clahe"] = round(q.contrast, 1)

    if pre.get("despeckle", True) and (always or q.noise > 6.5):
        work = despeckle(work)
        actions["despeckle"] = round(q.noise, 2)

    if pre.get("binarize", False):
        work = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 12)
        actions["binarize"] = True

    if not actions:
        return image_path, q, {}

    out_path = out_path or image_path.with_name(image_path.stem + "_clean.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        return image_path, q, {}
    buf.tofile(str(out_path))
    return out_path, q, actions


def _null_quality() -> PageQuality:
    return PageQuality(999.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# Module-level worker so it can be used from a ProcessPoolExecutor on Windows.
def preprocess_worker(args: tuple) -> dict:
    from .config import Config as _C

    image_path, out_path, cfg_data = args
    cfg = _C(cfg_data)
    path, q, actions = enhance_page(Path(image_path), cfg, Path(out_path) if out_path else None)
    return {"path": str(path), "quality": q.to_dict(), "actions": actions}
