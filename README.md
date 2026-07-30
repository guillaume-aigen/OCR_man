# OCR_man

**Scanned books → clean text. Fully offline, on your own GPU.**

[![Licence: AGPL v3](https://img.shields.io/badge/licence-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](#requirements)

Point it at image-only PDFs or EPUBs and it produces a clean Markdown file for
LLM use and a reflowable EPUB for reading. No cloud, no API keys, no page
limits, nothing leaves the machine.

Built for the hard cases: **badly scanned pages** and **multi-column layouts**
where the text runs down the left half of the page and continues on the right.

> **Just want to run it? → [START_HERE.md](START_HERE.md).** This file is the
> technical reference.

---

## Measured results

Verified against a two-column encyclopedia scan and a 52-page illustrated book
on an RTX 4090:

- **≥99.8% word accuracy** — 3 random pages audited against the source images,
  ~1,450 words, zero errors. Drop caps merged correctly, line-break hyphens
  rejoined, accents and typography preserved — including a typo in the original
  book, left uncorrected rather than "fixed".
- **~5 s/page** end to end, with no page limit.
- **Zero text loss or duplication** across page boundaries, checked over 736
  sentences.

---

## Outputs

| Output | For | What it is |
|---|---|---|
| `<name>.md` | LLMs | Clean full text with headings, page anchors and no page furniture |
| `<name>.epub` | Humans | Reflowable book with a table of contents and embedded figures |
| `<name>_searchable.pdf` | Humans | The scan itself, with an invisible text layer so it is searchable and copy-pasteable. Off by default; enable in `config.yaml`. |

---

## How long it takes

Measured on an RTX 4090 against a 98-page two-column encyclopedia scan:

| Stage | Rate | 300-page book | 1000-page book |
|---|---|---|---|
| Render + restore | ~0.05 s/page | ~15 s | ~50 s |
| OCR (MinerU) | ~4.9 s/page | ~25 min | ~80 min |
| LLM correction (qwen3.6:35b) | ~8.6 s/page | ~45 min | ~2h20m |

Add ~60 s of one-time model loading per run.

**The LLM pass costs more than the OCR itself**, and on clean OCR output it
often changes nothing: over the 98-page benchmark it accepted all 98 chunks
unchanged, having found nothing to correct. It earns its time on genuinely
degraded scans — it repaired `wnting`, `artides`, `thmk` and `sIightest` on
damaged text — but it is not free and not always useful. Judge it on your own
material with `--pages 20`, and run the book with `--no-llm` if the raw OCR is
already clean. A smaller model (`--llm-model mistral-small3.2:24b`) is
substantially faster if you want the pass without the cost.

Interrupting is safe — rerun and it resumes from the cache rather than
restarting the book.

## Quick start

```bash
python SETUP.py
```

One time only. Creates `.venv`, installs PyTorch with CUDA and the OCR engine,
downloads ~5 GB of model weights and runs a self-test. Then:

1. Drop your scanned PDFs / EPUBs into `INPUT/`
2. Run it:

```bash
.venv\Scripts\python.exe RUN_ME.py
```

3. Collect the results from `OUTPUT/`

---

## What makes it work on bad scans

**It ignores the PDF's existing text layer.** Scans from mass-digitisation
projects almost always ship an OCR layer, and it is almost always bad — one
word per line, `CONTENTS` read as `CONTEXTS`, hyphens left dangling. Tools that
trust it produce garbage. This pipeline re-reads the pixels every time. Page
images are wrapped into a fresh text-free PDF before the engine sees them, so a
junk layer physically cannot leak into the output.

**Multi-column pages are read in the right order.** A two-column page read
straight across gives you every word, correctly recognised, in an unreadable
order. The OCR engine is layout-aware, and its reading order is then checked
against an independent column analysis: if the blocks zig-zag between columns,
the order is rebuilt with a recursive XY-cut. A paragraph that runs off the
bottom of the left column and continues at the top of the right is rejoined
into one paragraph, and so is one that continues onto the next page.

Columns are found two different ways, because either one alone fails on real
books. The first looks for a whitespace valley in the horizontal ink
distribution. That misses pages where the columns sit flush against each other
or where the text detector pads its boxes across the gutter — on the
encyclopedia sample the ink profile is solid all the way across, with no gap to
find. So when it finds nothing, the left edges of the lines are clustered
instead: two columns produce two tight clusters of left edges no matter how
narrow the gutter is. Spans come from percentiles rather than extremes, so the
occasional heading or merged line running the full page width cannot swallow a
column and sink the split.

**Damaged scans are restored before recognition.** Each page is scored for
skew, contrast, uneven lighting, speckle and scanner border, and only the pages
that need it are corrected — deskewed by projection-profile analysis,
background-flattened, contrast-equalised, despeckled and trimmed.

**Page furniture is stripped.** Running heads, running feet and folios repeat
across pages; they are detected by repetition and removed from the prose, but
the folios are kept as `<!-- page 61 -->` anchors in the Markdown so you can
trace any passage back to the physical page.

**Words broken across lines are rejoined**, ligatures are normalised, and
headings are given levels so the EPUB gets a usable table of contents.

**An optional local LLM pass** repairs what is left. It is guarded: every
corrected chunk is compared against the original, and anything that drifted too
far in length, token similarity or digit count is thrown away and the original
OCR text kept. The model can fix `rn`→`m`; it cannot quietly rewrite your book.

---

## Typical run

```
[1/2] Sample-02.pdf
  Sample-02.pdf: 7 pages, image scan, native 300 DPI, text-layer quality 0.31
    ignoring the embedded text layer (quality 0.31) and re-reading the pixels
  restore done: 7/7 in 4.2s  3 page(s) restored, median scan quality 0.81
  OCR engine: mineru (MinerU 3.x layout+VLM, best multi-column reading order)
  OCR[mineru] done: 7/7 in 1m14s
  LLM cleanup with qwen3.6:35b
  cleanup done: 9/9  6 corrected, 2 unchanged, 1 rejected (diverged=1)
  wrote Sample-02.md (34.1KB)
  wrote Sample-02.epub (98.7KB)
  wrote Sample-02_searchable.pdf (4.2MB)
```

Interrupt it at any time. Every stage caches into `WORK/`, so rerunning picks up
where it stopped instead of starting the book again. Use `--fresh` to redo
everything.

---

## Options

```
python RUN_ME.py [files...]

  --pages N            only the first N pages (quick quality check)
  --no-llm             skip the LLM correction pass
  --llm-model NAME     use a different Ollama model
  --no-preprocess      skip scan restoration
  --force-preprocess   restore every page, not just damaged ones
  --only md|epub|pdf   limit which outputs are written (repeatable)
  --engine NAME        force an OCR engine
  --list-engines       show what is installed
  --fresh              ignore cached results
  --input-dir / --output-dir
```

Everything else lives in `config.yaml`, which is commented. Any setting can also
be overridden per run with an environment variable — `OCRMAN_LLM__ENABLED=0`,
`OCRMAN_RUN__MAX_PAGES=20`.

**Check quality before committing to a long book:**

```bash
.venv\Scripts\python.exe RUN_ME.py --pages 20
```

**After changing anything in `ocr_man/`**, run the self-checks. They cover
column detection, reading order, stitching, de-hyphenation, EPUB parsing and
the LLM guards in about a second, with no GPU:

```bash
.venv\Scripts\python.exe TEST.py
```

---

## Engines

| Engine | Speed | Notes |
|---|---|---|
| `mineru` | ~5 s/page | Default. Layout model + VLM recogniser. Multi-column reading order, tables, formulas, figure description. Needs the GPU. |
| `rapidocr` | ~1 s/page | Fallback. ONNX line recogniser with no layout model, so column analysis and reading order are done in this pipeline instead. Runs on CPU. Noticeably lower text accuracy — use it when MinerU is unavailable, not by preference. |

Pages that come back nearly empty are automatically retried on the next engine
in the list, and whichever result has more text wins. That only kicks in when
enough pages look wrong to suggest a systematic problem, so genuine blank leaves
and full-page plates are left alone.

---

## Requirements

- NVIDIA GPU, 8 GB VRAM or more (built and tested on an RTX 4090)
- Python 3.10–3.12
- ~15 GB disk for the environment and model weights
- Optional: [Ollama](https://ollama.com) running locally for the correction pass

CPU-only works (`python SETUP.py --cpu`) but is roughly 20× slower.

---

## Layout

```
INPUT/          put source documents here
OUTPUT/         .md, .epub, _searchable.pdf, and images/
WORK/           per-document cache; delete it to reclaim disk
config.yaml     settings
RUN_ME.py       entry point
SETUP.py        one-time installer
TEST.py         fast self-checks, no GPU needed
ocr_man/        the pipeline
  ingest.py         PDF/EPUB/image -> page images, text-layer quality probe
  preprocess.py     scan restoration (deskew, illumination, despeckle)
  engines/          OCR back-ends behind one interface
  reading_order.py  column detection, XY-cut, engine-order validation
  assemble.py       furniture removal, de-hyphenation, paragraph stitching
  htmltext.py       EPUB chapter XHTML -> structured elements
  llm_clean.py      guarded local-LLM correction pass
  pdfbuild.py       normalised PDF in, searchable PDF out
  exporters/        Markdown and EPUB3 writers
```

## Input types

| Input | Handling |
|---|---|
| Image-only PDF | Rendered at the scan's own DPI, restored, OCR'd. The main case. |
| PDF with a bad OCR layer | Same — the existing text layer is ignored entirely. |
| Born-digital PDF | Text extracted directly, no OCR (set `render.force_ocr: false`). |
| Image-scan EPUB | Page images pulled from the spine in order, then OCR'd. |
| Real-text EPUB | Converted directly with headings and paragraphs preserved; never OCR'd, since there are no pixels to re-read. |
| Loose page images | A folder of images is treated as one document, ordered naturally. Multi-page TIFFs work too. |

---

## Notes and limits

- **The LLM pass can make a plausible wrong guess.** The guard catches
  rewrites, summaries, truncation and invented content — it cannot catch a
  single word swapped for another word of the same shape. On the test corpus
  it correctly repaired `wnting`→`writing`, `artides`→`articles`,
  `thmk`→`think` and `sIightest`→`slightest`, but turned the unusual
  `automobile-devounng` into `automobile-driving` instead of `devouring`.
  Run with `--no-llm` if you need the OCR output untouched; the raw version is
  always kept in `WORK/<doc>/elements_raw.json` either way.
- **The OCR model occasionally hallucinates text on decorative artwork** —
  an ornamental header band produced "The quick brown fox jumps over the lazy
  dog". Such regions are detected as page furniture and stripped from the
  Markdown and EPUB, but they do land in the searchable PDF's hidden layer.
- **Searchable-PDF text alignment is block-level.** MinerU returns column
  segments rather than visual lines, so the hidden text is wrapped to fill each
  block's box. Search and copy-paste are exact; a click-drag selection lands in
  the right region but does not track individual word boundaries.
- The invisible layer embeds a subset of a system font (Arial on Windows,
  DejaVu on Linux) for Unicode coverage beyond Latin-1. Point
  `output.searchable_pdf_font` at any TTF to override it.
- **The searchable PDF is rebuilt from the restored page images**, so it is
  deskewed and usually much smaller than the original — but it is not a
  byte-identical copy of your source file. Keep the original if you need one.
- **Handwriting** is not a target. The engine will attempt it and the results
  are unreliable.
- On Windows, `SETUP.py` writes a `sitecustomize.py` into the venv to work
  around a `huggingface_hub` symlink race that otherwise aborts the model
  download with `WinError 1314`. It is inert unless `OCRMAN_HF_NO_SYMLINK=1` is
  set, which only this pipeline does.

---

## Licence

AGPL-3.0 — see [LICENSE](LICENSE).

This project depends on PyMuPDF, which is AGPL-3.0 unless you hold a
commercial licence from Artifex, so a more permissive licence here would not be
compatible. [NOTICE.md](NOTICE.md) lists every third-party component, its role
and its licence, and explains how to lift that constraint if you need to.

Recognition is done by [MinerU](https://github.com/opendatalab/MinerU) and
[RapidOCR](https://github.com/RapidAI/RapidOCR). What this project adds is
everything around them: ingest and scan restoration, column detection and
reading-order repair, cross-page document assembly, a guarded local-LLM
correction pass, and the exporters.

No documents, model weights or scans are distributed with this repository.
