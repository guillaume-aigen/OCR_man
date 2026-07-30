# Third-party components

OCR_man is a pipeline built on existing open-source models and libraries. It
contributes the ingest, scan restoration, column/reading-order analysis,
document assembly, guarded LLM correction and export stages; recognition
itself is done by the engines below.

| Component | Role | Licence |
|---|---|---|
| [MinerU](https://github.com/opendatalab/MinerU) | Default OCR engine (layout model + VLM recogniser) | MinerU Open Source License |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | PDF rendering and writing | AGPL-3.0 or commercial (Artifex) |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) | Fallback line recogniser | Apache-2.0 |
| [PyTorch](https://pytorch.org) | Model runtime | BSD-3-Clause |
| [OpenCV](https://opencv.org) | Image restoration | Apache-2.0 |
| [Pillow](https://python-pillow.org) | Image I/O | MIT-CMU |

## Why this project is AGPL-3.0

`ocr_man` imports PyMuPDF directly, and PyMuPDF is AGPL-3.0 unless you hold a
commercial licence from Artifex. A more permissive licence on this code would
not be compatible, so the project as a whole is AGPL-3.0.

If you need a permissive licence, the PDF layer (`ocr_man/pdfbuild.py`, plus
the PDF paths in `ocr_man/ingest.py`) is the only part that depends on
PyMuPDF; replacing it with `pypdfium2` (BSD/Apache) would remove the
constraint.

MinerU is invoked as a separate process through its command-line interface and
is not linked into this codebase. It is a dependency you install, not code
distributed here.

## Models

Model weights are downloaded at setup time from their upstream repositories and
are covered by their own licences. No weights are distributed with this
project.
