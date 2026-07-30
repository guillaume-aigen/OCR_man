#!/usr/bin/env python
"""OCR_man - put scanned books in INPUT/, run this, collect OUTPUT/.

    python RUN_ME.py

For every document in INPUT/ this writes three files to OUTPUT/:

    <name>.md                 clean full text, for LLM consumption
    <name>.epub               reflowable book, for reading
    <name>_searchable.pdf     the original scan with an invisible text layer

Run `python RUN_ME.py --help` for the options.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _check_interpreter() -> None:
    """Make sure we are running inside the project venv, not bare Python."""
    venv_py = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    try:
        import pymupdf  # noqa: F401
        return
    except ImportError:
        pass
    msg = ["OCR_man's dependencies are not available in this interpreter.", ""]
    if venv_py.exists():
        msg += ["Run it with the project environment instead:", "",
                f'    "{venv_py}" RUN_ME.py', ""]
    else:
        msg += ["Set the environment up first:", "", "    python SETUP.py", ""]
    sys.exit("\n".join(msg))


_check_interpreter()

from ocr_man.config import load_config  # noqa: E402
from ocr_man.engines.base import all_engine_classes, select_engine  # noqa: E402
from ocr_man.ingest import discover_inputs, group_inputs  # noqa: E402
from ocr_man.pipeline import process_document  # noqa: E402
from ocr_man.util import (  # noqa: E402
    LOG, banner, c, fmt_duration, human_size, setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="RUN_ME.py",
        description="OCR scanned PDFs and EPUBs into Markdown, EPUB and searchable PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Settings live in config.yaml; the flags below override it for one run.",
    )
    p.add_argument("inputs", nargs="*", type=Path,
                   help="specific files to process (default: everything in INPUT/)")
    p.add_argument("--input-dir", type=Path, help="override the INPUT directory")
    p.add_argument("--output-dir", type=Path, help="override the OUTPUT directory")
    p.add_argument("--engine", help="force an OCR engine (default: best available)")
    p.add_argument("--list-engines", action="store_true", help="show engines and exit")
    p.add_argument("--pages", type=int, metavar="N",
                   help="only process the first N pages (quick quality check)")
    p.add_argument("--no-llm", action="store_true", help="skip the LLM cleanup pass")
    p.add_argument("--llm-model", help="Ollama model for the cleanup pass")
    p.add_argument("--no-preprocess", action="store_true",
                   help="skip scan restoration (deskew, illumination, despeckle)")
    p.add_argument("--force-preprocess", action="store_true",
                   help="restore every page, not just the ones that look damaged")
    p.add_argument("--fresh", action="store_true",
                   help="ignore cached intermediate results and redo everything")
    p.add_argument("--only", choices=["md", "epub", "pdf"], action="append",
                   help="limit the outputs written (repeatable)")
    p.add_argument("--quiet", action="store_true", help="less console output")
    p.add_argument("--config", type=Path, help="path to an alternative config.yaml")
    return p


def apply_overrides(cfg, args) -> None:
    if args.input_dir:
        cfg.set("paths.input", str(args.input_dir))
    if args.output_dir:
        cfg.set("paths.output", str(args.output_dir))
    if args.engine:
        cfg.set("ocr.engine", args.engine)
    if args.pages:
        cfg.set("run.max_pages", args.pages)
    if args.no_llm:
        cfg.set("llm.enabled", False)
    if args.llm_model:
        cfg.set("llm.model", args.llm_model)
    if args.no_preprocess:
        cfg.set("preprocess.enabled", False)
    if args.force_preprocess:
        cfg.set("preprocess.mode", "always")
    if args.fresh:
        cfg.set("run.resume", False)
    if args.quiet:
        cfg.set("run.verbose", False)
    if args.only:
        cfg.set("output.markdown", "md" in args.only)
        cfg.set("output.epub", "epub" in args.only)
        cfg.set("output.searchable_pdf", "pdf" in args.only)


def list_engines(cfg) -> int:
    print("\nOCR engines:\n")
    for name, cls in all_engine_classes().items():
        try:
            ok, why = cls.check_available(cfg)
        except Exception as exc:
            ok, why = False, f"probe failed: {exc}"
        mark = c("available", "green") if ok else c("unavailable", "yellow")
        print(f"  {name:<12} {mark}   {cls.description}")
        if not ok:
            print(f"  {'':<12} {c(why, 'grey')}")
    print(f"\nPreference order: {' -> '.join(cfg.get('ocr.engines', []))}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, root=ROOT)
    apply_overrides(cfg, args)

    setup_logging(
        verbose=cfg.get("run.verbose", True),
        logfile=cfg.work_dir / "ocr_man.log",
    )

    if args.list_engines:
        return list_engines(cfg)

    banner("OCR_man")

    input_dir = cfg.input_dir
    if args.inputs:
        files = []
        for p in args.inputs:
            p = p if p.is_absolute() else (Path.cwd() / p)
            if not p.exists():
                LOG.error(f"no such file: {p}")
                return 2
            files.append(p)
        groups = group_inputs(files, input_dir)
    else:
        input_dir.mkdir(parents=True, exist_ok=True)
        found = discover_inputs(input_dir)
        if not found:
            LOG.warning(f"nothing to do: {input_dir} is empty")
            LOG.info("Put your scanned PDFs or EPUBs in there and run this again.")
            # Distinct code so RUN.bat can tell "nothing to do" apart from
            # "finished successfully" and not claim there are results.
            return 4
        groups = group_inputs(found, input_dir)

    total_bytes = sum(f.stat().st_size for g in groups for f in g)
    LOG.info(f"{len(groups)} document(s) to process, {human_size(total_bytes)} total")
    LOG.info(f"output -> {cfg.output_dir}")

    # One engine instance for the whole run: model load is the expensive part.
    engine = None
    if any(True for _ in groups):
        try:
            engine = select_engine(cfg)
        except SystemExit as exc:
            LOG.error(str(exc))
            return 3

    t0 = time.perf_counter()
    results = []
    for i, group in enumerate(groups, 1):
        name = group[0].name if len(group) == 1 else f"{group[0].parent.name}/ ({len(group)} images)"
        LOG.info("")
        LOG.info(c(f"[{i}/{len(groups)}] {name}", "bold"))
        results.append(process_document(cfg, group, engine=engine))

    if engine:
        engine.close()

    elapsed = time.perf_counter() - t0
    _report(results, elapsed, cfg)
    return 0 if all(r.ok for r in results) else 1


def _report(results, elapsed: float, cfg) -> None:
    LOG.info("")
    banner("Summary")
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    pages = sum(r.pages for r in ok)

    for r in results:
        if r.ok:
            kinds = ", ".join(sorted(r.outputs))
            per_page = f"{r.seconds / r.pages:.1f}s/page" if r.pages else "-"
            LOG.info(f"  {c('OK', 'green')}   {r.stem}: {r.pages} pages, "
                     f"{fmt_duration(r.seconds)} ({per_page}) -> {kinds}")
            if r.stats.get("llm_cleanup"):
                LOG.info(f"       LLM cleanup: {r.stats['llm_cleanup']}")
        else:
            LOG.info(f"  {c('FAIL', 'red')} {r.stem}: {r.error}")

    LOG.info("")
    LOG.info(f"  {len(ok)} succeeded, {len(bad)} failed, {pages} pages in {fmt_duration(elapsed)}")
    LOG.info(f"  outputs in {cfg.output_dir}")
    if bad:
        LOG.info(f"  see {cfg.work_dir / 'ocr_man.log'} for details")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted - rerun to resume from where it stopped")
        raise SystemExit(130)
