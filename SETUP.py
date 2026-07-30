#!/usr/bin/env python
"""One-time environment setup for OCR_man.

    python SETUP.py

Creates .venv, installs PyTorch with CUDA, installs the OCR engine, applies a
Windows-specific HuggingFace workaround, and pre-downloads the models so the
first real run does not stall on a 5 GB download.

Safe to re-run: every step checks whether it is already done.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = sys.platform == "win32"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")

#: CUDA wheel index. cu124 covers every RTX 30xx/40xx/50xx card.
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

CORE_PACKAGES = ["pymupdf", "pillow", "opencv-python", "numpy", "pyyaml"]
ENGINE_PACKAGES = ["mineru[core]"]
FALLBACK_PACKAGES = ["rapidocr", "onnxruntime"]

GREEN, YELLOW, RED, GREY, BOLD, RESET = (
    ("\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[1m", "\033[0m")
    if sys.stdout.isatty() else ("", "", "", "", "", "")
)


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n{BOLD}==> {msg}{RESET}", flush=True)


def ok(msg: str) -> None:
    say(f"{GREEN}OK{RESET}   {msg}")


def warn(msg: str) -> None:
    say(f"{YELLOW}!{RESET}    {msg}")


def fail(msg: str) -> None:
    say(f"{RED}X{RESET}    {msg}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    say(f"{GREY}$ {' '.join(str(c) for c in cmd[:6])}{'...' if len(cmd) > 6 else ''}{RESET}")
    return subprocess.run([str(c) for c in cmd], **kw)


def pip(*args: str, quiet: bool = False) -> int:
    cmd = [VENV_PY, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd += list(args)
    return run(cmd).returncode


def venv_has(module: str) -> bool:
    r = subprocess.run([str(VENV_PY), "-c", f"import {module}"], capture_output=True)
    return r.returncode == 0


# ---------------------------------------------------------------------------

def create_venv() -> None:
    step("Python environment")
    if VENV_PY.exists():
        ver = subprocess.run([str(VENV_PY), "-c",
                              "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
                             capture_output=True, text=True).stdout.strip()
        ok(f".venv already exists (Python {ver})")
        return

    if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
        warn(f"running Python {sys.version_info.major}.{sys.version_info.minor}; "
             "the OCR engine supports 3.10-3.12 best")
    say("creating .venv ...")
    r = run([sys.executable, "-m", "venv", str(VENV)])
    if r.returncode != 0 or not VENV_PY.exists():
        sys.exit("could not create .venv")
    run([VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    ok("created .venv")


def install_torch(cpu_only: bool = False) -> None:
    step("PyTorch")
    if venv_has("torch"):
        info = subprocess.run(
            [str(VENV_PY), "-c",
             "import torch,json;print(json.dumps({'v':torch.__version__,"
             "'cuda':torch.cuda.is_available(),"
             "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}))"],
            capture_output=True, text=True).stdout.strip()
        try:
            d = json.loads(info)
            if d["cuda"]:
                ok(f"torch {d['v']} with CUDA on {d['name']}")
                return
            if cpu_only:
                ok(f"torch {d['v']} (CPU build, as requested)")
                return
            warn(f"torch {d['v']} installed but CUDA is not available; reinstalling CUDA build")
        except Exception:
            pass

    args = ["torch", "torchvision"]
    if not cpu_only:
        args += ["--index-url", TORCH_INDEX]
    if pip(*args) != 0:
        sys.exit("torch installation failed")

    r = subprocess.run([str(VENV_PY), "-c",
                        "import torch;print(torch.cuda.is_available())"],
                       capture_output=True, text=True)
    if "True" in r.stdout:
        ok("torch installed, CUDA available")
    else:
        warn("torch installed but CUDA is NOT available - OCR will run on CPU and be very slow")


def install_packages() -> None:
    step("Core packages")
    missing = [p for p, m in
               [("pymupdf", "pymupdf"), ("pillow", "PIL"), ("opencv-python", "cv2"),
                ("numpy", "numpy"), ("pyyaml", "yaml")]
               if not venv_has(m)]
    if missing:
        if pip(*missing) != 0:
            sys.exit("core package installation failed")
    ok("pymupdf, pillow, opencv, numpy, pyyaml")

    step("OCR engine (MinerU)")
    if venv_has("mineru"):
        ok("mineru already installed")
    else:
        say("this pulls in a few hundred MB, give it a minute ...")
        if pip(*ENGINE_PACKAGES) != 0:
            fail("mineru installation failed - the RapidOCR fallback will be used instead")
        else:
            ok("mineru installed")
        # MinerU can pull a CPU-only torch over the CUDA build; put it back.
        r = subprocess.run([str(VENV_PY), "-c", "import torch;print(torch.cuda.is_available())"],
                           capture_output=True, text=True)
        if "True" not in r.stdout:
            warn("CUDA disappeared after installing mineru; reinstalling the CUDA torch build")
            pip("--force-reinstall", "torch", "torchvision", "--index-url", TORCH_INDEX)

    step("Fallback engine (RapidOCR)")
    if venv_has("rapidocr"):
        ok("rapidocr already installed")
    elif pip(*FALLBACK_PACKAGES, quiet=True) == 0:
        ok("rapidocr installed")
    else:
        warn("rapidocr not installed (optional; only used if MinerU fails on a page)")


def install_windows_hf_fix() -> None:
    """Force huggingface_hub to copy instead of symlink.

    Windows refuses symlink creation without Developer Mode, and
    huggingface_hub's own detection has a race that lets one download thread
    try anyway, aborting the model download with WinError 1314.  This makes
    the copy path unconditional for processes that set OCRMAN_HF_NO_SYMLINK,
    which the pipeline does when it launches the engine.
    """
    if not IS_WIN:
        return
    step("Windows HuggingFace symlink workaround")
    site = VENV / "Lib" / "site-packages"
    if not site.exists():
        warn("site-packages not found; skipping")
        return
    target = site / "sitecustomize.py"
    marker = "OCRMAN_HF_NO_SYMLINK"
    if target.exists() and marker in target.read_text(encoding="utf-8", errors="replace"):
        ok("already applied")
        return
    if target.exists():
        warn(f"{target} already exists and is not ours; appending")
        content = target.read_text(encoding="utf-8", errors="replace") + "\n\n"
    else:
        content = ""
    content += (
        '# --- added by OCR_man/SETUP.py ---\n'
        '# Windows blocks symlinks without Developer Mode. huggingface_hub can cope,\n'
        '# but its support probe optimistically caches "supported" before testing, so\n'
        '# a parallel download can still call os.symlink and die with WinError 1314.\n'
        'import os as _os\n'
        'if _os.environ.get("OCRMAN_HF_NO_SYMLINK") == "1":\n'
        '    try:\n'
        '        import huggingface_hub.file_download as _fd\n'
        '        _fd.are_symlinks_supported = lambda cache_dir=None: False\n'
        '    except Exception:\n'
        '        pass\n'
    )
    target.write_text(content, encoding="utf-8")
    ok(f"wrote {target.name}")


def predownload_models(skip: bool = False) -> None:
    step("OCR models")
    if skip:
        warn("skipped (--no-models); they will download on the first run")
        return
    if not venv_has("mineru"):
        warn("mineru not installed; nothing to pre-download")
        return

    sample = ROOT / "INPUT"
    sample.mkdir(exist_ok=True)
    say("downloading model weights (~5 GB, one time) and running a self-test ...")

    env = os.environ.copy()
    env["OCRMAN_HF_NO_SYMLINK"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env.setdefault("MINERU_MODEL_SOURCE", "huggingface")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from pathlib import Path\n"
        "import tempfile, pymupdf\n"
        "from ocr_man.config import load_config\n"
        "from ocr_man.engines.mineru_engine import MinerUEngine\n"
        "from ocr_man.engines.base import PageJob\n"
        "tmp = Path(tempfile.mkdtemp())\n"
        "doc = pymupdf.open(); page = doc.new_page(width=612, height=792)\n"
        "page.insert_text((72, 200), 'OCR_man self test', fontsize=28)\n"
        "page.insert_text((72, 260), 'The quick brown fox jumps over the lazy dog.', fontsize=16)\n"
        "pix = page.get_pixmap(dpi=200); img = tmp / 'p.jpg'; pix.save(img)\n"
        "doc.close()\n"
        "cfg = load_config(root=Path(r'%s'))\n"
        "eng = MinerUEngine(cfg)\n"
        "job = PageJob(index=0, image_path=img, width=pix.width, height=pix.height,"
        " source_page=0, dpi=200.0)\n"
        "pages = list(eng.run([job]))\n"
        "text = pages[0].plain_text() if pages else ''\n"
        "print('SELFTEST_TEXT:' + text.replace(chr(10), ' ')[:120])\n"
    ) % (str(ROOT), str(ROOT))

    r = subprocess.run([str(VENV_PY), "-c", code], env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    line = next((l for l in out.splitlines() if l.startswith("SELFTEST_TEXT:")), "")
    if r.returncode == 0 and "fox" in line.lower():
        ok("models downloaded and the engine round-trips correctly")
        say(f"{GREY}   recognised: {line.split(':', 1)[1].strip()[:80]}{RESET}")
    elif r.returncode == 0:
        warn("engine ran but the self-test text did not come back as expected")
        say(f"{GREY}   {line or out.strip().splitlines()[-1:] or ''}{RESET}")
    else:
        fail("engine self-test failed")
        for l in out.strip().splitlines()[-12:]:
            say(f"{GREY}   {l}{RESET}")


def check_ollama(cfg_url: str = "http://localhost:11434") -> None:
    step("Ollama (optional LLM cleanup pass)")
    try:
        with urllib.request.urlopen(f"{cfg_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        if models:
            ok(f"reachable, {len(models)} model(s): {', '.join(models[:4])}"
               + (" ..." if len(models) > 4 else ""))
        else:
            warn("reachable but no models installed (`ollama pull qwen3:14b`)")
    except Exception:
        warn(f"not reachable at {cfg_url}")
        say(f"{GREY}   the pipeline still works; it just skips the correction pass{RESET}")
        say(f"{GREY}   start it with `ollama serve`, or set llm.enabled: false in config.yaml{RESET}")


def check_gpu() -> None:
    step("GPU")
    exe = shutil.which("nvidia-smi")
    if not exe:
        warn("nvidia-smi not found; assuming no NVIDIA GPU")
        return
    r = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    for line in r.stdout.strip().splitlines():
        ok(line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up the OCR_man environment.")
    ap.add_argument("--no-models", action="store_true",
                    help="skip the model pre-download and self-test")
    ap.add_argument("--cpu", action="store_true",
                    help="install the CPU-only PyTorch build")
    args = ap.parse_args()

    print(f"\n{BOLD}OCR_man setup{RESET}")
    print(f"{GREY}  project: {ROOT}{RESET}")

    check_gpu()
    create_venv()
    install_torch(cpu_only=args.cpu)
    install_packages()
    install_windows_hf_fix()
    predownload_models(skip=args.no_models)
    check_ollama()

    for d in ("INPUT", "OUTPUT"):
        (ROOT / d).mkdir(exist_ok=True)

    print(f"\n{BOLD}Ready.{RESET}")
    print(f"  1. put your scanned PDFs / EPUBs in {ROOT / 'INPUT'}")
    print(f"  2. run:  {GREEN}\"{VENV_PY}\" RUN_ME.py{RESET}")
    print(f"  3. collect the results from {ROOT / 'OUTPUT'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
