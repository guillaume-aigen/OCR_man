"""Optional second pass: fix residual OCR damage with a local LLM.

Even a strong VLM leaves artefacts on a degraded scan -- a dropped diacritic,
'rn' read as 'm', a hyphen that should have closed up, a paragraph that
starts mid-word.  A local instruct model repairs those cheaply.

The obvious risk is that the model stops correcting and starts *writing*:
smoothing an author's prose, summarising, translating, inventing a sentence
to bridge a gap.  That would silently corrupt the archive, which is worse
than leaving the OCR noise in.  So every chunk that comes back is measured
against what went in, and anything that drifted too far is thrown away and
the original kept.  The guard is deliberately strict: a rejected chunk costs
nothing but a little OCR noise, an accepted hallucination costs the truth.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import Config
from .doctypes import CAPTION, Element, FOOTNOTE, LIST, TEXT
from .util import LOG, Progress

SYSTEM_PROMPT = (
    "You are an OCR correction tool. You repair text that was scanned from a printed book.\n"
    "\n"
    "RULES - follow every one:\n"
    "1. Output ONLY the corrected text. No preamble, no commentary, no code fences.\n"
    "2. Fix only OCR errors: misread characters (rn/m, l/1, O/0, c/e), missing or wrong\n"
    "   diacritics, words wrongly split or joined, and hyphens left over from line breaks.\n"
    "3. Preserve the author's wording exactly. Do not rephrase, modernise, summarise,\n"
    "   translate, shorten, expand, or 'improve' anything.\n"
    "4. Preserve paragraph breaks, capitalisation (including SMALL-CAPS words rendered in\n"
    "   capitals), spelling conventions, and punctuation style.\n"
    "5. Never add a sentence, a heading, a note, or an explanation.\n"
    "6. If a passage is garbled beyond repair, leave it exactly as it is.\n"
    "7. If the text is already correct, return it unchanged, character for character.\n"
)

USER_TEMPLATE = "Correct the OCR errors in the following text.\n\n<text>\n{chunk}\n</text>"

#: Element types whose prose is worth sending to the model.
CLEANABLE = {TEXT, LIST, CAPTION, FOOTNOTE}


@dataclass
class CleanStats:
    chunks: int = 0
    accepted: int = 0
    rejected: int = 0
    unchanged: int = 0
    errors: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def summary(self) -> str:
        bits = [f"{self.accepted} corrected", f"{self.unchanged} unchanged",
                f"{self.rejected} rejected"]
        if self.errors:
            bits.append(f"{self.errors} errors")
        s = ", ".join(bits)
        if self.reasons:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items()))
            s += f" ({detail})"
        return s


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, base_url: str, timeout_s: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        #: None = untested, True/False = whether the server accepts `think`.
        self._supports_think: bool | None = None

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5).close()
            return True
        except Exception:
            return False

    def generate(self, model: str, system: str, prompt: str,
                 temperature: float, num_ctx: int) -> str:
        payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "top_p": 0.9,
                "repeat_penalty": 1.0,
            },
        }
        if self._supports_think is not False:
            # Reasoning models burn a lot of tokens deliberating over a task
            # that needs none. Older models reject the flag outright, so the
            # first rejection turns it off for the rest of the run.
            payload["think"] = False
        try:
            data = self._post("/api/generate", payload)
            self._supports_think = self._supports_think is not False
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and self._supports_think is not False:
                self._supports_think = False
                payload.pop("think", None)
                data = self._post("/api/generate", payload)
            else:
                raise
        return data.get("response", "")


#: Models that cannot follow an instruction; picking one as a fallback would
#: silently destroy the text.
_NON_CHAT_HINTS = ("embed", "bge-", "gte-", "minilm", "reranker", "rerank",
                   "clip", "whisper", "moondream")


def resolve_model(client: OllamaClient, wanted: str, fallbacks: list[str]) -> str | None:
    """Pick the configured model, or the closest installed alternative."""
    installed = client.list_models()
    if not installed:
        return None
    names = {m.split(":")[0]: m for m in installed}
    for cand in [wanted, *fallbacks]:
        if cand in installed:
            return cand
        base = cand.split(":")[0]
        if base in names:
            return names[base]
    usable = [m for m in installed
              if not any(h in m.lower() for h in _NON_CHAT_HINTS)]
    if usable:
        LOG.warning(
            f"  none of the configured models are installed; falling back to {usable[0]}"
        )
        return usable[0]
    return None


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def similarity(a: str, b: str) -> float:
    """Token-level similarity, 0..1, robust to the small edits we expect.

    Uses difflib's ratio over token lists rather than characters, so a
    legitimate spelling fix barely moves the number while a rewrite or a
    dropped sentence moves it a lot.
    """
    import difflib

    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.S)
_TAGGED = re.compile(r"<text>\s*(.*?)\s*</text>", re.S)
_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is)[^\n:]*:|corrected text:|sure[^\n]*:|output:)\s*\n",
    re.I,
)
_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)


def strip_wrapper(response: str) -> str:
    """Remove the chatter a model wraps around the answer despite instructions.

    The layers nest in no fixed order -- a preamble can sit outside a code
    fence, or the other way round -- so peeling repeats until nothing more
    comes off.
    """
    out = _THINK.sub("", response or "").strip()
    for _ in range(4):
        before = out
        m = _TAGGED.search(out)
        if m:
            out = m.group(1).strip()
        out = _PREAMBLE.sub("", out).strip()
        m = _FENCE.match(out)
        if m:
            out = m.group(1).strip()
        if out == before:
            break
    return out.strip()


def check_result(original: str, candidate: str, cfg: Config) -> tuple[bool, str]:
    """Decide whether a corrected chunk is safe to accept."""
    if not candidate.strip():
        return False, "empty"

    lo = float(cfg["llm.min_length_ratio"])
    hi = float(cfg["llm.max_length_ratio"])
    ratio = len(candidate) / max(1, len(original))
    if ratio < lo:
        return False, "too_short"
    if ratio > hi:
        return False, "too_long"

    sim = similarity(original, candidate)
    if sim < float(cfg["llm.min_similarity"]):
        return False, "diverged"

    # A refusal or meta-comment can still pass the length test; catch the
    # common shapes explicitly.
    head = candidate[:180].lower()
    for marker in ("i cannot", "i can't", "as an ai", "i'm sorry", "note:",
                   "the text appears", "this text contains"):
        if head.startswith(marker):
            return False, "meta_response"

    # Numbers are the easiest thing for a model to silently "tidy"; if the
    # multiset of digits changed a lot, something was invented.
    d_orig = re.findall(r"\d+", original)
    d_cand = re.findall(r"\d+", candidate)
    if d_orig and abs(len(d_cand) - len(d_orig)) > max(1, len(d_orig) * 0.25):
        return False, "numbers_changed"

    return True, "ok"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def build_chunks(elements: list[Element], max_chars: int) -> list[list[int]]:
    """Group consecutive cleanable elements into chunks of ~max_chars."""
    chunks: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for i, el in enumerate(elements):
        if el.type not in CLEANABLE or not el.text.strip():
            continue
        n = len(el.text)
        if cur and size + n > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


SEP = "\n\n"


# ---------------------------------------------------------------------------

def clean_elements(elements: list[Element], cfg: Config) -> tuple[list[Element], CleanStats]:
    """Run the correction pass over `elements` in place, returning stats."""
    st = CleanStats()
    if not cfg.get("llm.enabled", True):
        return elements, st

    client = OllamaClient(cfg["llm.base_url"], int(cfg["llm.timeout_s"]))
    if not client.available():
        LOG.warning(
            f"  LLM cleanup skipped: no Ollama server at {cfg['llm.base_url']} "
            "(start it with `ollama serve`, or set llm.enabled: false)"
        )
        return elements, st

    model = resolve_model(client, cfg["llm.model"], list(cfg.get("llm.fallback_models", [])))
    if not model:
        LOG.warning("  LLM cleanup skipped: Ollama has no models installed")
        return elements, st
    LOG.info(f"  LLM cleanup with {model}")

    chunks = build_chunks(elements, int(cfg["llm.chunk_chars"]))
    if not chunks:
        return elements, st

    prog = Progress(len(chunks), label="cleanup", enabled=cfg.get("run.verbose", True))
    temperature = float(cfg["llm.temperature"])
    num_ctx = int(cfg["llm.num_ctx"])
    retries = int(cfg.get("llm.max_retries", 2))

    for idx_list in chunks:
        st.chunks += 1
        original = SEP.join(elements[i].text for i in idx_list)

        candidate, err = None, None
        for attempt in range(retries + 1):
            try:
                raw = client.generate(
                    model, SYSTEM_PROMPT,
                    USER_TEMPLATE.format(chunk=original),
                    temperature, num_ctx,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                err = str(exc)
                continue
            except Exception as exc:
                err = str(exc)
                continue
            cand = strip_wrapper(raw)
            ok, reason = check_result(original, cand, cfg)
            if ok:
                candidate = cand
                break
            err = reason
            if reason in {"too_long", "meta_response", "empty"} and attempt < retries:
                continue
            break

        if candidate is None:
            if err in {"empty", "too_short", "too_long", "diverged",
                       "meta_response", "numbers_changed"}:
                st.rejected += 1
                st.note(err)
            else:
                st.errors += 1
                st.note("request_failed")
                LOG.debug(f"    LLM request failed: {err}")
            prog.update(1)
            continue

        parts = candidate.split(SEP)
        if len(parts) != len(idx_list):
            # The model merged or split paragraphs. Rather than guess which
            # text belongs to which element, only accept when there is exactly
            # one element in the chunk.
            if len(idx_list) == 1:
                parts = [candidate]
            else:
                st.rejected += 1
                st.note("paragraph_count")
                prog.update(1)
                continue

        changed = False
        for i, new_text in zip(idx_list, parts):
            new_text = new_text.strip()
            if new_text and new_text != elements[i].text:
                elements[i].text = new_text
                changed = True
        if changed:
            st.accepted += 1
        else:
            st.unchanged += 1
        prog.update(1)

    prog.close(st.summary())
    return elements, st
