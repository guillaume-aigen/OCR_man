# How to use this — start here

Run `python SETUP.py` once. After that:

---

## The whole thing in three steps

**1. Put your files in the `INPUT` folder**

Scanned PDFs or EPUBs. As many as you like — they are processed one after
another. Subfolders are fine.

**2. Double-click `TRY_FIRST.bat`**

This does only the first 20 pages of each document, so you get something to
look at in a couple of minutes instead of waiting hours for a book that might
have come out wrong.

**3. Look at the `.md` file in `OUTPUT`, then run the real thing**

If the sample looks right (see *Is it any good?* below), just double-click
**`RUN.bat`**. It carries on from the 20 pages already done rather than
redoing them.

That's it. When it finishes you get three files per document in `OUTPUT`.

---

## What you get

For a book called `MyBook.pdf`:

| File | Use it for |
|---|---|
| `MyBook.md` | Feeding to an LLM. Plain text, headings, no page clutter. |
| `MyBook.epub` | Reading. Works on Kindle, Kobo, phone, Calibre. |
| `MyBook_searchable.pdf` | Looks exactly like the scan, but you can search and copy text. |

---

## Is it any good? — what to check in the `.md`

Open the `.md` in any text editor and check four things:

1. **Reading order.** On a two-column page, does the text read straight
   through, or does it jump back and forth mid-sentence? Jumping is the one
   failure that ruins a book.
2. **Whole paragraphs.** Sentences should not stop dead and restart.
3. **No page furniture.** The running header and page numbers should not be
   sprinkled through the text.
4. **Spelling.** A few OCR slips are normal. Lots of mangled words means the
   scan needs help — see *If the output is bad* below.

---

## How long it takes

Roughly **5 seconds per page**, plus about a minute of startup.

| Book size | Time |
|---|---|
| 20 pages | ~2 min |
| 300 pages | ~25 min |
| 1000 pages | ~1h 20m |

**Add roughly 9 seconds per page on top if the LLM correction pass is on** —
it is on by default, and it more than doubles the total. On the test book it
took an extra 14 minutes and changed nothing, because the OCR was already
clean. To skip it:

```
RUN.bat --no-llm
```

Try a document both ways with `TRY_FIRST.bat` and see whether it actually
improves anything on your material.

**You can stop it at any time** (close the window or press Ctrl+C). Run it
again and it picks up where it left off instead of starting the book over.

---

## If the output is bad

| Problem | Try |
|---|---|
| Text jumps between columns | `RUN.bat --force-preprocess` |
| Lots of misread letters | `RUN.bat --force-preprocess` (straightens and cleans every page) |
| Pages look crooked or dark | same as above |
| Words look "corrected" into wrong words | `RUN.bat --no-llm` |
| Want to redo from scratch | `RUN.bat --fresh` |
| Only want the Markdown | `RUN.bat --only md` |

Combine them freely: `RUN.bat --no-llm --force-preprocess`

---

## Housekeeping

- **`WORK\`** holds the cache that makes resuming possible. It gets large —
  roughly 100 MB per 100 pages. Delete the whole folder once you are happy
  with a book's output. Deleting it is always safe; it only means the next
  run starts from scratch.
- Files are matched by name, so processing `MyBook.pdf` twice overwrites
  `MyBook.md`. Old outputs from documents you have since removed from `INPUT`
  are not cleaned up automatically.
- Take documents out of `INPUT` once they are done, or every run will check
  them again.

---

## Running it from a terminal instead

The `.bat` files just wrap this:

```bash
.venv\Scripts\python.exe RUN_ME.py
```

Every option: `.venv\Scripts\python.exe RUN_ME.py --help`

---

## If something breaks

1. Check `WORK\ocr_man.log` — the full error is at the bottom.
2. Confirm the machinery still works: `.venv\Scripts\python.exe TEST.py`
   (should print *all 38 checks passed*).
3. Confirm the OCR engine is found:
   `.venv\Scripts\python.exe RUN_ME.py --list-engines`

`README.md` has the technical detail — how it handles columns, what the LLM
guard rails do, and the known limitations.
