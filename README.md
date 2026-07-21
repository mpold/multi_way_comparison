# Knowledge-graph comparison tools

Small, standard-library-only scripts that uses input `*.html` knowledge/relationship 
graphs and either **repair** them or **compare** them. Each graph is a
self-contained vis-network page with its data embedded as a `const DATA={...}`
block; these scripts read that block, transform it, and write a new page.

Everything file-related is asked for interactively at the prompt — nothing is
hardcoded, and inputs are never modified in place.

| Script | Purpose |
|---|---|
| `backfill_years.py` | Fill in publication years missing from a source graph. |
| `type_analysis.py`  | Build one interactive graph over **N** source graphs, colouring each gene by which type it belongs to and how exclusively. |

## Requirements

- Python 3 (standard library only — no `pip install` needed).
- `backfill_years.py` needs network access (Europe PMC and NCBI).

---

## `backfill_years.py`

Fills the publication years that the extraction step dropped. In the set this
was written for, 36.5% of papers carried no year at all — not because the source
lacked one, but because a reduced API record was swallowed as `null` rather than
retried.

```
python backfill_years.py
```

You are prompted for the input graphs (based on individual PubMed searches).
The input graphs are produced the pipeline provided in https://github.com/mpold/knowledge_graphs/blob/main/README.md

**What it does**

1. **Survey.** Reads each graph's `const DATA=` block and splits every paper's
   PMCID into *already dated* and *missing a year* (`yr: null`).
2. **Convention check (before writing anything).** Re-fetches years for papers
   the graph **already** has and compares. If more than 5% disagree, the source
   and the pipeline are using different definitions of "year", so it **stops** —
   backfilling would silently mix two conventions in one column.
3. **Fetch.** Looks up the missing years from **Europe PMC** (no API key,
   batches of 50, retried with backoff). IDs Europe PMC has never heard of fall
   back to **NCBI E-utilities** (throttled under 3 req/s), which gets its own
   convention check — a failed NCBI check drops only the fallback, not the whole
   run.
4. **Rewrite.** Splices the patched data into a **new** file with the suffix.
   Inputs are untouched; unresolved papers are counted and named rather than left
   looking like an absent year.

**Guarantees**

- Inputs are never modified — every graph is written to a new, suffixed file.
- No failure is swallowed: every request is retried, and anything still
  unresolved is reported rather than written back as `null`.

---

## `type_analysis.py`

Builds a single interactive knowledge graph over **two or more** source graphs.
Node **hue** is the type a gene belongs to most; node **saturation** is how
exclusively it belongs there. It is the N-way sibling of `diff_two.py` and, at
N = 2, produces a recognisably identical picture.

```
python type_analysis.py
```

Prompts: the input graphs (names or a glob; **two or more** required — the last
one read supplies the inlined vis-network runtime), the output file name, the
graph's heading, and a label for each type (blank accepts a name derived from the
file name).

**What it computes**

- **Pairs.** Folds each graph's directed, category-split edges into undirected
  gene pairs, keeping every sentence at or above the score floor (`0.50`) tagged
  with its own score, so the page can re-filter at any slider position. Nothing
  is filtered in Python — the whole union of pairs ships and every threshold is
  applied in the browser.
- **Shares.** For each gene, per graph:
  `share = degree_in_graph / total_degree_of_graph`. Normalising by each
  corpus's own connectivity means a large size difference between literatures
  (320× in the sample set) does not decide the answer before the biology does.
- **Dominant type** = the graph with the largest share.
- **Specificity T** = the Yanai tissue-specificity index over those shares:
  `0` for a gene spread evenly across every graph, `1` for a gene seen in only
  one. It depends only on ratios, so the graphs need not be the same size.
- **Share floor.** A corpus is treated as no smaller than 15% of the median
  corpus size, so a handful of pairs in a tiny literature cannot win every
  comparison on a single paper. Floored corpora are named at the command line.
- **Significance (q values).** A binomial-tail test per gene against its
  dominant corpus, corrected across all tested genes with Benjamini–Hochberg.
  It counts distinct **papers**, not partner slots, so one author's phrasing is
  not treated as several independent observations. Paper **overlap** between
  corpora is measured, and the run warns if the disjoint-corpora assumption the
  q values rest on is violated.

**Colour.** N ramps fanning out from one shared neutral, interpolated in OKLab so
each arm covers the same perceptual distance. Poles are hand-picked for
colourblind separability; past eight types, hues are spaced mechanically and the
run prints a warning.

**Output.** A single offline HTML page. It ships the **raw** per-step degree and
paper tables rather than baked answers, so every control (score, publication
count per type, cluster size, specificity range, FDR ceiling, year window)
recomputes in the browser without regenerating the file. The vis-network runtime
is lifted from the last input and inlined.

**Guarantees**

- Inputs are read, never written; the output name is refused if it collides with
  an input.
- Candidates are validated by parsing, not by file name — a `diff_two.py` output
  (same `*_G.html` name, different edge shape) is detected and rejected at the
  prompt.

---

## The graph format, briefly

A source graph embeds `const DATA={ "nodes": [...], "edges": [...] }`. Each edge
carries `from`/`to` genes, a category `cat`, and a list of `sents`, where each
sentence has a PMID (`pmid`), score (`sc`), text (`text`), and year (`yr`, which
may be `null` — the gap `backfill_years.py` repairs). `diff_two.py` outputs reuse
the `*_G.html` name but collapse edges to a different (`f`/`t`) shape, which both
scripts detect and refuse.
