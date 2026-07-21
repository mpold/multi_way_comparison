"""Fill in the publication years missing from a source graph.

Reads one or more *_G.html source graphs, finds the sentences whose `yr` is
null, asks Europe PMC for the year of every paper involved — then NCBI for any
Europe PMC has never heard of — and writes out copies with the years filled in.

    python backfill_years.py

Everything is asked for at the prompt — the graphs, and the suffix for the
copies. Only the standard library is used.

WHY THIS EXISTS
---------------
In the four-subtype set this was written for, 36.5% of papers (2,437 of 6,677)
carried no year at all, which is 19.3% of sentences. That is not missing source
data: spot checks against Europe PMC found a year for every one of them. The
extraction step had the years available and dropped them.

The pattern says the loss was mechanical rather than about the papers:

  * it is per paper, absolutely — no paper is dated in one sentence and null in
    another, and no paper carries two different years
  * it is not recency — undated PMC ids sit inside the dated range, not above it
  * it is not open access, and not the journal — two papers from one journal,
    both open access, one dated and one not

The one correlate is that undated papers are thinner (4.25 sentences against
10.24), which suggests one retrieval returning a reduced record — less text and
no metadata — with the failure swallowed as null rather than retried.

So this script is deliberately built not to repeat that mistake. Every request
is retried with backoff, anything still unresolved is counted and named rather
than left to look like an absent year, and it refuses to report success it did
not have.

CHECK BEFORE TRUST
------------------
Before filling anything in, the script re-fetches papers whose year the graph
ALREADY has and compares. If the years it fetches disagree with the years the
pipeline recorded, the two are using different conventions — Europe PMC's
`pubYear` against, say, an e-pub-ahead-of-print date — and mixing them would put
two different meanings in one field. That check runs every time and its result
is printed before a single year is written.

SAFETY
------
The inputs are never modified. Each graph is written to a new file with a
suffix, in keeping with the rest of this directory: a typo here would otherwise
destroy megabytes of upstream pipeline output with no way back.
"""

import glob as globmod
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

# Bare names resolve here; an absolute path is taken as given.
HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Europe PMC
# --------------------------------------------------------------------------
#
# Chosen over NCBI E-utilities because it needs no API key and its throttling is
# far more forgiving — and a key-less E-utilities client capped at 3 requests a
# second is exactly the shape of thing that plausibly lost these years to begin
# with.
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Ids per request. The query goes in the URL, and each id costs about 23
# characters of `PMCID:PMC12345678 OR `, so 50 keeps a request near 1 KB —
# comfortably inside every proxy's URL limit while still being 50x fewer round
# trips than asking one at a time.
BATCH = 50

TIMEOUT = 30                 # seconds per request
RETRIES = 4                  # attempts per batch before giving up on it
BACKOFF = 2.0                # seconds, doubled each retry
PAUSE = 0.2                  # between successful requests, to stay welcome

# How many already-dated papers to re-fetch as a convention check. Enough to
# make disagreement obvious, small enough to add only a few seconds.
VALIDATE = 150

# Identifies the client, which is the courtesy every public API asks for and the
# thing that gets you unblocked rather than throttled if a run misbehaves.
AGENT = "backfill_years.py (knowledge-graph year repair; stdlib urllib)"

# --------------------------------------------------------------------------
# NCBI, as a fallback for what Europe PMC has never heard of
# --------------------------------------------------------------------------
#
# Europe PMC resolved 2,393 of 2,437 papers on the set this was written for. The
# 44 it missed are not papers without a year — Europe PMC returns no record for
# them at all, and NCBI has every one that was spot-checked. They sit at the two
# ends of the id range: the newest, presumably not yet mirrored, and a handful of
# very old ones.
#
# This is a second source, so it gets the same scepticism as the first: its years
# are checked against the ones already in the graph before any of them is used.
# A source that resolves an id is not the same as a source that agrees with the
# rest of the column.
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# NCBI throttles at 3 requests a second without an API key and will start
# refusing above it. That refusal, swallowed, is a plausible origin of the very
# gap this script repairs — so the limit is enforced here as a hard floor on the
# interval between requests rather than left to a hopeful sleep.
NCBI_MIN_GAP = 0.4           # seconds between requests: 2.5/s, under the limit
NCBI_BATCH = 100             # esummary is happy with this many ids per GET

# NCBI asks callers to identify themselves so they can make contact before
# blocking anyone. `tool` is always sent. `email` is left empty deliberately —
# it is a personal detail and belongs to whoever runs this, not to the file.
# Filling it in raises your standing if a run ever misbehaves.
NCBI_TOOL = "backfill_years"
NCBI_EMAIL = ""

_ncbi_last = [0.0]


def _ncbi_throttle():
    """Block until NCBI_MIN_GAP has passed since the previous NCBI request."""
    gap = time.time() - _ncbi_last[0]
    if gap < NCBI_MIN_GAP:
        time.sleep(NCBI_MIN_GAP - gap)
    _ncbi_last[0] = time.time()


# --------------------------------------------------------------------------
# Reading and rewriting a source graph
# --------------------------------------------------------------------------

def locate_data(path):
    """Return (source_text, start, end, data) for the embedded DATA object.

    The span is returned as well as the object so the file can be rebuilt by
    splicing — everything outside those two offsets, the vis-network runtime
    included, is passed through byte for byte. Re-emitting the whole page from a
    template would risk changing parts of it this script has no business
    touching.
    """
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()
    start = src.index("{", src.index("const DATA="))
    data, length = json.JSONDecoder().raw_decode(src[start:])
    return src, start, start + length, data


def sentences(data):
    """Every sentence object in the graph, in document order."""
    for edge in data.get("edges", ()):
        for sent in edge.get("sents", ()):
            yield sent


def survey(data):
    """Split the paper ids into those already dated and those missing a year."""
    dated, undated = {}, set()
    for sent in sentences(data):
        pmid = sent.get("pmid")
        if not pmid:
            continue
        if sent.get("yr") is None:
            undated.add(pmid)
        else:
            dated[pmid] = sent["yr"]
    # A paper dated in one sentence needs no lookup, whatever its other
    # sentences say. Nothing in the sample set was inconsistent this way, but
    # nothing guarantees that either.
    return dated, undated - set(dated)


def rewrite(path, out, data, start, end, src):
    """Write the graph back out with the patched DATA spliced in."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    # `</script>` inside a sentence would close the block early. `\/` is a valid
    # JSON escape and a valid JS one, so this is safe in both directions and
    # costs nothing when — as usual — there is nothing to escape.
    payload = payload.replace("</", "<\\/")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(src[:start])
        fh.write(payload)
        fh.write(src[end:])


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _request(query):
    """One search request, retried with backoff. Returns [] only after RETRIES."""
    url = API + "?" + urllib.parse.urlencode({
        "query": query,
        "format": "json",
        "resultType": "lite",
        "pageSize": BATCH,
    })
    wait = BACKOFF
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body).get("resultList", {}).get("result", [])
        except (urllib.error.URLError, urllib.error.HTTPError,
                ValueError, OSError) as err:
            if attempt == RETRIES - 1:
                # Surfaced, never swallowed — a silent failure here would put
                # back exactly the nulls this script exists to remove.
                print("    request failed after %d attempts: %s" % (RETRIES, err))
                return []
            time.sleep(wait)
            wait *= 2
    return []


def fetch_years(pmcids, label):
    """Look up {pmcid: year} for a set of ids, in batches, with progress."""
    ids = sorted(pmcids)
    found = {}
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        query = " OR ".join("PMCID:%s" % p for p in chunk)
        for row in _request(query):
            pmcid, year = row.get("pmcid"), row.get("pubYear")
            if not pmcid or not year:
                continue
            try:
                found[pmcid] = int(year)
            except (TypeError, ValueError):
                continue
        done = min(i + BATCH, len(ids))
        print("\r    %s %d/%d resolved %d" % (label, done, len(ids), len(found)),
              end="", flush=True)
        if done < len(ids):
            time.sleep(PAUSE)
    print()
    return found


def _ncbi_year(rec):
    """Pull a four-digit year out of an esummary record.

    `pubdate` first because it is the record's own idea of when the paper was
    published, which is what Europe PMC's pubYear reports and therefore what the
    graph's existing years were validated against. The others are fallbacks for
    records that leave it blank; the convention check is what confirms the
    choice was right rather than merely plausible.
    """
    for key in ("pubdate", "sortdate", "printpubdate", "epubdate"):
        text = rec.get(key) or ""
        for token in str(text).replace("/", " ").split():
            if len(token) == 4 and token.isdigit() and 1800 < int(token) < 2100:
                return int(token)
    return None


def _ncbi_request(chunk):
    """One esummary call for a list of bare numeric ids, retried with backoff."""
    params = {"db": "pmc", "id": ",".join(chunk), "retmode": "json",
              "tool": NCBI_TOOL}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    url = NCBI + "?" + urllib.parse.urlencode(params)
    wait = BACKOFF
    for attempt in range(RETRIES):
        _ncbi_throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body).get("result", {})
        except (urllib.error.URLError, urllib.error.HTTPError,
                ValueError, OSError) as err:
            if attempt == RETRIES - 1:
                print("    NCBI request failed after %d attempts: %s"
                      % (RETRIES, err))
                return {}
            time.sleep(wait)
            wait *= 2
    return {}


def fetch_years_ncbi(pmcids, label):
    """Look up {pmcid: year} at NCBI. Same contract as fetch_years."""
    ids = sorted(pmcids)
    found = {}
    for i in range(0, len(ids), NCBI_BATCH):
        chunk = ids[i:i + NCBI_BATCH]
        # esummary wants the bare number; the graph carries the PMC prefix
        result = _ncbi_request([p[3:] for p in chunk if p.startswith("PMC")])
        for uid, rec in result.items():
            if uid == "uids" or not isinstance(rec, dict) or "error" in rec:
                continue
            year = _ncbi_year(rec)
            if year is not None:
                found["PMC" + uid] = year
        done = min(i + NCBI_BATCH, len(ids))
        print("\r    %s %d/%d resolved %d" % (label, done, len(ids), len(found)),
              end="", flush=True)
    print()
    return found


def check_convention(dated, fetch, source, size=VALIDATE):
    """Re-fetch papers we already have years for and compare.

    Agreement means the pipeline's year and this source's mean the same thing,
    so filling the gaps from it leaves one consistent field. If they disagree,
    backfilling would silently mix two definitions of "year", and the caller is
    told rather than being handed a quietly corrupted file.

    Run once per source, not once per run. A second source that can resolve an
    id is not the same as a second source that agrees with the column it is
    about to write into.
    """
    sample = sorted(dated)[:size]
    if not sample:
        return None
    print("  checking %s against %d papers that are already dated"
          % (source, len(sample)))
    fetched = fetch(set(sample), "checked")
    both = [(p, dated[p], fetched[p]) for p in sample if p in fetched]
    if not both:
        return None
    agree = [x for x in both if x[1] == x[2]]
    off = [x for x in both if x[1] != x[2]]
    print("    %d of %d agree (%.1f%%)"
          % (len(agree), len(both), 100.0 * len(agree) / len(both)))
    for pmcid, mine, theirs in off[:5]:
        print("      %s: graph says %s, %s says %s" % (pmcid, mine, source, theirs))
    if len(off) > 5:
        print("      ... and %d more" % (len(off) - 5))
    return len(agree) / float(len(both))


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def _ask(label):
    """One line from stdin. EOF is an abort, not an empty answer."""
    try:
        return input(label).strip().strip('"\'')
    except EOFError:
        raise SystemExit("\naborted: %s needs a value" % label.strip().rstrip(":"))


def _resolve(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def usable(path):
    """None if this is a source graph this script can patch, else why not."""
    try:
        _, _, _, data = locate_data(path)
    except ValueError:
        return "no readable `const DATA=` block — not a graph from this pipeline"
    except (IOError, OSError, UnicodeDecodeError):
        return "could not be read as text"
    edges = data.get("edges") if isinstance(data, dict) else None
    if not edges:
        return "no relationships to patch"
    if "from" not in edges[0]:
        if "f" in edges[0]:
            return "a diff_two.py output — patch the source graphs it was built from"
        return "edges are not in this pipeline's format"
    return None


def ask_input_graphs():
    """Collect one or more source graphs, by name or by glob, blank to finish."""
    print("graphs to patch — one name or glob per line, blank line when done:")
    chosen = []
    while True:
        name = _ask("  graph %d: " % (len(chosen) + 1))
        if not name:
            if chosen:
                return chosen
            print("    at least one graph is required")
            continue
        pattern = _resolve(name)
        if any(ch in name for ch in "*?["):
            hits = sorted(p for p in globmod.glob(pattern) if os.path.isfile(p))
            if not hits:
                print("    nothing matches: %s" % pattern)
                continue
            for path in hits:
                if path in chosen:
                    continue
                why = usable(path)
                if why:
                    print("    - %s — %s" % (os.path.basename(path), why))
                else:
                    print("    + %s" % os.path.basename(path))
                    chosen.append(path)
            continue
        if not os.path.isfile(pattern):
            print("    no such file: %s" % pattern)
            continue
        if pattern in chosen:
            print("    already listed")
            continue
        why = usable(pattern)
        if why:
            print("    not usable: %s" % why)
            continue
        chosen.append(pattern)


def ask_suffix(paths):
    """Suffix for the copies. Never allowed to resolve onto an input."""
    while True:
        raw = _ask("suffix for the patched copies [_dated]: ")
        suffix = raw or "_dated"
        outs = [out_path(p, suffix) for p in paths]
        clash = [o for o in outs
                 if any(os.path.abspath(o) == os.path.abspath(p) for p in paths)]
        if clash:
            print("  that would overwrite an input — pick another suffix")
            continue
        return suffix


def out_path(path, suffix):
    stem, ext = os.path.splitext(path)
    return stem + suffix + ext


# --------------------------------------------------------------------------

def main():
    paths = ask_input_graphs()
    suffix = ask_suffix(paths)

    # One survey pass first, so the convention check and the lookups can both be
    # done once over the whole run rather than per file. Corpora from disjoint
    # queries share no papers, but a reader may well point this at graphs that
    # do, and paying for the same id twice would be silly.
    print("\nreading %d graph%s" % (len(paths), "" if len(paths) == 1 else "s"))
    loaded, all_dated, all_undated = [], {}, set()
    for path in paths:
        src, start, end, data = locate_data(path)
        dated, undated = survey(data)
        loaded.append((path, src, start, end, data))
        all_dated.update(dated)
        all_undated |= undated
        total = sum(1 for _ in sentences(data))
        gaps = sum(1 for s in sentences(data) if s.get("yr") is None)
        print("  %-46s %6d sentences, %5d undated (%4.1f%%), %4d papers to look up"
              % (os.path.basename(path)[:44], total, gaps,
                 100.0 * gaps / max(1, total), len(undated)))

    if not all_undated:
        raise SystemExit("\nnothing to do — every sentence already has a year")

    print("\n%d distinct papers need a year" % len(all_undated))
    agreement = check_convention(all_dated, fetch_years, "Europe PMC")
    if agreement is not None and agreement < 0.95:
        print("\nSTOPPING: the years already in these graphs disagree with Europe PMC")
        print("for %.0f%% of the papers checked, so the two are not the same field."
              % (100 * (1 - agreement)))
        print("Filling the gaps from Europe PMC would mix two conventions in one")
        print("column. Work out which definition the pipeline used first.")
        raise SystemExit(1)

    print("\nlooking up %d papers" % len(all_undated))
    found = fetch_years(all_undated, "fetched")
    missing = all_undated - set(found)
    print("  Europe PMC resolved %d, unresolved %d" % (len(found), len(missing)))

    # Europe PMC returns no record at all for a small tail of ids — the newest,
    # not yet mirrored, and a few very old ones. NCBI has them. Only the tail is
    # sent, so the slower of the two APIs is asked for the least.
    if missing:
        print("\nfalling back to NCBI for %d unresolved" % len(missing))
        nagree = check_convention(all_dated, fetch_years_ncbi, "NCBI",
                                  size=min(50, VALIDATE))
        if nagree is not None and nagree < 0.95:
            # Not fatal: Europe PMC's years are already validated and in hand,
            # so the fallback is dropped rather than the whole run. Discarding
            # 2,393 good years over a disagreement about 44 would be the wrong
            # trade by two orders of magnitude.
            print("  SKIPPING the NCBI fallback: its years disagree with this graph")
            print("  for %.0f%% of the papers checked. Keeping the Europe PMC results"
                  % (100 * (1 - nagree)))
            print("  only; the %d unresolved stay undated." % len(missing))
        else:
            extra = fetch_years_ncbi(missing, "fetched")
            # a source cannot overwrite another's answer; it only fills a gap
            for pmcid, year in extra.items():
                found.setdefault(pmcid, year)
            missing = all_undated - set(found)
            print("  NCBI resolved %d, still unresolved %d" % (len(extra), len(missing)))

    print("  total resolved %d of %d" % (len(found), len(all_undated)))
    if missing:
        shown = sorted(missing)[:8]
        print("  still undated: %s%s"
              % (", ".join(shown), " ..." if len(missing) > len(shown) else ""))
    if not found:
        raise SystemExit("\nnothing resolved — leaving every file untouched")

    print()
    for path, src, start, end, data in loaded:
        filled = 0
        for sent in sentences(data):
            if sent.get("yr") is None and sent.get("pmid") in found:
                sent["yr"] = found[sent["pmid"]]
                filled += 1
        left = sum(1 for s in sentences(data) if s.get("yr") is None)
        out = out_path(path, suffix)
        rewrite(path, out, data, start, end, src)
        size = os.path.getsize(out) / 1048576.0
        print("wrote %s (%.1f MB) — filled %d sentences, %d still undated"
              % (os.path.basename(out), size, filled, left))

    print("\nInputs are untouched. Check a patched copy opens and reads correctly,")
    print("then swap the originals for it if you are satisfied.")


if __name__ == "__main__":
    main()
