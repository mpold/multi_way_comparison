"""Build one knowledge graph over any number of source graphs.

Reads N vis-network graphs, keeps the pairs that are prominent in any of them,
and renders them as a single network in which node *hue* is the type the gene
belongs to most and node *saturation* is how exclusively it belongs there.

    python type_analysis.py

Everything file-related is asked for at the prompt — the inputs, the output, the
graph's name and what to call each type. Nothing is hardcoded, so the script has
no idea which diseases it is analysing until it is run.

    input graphs:        one per line, or a glob like lung_*_G.html;
                         blank line ends the list. Two or more are required.
                         The LAST one read supplies the inlined vis-network
                         runtime.
    output file name:
    graph name:          the page's heading
    label for <file>:    what to call that type everywhere in the page; blank
                         accepts the name derived from the file

Only the standard library is used. The vis-network runtime is lifted out of the
last input and inlined, so the output opens offline.

This is the N-way sibling of diff_two.py, and it inherits that script's model
wholesale: the same score steps, and the same decision to ship *raw* evidence
rather than a baked snapshot so the browser can recompute everything as the
controls move.

It goes further than diff_two.py on that last point. That script still made one
decision on the reader's behalf before writing the file — it dropped any pair
under two publications. Here that judgement is the reader's, as one publication
slider per type, so the whole union of pairs ships and nothing is filtered in
Python at all. The file is about five times larger for it.

What it cannot inherit is the balance index. That index is a signed position
between two poles; three graphs have no axis to be signed along. So the single
number splits into the two independent questions it was really answering:

    dominant type  — which graph is this gene most central to (the sign)
    specificity T  — how exclusively (the magnitude)

Both are read off the same per-graph shares diff_two.py uses, and the split is
faithful: at N=2, T = 1 - min_share/max_share, while diff_two's |index| is
(max-min)/(max+min). Each is a monotone function of the other
(index = T/(2-T)), so the two pages order genes identically on two inputs. This
one just stops pretending the ordering is a line when there are more than two
ends to it.

T is the Yanai tissue-specificity index, borrowed from exactly this problem in
expression data: one gene, many tissues, how tissue-restricted is it. It is
scale-free (it depends only on ratios between shares), it is 0 for a gene spread
evenly over every graph and 1 for a gene seen in only one, and unlike Shannon
entropy it does not need every graph to be the same size to behave.

Normalising each degree by its own graph's connectivity is even more load-bearing
here than in the two-graph case. These corpora are not remotely comparable in
size — the set this was first built for ran from 11,614 pairs down to 36, a 320x
spread — so on raw counts every gene would be "adenocarcinoma-specific" and the
answer would be a restatement of which literature is largest.

That normalisation is necessary and not sufficient. It corrects the bias of
unequal literatures; it leaves the noise, and a corpus of a few dozen pairs
produces shares so unstable that it wins every comparison on a single paper. See
SHARE_FLOOR_FRACTION, which is what stops it.
"""

import glob as globmod
import io
import json
import math
import os
from collections import defaultdict

# Bare names resolve here; an absolute path is taken as given.
HERE = os.path.dirname(os.path.abspath(__file__))

# Selectable minimum relationship scores, following the source graphs' own
# scheme: coarse 0.05 steps through the low band, then 0.01 steps from 0.95 up,
# where a single point still changes the graph materially.
# The floor drives the payload — every sentence at or above it ships.
#
# The ladder starts at 0.50 because that is where the upstream pipeline's own
# cutoff sits: the lowest score present in any source graph is exactly 0.500.
# Starting at 0.75, as this did, silently discarded 37.8% of the sentences the
# pipeline had already extracted and shipped — evidence that was in the input
# files and simply could not be reached from the page at any slider position.
# Matching the floor to the data's floor means nothing provided is thrown away.
#
# The cost is real and worth stating: it takes the payload from 8.7 MB to 14.1
# MB on the four-subtype set, and the union from 14,566 pairs to 21,441. Since
# the default score moved to this floor, that union IS the opening view; the
# cluster and T defaults below are what make it readable.
SCORE_STEPS = ([round(0.50 + 0.05 * i, 2) for i in range(9)]      # 0.50 .. 0.90
               + [round(0.95 + 0.01 * i, 2) for i in range(5)])   # 0.95 .. 0.99
FLOOR = min(SCORE_STEPS)
# Where the slider starts. Resolved to the nearest step rather than a hardcoded
# index, so it survives edits to SCORE_STEPS.
DEFAULT_SCORE = 0.5
DEFAULT_STEP = min(range(len(SCORE_STEPS)),
                   key=lambda i: abs(SCORE_STEPS[i] - DEFAULT_SCORE))

# The rest of the opening position, in one place so the markup's initial values
# and the "Default settings" button cannot drift apart. Cluster and T min carry
# the thinning that the score slider used to do at 0.95: the whole graph at 0.5
# is a hairball, and these two cut it to something a reader can look at without
# hiding any evidence the sliders cannot bring back.
DEFAULT_CLUSTER = 6
DEFAULT_TMIN = 0.35
DEFAULT_TMAX = 1.0

# How much evidence a pair needs is the reader's call, not this script's, so it
# is a slider per type rather than a constant. A pair is drawn when at least one
# type backs it with that type's own minimum — see the rule in build().
#
# The sliders open at 1: every pair the corpora assert is on the table from the
# start, and raising a slider is the reader's deliberate act of asking for
# corroboration.
#
# The cost is real and worth stating. The overwhelming majority of pairs in
# these corpora rest on one paper — 10,777 of adenocarcinoma's 11,614 — so
# opening at 1 draws ~10,000 pairs over ~4,500 genes, and a sentence count there
# mostly ranks how often one author repeated themselves rather than how well
# attested a relation is. Opening at 2 would draw ~560. The cluster and T
# defaults above carry the thinning instead, and raising a publication slider
# remains the way to demand replication of one type.
DEFAULT_PUB = 1

# Slider ceiling. Publication counts have a long, thin tail — adenocarcinoma
# reaches 58, but only single digits of pairs sit above 8 — so a slider running
# to the true maximum would spend 80% of its travel on a handful of pairs. Each
# type's slider stops at its own maximum or here, whichever is lower, and the
# true maximum is still reported in the control's tooltip.
PUB_CAP = 10

# Smallest denominator a share may be divided by, as a fraction of the median
# corpus size at that score step.
#
# Normalising by each corpus's own connectivity removes the *bias* of unequal
# literature sizes. It does nothing about the *variance*, and below a few
# hundred endpoints that variance swamps the signal: degree/total is a ratio
# estimator, and at n=72 one relationship moves a gene's share by 1.4% when the
# best-attested gene in a well-populated corpus only reaches about 1.3%. A
# corpus that thin therefore wins every argument it enters, on one paper.
#
# It did exactly that. On the sample set the 36-pair large-cell corpus claimed
# all 48 of the genes it contained, including ASCL1, NEUROD1 and YAP1 — the
# small-cell subtype transcription factors — on the strength of a single
# sentence from one LCNEC paper, against 53 relationships in the small-cell
# corpus. The biology in that sentence is real (LCNEC does share the
# neuroendocrine program); the weight put on it was not.
#
# So a corpus is treated as no smaller than this fraction of the median corpus.
# Anchored to the median rather than the mean so one very large literature
# cannot drag the floor up, and expressed as a fraction rather than a constant
# so it scales to whatever is loaded — a set of uniformly small graphs floors
# nobody. This is deliberately not a size cutoff: a floored corpus keeps every
# claim it can still win on relative evidence, and only loses the ones it was
# winning on arithmetic alone.
SHARE_FLOOR_FRACTION = 0.15

# Selectable false-discovery-rate ceilings for the q filter. Opens at the
# conventional 5%: genes above it are still drawn, as faded context, so this
# qualifies the specificity scale on first paint without hiding anything. Only
# "significant only" actually removes them. Resolved by value, not by index, so
# it survives edits to the list.
FDR_STEPS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
DEFAULT_FDR = 0.05
DEFAULT_FDR_STEP = FDR_STEPS.index(DEFAULT_FDR)

# Below this many inputs there is no N-way question to ask; diff_two.py answers
# the two-graph case with a scale built for it.
MIN_INPUTS = 2

# Below this the score slider says so. Not a calibrated threshold — it is where
# this script's floor used to sit, so it marks exactly the band that was
# unreachable before and holds the extractor's least confident sentences. The
# page states it rather than letting a reader assume every score is alike.
LOW_SCORE = 0.75


# --------------------------------------------------------------------------
# Reading the source graphs
# --------------------------------------------------------------------------

def extract_data(path):
    """Return the `const DATA={...}` object embedded in a source graph."""
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()
    start = src.index("{", src.index("const DATA="))
    return json.JSONDecoder().raw_decode(src[start:])[0]


def extract_vis_runtime(path):
    """Return the vis-network bundle — the first <script> of a source graph."""
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()
    start = src.index("<script>") + len("<script>")
    return src[start:src.index("</script>", start)]


def collapse_to_pairs(data):
    """Fold directed, category-split edges into undirected gene pairs.

    Keeps every sentence at or above FLOOR, tagged with its own score, so the
    page can re-filter at any step. Returns {(gene_a, gene_b): [sentence, ...]}
    with gene_a < gene_b, so A->B and B->A land on one key.
    """
    edges = data["edges"]
    # One check rather than a guard per edge: the same mistake reached here as a
    # bare KeyError on the first edge, which named a missing dict key and not the
    # actual problem. Callers using this as a module get the diagnosis too.
    if edges and "from" not in edges[0]:
        raise ValueError(
            "these edges carry no 'from' key, so this is not a source graph — "
            "a diff_two.py output has the same *_G.html name and reaches here "
            "looking plausible")
    pairs = defaultdict(list)
    for edge in edges:
        key = tuple(sorted((edge["from"], edge["to"])))
        for s in edge["sents"]:
            if s["sc"] >= FLOOR:
                pairs[key].append({
                    "p": s["pmid"],
                    "t": s["text"],
                    "c": round(s["sc"], 4),
                    "y": s["yr"],
                    "g": edge["cat"],
                })
    return dict(pairs)


def present_at(sents, score):
    return any(s["c"] >= score for s in sents)


def pubs_at(sents, score):
    return len({s["p"] for s in sents if s["c"] >= score})


# --------------------------------------------------------------------------
# Colour — N ramps radiating from one shared neutral
# --------------------------------------------------------------------------
#
# diff_two.py encodes two types as the two ends of a diverging red-grey-blue
# ramp. That shape does not extend: a third type has nowhere to go. So the ramp
# is cut in half at the grey and rebuilt as a fan — N arms, all starting from
# the SAME neutral midpoint and running out to their own pole.
#
# The two encodings then say the same thing with the same ink:
#
#     hue        = which type dominates      (was: which side of the grey)
#     saturation = specificity T             (was: distance from the grey)
#
# A gene equally central to every graph lands on the shared neutral and reads as
# uncommitted, exactly as a balanced gene does in the two-graph page. The first
# two poles below ARE diff_two.py's red and blue, so a two-input run of this
# script and a run of that one are recognisably the same picture.
#
# The poles are hand-picked rather than machine-spaced round the hue circle.
# Evenly spaced hues look principled and fail in practice — they land pairs in
# the blue/purple and red/orange confusions that dominate deuteranopia and
# protanopia. These eight are ordered so that any prefix of the list is a usable
# palette: the first two are the maximally separated diverging pair, and each
# addition is placed against everything already in play. Past eight, hues are
# rotated mechanically and the run prints a warning, because no fixed palette
# keeps that many categories apart and the honest answer is to say so.

RAMP_STEPS = 41

# Neutral midpoint per mode — near-white on a light surface, near-background on
# a dark one, both taken from diff_two.py's ramp so the greys agree.
NEUTRAL = {"light": "#f0efec", "dark": "#383835"}

# Outline shifts away from the surface. Without it the neutral end (1.12:1
# against the light surface) would vanish; with it every node clears 2.67:1.
BORDER_DL = {"light": -0.26, "dark": +0.24}

# Outline weight, in pixels. The outline is doing contrast work rather than
# decoration — it is the whole reason a near-neutral node is visible against the
# surface at all — so it thins rather than disappears. One pixel also matches
# the legend and explainer swatches, which have always been 1px, so a node on
# the canvas and its key in the rail are now drawn the same way.
#
# Significant genes keep a heavier ring. That is not emphasis for its own sake:
# they are redrawn over the finished scene, and a slightly stronger outline is
# what separates them from whatever they land on top of.
NODE_BORDER = 1
NODE_BORDER_SIG = 2
# Selection has to remain visible at these weights. Left unset, vis-network
# applies its own borderWidthSelected of 2, which would be no change at all for
# a significant node and so would make clicking one look broken.
NODE_BORDER_SEL = NODE_BORDER + 1
NODE_BORDER_SIG_SEL = NODE_BORDER_SIG + 1

# Poles, in priority order. Each entry is (light pole, dark pole): dark, dense
# colours to sit on a light surface; brighter, lighter ones to sit on a dark
# surface. Same lightness discipline as the source ramp, mirrored per mode.
POLES = [
    ("#a6272a", "#e14c4a"),   # red      — diff_two.py's first pole
    # Lighter than diff_two.py's #1c5cab second pole, which read as navy on the
    # light canvas. L 0.645 puts it near the orange pole below, so the two carry
    # similar weight rather than one sitting back while the other comes forward.
    ("#3a90ea", "#3987e5"),   # blue
    ("#1d6b4f", "#46a97f"),   # green
    # Orange is the one hue that cannot honour the lightness band above: brown IS
    # dark orange, so an L near 0.5 like the other light poles reads brown no matter
    # how the hue is set, and the ramp's middle (#c28c6d, #b16c42) reads browner
    # still. This pole is deliberately lighter (L 0.68) and more saturated so the
    # whole ramp stays orange; it costs contrast against the light canvas, which the
    # border ramp — which does keep the band — carries instead.
    ("#e8730d", "#ee8b33"),   # orange
    ("#6b3fa0", "#a284dd"),   # purple
    ("#0d6470", "#3fabbc"),   # teal
    ("#a33069", "#dd6fa5"),   # magenta
    ("#55591a", "#a8ae4a"),   # olive
]

# Edge ink for a pair two or more graphs agree on. Deliberately the same grey as
# the neutral node midpoint's neighbourhood: "shared" is one idea, and it should
# not arrive in two unrelated colours.
SHARED_EDGE = {"light": "#a5a49d", "dark": "#7a7972"}


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c):
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _cbrt(x):
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def _hex_to_linear(h):
    h = h.lstrip("#")
    return [_srgb_to_linear(int(h[i:i + 2], 16) / 255.0) for i in (0, 2, 4)]


def _linear_to_oklab(rgb):
    r, g, b = rgb
    l = _cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = _cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = _cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]


def _oklab_to_linear(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]


def _oklab_to_hex(lab):
    # floor(x + 0.5) to match the JS Math.round the ramp was authored against;
    # Python's round() breaks .5 ties to even and would shift some steps.
    chans = (
        int(math.floor(_linear_to_srgb(c) * 255 + 0.5))
        for c in _oklab_to_linear(lab)
    )
    return "#" + "".join("%02x" % min(255, max(0, c)) for c in chans)


def _rotate_hue(lab, turns):
    """Spin an OKLab colour round the hue circle, keeping lightness and chroma."""
    L, a, b = lab
    ang = math.atan2(b, a) + turns * 2 * math.pi
    chroma = math.hypot(a, b)
    return [L, chroma * math.cos(ang), chroma * math.sin(ang)]


def pole_hexes(count, mode):
    """`count` pole colours for one mode, hand-picked while they last."""
    col = 0 if mode == "light" else 1
    fixed = [p[col] for p in POLES[:count]]
    if count <= len(POLES):
        return fixed
    # Past the hand-picked set, spin the first pole round the circle. Evenly
    # spaced hues are the only thing left that is at least deterministic and
    # deliberate; main() warns that they are no longer separation-checked.
    base = _linear_to_oklab(_hex_to_linear(POLES[0][col]))
    extra = count - len(POLES)
    return fixed + [
        _oklab_to_hex(_rotate_hue(base, (i + 1) / float(extra + 1)))
        for i in range(extra)
    ]


def build_ramps(count):
    """One neutral -> pole ramp per type per mode, as fill/outline pairs.

    Interpolated in OKLab so the arms are perceptually even and every arm covers
    the same perceptual distance — a gene at T=0.5 looks half-committed whatever
    type it belongs to, which it would not if the poles were mixed in sRGB.
    """
    ramps = {}
    for mode in ("light", "dark"):
        neutral = _linear_to_oklab(_hex_to_linear(NEUTRAL[mode]))
        arms = []
        for pole in pole_hexes(count, mode):
            end = _linear_to_oklab(_hex_to_linear(pole))
            fills, borders = [], []
            for i in range(RAMP_STEPS):
                t = i / (RAMP_STEPS - 1.0)
                lab = [a + (b - a) * t for a, b in zip(neutral, end)]
                fills.append(_oklab_to_hex(lab))
                lightness = min(0.96, max(0.06, lab[0] + BORDER_DL[mode]))
                borders.append(_oklab_to_hex([lightness, lab[1], lab[2]]))
            arms.append({"fill": fills, "border": borders})
        ramps[mode] = arms
    return ramps


# --------------------------------------------------------------------------
# The type model
# --------------------------------------------------------------------------
#
# For each gene the page holds one share per graph, at every score step:
#
#     share[g] = degree_in_graph_g / total_degree_of_graph_g
#
# "What fraction of this graph's wiring runs through this gene?" — a question
# each graph answers on its own terms, so a 320x size difference between corpora
# does not decide the answer before the biology does. Everything below is
# computed from those shares and nothing else.
#
#     dominant = argmax share
#     T        = sum_g (1 - share[g] / max_share) / (N - 1)
#
# T is 0 when every graph gives the gene the same share (a gene the whole field
# writes about, wherever it looks) and 1 when exactly one graph has it at all (a
# marker). Note it depends only on RATIOS of shares, so it never needs the
# graphs to be the same size — which is the property that makes it survive this
# data set at all.
#
# The alternative was normalised Shannon entropy over the share distribution.
# It was rejected because it saturates: with eight types, one gene at 90%/10%
# and another at 55%/45%-across-two both score as "fairly specific", and the
# distinction between a marker and a merely lopsided gene is the whole point.
# T keeps that distinction because it measures every type against the leader
# rather than measuring the spread as a whole.
#
# Python ships the degree tables, not the answers, because the score slider
# moves: degrees AND totals both change at every step, so T is not a property of
# a gene, it is a property of a gene at a threshold.
#
# Note what the degree tables are counted over: every pair in each graph, not
# the pairs currently drawn. A gene's share is its standing in its whole corpus,
# so raising a publication slider changes what is on screen without changing
# what any gene *is*. That was already true when Python pre-filtered the payload
# (the tables were always built from the full pairs dict), which is why removing
# the filter left every T value in the sample set bit-for-bit identical.

def all_pairs(per_graph):
    """Every pair any graph asserts at FLOOR — a superset of every higher step.

    Python no longer decides which pairs are worth drawing. It used to drop
    anything under two publications before writing the file, which is fine right
    up until the reader is handed a publication slider: a control that reaches 1
    is a lie if the single-publication pairs were discarded upstream. So the
    whole union ships and every threshold is applied in the browser.

    That is the same bargain the score slider already struck, at the same price.
    The payload grows about fivefold — 14,566 pairs instead of 1,146 on the
    sample set, 8 MB instead of 1.6 — and in exchange every control on the page
    can be moved in both directions without regenerating anything.
    """
    return {key for pairs in per_graph for key in pairs}


def pub_ceiling(pairs):
    """Where this graph's publication slider should stop.

    Its own busiest pair, capped at PUB_CAP, floored at 2 so the control is
    never a single point. A graph whose pairs are all single-publication gets a
    1-to-2 slider, where 2 shows nothing — which is the true answer about that
    corpus, not a bug.
    """
    counts = [pubs_at(sents, FLOOR) for sents in pairs.values()]
    return max(2, min(PUB_CAP, max(counts) if counts else 2))


def degree_tables(pairs, genes):
    """Per-gene degree and whole-graph total degree, one entry per score step."""
    table = {g: [0] * len(SCORE_STEPS) for g in genes}
    totals = [0] * len(SCORE_STEPS)
    for key, sents in pairs.items():
        for i, score in enumerate(SCORE_STEPS):
            if not present_at(sents, score):
                continue
            totals[i] += 2                      # a pair contributes two endpoints
            for gene in key:
                if gene in table:
                    table[gene][i] += 1
    return table, totals


# --------------------------------------------------------------------------
# Significance — is a specificity claim more than the corpus sizes talking?
# --------------------------------------------------------------------------
#
# T is an effect size. It says how concentrated a gene's wiring is and nothing
# whatever about whether that concentration could have arisen by chance, which
# is why a gene resting on one paper scores 1.00 beside a well-attested marker.
# The q value below is the missing half.
#
# The test. Under the null a gene has no type preference, so the papers
# mentioning it fall across the corpora in proportion to how large those corpora
# are. For a gene with D papers in total and d in its dominant corpus, which
# holds a fraction p of all papers, the evidence against that null is the
# binomial tail P(X >= d | D, p). Benjamini-Hochberg over every gene tested at
# that step turns those into q values.
#
# Two choices in there are load-bearing.
#
# PAPERS, not partners. Sentences cluster inside publications — one sentence
# naming three genes creates three pairs at once — so counting partner slots
# treats one author's phrasing as several independent observations. On the
# sample set the partner-level test called 203 genes significant and the
# paper-level test 22. The nine-fold gap is that clustering, and the paper-level
# number is the honest one. This is only sound because the corpora turn out to
# be disjoint at the paper level: the sample set shares zero PMIDs between its
# four queries. If a pipeline ever produced overlapping corpora, the test would
# need rethinking and the q column would be a lie.
#
# What it does NOT license. The null concerns publishing, not biology. A small q
# says this gene is written about disproportionately in this subtype's
# literature — which is confounded by research fashion, by how the queries were
# drawn, and by extraction error. It does not say the gene is biologically
# restricted to that subtype.
#
# Power is wildly asymmetric and that asymmetry is real, not a defect. Where one
# corpus holds 78% of all papers, "found only there" is barely surprising and
# takes ~45 exclusive papers to establish, while two papers suffice in a corpus
# holding 0.3%. A gene failing FDR in the dominant literature has not been shown
# to be unspecific; it has not been shown to be anything.

def paper_tables(pairs, genes):
    """Distinct publications per gene, and per corpus, at each score step.

    The inference counterpart of degree_tables: that one counts partners, this
    counts the independent sources behind them.
    """
    table = {g: [0] * len(SCORE_STEPS) for g in genes}
    totals = [0] * len(SCORE_STEPS)
    for i, score in enumerate(SCORE_STEPS):
        per_gene = defaultdict(set)
        seen = set()
        for key, sents in pairs.items():
            pmids = {s["p"] for s in sents if s["c"] >= score}
            if not pmids:
                continue
            seen |= pmids
            for gene in key:
                if gene in table:
                    per_gene[gene] |= pmids
        for gene, pmids in per_gene.items():
            table[gene][i] = len(pmids)
        totals[i] = len(seen)
    return table, totals


def year_span(per_graph):
    """Earliest and latest publication year in the evidence, and how much of it
    has no year at all.

    The undated share is not a rounding error — nearly a fifth of the sentences
    in the sample set carry `yr: null` — so the page cannot quietly decide their
    fate. Dropping them the moment the window narrows would delete a fifth of
    the evidence for a reason never stated; keeping them always would make the
    filter silently leaky. The count is therefore surfaced and the choice is the
    reader's.
    """
    years, unknown = [], 0
    for pairs in per_graph:
        for sents in pairs.values():
            for s in sents:
                if s["y"] is None:
                    unknown += 1
                else:
                    years.append(s["y"])
    if not years:
        return None, None, unknown
    return min(years), max(years), unknown


def paper_overlap(per_graph):
    """Publications counted in more than one corpus, per score step.

    The q column's null treats the corpora as independent samples of the
    literature. That holds only if no publication is counted twice, and nothing
    upstream guarantees it — two PubMed queries that are not mutually exclusive
    will share papers, and a shared paper is then counted as two independent
    observations, making every q derived from it optimistic.

    So it is measured rather than assumed. Returns the duplicated-paper count
    per step (by inclusion of the union, so a paper in three corpora counts
    twice over) and the pairwise intersection matrix, which is what lets the
    page name the corpora responsible instead of just raising an alarm.
    """
    dup, matrix = [], []
    for score in SCORE_STEPS:
        sets = []
        for pairs in per_graph:
            pmids = set()
            for sents in pairs.values():
                pmids |= {s["p"] for s in sents if s["c"] >= score}
            sets.append(pmids)
        union = set()
        for pmids in sets:
            union |= pmids
        dup.append(sum(len(p) for p in sets) - len(union))
        matrix.append([[0 if a == b else len(sets[a] & sets[b])
                        for b in range(len(sets))]
                       for a in range(len(sets))])
    return dup, matrix


def _binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), summed in log space."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))
    return min(1.0, total)


def _bh(pvals):
    """Benjamini-Hochberg: {key: p} -> {key: q}, monotone from the top down."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    qs, running = {}, 1.0
    for rank in range(m, 0, -1):
        key = order[rank - 1]
        running = min(running, pvals[key] * m / rank)
        qs[key] = running
    return qs


def qvalue_tables(ptables, ptotals, genes):
    """Per-gene BH q values, one per score step.

    The gene is tested against its own dominant corpus — the one claiming it —
    so the q answers the question the colour on screen is already asserting.
    Genes with no publications at a step are untested and carry q = 1.
    """
    qtable = {g: [1.0] * len(SCORE_STEPS) for g in genes}
    # how many genes were actually tested at each step — the m in BH, and the
    # number the table's note needs to explain what the correction is for
    tested = [0] * len(SCORE_STEPS)
    for i in range(len(SCORE_STEPS)):
        grand = sum(t[i] for t in ptotals)
        if not grand:
            continue
        pvals = {}
        for gene in genes:
            counts = [t[gene][i] for t in ptables]
            total = sum(counts)
            if not total:
                continue
            # dominant by share of its own corpus, matching profile() in the page
            shares = [
                counts[c] / float(ptotals[c][i]) if ptotals[c][i] else 0.0
                for c in range(len(counts))
            ]
            dom = shares.index(max(shares))
            pvals[gene] = _binom_sf(counts[dom], total,
                                    ptotals[dom][i] / float(grand))
        tested[i] = len(pvals)
        for gene, q in _bh(pvals).items():
            qtable[gene][i] = float("%.3g" % q)
    return qtable, tested


def share_floors(totals):
    """The minimum share denominator at each score step.

    Per step, not once: every corpus shrinks as the score rises, so a constant
    floor would be weightless at 0.75 and crushing at 0.99.
    """
    floors = []
    for i in range(len(SCORE_STEPS)):
        col = sorted(t[i] for t in totals)
        mid = len(col) // 2
        median = col[mid] if len(col) % 2 else (col[mid - 1] + col[mid]) / 2.0
        floors.append(int(round(SHARE_FLOOR_FRACTION * median)))
    return floors


def build_model(per_graph):
    """Assemble the payload: one evidence list and one degree row per graph.

    Per-graph values are kept as parallel arrays indexed by graph rather than
    keyed by label — labels come from the prompt and can be anything, including
    the same word twice, and an index can never collide with itself.
    """
    selected = all_pairs(per_graph)
    edges = [
        {"f": key[0], "t": key[1],
         "v": [pairs.get(key, []) for pairs in per_graph]}
        for key in sorted(selected)
    ]
    drawn = sorted({g for key in selected for g in key})
    tables, totals = [], []
    for pairs in per_graph:
        table, total = degree_tables(pairs, set(drawn))
        tables.append(table)
        totals.append(total)
    ptables, ptotals = [], []
    for pairs in per_graph:
        table, total = paper_tables(pairs, set(drawn))
        ptables.append(table)
        ptotals.append(total)
    qtable, tested = qvalue_tables(ptables, ptotals, drawn)
    nodes = [{"id": g, "d": [t[g] for t in tables], "q": qtable[g]}
             for g in drawn]
    return nodes, edges, totals, ptotals, tested


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script>__VIS__</script>
<style>
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --surface-2:#f2f2ef; --border:#dcdcd6;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#77766f;
  --accent:#1c5cab; --bad:#c0392b;
  --net-bg:#fcfcfb;
__VARSL__
  /* evidence tooltip — the source graphs' own palette */
  --tip-bg:#ffffff; --tip-fg:#1a1a1a; --tip-bd:#999999; --tip-rule:#e3e3e3;
  --tip-mut:#5b6677; --tip-more:#888888; --tip-head:#333333;
  --tip-pm-bg:#eef3fb; --tip-pm-fg:#2b6cb0; --tip-pm-bg-hover:#d6e6fb;
  --tip-mark:#ffe680;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#232322; --border:#3a3a37;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
    --accent:#3987e5; --bad:#e66767;
    --net-bg:#1a1a19;
__VARSD__
    --tip-bg:#232322; --tip-fg:#e9e8e1; --tip-bd:#55554f; --tip-rule:#3a3a37;
    --tip-mut:#a9a89f; --tip-more:#8f8e85; --tip-head:#d8d7cf;
    --tip-pm-bg:#1e3555; --tip-pm-fg:#8fbdf0; --tip-pm-bg-hover:#27456e;
    --tip-mark:#ffe680;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232322; --border:#3a3a37;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
  --accent:#3987e5; --bad:#e66767;
  --net-bg:#1a1a19;
__VARSD__
  --tip-bg:#232322; --tip-fg:#e9e8e1; --tip-bd:#55554f; --tip-rule:#3a3a37;
  --tip-mut:#a9a89f; --tip-more:#8f8e85; --tip-head:#d8d7cf;
  --tip-pm-bg:#1e3555; --tip-pm-fg:#8fbdf0; --tip-pm-bg-hover:#27456e;
  --tip-mark:#ffe680;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-1);color:var(--text-primary);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  /* a hard viewport box, so the rail scrolls internally instead of the page
     growing past the fold */
  display:flex;flex-direction:column;height:100vh;overflow:hidden;}

/* One rail on the right carries title, controls and legend; the graph gets
   everything else. The overlays swap in over the canvas rather than stacking
   below it — parked under a full-height canvas they opened off-screen, which
   read as a dead button. */
.stage{flex:1;display:flex;align-items:stretch;min-height:0;}
.canvas{flex:1;min-width:0;position:relative;background:var(--net-bg);}
#net{position:absolute;inset:0;}
/* The rail scrolls internally so the legend never pushes the page past the fold. */
.side{flex:none;width:286px;overflow:hidden;
  border-left:1px solid var(--border);background:var(--surface-1);
  display:flex;flex-direction:column;}
/* The left rail (title, intro, the three overlay buttons) mirrors the right one,
   so its divider sits on its right edge instead of its left. */
.sideleft{border-left:none;border-right:1px solid var(--border);}
.railtop{flex:1;min-height:0;overflow-y:auto;padding:16px 18px;
  display:flex;flex-direction:column;gap:12px;}
h1{margin:0;font-size:20px;font-weight:700;letter-spacing:-0.01em;line-height:1.25;}
.sub{margin:9px 0 0;color:var(--text-secondary);font-size:12.5px;}
.sub + .sub{margin-top:3px;}
.controls{display:flex;flex-wrap:wrap;gap:9px;align-items:center;}
/* These three buttons are the only controls that open a view instead of changing
   the graph, so they are framed and captioned as a group. The border is the plain
   rail border, not an accent: it should read as a grouping, not as emphasis. */
.panelbox{border:1px solid var(--border);border-radius:8px;padding:9px 10px 10px;}
.panelbox .boxcap{margin:0 0 7px;font-size:11.5px;font-weight:600;
  color:var(--text-muted);}
select,button{font:inherit;font-size:13px;padding:5px 9px;border-radius:6px;
  border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary);}
button{cursor:pointer} button:hover{background:var(--surface-2)}
/* A pressed toggle has to look held down, not merely hovered — it changes what
   the graph contains, and the reader needs to see that from across the rail. */
button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);
  color:var(--net-bg);font-weight:600;}
button[aria-pressed=true]:hover{filter:brightness(1.08);}
/* The view/tool buttons (Remove isolated, Show table, Summary, How specificity)
   carry an accent outline — distinct from the plain rail buttons, a tier below the
   filled Default settings. Only colour and weight change, so the box is no larger.
   A pressed tool (Remove isolated when on) still fills accent: [aria-pressed=true]
   above is more specific than .tool, so it wins. */
.tool{border-color:var(--accent);color:var(--accent);font-weight:600;}
.tool:hover{background:var(--surface-2);}
.sliders{display:flex;flex-direction:column;gap:7px;}
.srow{display:flex;align-items:center;gap:8px;}
.slab{flex:none;width:66px;font-size:12.5px;color:var(--text-secondary);}
.srow input[type=range]{flex:1;min-width:0;accent-color:var(--accent);cursor:pointer;}
.sval{flex:none;font-size:13px;font-weight:600;color:var(--text-primary);
  font-variant-numeric:tabular-nums;min-width:30px;text-align:right;}
.slabel{font-size:13px;font-weight:600;color:var(--text-secondary);display:block;
  margin-bottom:5px;}
/* one publication row per type: swatch · label · track · value. The label is
   prompt-supplied, so it must survive any length without shoving the track. */
.prow{display:flex;align-items:center;gap:7px;margin-bottom:6px;}
.prow:last-of-type{margin-bottom:0;}
.prow .pdot{width:9px;height:9px;border-radius:50%;flex:none;}
.prow label{flex:none;width:74px;font-size:11.5px;line-height:1.25;color:var(--text-secondary);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer;}
.prow input[type=range]{flex:1;min-width:0;accent-color:var(--accent);cursor:pointer;}
.prow .sval{min-width:16px;}
/* four-digit years need more room than the 2-char values elsewhere */
.yrow .sval{min-width:38px;}
.ychk{display:flex;align-items:flex-start;gap:7px;margin:10px 0 0;font-size:12px;
  line-height:1.4;color:var(--text-secondary);cursor:pointer;}
.ychk input{accent-color:var(--accent);cursor:pointer;flex:none;margin-top:2px;}
/* compact preview of the thresholds while the section is collapsed */
.mnums{flex:none;font-size:11.5px;color:var(--text-muted);
  font-variant-numeric:tabular-nums;white-space:nowrap;}
.leg[open] .mnums{display:none;}
input[type=search]{width:100%;font:inherit;font-size:13px;padding:5px 8px;border-radius:6px;
  border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary);}
input[type=search]:focus{outline:2px solid var(--accent);outline-offset:1px;}
/* context in secondary ink, the actionable instruction bold in primary */
.hgnc{margin:14px 0 0;font-size:12px;color:var(--text-secondary);line-height:1.45;}
.hgnc strong{font-weight:600;color:var(--text-primary);}
.hgnc a{color:inherit;text-decoration:underline;text-underline-offset:2px;}
.hgnc a:hover{text-decoration-thickness:2px;}
.hgnc a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.hint{font-size:11.5px;color:var(--text-muted);margin:5px 0 0;}
.hint code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;}
/* error ink is its own variable rather than a literal: every hue on this page
   names a type, so the one colour that must never be mistaken for an encoding
   still has to stay legible in both themes */
.bad{color:var(--bad);}
.pair{display:flex;gap:7px;align-items:center;}
.pair input[type=search]{flex:1;min-width:0;}
.pair select{flex:none;padding:5px 4px;}
.count{margin:10px 0 0;font-size:12.5px;color:var(--text-secondary);}
.count b{color:var(--text-primary);font-weight:600;}
/* Filled accent to stand out from the plain rail buttons — the reset is the one
   control a lost reader reaches for. Only colour and weight change; padding,
   font-size, border width and width are untouched, so the button is no larger. */
.resetbtn{margin-top:10px;width:100%;font-size:12.5px;font-weight:700;
  background:var(--accent);border-color:var(--accent);color:var(--net-bg);
  letter-spacing:.02em;}
.resetbtn:hover{background:var(--accent);filter:brightness(1.08);}
/* 'Significant only' borrows the filled look of Default settings so it reads as a
   primary action, in green to stay clear of the accent-blue toggles around it. The
   ID + attribute selectors below out-specify the generic button[aria-pressed=true]
   rule, so its pressed state keeps the green fill instead of flipping to accent. */
#sigbtn{width:100%;margin-top:0;font-size:12.5px;font-weight:700;letter-spacing:.02em;
  background:#42993f;border-color:#42993f;color:var(--net-bg);}
#sigbtn:hover{background:#42993f;filter:brightness(1.08);}
#sigbtn[aria-pressed=true]{background:#42993f;border-color:#42993f;color:var(--net-bg);
  filter:brightness(0.88);}
#sigbtn[aria-pressed=true]:hover{filter:brightness(0.96);}
.sect{font-size:11px;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;
  color:var(--text-muted);margin:0 0 9px;}
/* Collapsed legends keep a colour preview in the summary, so the encoding is
   still readable at a glance without spending the whole rail on it. */
.leg summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:9px;
  padding:1px 0;}
.leg summary::-webkit-details-marker{display:none;}
.leg summary .sect{margin:0;}
.leg summary::after{content:'';margin-left:auto;width:6px;height:6px;flex:none;
  border-right:1.5px solid var(--text-muted);border-bottom:1.5px solid var(--text-muted);
  transform:rotate(45deg) translate(-2px,-2px);transition:transform .15s;}
.leg[open] summary::after{transform:rotate(-135deg);}
.leg summary:hover .sect{color:var(--text-secondary);}
.leg .body{padding-top:12px;}
.mkeys{display:flex;gap:4px;flex:none;flex-wrap:wrap;max-width:120px;}
.mkeys i{width:13px;height:3px;border-radius:2px;}
.mdots{display:flex;gap:3px;flex:none;flex-wrap:wrap;max-width:120px;}
.mdots i{width:9px;height:9px;border-radius:50%;}
.leg[open] .mkeys,.leg[open] .mdots{display:none;}
/* One ramp arm per type: neutral on the left, that type's pole on the right.
   Stacked rather than fanned, because a fan needs radial reading and this is
   the same comparison the table makes — row against row. */
.arm{margin-bottom:11px;}
.arm:last-child{margin-bottom:0;}
.armhd{display:flex;align-items:center;justify-content:space-between;gap:8px;
  font-size:12.5px;margin-bottom:4px;}
.armhd label{display:flex;align-items:center;gap:6px;min-width:0;cursor:pointer;}
.armhd .nmx{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  color:var(--text-primary);}
.armhd input[type=checkbox]{accent-color:var(--accent);cursor:pointer;flex:none;}
.armhd .cnt{flex:none;font-size:11.5px;color:var(--text-muted);
  font-variant-numeric:tabular-nums;}
.hramp{position:relative;display:block;width:100%;height:11px;border-radius:6px;
  border:1px solid var(--border);}
/* the excluded ends of the scale, greyed back over the ramp itself */
.cut{position:absolute;top:0;bottom:0;background:var(--surface-1);opacity:.76;
  pointer-events:none;border-radius:6px;}
.hticks{display:flex;justify-content:space-between;gap:6px;margin-top:7px;}
.hticks span{font-size:11px;color:var(--text-secondary);line-height:1.2;white-space:nowrap;}
.hticks b{color:var(--text-primary);font-weight:600;}
.skey{display:flex;flex-direction:column;gap:9px;}
.lg{display:flex;align-items:center;gap:9px;font-size:13px;color:var(--text-primary);}
.lg .nmx{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.ln{width:22px;height:3px;border-radius:2px;flex:none;}
.ln.dash{background:repeating-linear-gradient(to right,
  var(--text-secondary) 0 5px,transparent 5px 9px);}
.dot{width:11px;height:11px;border-radius:50%;flex:none;}
.note{font-size:12px;color:var(--text-muted);margin:9px 0 0;}
/* narrow viewports: the panels stack — left panel (title + buttons) above the
   graph as a header, the right rail (sliders + legends) below it. */
@media (max-width:820px){
  body{height:auto;overflow:visible;}
  .stage{flex-direction:column;}
  .canvas{height:60vh;min-height:340px;flex:none;}
  .side{width:auto;overflow:visible;
    border-left:none;border-right:none;border-bottom:1px solid var(--border);}
  .sideleft{order:-1;border-bottom:1px solid var(--border);}
  .railtop{overflow:visible;}
}
/* cover the canvas exactly, so the toggles are always visible */
.tblwrap,.docwrap,.sumwrap{position:absolute;inset:0;display:none;overflow:auto;
  padding:14px 18px 24px;background:var(--surface-1);}
.tblwrap.on,.docwrap.on,.sumwrap.on{display:block}
.docwrap,.sumwrap{padding:24px 28px 48px;}
.doc{max-width:80ch;}
.doc h2{margin:0 0 4px;font-size:19px;font-weight:700;letter-spacing:-0.01em;}
.doc .std{margin:0 0 22px;color:var(--text-secondary);font-size:13px;}
.doc h3{margin:24px 0 6px;font-size:14px;font-weight:600;}
.doc h3 .n{display:inline-block;min-width:20px;color:var(--text-muted);font-weight:700;}
.doc h3 .mut{color:var(--text-muted);font-weight:400;}
.doc p{margin:0 0 8px;font-size:13.5px;color:var(--text-secondary);max-width:74ch;}
.doc b{color:var(--text-primary);}
.doc .fml{display:block;margin:9px 0;padding:9px 12px;border-radius:7px;
  background:var(--surface-2);border:1px solid var(--border);
  font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px;
  color:var(--text-primary);white-space:pre;overflow-x:auto;}
.doc .why{margin:22px 0 0;padding:12px 14px;border-radius:8px;
  background:var(--surface-2);border:1px solid var(--border);}
.doc .why p{margin:0;}
.doc table{margin:10px 0 0;}
.doc td.g{font-weight:600;color:var(--text-primary);}
/* the discredited column: muted, never coloured — every hue on this page names
   a type, and a coloured "wrong answer" would read as an encoding */
.doc .raw{color:var(--text-muted);text-decoration:line-through;
  text-decoration-color:var(--text-muted);text-decoration-thickness:1px;}
.doc th.raw{text-decoration:none;}
table{border-collapse:collapse;font-size:13px;}
th,td{text-align:left;padding:5px 12px 5px 0;border-bottom:1px solid var(--border);white-space:nowrap;}
th{color:var(--text-secondary);font-weight:600;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;padding-right:18px;}
.tag{display:inline-flex;align-items:center;gap:6px;}
/* the pairwise overlap matrix: a heatmap of Jaccard between corpora */
.mx td.c{text-align:center;font-variant-numeric:tabular-nums;padding:5px 10px;
  border:1px solid var(--border);color:var(--text-primary);}
.mx td.self{color:var(--text-muted);}
.mx th.rh{padding-right:14px;font-weight:600;color:var(--text-primary);}
.mx th.ch{text-align:center;padding:5px 10px;}
/* q is weighted rather than coloured: every hue on this page names a type, and
   a green "significant" would read as small-cell membership */
/* the table's own header note: what the two columns mean, before the numbers.
   Sits above a table that can be 640px wide and thousands of rows long, so it
   is capped to a readable measure rather than stretching with the table. */
.tnote{max-width:78ch;margin:0 0 18px;font-size:12.5px;line-height:1.5;
  color:var(--text-secondary);}
.tnote h2{margin:0 0 10px;font-size:15px;font-weight:700;letter-spacing:-0.01em;
  color:var(--text-primary);}
.tnote dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;
  align-items:baseline;}
.tnote dt{font-weight:600;color:var(--text-primary);white-space:nowrap;}
.tnote dd{margin:0;}
.tnote p{margin:10px 0 0;}
.tnote b{color:var(--text-primary);}
.tnote .base{margin:9px 0 0;padding:9px 12px;border-radius:7px;
  background:var(--surface-2);border:1px solid var(--border);}
.tnote .base table{min-width:0;font-size:12px;margin:0;}
.tnote .base th,.tnote .base td{padding:2px 14px 2px 0;border-bottom:none;}
.tnote .warn{margin-top:10px;color:var(--text-muted);}
/* The overlap verdict. Carried by a rule and weight rather than a hue, because
   every colour on this page names a cancer type and a red banner here would
   read as one of them. */
.tnote .alert,.tnote .ok{margin:12px 0 0;padding:9px 12px;border-radius:7px;
  background:var(--surface-2);border:1px solid var(--border);}
.tnote .alert{border-left:3px solid var(--text-primary);color:var(--text-primary);}
.tnote .ok{color:var(--text-muted);}
td.sig{font-weight:600;color:var(--text-primary);}
td.ins{color:var(--text-muted);}
.sortable th{cursor:pointer;user-select:none;}
.sortable th:hover{color:var(--text-primary);}
.sortable th .ar{color:var(--text-muted);font-size:10px;}

/* Evidence tooltip — same anatomy and styling as the source graphs (PMID chip,
   score/year in muted brackets, rule between sentences), themed for dark. */
div.vis-tooltip{max-width:480px!important;white-space:normal!important;
  background:var(--tip-bg)!important;color:var(--tip-fg)!important;
  border:1px solid var(--tip-bd)!important;border-radius:8px!important;
  padding:8px 10px!important;box-shadow:0 4px 16px rgba(0,0,0,.35)!important;
  font:12px/1.45 Segoe UI,Arial,sans-serif!important}
@media (max-width:600px){div.vis-tooltip{max-width:88vw!important}}
div.vis-tooltip .eth{font-size:13px;margin-bottom:6px}
div.vis-tooltip .esrc{margin-top:8px;font-size:12px;font-weight:600;color:var(--tip-head)}
div.vis-tooltip .stip{padding:3px 0;border-top:1px solid var(--tip-rule)}
div.vis-tooltip .mut{color:var(--tip-mut);font-size:12px}
div.vis-tooltip .more{margin-top:5px;color:var(--tip-more);font-style:italic}
div.vis-tooltip .pm{display:inline-block;background:var(--tip-pm-bg);color:var(--tip-pm-fg);
  border-radius:4px;padding:0 5px;margin-right:5px;font-weight:600;font-size:11px;
  text-decoration:none}
div.vis-tooltip .pm:hover{background:var(--tip-pm-bg-hover);text-decoration:underline}
/* forced dark ink: the source leaves this at `inherit`, which is fine on its
   always-white tooltip but would put pale text on yellow in dark mode */
div.vis-tooltip mark{background:var(--tip-mark);color:#1a1a1a;border-radius:2px;padding:0 1px}
</style></head><body>
<div class="stage">
  <aside class="side sideleft">
   <div class="railtop">
    <div>
      <h1>__TITLE__</h1>
      <p class="sub">__NTYPES__ types. A pair is shown when any one type meets its publication minimum.</p>
      <p class="sub">Hover a connector for its sentences.</p>
    </div>
    <div>
      <p class="hgnc">Gene symbols and names in the sentences are normalized to HGNC symbols displayed
      as nodes. <strong>Use the <a href="https://www.genenames.org/" target="_blank" rel="noopener">HGNC
      website</a> to look up HGNC symbols when the nodes and sentences do not use the same
      terminology.</strong></p>
      <div style="margin-top:12px">
        <label class="slabel" for="genefilter">Focus on gene</label>
        <div class="pair">
          <input id="genefilter" type="search" autocomplete="off" spellcheck="false"
                 list="genelist" placeholder="gene symbol">
          <select id="hops" aria-label="Neighbourhood depth">
            <option value="1">1 hop</option>
            <option value="2">2 hops</option>
            <option value="3">3 hops</option>
          </select>
        </div>
        <datalist id="genelist"></datalist>
        <p class="hint" id="ghint">Exact name, else prefix match.</p>
        <p class="hint" style="font-weight:700"><strong>Significant nodes or node-pairs may stand alone when they don’t connect to other significant nodes. Hop count can also add or remove edges between isolated significant nodes.</strong></p>
        <div class="controls" style="margin-top:8px">
          <button id="isobtn" class="tool" aria-pressed="false">Remove isolated nodes</button>
        </div>
      </div>
      <div style="margin-top:12px">
        <label class="slabel" for="textfilter">Match text in sentence</label>
        <input id="textfilter" type="search" autocomplete="off" spellcheck="false"
               placeholder="e.g. phosphorylat or /inhibit(s|ed)?/">
        <p class="hint" id="thint">Substring, or <code>/regex/</code>.</p>
      </div>
      <div class="panelbox" style="margin-top:12px">
        <p class="boxcap" id="boxcap">Click for more details</p>
        <div class="controls" role="group" aria-labelledby="boxcap">
          <button id="tbtn" class="tool" aria-expanded="false" aria-controls="tblwrap">Show table</button>
          <button id="sbtn" class="tool" aria-expanded="false" aria-controls="sumwrap">Summary</button>
          <button id="dbtn" class="tool" aria-expanded="false" aria-controls="docwrap">How specificity works</button>
        </div>
      </div>
    </div>
    <details class="leg" id="legpub" open>
      <summary><span class="sect">Min publications</span>
        <span class="mnums" id="pubsum"></span>
      </summary>
      <div class="body">
__PUBROWS__
        <p class="hint" id="pubhint"></p>
        <p class="note">A pair survives if <b>any one</b> type backs it with at least that
        type's minimum — so raising one slider thins that type's exclusive pairs without
        touching relationships another type attests on its own.</p>
      </div>
    </details>
    <details class="leg" id="legyear">
      <summary><span class="sect">Publication years</span>
        <span class="mnums" id="yearsum"></span>
      </summary>
      <div class="body">
        <div class="sliders yrow">
          <div class="srow">
            <label class="slab" for="yrfrom">From</label>
            <input id="yrfrom" type="range" min="__YMIN__" max="__YMAX__" step="1"
                   value="__YMIN__" aria-label="Earliest publication year">
            <span class="sval" id="yrfromv">__YMIN__</span>
          </div>
          <div class="srow">
            <label class="slab" for="yrto">To</label>
            <input id="yrto" type="range" min="__YMIN__" max="__YMAX__" step="1"
                   value="__YMAX__" aria-label="Latest publication year">
            <span class="sval" id="yrtov">__YMAX__</span>
          </div>
        </div>
        <label class="ychk"><input type="checkbox" id="yrnull" checked>
          Include the __YNULL__ undated sentences</label>
        <p class="hint" id="yrhint"></p>
        <p class="note">Filters the evidence itself, so publication counts, edge widths and
        which pairs survive all follow the window. Specificity and q do not — they describe
        the whole corpus, not the slice on screen.</p>
      </div>
    </details>
   </div>
  </aside>
  <div class="canvas">
    <div id="net"></div>
    <div class="tblwrap" id="tblwrap">
      <div class="tnote" id="tblnote"></div>
      <table id="tbl" class="sortable"></table>
    </div>
    <div class="sumwrap" id="sumwrap"><div class="doc" id="sum"></div></div>
    <div class="docwrap" id="docwrap"><div class="doc" id="doc"></div></div>
  </div>
  <aside class="side">
   <div class="railtop">
    <div>
      <div class="sliders" style="margin-top:0">
        <div class="srow">
          <label class="slab" for="conf">Min score</label>
          <input id="conf" type="range" min="0" max="__MAXSTEP__" step="1" value="__DEFSTEP__"
                 aria-label="Minimum relationship score">
          <span class="sval" id="confval">__DEFCONF__</span>
        </div>
        <p class="hint" id="confhint" hidden></p>
        <div class="srow">
          <label class="slab" for="mincluster">Min cluster</label>
          <input id="mincluster" type="range" min="2" max="12" step="1" value="__DEFCLUSTER__"
                 aria-label="Minimum connected cluster size">
          <span class="sval" id="mcval">__DEFCLUSTER__</span>
        </div>
        <p class="hint" id="mchint" hidden></p>
        <div class="panelbox" style="margin-top:10px">
          <button class="resetbtn" id="rbtn" style="margin-top:0"
                  title="Score ≥ __DEFCONF__, min cluster __DEFCLUSTER__, T __DEFTMIN__–__DEFTMAX__, max q __DEFFDRV__, publications __DEFPUB__, all types and years, no gene focus or text search">Default settings</button>
          <p class="boxcap" style="margin:7px 0 0">Controls become stricter as you move right. Scroll down for all control options. Some settings can slow graph rendering.</p>
        </div>
      </div>
      <p class="count" id="count"></p>
    </div>
    <details class="leg" id="legspec" open>
      <summary><span class="sect">Gene specificity</span>
        <span class="mdots" role="img" aria-label="one colour per type">__MDOTS__</span>
      </summary>
      <div class="body">
__ARMS__
        <div class="hticks">
          <span><b>ubiquitous</b></span>
          <span>T</span>
          <span><b>exclusive</b></span>
        </div>
        <div class="sliders" style="margin-top:11px">
          <div class="srow">
            <label class="slab" for="spechi">T max</label>
            <input id="spechi" type="range" min="0" max="1" step="0.05" value="__DEFTMAX__"
                   aria-label="Show genes up to this specificity">
            <span class="sval" id="spechiv">__DEFTMAX__</span>
          </div>
          <div class="srow">
            <label class="slab" for="speclo">T min</label>
            <input id="speclo" type="range" min="0" max="1" step="0.05" value="__DEFTMIN__"
                   aria-label="Show genes down to this specificity">
            <span class="sval" id="speclov">__DEFTMIN__</span>
          </div>
        </div>
        <p class="hint" id="spechint" hidden></p>
        <div class="sliders" style="margin-top:11px">
          <div class="srow">
            <label class="slab" for="fdr" title="Benjamini-Hochberg false discovery rate">Max q (FDR)</label>
            <input id="fdr" type="range" min="0" max="__MAXFDR__" step="1" value="__DEFFDR__"
                   aria-label="Maximum false discovery rate">
            <span class="sval" id="fdrval">off</span>
          </div>
        </div>
        <p class="hint" id="fdrhint">T is the effect size; q is whether it is more than corpus size talking.</p>
        <div class="controls" style="margin-top:9px">
          <button id="sigbtn" aria-pressed="false">Significant only</button>
          <button id="spreadbtn" class="tool" aria-pressed="false" hidden>Spread out</button>
        </div>
        <p class="hint" id="sighint">Off — genes above the ceiling are drawn as context.</p>
        <p class="note">Untick a type to hide the genes it dominates. Share of each
        graph's own wiring, so the biggest corpus doesn't skew it — and floored, so
        the smallest can't either. See the explainer.</p>
      </div>
    </details>
    <details class="leg" id="legrel">
      <summary><span class="sect">Relationship</span>
        <span class="mkeys" role="img" aria-label="one colour per type, grey shared">__MKEYS__</span>
      </summary>
      <div class="body">
        <div class="skey">
__ELEG__
          <span class="lg"><span class="ln" style="background:var(--edge-shared)"></span>shared pair</span>
          <span class="lg"><span class="ln dash"></span>negated</span>
        </div>
        <p class="note">Width = publications. Node size = partners.</p>
      </div>
    </details>
   </div>
  </aside>
</div>
<script>
const DATA=__PAYLOAD__, RAMP=__RAMPJS__, STEPS=__STEPSJS__, TOT=__TOTALSJS__,
      DEFSTEP=__DEFSTEP__, DEFPUB=__DEFPUB__, PUBTRUE=__PUBTRUEJS__,
      DEFCLUSTER=__DEFCLUSTER__, DEFTMIN=__DEFTMIN__, DEFTMAX=__DEFTMAX__,
      FLOORS=__FLOORSJS__, FDRS=__FDRSJS__, DEFFDR=__DEFFDR__,
      PTOT=__PTOTJS__, NTEST=__NTESTJS__,
      OVDUP=__OVDUPJS__, OVMAT=__OVMATJS__;
// node outline weights: plain, significant, and each one's selected form
const BD=__BD__, BDSIG=__BDSIG__, BDSEL=__BDSEL__, BDSIGSEL=__BDSIGSEL__;
const LOWSCORE=__LOWSCORE__;   // below this the score slider flags the band
const YMIN=__YMIN__, YMAX=__YMAX__, YNULL=__YNULL__;
// the types, named at the prompt; every label below is built from these
const LABS=__LABSJS__, N=LABS.length;
// formula-safe form of a label: 'some disease' -> 'some_disease'
function slug(s){return s.trim().replace(/[^A-Za-z0-9]+/g,'_').replace(/^_|_$/g,'').toLowerCase()||'x';}
// slugs must stay distinct or the worked formulae would name two graphs alike
const SLUGS=(()=>{const seen={},out=[];LABS.forEach(l=>{let s=slug(l);
  if(seen[s]!==undefined){seen[s]++;s=s+'_'+seen[s];}else seen[s]=1;out.push(s);});return out;})();
const NODE_BY_ID={}; DATA.nodes.forEach(n=>{NODE_BY_ID[n.id]=n;});
function cvar(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
function isDark(){const t=document.documentElement.getAttribute('data-theme');
  if(t)return t==='dark';return matchMedia('(prefers-color-scheme: dark)').matches;}
function ramp(g){return RAMP[isDark()?'dark':'light'][g];}
function stepIdx(){const v=parseInt(document.getElementById('conf').value);
  return isNaN(v)?DEFSTEP:Math.max(0,Math.min(STEPS.length-1,v));}

// --- the type profile of one gene at one score -----------------------------
// Everything the page encodes about a gene comes out of here, and all of it is
// derived from shares — never from raw degrees. TOT[g][i] is 2x the pair count
// of graph g at step i, so share is the fraction of that graph's endpoints
// landing on this gene.
function profile(n,i){
  const sh=[]; let sum=0,mx=0,dom=0;
  for(let g=0;g<N;g++){
    // FLOORS[i] is the crux: a corpus too thin to estimate a share is divided
    // by the floor instead of by its own tiny total, so it cannot win on
    // arithmetic. TOT stays the true total everywhere it is reported.
    const t=Math.max(TOT[g][i]||0,FLOORS[i]), v=t?n.d[g][i]/t:0;
    sh.push(v); sum+=v;
    if(v>mx){mx=v;dom=g;}
  }
  // T: every share measured against the leader, averaged. 0 = the graphs agree
  // completely about how central this gene is; 1 = only one graph has it.
  let tau=0;
  if(mx>0&&N>1){for(let g=0;g<N;g++)tau+=1-sh[g]/mx;tau/=(N-1);}
  return {sh:sh,tau:tau,dom:dom,sum:sum,mx:mx,
          q:sh.map(v=>sum?v/sum:0),          // share of share — reads as a %
          seen:sh.filter(v=>v>0).length};
}
function stepOf(tau){const n=ramp(0).fill.length;
  return Math.max(0,Math.min(n-1,Math.round(tau*(n-1))));}
function fill(p){return ramp(p.dom).fill[stepOf(p.tau)];}
function bord(p){return ramp(p.dom).border[stepOf(p.tau)];}
// Words for a number, bucketed only for readability — colour uses the
// continuous value. Below the first cut the dominant type is not worth naming:
// the gene is not that type's, it is everyone's.
function label(p){const L=LABS[p.dom];
  return p.tau>=0.999?'exclusive to '+L:p.tau>=0.7?'strongly '+L+'-specific':
         p.tau>=0.35?L+'-leaning':p.tau>=0.12?'weakly '+L+'-leaning':'ubiquitous';}
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function pmA(p){return '<a class=pm target=_blank rel=noopener href="https://www.ncbi.nlm.nih.gov/pmc/articles/'+
  encodeURIComponent(p)+'/">'+esc(p)+'</a>';}
// --- text search, same contract as the source graphs -----------------------
// plain query = case-insensitive substring; /.../flags = regex (bad regex falls
// back to a literal match, as in the source)
function reEsc(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');}
function activeText(){return (document.getElementById('textfilter').value||'').trim();}
function textMatcher(q){
  if(!q)return null;
  const m=/^\/(.*)\/([a-z]*)$/.exec(q);
  let src=null,flags='i';
  if(m){try{new RegExp(m[1],m[2]);src=m[1];
    flags=(m[2].indexOf('i')>=0?m[2]:m[2]+'i').replace(/g/g,'');}catch(err){src=null;}}
  if(src===null)src=reEsc(q);
  try{return {test:t=>new RegExp(src,flags).test(t||''),hlre:new RegExp(src,flags+'g')};}
  catch(err){return null;}
}
let TM=null;   // matcher in force for the current view
// --- specificity brush -----------------------------------------------------
// A brush over the ramp: keep only the genes whose T falls inside it. The
// handles are independent sliders, so they can cross — read them as a set
// rather than trusting which is which.
function specRange(){
  let hi=parseFloat(document.getElementById('spechi').value);
  let lo=parseFloat(document.getElementById('speclo').value);
  if(isNaN(hi))hi=1;
  if(isNaN(lo))lo=0;
  return [Math.min(lo,hi),Math.max(lo,hi)];
}
// every arm runs 0 at the left to 1 at the right, so the cuts are the same on
// each — one range, N ramps
function paintRange(lo,hi){
  for(let g=0;g<N;g++){
    const A=document.getElementById('cutL'+g), B=document.getElementById('cutR'+g);
    if(!A||!B)continue;
    A.style.left='0%';        A.style.width=Math.max(0,lo*100)+'%';
    B.style.left=(hi*100)+'%';B.style.width=Math.max(0,(1-hi)*100)+'%';
  }
}
// --- type selection --------------------------------------------------------
function typeOn(g){const el=document.getElementById('tchk'+g);return !el||el.checked;}
function typesOn(){const s=[];for(let g=0;g<N;g++)if(typeOn(g))s.push(g);return s;}
// --- publication minimums --------------------------------------------------
// One per type, because the corpora are nowhere near comparable in depth: on
// the sample set adenocarcinoma reaches 58 publications on a pair while large
// cell never passes 2, so a single shared threshold would either wave through
// everything adenocarcinoma says or erase large cell entirely.
function pubMin(g){const el=document.getElementById('pub'+g);
  const v=el?parseInt(el.value):DEFPUB;return isNaN(v)?DEFPUB:v;}
function pubMins(){const a=[];for(let g=0;g<N;g++)a.push(pubMin(g));return a;}
// --- publication years -----------------------------------------------------
// Two independent handles that may cross, read as a set, exactly like the T
// brush. Undated sentences are governed by their own tick rather than by the
// range, because a year filter has nothing to say about a sentence with no
// year: excluding them silently would drop a fifth of the evidence unannounced.
function yearRange(){
  let a=parseInt(document.getElementById('yrfrom').value);
  let b=parseInt(document.getElementById('yrto').value);
  if(isNaN(a))a=YMIN;
  if(isNaN(b))b=YMAX;
  return [Math.min(a,b),Math.max(a,b)];
}
function keepUndated(){const el=document.getElementById('yrnull');return !el||el.checked;}
function yearFilter(){
  const [lo,hi]=yearRange(), full=(lo<=YMIN&&hi>=YMAX), keep=keepUndated();
  return {lo:lo,hi:hi,
          // nothing to do at all only when the window is open AND undated stay
          off:full&&keep,
          ok:y=>y==null?keep:(y>=lo&&y<=hi)};
}
// --- significance ----------------------------------------------------------
// q is precomputed per gene per score step: a Benjamini-Hochberg value over the
// binomial tail P(X >= papers in the dominant corpus | all its papers, that
// corpus's share of all papers). Precomputed because BH needs the whole gene
// set, and because a binomial tail over several thousand genes on every slider
// move would stall the page.
function fdrIdx(){const v=parseInt(document.getElementById('fdr').value);
  return isNaN(v)?DEFFDR:Math.max(0,Math.min(FDRS.length-1,v));}
function fdrMax(){return FDRS[fdrIdx()];}
function sigOnly(){const el=document.getElementById('sigbtn');
  return !!el&&el.getAttribute('aria-pressed')==='true';}
// Spreading only makes sense over a set small enough to have somewhere to go, so
// the control appears with "significant only" and answers only while it is on —
// a pressed state left behind by an earlier session must not silently loosen the
// layout of the full graph.
function spreadOn(){const el=document.getElementById('spreadbtn');
  return sigOnly()&&!!el&&el.getAttribute('aria-pressed')==='true';}
// Drop nodes that have no edge in the current view. The only edge-less nodes here
// are the isolated significant genes "significant only" adds, so this reverses
// that addition when the reader wants just the connected significant graph.
function removeIsolated(){const el=document.getElementById('isobtn');
  return !!el&&el.getAttribute('aria-pressed')==='true';}
// The ceiling emphasis and "significant only" both answer to. With the slider
// off there is no ceiling to read, so both fall back to the conventional 5%
// rather than treating every gene as significant — a button that did nothing
// when the slider sat at "off" would read as broken.
function sigCeiling(){const q=fdrMax();return q>=1?0.05:q;}
function fmtQ(q){return q>=0.001?q.toFixed(3):q.toExponential(1);}
// blend toward the canvas so a gene that failed the FDR reads as context
function mixHex(a,b,t){
  const p=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
  const A=p(a),B=p(b);
  return '#'+[0,1,2].map(i=>Math.round(A[i]+(B[i]-A[i])*t)
    .toString(16).padStart(2,'0')).join('');
}

// --- gene focus ------------------------------------------------------------
function activeGene(){return (document.getElementById('genefilter').value||'').trim();}
// resolve like the source graphs: exact name first, else the first prefix match
function findGene(q){
  q=q.toLowerCase();
  const ids=DATA.nodes.map(n=>n.id);
  return ids.find(id=>id.toLowerCase()===q) ||
         ids.find(id=>id.toLowerCase().indexOf(q)===0) || null;
}
// --- cluster size ----------------------------------------------------------
function minCluster(){
  const v=parseInt(document.getElementById('mincluster').value);
  return isNaN(v)?2:v;
}
// Drop connected groups smaller than `mc`. Components are found over the edges
// still standing, so the sizes are the ones on screen. At mc=2 this is a no-op:
// every component built from edges already has two nodes.
function dropSmallClusters(view,mc){
  const adj={};
  view.forEach(o=>{(adj[o.f]=adj[o.f]||[]).push(o.t);(adj[o.t]=adj[o.t]||[]).push(o.f);});
  const comp={};let cid=0;
  for(const n in adj){
    if(comp[n]!==undefined)continue;
    const stack=[n];comp[n]=cid;
    while(stack.length){const x=stack.pop();
      (adj[x]||[]).forEach(y=>{if(comp[y]===undefined){comp[y]=cid;stack.push(y);}});}
    cid++;
  }
  const size={};
  for(const n in comp)size[comp[n]]=(size[comp[n]]||0)+1;
  return view.filter(o=>size[comp[o.f]]>=mc);
}
// breadth-first over the edges that survived every other filter, so the
// neighbourhood is the one actually on screen — not the whole graph's
function focusOn(view,seed,hops){
  const adj={};
  view.forEach(o=>{(adj[o.f]=adj[o.f]||[]).push(o.t);(adj[o.t]=adj[o.t]||[]).push(o.f);});
  const seen=new Set([seed]);
  let front=[seed];
  for(let h=0;h<hops;h++){
    const next=[];
    front.forEach(x=>(adj[x]||[]).forEach(y=>{if(!seen.has(y)){seen.add(y);next.push(y);}}));
    front=next;
  }
  return view.filter(o=>seen.has(o.f)&&seen.has(o.t));
}
// mark hits on the RAW text, escaping each piece, so a query containing < or &
// still highlights without injecting markup
function hl(t){
  t=t||'';
  if(!TM)return esc(t);
  const re=TM.hlre;re.lastIndex=0;
  let out='',last=0,m;
  while((m=re.exec(t))!==null){
    if(!m[0].length){re.lastIndex++;continue;}   // zero-length match: never loop
    out+=esc(t.slice(last,m.index))+'<mark>'+esc(m[0])+'</mark>';
    last=m.index+m[0].length;
  }
  return out+esc(t.slice(last));
}
// evidence block for one source graph, mirroring the source files' tooltip.
// `all` = sentences at the current score; `shown` = those matching the search.
function srcBlock(name,all,shown){
  if(!all.length)return '';
  const np=new Set(shown.map(s=>s.p)).size, lim=10;
  const cats=[...new Set(all.map(s=>s.g))].join(', ');
  const cnt=TM?(shown.length+' of '+all.length+' sentences match')
             :(all.length+' sentence'+(all.length===1?'':'s'));
  let h='<div class=esrc>'+esc(name)+' &middot; '+cnt+' &middot; '+np+' PMID'+(np===1?'':'s')+
    ' &middot; '+esc(cats)+'</div>';
  shown.slice(0,lim).forEach(s=>{h+='<div class=stip>'+pmA(s.p)+' <span class=mut>['+
    s.c.toFixed(3)+(s.y?(' &middot; '+s.y):'')+']</span> '+hl(s.t)+'</div>';});
  if(shown.length>lim)h+='<div class=more>+'+(shown.length-lim)+' more</div>';
  return h;
}
// with N graphs the provenance is a set, not a side: name it by how many agree
function eName(o){
  return o.c<0?('shared by '+o.seen+' types'):(LABS[o.c]+' only');
}
function edgeTip(o){
  const d=document.createElement('div');
  let h='<div class=eth><b>'+esc(o.f)+' &ndash; '+esc(o.t)+'</b> ('+esc(eName(o))+')</div>';
  for(let g=0;g<N;g++)h+=srcBlock(LABS[g],o.v[g],o.vm[g]);
  d.innerHTML=h;
  return d;
}
let network=null, VIEW=[];
// The set of genes actually drawn as nodes. Normally the endpoints of VIEW's
// edges, but under "significant only" it also carries isolated significant genes
// that have no surviving edge. The table reads this so it lists exactly what the
// graph draws, isolated nodes included.
let KEEP=new Set();
// Everything below the score slider is derived, so it all recomputes here.
function build(){
  const si=stepIdx(), T=STEPS[si];
  document.getElementById('confval').textContent=T.toFixed(2);
  // the low band is reachable now, so say what is in it rather than let every
  // score read as equally trustworthy
  const ch=document.getElementById('confhint');
  if(T<LOWSCORE){ch.hidden=false;
    ch.textContent='The extractor’s least confident calls — hover the connectors and '+
      'read the sentences before trusting the shape.';}
  else ch.hidden=true;
  TM=textMatcher(activeText());
  const EC=[];for(let g=0;g<N;g++)EC.push(cvar('--etype'+g));
  const ESH=cvar('--edge-shared');
  // the profile is per-gene and per-step, so resolve it once rather than per edge
  const P={};
  DATA.nodes.forEach(n=>{P[n.id]=profile(n,si);});
  const [slo,shi]=specRange();
  const fullSpec=(slo<=0&&shi>=1);
  document.getElementById('spechiv').textContent=shi.toFixed(2);
  document.getElementById('speclov').textContent=slo.toFixed(2);
  paintRange(slo,shi);
  const YF=yearFilter();
  document.getElementById('yrfromv').textContent=YF.lo;
  document.getElementById('yrtov').textContent=YF.hi;
  document.getElementById('yearsum').textContent=
    YF.off?(YMIN+'–'+YMAX):(YF.lo+'–'+YF.hi+(keepUndated()?'':' *'));
  const yh=document.getElementById('yrhint');
  if(YF.off)yh.textContent='All '+YMIN+'–'+YMAX+', undated included.';
  else yh.textContent=YF.lo+'–'+YF.hi+', '+
    (keepUndated()?'undated included':'undated excluded ('+fmtInt(YNULL)+' sentences)')+'.';
  const PMIN=pubMins();
  for(let g=0;g<N;g++){const el=document.getElementById('pubv'+g);
    if(el)el.textContent=PMIN[g];}
  document.getElementById('pubsum').textContent=PMIN.join(' · ');
  const on=typesOn(), allTypes=(on.length===N);
  const okType=id=>allTypes||on.indexOf(P[id].dom)>=0;
  const okSpec=id=>{const t=P[id].tau;return t>=slo-1e-9&&t<=shi+1e-9;};
  // The FDR filter keeps an edge when EITHER end is significant, unlike every
  // other gene-level filter here, which need both. The difference is deliberate.
  // The specificity brush selects a band of the colour scale, so drawing a node
  // outside it would contradict the legend. This selects which claims are
  // trustworthy, and a significant gene's partners are the context that makes it
  // readable — requiring both ends showed 9 of 561 edges on the sample set and
  // hid most of the very genes it exists to highlight. The partners are dimmed
  // below so the view never implies they passed the test too.
  const qmax=fdrMax(), fullQ=(qmax>=1);
  const okQ=id=>NODE_BY_ID[id].q[si]<=qmax+1e-12;
  // "Significant only" overrides the either-end rule above: both ends must clear
  // the ceiling, so nothing is drawn as context and the graph shows only claims
  // the corpus sizes cannot explain.
  const SIG=sigOnly(), sigQ=sigCeiling();
  const okSig=id=>NODE_BY_ID[id].q[si]<=sigQ+1e-12;
  VIEW=[];
  DATA.edges.forEach(e=>{
    // both endpoints must survive the gene-level filters — an edge with one end
    // excluded would draw a node the legend says is not being shown
    if(!fullSpec&&(!okSpec(e.f)||!okSpec(e.t)))return;
    if(!allTypes&&(!okType(e.f)||!okType(e.t)))return;
    if(SIG){if(!okSig(e.f)||!okSig(e.t))return;}
    else if(!fullQ&&!okQ(e.f)&&!okQ(e.t))return;
    // year and score are both filters on the evidence itself, so they apply
    // together and before anything is counted — asking for 2020-2024 with a
    // two-publication minimum should mean two publications in that window, not
    // two anywhere with one of them merely visible
    const v=e.v.map(list=>list.filter(x=>x.c>=T&&YF.ok(x.y)));
    const pubs=v.map(list=>new Set(list.map(x=>x.p)).size);
    const backing=[];
    for(let g=0;g<N;g++)if(v[g].length)backing.push(g);
    if(!backing.length)return;
    // Provenance, width and specificity stay measured on the FULL evidence at
    // this score. The text search picks which pairs to show; it must not
    // restate what they are — calling a shared pair exclusive because the query
    // happened to hit one type's sentences would be a false claim about the
    // biology.
    const c=backing.length===1?backing[0]:-1;
    // The publication rule: ANY one type clearing its own minimum keeps the
    // pair. Requiring every backing type to clear its bar instead would punish
    // a relationship for being reported by a second, thinner literature — the
    // pair would vanish the moment a small corpus corroborated it, which is
    // precisely backwards. There is deliberately no exemption for pairs several
    // types agree on: an exemption would make the sliders unable to thin the
    // shared core, and a control that cannot move some of the graph reads as
    // broken rather than as principled.
    if(!backing.some(g=>pubs[g]>=PMIN[g]))return;
    const vm=TM?v.map(list=>list.filter(x=>TM.test(x.t))):v;
    if(TM&&!vm.some(list=>list.length))return;
    VIEW.push({f:e.f,t:e.t,c:c,seen:backing.length,back:backing,v:v,vm:vm,pubs:pubs,
      sent:v.map(list=>list.length),
      neg:v.some(list=>list.some(x=>x.g==='negated'))});
  });
  // gene focus runs last, over whatever the other filters left standing
  const gq=activeGene(), gh=document.getElementById('ghint');
  if(gq){
    const seed=findGene(gq), hops=parseInt(document.getElementById('hops').value)||1;
    if(!seed){VIEW=[];gh.className='hint bad';gh.textContent='No gene named or starting with “'+gq+'”.';}
    else{
      VIEW=focusOn(VIEW,seed,hops);
      gh.className='hint';
      gh.textContent=VIEW.length?('Showing '+seed+' + '+hops+' hop'+(hops>1?'s':'')+'.')
        :(seed+' has no relationships in the current view.');
    }
  }else{gh.className='hint';
    gh.textContent='Exact name, else prefix match.';}
  // Name the types whose slider is actually doing something, and flag any whose
  // ceiling is hiding a longer tail than the control can reach.
  const ph=document.getElementById('pubhint');
  // measured against the slider's floor of 1, not against DEFPUB: the default
  // is itself a filter, and reporting "no filter" while one is in force would
  // be a false statement about what is on screen
  const raised=[];for(let g=0;g<N;g++)if(PMIN[g]>1)raised.push(LABS[g]+' ≥ '+PMIN[g]);
  const capped=[];for(let g=0;g<N;g++){const el=document.getElementById('pub'+g);
    if(el&&PMIN[g]>=+el.max&&PUBTRUE[g]>+el.max)capped.push(LABS[g]+' reaches '+PUBTRUE[g]);}
  ph.textContent=(raised.length?raised.join(', ')+'.':'No publication filter.')+
    (capped.length?' Slider ends here; '+capped.join(', ')+'.':'');
  const sh=document.getElementById('spechint');
  if(fullSpec&&allTypes)sh.hidden=true;
  else{sh.hidden=false;
    sh.textContent='Showing T '+slo.toFixed(2)+' to '+shi.toFixed(2)+
      (allTypes?'':' · '+on.length+' of '+N+' types')+'.';}
  // Cluster pruning is skipped while a gene is focused, as in the source graphs:
  // the focus already states which neighbourhood you want, and a size rule laid
  // over it could silently delete the very gene you asked for. "Significant only"
  // is skipped for the same reason: it is itself a strong, deliberate filter, and
  // a type's significant genes are often a handful (3-4 for adenocarcinoma, large
  // cell and squamous here) that form components below the default of 6 — laying
  // min-cluster over them deleted the whole view, which read as the button being
  // broken. The significant set is exactly what the reader asked to see.
  const mc=minCluster(), mch=document.getElementById('mchint');
  document.getElementById('mcval').textContent=mc;
  if(gq){mch.hidden=false;mch.textContent='Cluster size not applied while a gene is focused.';}
  else if(SIG){mch.hidden=false;mch.textContent='Cluster size not applied while showing significant only.';}
  else if(mc>2){VIEW=dropSmallClusters(VIEW,mc);
    mch.hidden=false;mch.textContent='Hiding groups smaller than '+mc+' genes.';}
  else mch.hidden=true;
  const keep=new Set(); VIEW.forEach(o=>{keep.add(o.f);keep.add(o.t);});
  // every VIEW endpoint has an edge; snapshot that before adding lone nodes, so
  // "remove isolated nodes" below can tell the two apart
  const connected=new Set(keep);
  // Significant only also surfaces significant genes that have NO significant
  // partner. The graph is edge-driven, so without this they vanish simply for
  // lacking a qualifying edge — not what "show the significant genes" should mean.
  // They are added as isolated nodes (and table rows). q, T and dominant type are
  // per-score, not evidence-filtered, so year/publication filters — which act on
  // edges — do not gate a lone node; spec, type and a partner in a selected type
  // at this score do.
  if(SIG){
    DATA.nodes.forEach(n=>{
      if(keep.has(n.id)||!okSig(n.id)||!okSpec(n.id)||!okType(n.id))return;
      if(!on.some(g=>n.d[g][si]>0))return;
      keep.add(n.id);
    });
  }
  // "Remove isolated nodes": strip anything without an edge in the view. Applied
  // last so it overrides the addition above when the reader wants the connected
  // significant graph only.
  if(removeIsolated())[...keep].forEach(id=>{if(!connected.has(id))keep.delete(id);});
  KEEP=keep;
  // Say plainly how much of what is drawn is actually supported — the whole
  // point of the column is that "specific" and "shown to be specific" differ.
  // Must follow `keep`: the counts describe the genes that survived every
  // filter, not the whole model.
  const fv=document.getElementById('fdrval'), fh=document.getElementById('fdrhint');
  fv.textContent=fullQ?'off':(qmax>=0.01?qmax.toFixed(2):qmax.toString());
  if(SIG)
    fh.textContent='Significant only is on — the ceiling in force is q ≤ '+sigQ+'.';
  else if(fullQ)
    fh.textContent='T is the effect size; q is whether it is more than corpus size talking.';
  else{
    const pass=[...keep].filter(okQ).length;
    fh.textContent=pass+' gene'+(pass===1?'':'s')+' at q ≤ '+qmax+
      ', plus '+(keep.size-pass)+' shown as context.';
  }
  // The button reports what it produced, not merely that it is on: "significant
  // only" over a set with nothing significant in it is an empty graph, and an
  // empty graph with no explanation reads as a bug.
  const shn=document.getElementById('sighint'), spb=document.getElementById('spreadbtn');
  spb.hidden=!SIG;
  if(SIG)shn.textContent=keep.size?
    ('Showing '+keep.size+' significant gene'+(keep.size===1?'':'s')+' (q ≤ '+sigQ+
     '); relationships need both ends significant, and genes with none stand alone.'+
     (spreadOn()?' Spread out.':'')):
    ('No gene clears q ≤ '+sigQ+' here.');
  else shn.textContent='Off — genes above the ceiling are drawn as context.';
  // acting on q happens here, so the caveat has to be here too, not only in the
  // table a reader may never open
  if(OVDUP[si]>0)
    fh.textContent+=' ⚠ Papers overlap — q is optimistic; see the table.';
  const bg=cvar('--net-bg');
  // What counts as significant for emphasis is sigQ, computed with the filter
  // above: the ceiling in force, or the conventional 5% when the slider is off.
  // Emphasis is independent of filtering — a supported gene should stand out
  // whether or not the reader has asked to hide anything.
  const isSig=okSig;
  // Draw order IS z-order in vis-network: nodes paint in array order, each one
  // drawing its own label, so a later node covers an earlier node's name as
  // well as its body. Significant genes therefore go last and land on top of
  // everything else. Alphabetical within each band keeps the order stable
  // between rebuilds, so the picture does not reshuffle as sliders move.
  const ordered=[...keep].sort((a,b)=>{
    const sa=isSig(a), sb=isSig(b);
    if(sa!==sb)return sa?1:-1;
    return a.localeCompare(b);
  });
  const nodes=ordered.map(id=>{
    const n=NODE_BY_ID[id], p=P[id], deg=Math.max.apply(null,n.d.map(r=>r[si]));
    const q=n.q[si], faded=!fullQ&&!okQ(id);
    let tip=id+' — '+label(p)+' (T '+p.tau.toFixed(2)+', q '+fmtQ(q)+')';
    if(faded)tip+='\nshown as context — above the q ceiling';
    tip+='\npartners:';
    for(let g=0;g<N;g++)if(n.d[g][si])tip+='\n  '+LABS[g]+' '+n.d[g][si];
    // a gene kept only as its neighbour's context recedes toward the canvas,
    // so the graph never implies it carries a supported claim of its own.
    // The border keeps slightly more ink than the fill: at these mixes a shape
    // with no outline at all stops reading as a node, and the point is to
    // recede, not to disappear.
    const bgc=faded?mixHex(fill(p),bg,0.88):fill(p);
    const bdc=faded?mixHex(bord(p),bg,0.8):bord(p);
    // A wider halo is what actually keeps a name readable where it overlaps a
    // neighbour, and a heavier outline separates the node from whatever it is
    // sitting on. Font SIZE is left alone deliberately: it already encodes
    // partner count, and borrowing it for significance would collide two
    // meanings on one channel.
    const sig=isSig(id);
    return {id:id, label:id, size:8+Math.sqrt(Math.max(deg,1))*3.2,
      color:{background:bgc,border:bdc,
             highlight:{background:bgc,border:cvar('--text-primary')}},
      borderWidth:sig?BDSIG:BD, borderWidthSelected:sig?BDSIGSEL:BDSEL,
      font:{color:faded?cvar('--text-muted'):cvar('--text-primary'),
            size:Math.max(14,Math.min(14+deg*0.2,26)),
            strokeWidth:sig?9:5,strokeColor:bg,vadjust:-1},
      title:tip};
  });
  const edges=VIEW.map((o,i)=>({id:i,from:o.f,to:o.t,
    width:Math.min(1.4+Math.max.apply(null,o.pubs)*0.75,9),
    color:{color:o.c<0?ESH:EC[o.c],opacity:o.c<0?0.9:0.62},
    dashes:o.neg?[5,4]:false,
    title:edgeTip(o)}));
  const data={nodes:new vis.DataSet(nodes),edges:new vis.DataSet(edges)};
  // Stabilisation cost scales with nodes x iterations, and with the publication
  // floor at its default of 1 this view opens on ten thousand edges rather than
  // the thousand the two-graph page ever saw. 400 iterations there is a long
  // freeze here, and buys nothing: a graph this dense has settled into its
  // hairball long before then. Thin it with the sliders and the full count
  // comes back for the smaller graph, where it does visibly help.
  const iters=edges.length>2000?150:400;
  // Spread mode trades compactness for legibility: harder repulsion, longer
  // springs, and much less central pull, so a small dense set opens out instead
  // of balling up. Nothing is hidden — this only changes where things sit.
  const SPREAD=spreadOn();
  const bh=SPREAD
    ?{gravitationalConstant:-22000,centralGravity:0.25,springLength:190,
      springConstant:0.02,damping:0.45,avoidOverlap:0.6}
    :{gravitationalConstant:-7000,centralGravity:0.62,springLength:95,
      springConstant:0.035,damping:0.4,avoidOverlap:0.28};
  const options={layout:{improvedLayout:false},
    physics:{stabilization:{iterations:iters},barnesHut:bh},
    interaction:{hover:true,tooltipDelay:120},
    nodes:{shape:'dot'},
    edges:{smooth:false,arrows:{to:{enabled:false}},hoverWidth:0,selectionWidth:0}};
  if(network)network.destroy();
  network=new vis.Network(document.getElementById('net'),data,options);
  // Node order is only half of z-order here. vis draws every node BODY first
  // and then every external label in a second pass — 'dot' is an external-label
  // shape — so a label always lands on top of every circle in the scene, no
  // matter how the nodes were ordered. An unsupported gene's name could
  // therefore still cover a supported gene's node.
  //
  // The two passes cannot be interleaved from outside, so the significant genes
  // are simply drawn again once the whole scene is finished. afterDrawing fires
  // inside the canvas transform (save/translate/scale happen before it and
  // restore after), so re-running the node's own draw puts it in the right
  // place with the right colours, borders and font — no reimplementation of the
  // node's appearance to drift out of step with the options above.
  const TOP=[...keep].filter(isSig);
  let topBroken=false;
  network.on('afterDrawing',ctx=>{
    // The runtime is lifted from whichever graph was read last, so a future
    // source file could carry a vis version whose internals differ. If this
    // ever throws, give up on the overlay rather than throwing once per frame.
    if(topBroken)return;
    try{
      for(const id of TOP){
        const n=network.body.nodes[id];
        if(!n)continue;
        const r=n.draw(ctx);
        if(r&&r.drawExternalLabel)r.drawExternalLabel();
      }
    }catch(err){topBroken=true;}
  });
  network.on('stabilizationIterationsDone',()=>{
    network.setOptions({physics:false});network.fit({animation:false});network.redraw();});
  // Node and edge detail lives in the hover tooltips (title:tip / edgeTip); the
  // click-to-inspect panel that used to mirror it was removed with the Selection
  // box, so there is no click handler here.
  document.getElementById('count').innerHTML='<b>'+nodes.length+'</b> genes · <b>'+
    edges.length+'</b> relationships'+(TM?' matching':'');
  // per-type counts in the legend, so unticking a type says what it costs
  for(let g=0;g<N;g++){
    const el=document.getElementById('acnt'+g);
    if(el)el.textContent=[...keep].filter(id=>P[id].dom===g).length+'';
  }
  const th=document.getElementById('thint');
  if(TM&&!VIEW.length){th.className='hint bad';th.textContent='No sentences match that query.';}
  else{th.className='hint';
    th.innerHTML='Substring, or <code>/regex/</code>.';}
  table(si,P);
  renderSummary(si,P,keep,okQ,fullQ,qmax);
  renderDoc(si,P);
}
function fmtInt(n){return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g,',');}
function swatch(p){return '<span class=dot style="background:'+fill(p)+';border:1px solid '+
  bord(p)+'"></span>';}
// swatch for a bare (type, T) rather than a real gene — legend and doc rows
function swatchAt(g,tau){const s=stepOf(tau),R=ramp(g);
  return '<span class=dot style="background:'+R.fill[s]+';border:1px solid '+R.border[s]+'"></span>';}

// --- table -----------------------------------------------------------------
// Sortable, because with N types "most specific" is only one of the questions
// and the interesting one is often "most connected in type k".
let SORT={k:'tau',dir:-1};
let RETABLE=null;
// Roughly how many papers a gene needs in one type alone to clear a given FDR,
// using the most conservative BH rank (p0^d <= q/m). Approximate by design —
// it is here to show how far apart the four bars are, not to be a threshold.
function papersFor(g,si,q){
  const grand=PTOT.reduce((a,r)=>a+r[si],0), m=NTEST[si];
  if(!grand||!m)return null;
  const p0=PTOT[g][si]/grand;
  if(!(p0>0)||p0>=1)return null;
  let d=1;
  while(Math.pow(p0,d)>q/m&&d<400)d++;
  return d;
}
// Where the wording in the "Reads as" column changes. Probed from label()
// rather than restated, so the bands quoted in the note and the words actually
// printed in the table cannot drift apart when one of them is edited.
function tBands(){
  const cuts=[]; let prev=null;
  for(let i=0;i<=2000;i++){
    const t=i/2000, l=label({dom:0,tau:t});
    if(l!==prev){cuts.push([t,l]);prev=l;}
  }
  return cuts.map((c,i)=>({lo:c[0],
    hi:i+1<cuts.length?cuts[i+1][0]:1.0001, txt:c[1]}));
}
// Two decimals unless that would collapse an edge onto its neighbour: the top
// cut sits at 0.999, and printing it as "1.00" would show the exclusive band as
// 1.00-1.00 and imply the band below it reaches 1.
function fmtEdge(x){
  return Math.abs(x-Number(x.toFixed(2)))<1e-9?x.toFixed(2):x.toFixed(3);
}
// The note above the table. Everything in it is read off the model at the
// current score, because the baselines it turns on move with the slider.
function tableNote(si){
  const qmax=fdrMax(), on=qmax<1, grand=PTOT.reduce((a,r)=>a+r[si],0);
  const [slo,shi]=specRange(), fullSpec=(slo<=0&&shi>=1);
  const bands=tBands();
  let h='<h2>Reading this table</h2><dl>'+
    '<dt>T</dt><dd><b>Effect size.</b> How concentrated the gene is: 0 = every type '+
    'gives it the same share of their wiring, 1 = only one type mentions it at all. '+
    'It says nothing about how much evidence there is. The <b>Reads as</b> column is '+
    'just this number bucketed for readability — '+
    bands.map(b=>'<b>'+fmtEdge(b.lo)+'</b>–<b>'+(b.hi>1?'1.00':fmtEdge(b.hi))+
      '</b> ' + esc(b.txt)).join(' · ')+
    ' — with the dominant type’s name filled in (shown here for '+esc(LABS[0])+').</dd>'+
    '<dt>T min / T max</dt><dd><b>The brush in the rail.</b> Keeps only genes whose T '+
    'falls inside the band, so the two ends ask opposite questions: raise <b>T min</b> to '+
    'isolate markers, lower <b>T max</b> to isolate the shared core every type writes '+
    'about. <b>Both</b> ends of a relationship must be inside the band — unlike Max q, '+
    'which keeps an edge when <i>either</i> end passes — because this selects a band of '+
    'the colour scale, and drawing a node outside it would contradict the legend. The two '+
    'handles are independent and may cross; they are read as a set. Currently <b>'+
    slo.toFixed(2)+' to '+shi.toFixed(2)+'</b>'+(fullSpec?' (the whole scale)':'')+'.</dd>'+
    '<dt>q</dt><dd><b>Evidence.</b> A Benjamini-Hochberg false discovery rate for the '+
    'one claim the gene’s colour is making — that its dominant type really does own it. '+
    'The null is that a gene has no preference, so its papers fall across the '+N+
    ' corpora in proportion to their size. Counted in <b>publications, not partners</b>, '+
    'because one sentence naming three genes is one observation, not three.</dd>'+
    '<dt>Max q</dt><dd><b>The slider in the rail.</b> Keeps genes at or below the ceiling '+
    'you set. Because it is a false discovery <i>rate</i>, at 0.05 it means: of the genes '+
    'kept, at most 5% are expected to be spurious — not that each has a 5% chance of being '+
    'wrong. It is currently <b>'+(on?'≤ '+qmax:'off')+'</b>.</dd></dl>';

  h+='<div class=base><table><thead><tr><th>Type</th><th class=num>Papers</th>'+
     '<th class=num>Share</th><th class=num>Papers alone for q ≤ 0.05</th></tr></thead><tbody>';
  for(let g=0;g<N;g++){
    const need=papersFor(g,si,0.05);
    h+='<tr><td><span class=tag>'+swatchAt(g,1)+esc(LABS[g])+'</span></td>'+
       '<td class=num>'+fmtInt(PTOT[g][si])+'</td>'+
       '<td class=num>'+(grand?(PTOT[g][si]/grand*100).toFixed(1):'0.0')+'%</td>'+
       '<td class=num>'+(need===null?'—':'~'+need)+'</td></tr>';
  }
  h+='</tbody></table></div>';

  // The asymmetry is the whole point of reading q in a four-way analysis, so it
  // is stated in the reader's own numbers rather than left to be inferred.
  const sh=[];for(let g=0;g<N;g++)sh.push(PTOT[g][si]);
  const big=sh.indexOf(Math.max.apply(null,sh)), sml=sh.indexOf(Math.min.apply(null,sh));
  const nb=papersFor(big,si,0.05), ns=papersFor(sml,si,0.05);
  if(nb!==null&&ns!==null&&big!==sml)
    h+='<p>The bar is not the same for each type, because the baseline is not. Being found '+
       'only in <b>'+esc(LABS[big])+'</b> is barely surprising when it already holds '+
       (grand?(sh[big]/grand*100).toFixed(0):'?')+'% of the papers, so it takes about <b>'+
       nb+'</b> of them to clear 5%; in <b>'+esc(LABS[sml])+'</b>, <b>'+ns+'</b> will do. '+
       'The same q therefore means the same <i>surprise</i>, not the same amount of '+
       'evidence — and a gene failing in '+esc(LABS[big])+' has not been shown to be '+
       'unspecific, it has simply not been shown to be anything.</p>';
  // The disjointness the q column rests on, checked rather than assumed. Stated
  // either way: an unqualified q column would otherwise be read as "this was
  // verified", which is exactly the reading a silent failure would earn.
  const papers=PTOT.reduce((a,r)=>a+r[si],0), dup=OVDUP[si];
  if(dup>0){
    let wa=-1,wb=-1,worst=0;
    for(let a=0;a<N;a++)for(let b=a+1;b<N;b++)
      if(OVMAT[si][a][b]>worst){worst=OVMAT[si][a][b];wa=a;wb=b;}
    h+='<p class=alert><b>⚠ The q column is optimistic here.</b> '+fmtInt(dup)+' of '+
       fmtInt(papers)+' papers at this score ('+(papers?(dup/papers*100).toFixed(1):'0')+
       '%) appear in <b>more than one corpus</b>'+
       (wa>=0?' — most between <b>'+esc(LABS[wa])+'</b> and <b>'+esc(LABS[wb])+
        '</b>, which share '+fmtInt(worst):'')+'. The test treats the corpora as '+
       'independent samples, so a paper counted twice is counted as two independent '+
       'observations and every q resting on it is smaller than it should be. Treat the '+
       'column as a ranking, not as a rate, until the queries are made mutually '+
       'exclusive.</p>';
  }else{
    h+='<p class=ok>No paper appears in more than one corpus at this score, so the '+
       'independence the q column assumes actually holds here.</p>';
  }
  h+='<p class=warn>'+fmtInt(NTEST[si])+' genes are tested at this score, which is why the '+
     'column is an FDR and not a p-value: at raw p &lt; 0.05 about '+
     fmtInt(NTEST[si]*0.05)+' would look significant by chance alone. The null concerns '+
     '<b>publishing, not biology</b> — a small q says a gene is written about '+
     'disproportionately in that subtype’s literature.</p>';
  document.getElementById('tblnote').innerHTML=h;
}
function table(si,P){
  // Re-render hook for callers that have no si/P of their own — showPanel needs
  // to redraw the table after setting the sort, and closing over the current
  // build's arguments is cheaper than making them global.
  RETABLE=()=>table(si,P);
  tableNote(si);
  // Rows are the genes the graph drew (KEEP), not VIEW's edge endpoints, so the
  // table lists exactly what is on the canvas — including the isolated significant
  // genes that "significant only" adds, which have no edge to be found through.
  const rows=[...KEEP].map(id=>{const n=NODE_BY_ID[id];return {n:n,p:P[id]};});
  const key=r=>SORT.k==='id'?r.n.id:SORT.k==='tau'?r.p.tau:
    SORT.k==='q'?-r.n.q[si]:
    SORT.k==='dom'?r.p.dom:r.n.d[SORT.k|0][si];
  rows.sort((a,b)=>{const x=key(a),y=key(b);
    const c=(typeof x==='string')?x.localeCompare(y):(x-y);
    return (c*SORT.dir)||a.n.id.localeCompare(b.n.id);});
  const ar=k=>SORT.k===k?(' <span class=ar>'+(SORT.dir<0?'▼':'▲')+'</span>'):'';
  let h='<thead><tr><th data-k="id">Gene'+ar('id')+'</th>'+
        '<th data-k="dom">Reads as'+ar('dom')+'</th>'+
        '<th class=num data-k="tau">T'+ar('tau')+'</th>'+
        '<th class=num data-k="q" title="Benjamini-Hochberg q for the dominant type, '+
        'from publication counts">q'+ar('q')+'</th>';
  for(let g=0;g<N;g++)h+='<th class=num data-k="'+g+'">'+esc(LABS[g])+ar(String(g))+'</th>';
  h+='</tr></thead><tbody>';
  rows.forEach(r=>{
    const q=r.n.q[si];
    h+='<tr><td>'+esc(r.n.id)+'</td><td><span class=tag>'+swatch(r.p)+label(r.p)+
       '</span></td><td class=num>'+r.p.tau.toFixed(2)+'</td>'+
       '<td class="num'+(q<0.05?' sig':' ins')+'">'+fmtQ(q)+'</td>';
    for(let g=0;g<N;g++)h+='<td class=num>'+r.n.d[g][si]+'</td>';
    h+='</tr>';});
  const tb=document.getElementById('tbl');
  tb.innerHTML=h+'</tbody>';
  tb.querySelectorAll('th[data-k]').forEach(th=>{
    th.addEventListener('click',()=>{const k=th.getAttribute('data-k');
      // a fresh column starts descending except the name, which starts A-Z
      if(SORT.k===k)SORT.dir=-SORT.dir;else{SORT.k=k;SORT.dir=(k==='id')?1:-1;}
      table(si,P);});});
}

// --- summary ---------------------------------------------------------------
// The part diff_two.py had no need for. With two graphs the network IS the
// comparison; with N, the pairwise question ("which of these subtypes actually
// resemble each other?") is not answerable by looking at a hairball, so it is
// answered in numbers. Everything here is recomputed at the live score.
// `keep` is the gene set the graph actually drew — it already carries the year,
// publication, type, T-range, cluster, focus and text filters. The gene-level
// sections below read from it rather than from DATA, so what the summary names
// and what the graph shows cannot drift apart. The corpus-level sections above
// them (counts, Jaccard) stay global: they describe the literatures themselves,
// which a filter on the drawing does not change.
function renderSummary(si,P,keep,okQ,fullQ,qmax){
  const T=STEPS[si];
  // pair sets per type, restricted to what is drawable at this score
  const sets=[];for(let g=0;g<N;g++)sets.push(new Set());
  const excl=new Array(N).fill(0);
  DATA.edges.forEach(e=>{
    const back=[];
    for(let g=0;g<N;g++)if(e.v[g].some(x=>x.c>=T))back.push(g);
    const k=e.f+'	'+e.t;   // tab: no HGNC symbol contains one, so keys cannot collide
    back.forEach(g=>sets[g].add(k));
    if(back.length===1)excl[back[0]]++;
  });
  const genes=[];for(let g=0;g<N;g++)genes.push(new Set());
  DATA.nodes.forEach(n=>{for(let g=0;g<N;g++)if(n.d[g][si])genes[g].add(n.id);});
  let h='<h2>Summary at score ≥ '+T.toFixed(2)+'</h2>'+
    '<p class=std><b>The corpora</b> and <b>how much the types overlap</b> describe the whole '+
    'model at the score you have selected — they are properties of the literatures, which '+
    'hiding part of the drawing does not change. <b>Core genes</b> and <b>Markers</b> name '+
    'individual genes, so they are computed over the '+fmtInt(keep.size)+' gene'+
    (keep.size===1?'':'s')+' currently drawn and follow every filter in the panel.</p>';

  h+='<h3>The corpora</h3><table><thead><tr><th>Type</th><th class=num>Genes</th>'+
     '<th class=num>Pairs</th><th class=num>Exclusive pairs</th>'+
     '<th class=num>Share of pairs</th></tr></thead><tbody>';
  const allPairs=sets.reduce((a,s)=>a+s.size,0);
  for(let g=0;g<N;g++){
    h+='<tr><td class=g><span class=tag>'+swatchAt(g,1)+esc(LABS[g])+'</span></td>'+
       '<td class=num>'+fmtInt(genes[g].size)+'</td>'+
       '<td class=num>'+fmtInt(sets[g].size)+'</td>'+
       '<td class=num>'+fmtInt(excl[g])+'</td>'+
       '<td class=num>'+(allPairs?(sets[g].size/allPairs*100).toFixed(1):'0.0')+'%</td></tr>';
  }
  h+='</tbody></table>';
  const sz=sets.map(s=>s.size), big=sz.indexOf(Math.max.apply(null,sz)),
        sml=sz.indexOf(Math.min.apply(null,sz));
  if(sz[sml]>0&&big!==sml)
    h+='<p style="margin-top:8px">The largest corpus is <b>'+(sz[big]/sz[sml]).toFixed(1)+
       '×</b> the smallest ('+esc(LABS[big])+' vs '+esc(LABS[sml])+'). This is exactly '+
       'the imbalance the specificity scale normalises away — see the explainer.</p>';

  // Jaccard over pair sets: |A n B| / |A u B|. Pairs, not genes, because two
  // corpora can name the same genes while asserting entirely different wiring.
  h+='<h3>How much the types overlap <span class=mut>— Jaccard over pairs</span></h3>'+
     '<p>Of every relationship either type asserts, the fraction both do. 0 = no shared '+
     'relationship at all; 1 = identical wiring.</p><table class=mx><thead><tr><th></th>';
  for(let g=0;g<N;g++)h+='<th class=ch>'+esc(LABS[g])+'</th>';
  h+='</tr></thead><tbody>';
  for(let a=0;a<N;a++){
    h+='<tr><th class=rh><span class=tag>'+swatchAt(a,1)+esc(LABS[a])+'</span></th>';
    for(let b=0;b<N;b++){
      if(a===b){h+='<td class="c self">—</td>';continue;}
      let inter=0;sets[a].forEach(k=>{if(sets[b].has(k))inter++;});
      const uni=sets[a].size+sets[b].size-inter, j=uni?inter/uni:0;
      // tinted by value using the neutral-to-pole ramp of the row's own type,
      // so the matrix reads with the same ink as everything else
      const s=stepOf(Math.min(1,j*3));   // x3: real overlaps here are small
      h+='<td class=c style="background:'+ramp(a).fill[s]+'">'+j.toFixed(3)+'</td>';
    }
    h+='</tr>';
  }
  h+='</tbody></table><p class=note style="margin-top:6px">Tint is scaled ×3 — '+
     'overlaps between literatures are small in absolute terms, and an unscaled '+
     'ramp would render the whole matrix neutral.</p>';

  // the two ends of the T scale, named from the data — from the genes on screen,
  // so every name here is one the reader can actually find in the graph
  const drawn=DATA.nodes.filter(n=>keep.has(n.id));
  const deg=n=>Math.max.apply(null,n.d.map(r=>r[si]));
  const core=drawn.filter(n=>P[n.id].seen===N).sort((a,b)=>
    P[a.id].tau-P[b.id].tau||deg(b)-deg(a)).slice(0,12);
  h+='<h3>Core genes <span class=mut>— lowest T, present in every type</span></h3>';
  if(!core.length)h+='<p>No gene on screen appears in all '+N+' types at this score.</p>';
  else{
    h+='<p>The field writes about these wherever it looks. They are the shared spine: '+
       'high connectivity, no allegiance.</p><table><thead><tr><th>Gene</th>'+
       '<th class=num>T</th><th class=num>Partners (max)</th></tr></thead><tbody>';
    core.forEach(n=>{h+='<tr><td class=g><span class=tag>'+swatch(P[n.id])+n.id+
      '</span></td><td class=num>'+P[n.id].tau.toFixed(3)+'</td><td class=num>'+
      deg(n)+'</td></tr>';});
    h+='</tbody></table>';
  }

  // Emphasis follows the ceiling in force, exactly as the graph's isSig does, so a
  // gene cannot be drawn as significant and listed here as not. With the filter off
  // both fall back to the conventional 5%.
  const sigQ=fullQ?0.05:qmax;
  const isSig=n=>n.q[si]<=sigQ+1e-12;
  const nsig=drawn.filter(isSig).length;
  h+='<h3>Markers <span class=mut>— specific <i>and</i> well-connected, per type</span></h3>'+
     '<p>Supported genes first — those clearing q &le; '+sigQ+' are ranked ahead of the rest, '+
     'because a gene the corpus sizes alone can explain is not a marker however exclusive it '+
     'looks. Within each band the order is <b>T &times; log(1 + partners)</b>, so specificity '+
     'and wiring are weighed together and a gene attested once does not outrank a hub. Ranking '+
     'on T alone would be a ranking on exclusivity: most markers sit at T = 1 exactly, which '+
     'means only that no other corpus mentions the gene — <b>not</b> that it has been shown to '+
     'be specific. Each entry reads '+
     '<span class=raw style="text-decoration:none">gene (T &middot; partners)</span>; genes in '+
     '<b>bold</b> clear the q ceiling. <b>'+nsig+'</b> of '+fmtInt(drawn.length)+' genes on '+
     'screen clear it at this score.</p><table><thead><tr>'+
     '<th>Type</th><th>Top genes</th></tr></thead><tbody>';
  // Rank on specificity BLENDED with wiring, not on specificity with wiring as a
  // tie-break. Sorting on T alone is effectively a sort on exclusivity: most markers
  // sit at T = 1 exactly (no other corpus mentions them), so the tie-break only ever
  // orders genes inside that block and the top 10 fills up with genes attested once.
  // A hub that one other corpus happens to mention drops below all of them — MALAT1
  // (T 0.98, 493 partners) landed at rank 3221 of adenocarcinoma's 3456 markers while
  // singletons held the list. log1p keeps the degree term from swamping T outright.
  // Support is the outer key, the blend the inner one. Ranking on the blend alone
  // let unsupported genes hold the list: at score 0.5 only 4 of adenocarcinoma's
  // 3456 markers clear q, so 7 of the 10 slots went to genes at q 0.11-0.77 while
  // EGFR (q 1.4e-11, 485 partners) sat at rank 35, cut by its T of 0.49. There is
  // no T floor here any more — the panel's T brush already decides that, and a
  // second hidden threshold would contradict the control.
  const rank=(n,g)=>P[n.id].tau*Math.log1p(n.d[g][si]);
  for(let g=0;g<N;g++){
    const mk=drawn.filter(n=>P[n.id].dom===g)
      .sort((a,b)=>(isSig(b)-isSig(a))||rank(b,g)-rank(a,g)||P[b.id].tau-P[a.id].tau)
      .slice(0,10);
    h+='<tr><td class=g><span class=tag>'+swatchAt(g,1)+esc(LABS[g])+'</span></td><td>'+
       (mk.length?mk.map(n=>{const s=isSig(n);
         return (s?'<b>':'<span class=raw style="text-decoration:none">')+esc(n.id)+
           ' ('+P[n.id].tau.toFixed(2)+' &middot; '+n.d[g][si]+')'+(s?'</b>':'</span>');}).join(', ')
        :'<span class=raw style="text-decoration:none">no gene of this type is on screen</span>')+'</td></tr>';
  }
  h+='</tbody></table>';
  document.getElementById('sum').innerHTML=h;
}

// --- explainer -------------------------------------------------------------
// Generated from the live model, not written out as prose, so its worked numbers
// are always the ones behind the graph on screen. Exemplars are chosen from the
// data: a hardcoded gene list would be wrong for any other set of graphs.
function docGenes(si,P){
  const deg=n=>Math.max.apply(null,n.d.map(r=>r[si]));
  const pool=DATA.nodes.filter(n=>deg(n)>0).sort((a,b)=>deg(b)-deg(a)).slice(0,60);
  pool.sort((a,b)=>P[b.id].tau-P[a.id].tau);
  if(pool.length<=7)return pool;
  const out=[];
  for(let i=0;i<7;i++)out.push(pool[Math.round(i*(pool.length-1)/6)]);
  return out;
}
function buckets(){return [
  [1,'T = 1 — no other type mentions the gene at all'],
  [0.85,'0.70 to 1.00'],[0.5,'0.35 to 0.70'],[0.22,'0.12 to 0.35'],
  [0.05,'below 0.12 — every type gives it a comparable share']];}
function renderDoc(si,P){
  const T=STEPS[si];
  const ex=docGenes(si,P).map(n=>{
    const p=P[n.id], d=n.d.map(r=>r[si]);
    const dmax=Math.max.apply(null,d), dsum=d.reduce((a,b)=>a+b,0);
    // the same index computed on RAW degrees — the mistake this scale avoids
    let raw=0;
    if(dmax>0&&N>1){for(let g=0;g<N;g++)raw+=1-d[g]/dmax;raw/=(N-1);}
    return {id:n.id,d:d,p:p,raw:raw,rawdom:d.indexOf(dmax),dsum:dsum};
  });
  const tot=TOT.map(r=>r[si]);
  const big=tot.indexOf(Math.max.apply(null,tot)), sml=tot.indexOf(Math.min.apply(null,tot));
  const ratio=(tot[big]/Math.max(1,tot[sml])).toFixed(1);
  // illustrate step 1 with a gene several types actually have
  const eg=ex.filter(e=>e.p.seen>1).sort((a,b)=>b.dsum-a.dsum)[0]||ex[0];

  let h='<h2>How the specificity scale works</h2>'+
    '<p class=std>Node <b>hue</b> is the type a gene belongs to most; node '+
    '<b>saturation</b> is how exclusively it belongs there. Every number below is '+
    'computed at the score you currently have selected (<b>≥ '+T.toFixed(2)+'</b>), so it '+
    'describes the graph on screen.</p>';

  h+='<h3><span class=n>1.</span>Count partners, per type</h3>'+
     '<p>For each gene, count how many distinct genes it is connected to, in each graph '+
     'separately. At ≥ '+T.toFixed(2)+', <b>'+esc(eg.id)+'</b> has ';
  h+=eg.d.map((v,g)=>'<b>'+v+'</b> in '+esc(LABS[g])).join(', ')+'.</p>';

  h+='<h3><span class=n>2.</span>Divide by each graph’s own size <span class=mut>— the crux</span></h3>'+
     '<p>These corpora are nowhere near the same size: '+esc(LABS[big])+'’s is <b>'+ratio+
     '×</b> '+esc(LABS[sml])+'’s at this score ('+fmtInt(tot[big]/2)+' pairs vs '+
     fmtInt(tot[sml]/2)+'). Raw counts are therefore not comparable — 20 partners means '+
     'far more in '+esc(LABS[sml])+' than in '+esc(LABS[big])+'. So each degree becomes a '+
     '<b>share of its own graph’s total connectivity</b>:</p><span class=fml>';
  for(let g=0;g<N;g++)h+=SLUGS[g]+'_share = '+SLUGS[g]+'_degree / '+
    (tot[g]<FLOORS[si]?fmtInt(FLOORS[si])+'   ← floored, see below':fmtInt(tot[g]))+'\n';
  h+='\n(total = 2 × pairs — every pair has two endpoints)</span>'+
     '<p>This asks <i>“what fraction of this type’s wiring runs through this gene?”</i> of '+
     'every type on equal terms, however lopsided the literatures are.</p>';

  // The floor is stated wherever it is in force, and named. A correction this
  // consequential must not be something the reader has to infer from the code.
  const under=[];
  for(let g=0;g<N;g++)if(tot[g]<FLOORS[si])under.push(g);
  h+='<h3><span class=n>2b.</span>…but never by a denominator this small '+
     '<span class=mut>— the correction</span></h3>'+
     '<p>Dividing by a corpus’s own size fixes the <b>bias</b> of unequal literatures. It '+
     'does nothing about the <b>noise</b>. A share is a ratio estimated from however many '+
     'endpoints the corpus has, and in a very small one a single relationship moves it more '+
     'than the best-attested gene in a large one ever reaches — so the thin corpus wins '+
     'every comparison it enters, on one paper. Each denominator is therefore held at a '+
     'floor of <b>'+fmtInt(FLOORS[si])+'</b> here (15% of the median corpus at this '+
     'score):</p><span class=fml>share = degree / max(total, '+fmtInt(FLOORS[si])+')</span>';
  if(under.length){
    const named=under.map(g=>esc(LABS[g])+' has '+fmtInt(tot[g])).join('; ');
    h+='<p>At this score that binds on <b>'+under.map(g=>esc(LABS[g])).join(', ')+
       '</b> — '+named+'.</p>'+
       '<p>It is not a size cutoff. A floored type keeps every gene it can still win on '+
       'relative evidence — where the other corpora are genuinely silent it still takes the '+
       'gene — and loses only the ones it was winning on arithmetic alone.</p>';
  } else
    h+='<p>No corpus is small enough for that to bind at this score; every share below is '+
       'divided by its own true total.</p>';

  h+='<h3><span class=n>3.</span>Measure every type against the leader</h3>'+
     '<span class=fml>max_share = the largest of those '+N+' shares\n'+
     'dominant  = the type it belongs to\n\n'+
     'T = Σ (1 − share[type] / max_share) / '+(N-1)+'</span>'+
     '<p>Bounded to <b>0 … 1</b>. A gene every type gives the same share scores <b>0</b>: '+
     'each term is 1−1 = 0. A gene only one type has scores <b>1</b>: every other term is '+
     '1−0 = 1. In between, T is the average distance the other types sit below the leader.</p>'+
     '<p>Because T uses only <i>ratios</i> of shares, it never needs the graphs to be the '+
     'same size — which is why it survives a '+ratio+'× spread. It is the Yanai '+
     'specificity index, borrowed from the identical problem in expression data: one gene, '+
     'many tissues, how restricted is it.</p>'+
     '<div class=why><p><b>Where the two-graph page fits.</b> With exactly two types this '+
     'reduces to T = 1 − smaller_share/larger_share, while <i>diff_two.py</i>’s balance '+
     'index is (larger−smaller)/(larger+smaller). Each is a monotone function of the other '+
     '(index = T/(2−T)), so the two pages rank genes identically — one just splits the '+
     'signed axis into a hue and a magnitude, because three types have no axis to be signed '+
     'along.</p></div>';

  h+='<h3><span class=n>4.</span>Name the number</h3><p>Colour uses the continuous value; '+
     'the words are just bucketed for readability. Below 0.12 the dominant type is not '+
     'named at all — a gene that flat is not that type’s, it is everyone’s.</p>'+
     '<table><tbody>';
  buckets().forEach(b=>{
    h+='<tr><td><span class=tag>'+swatchAt(eg.p.dom,b[0])+'<b>'+
      esc(label({dom:eg.p.dom,tau:b[0]}))+'</b></span></td>'+
      '<td style="padding-left:14px">'+b[1]+'</td></tr>';});
  h+='</tbody></table><p class=note>Swatches shown in '+esc(LABS[eg.p.dom])+
     '’s hue; every type has the same five steps in its own colour.</p>';

  h+='<h3>Worked examples at ≥ '+T.toFixed(2)+'</h3><table><thead><tr><th>Gene</th>';
  for(let g=0;g<N;g++)h+='<th class=num>'+esc(LABS[g])+'</th>';
  h+='<th class="num raw">raw T</th><th class=num>T</th><th>reads as</th></tr></thead><tbody>';
  ex.forEach(e=>{
    h+='<tr><td class=g>'+esc(e.id)+'</td>';
    for(let g=0;g<N;g++)h+='<td class=num>'+e.d[g]+'</td>';
    h+='<td class="num raw">'+e.raw.toFixed(2)+'</td><td class=num><b>'+
       e.p.tau.toFixed(2)+'</b></td><td><span class=tag>'+swatch(e.p)+label(e.p)+
       '</span></td></tr>';});
  h+='</tbody></table>';

  // "wrong" is measured, not asserted: where the two columns disagree
  const misread=ex.filter(e=>label({dom:e.rawdom,tau:e.raw})!==label(e.p));
  h+='<div class=why><p><b>Why bother?</b> The struck-through <b>raw T</b> column is the '+
     'same formula computed on unnormalised degrees. It disagrees with the normalised value '+
     'for <b>'+misread.length+' of these '+ex.length+' genes</b>'+
     (misread.length?' — '+misread.map(e=>esc(e.id)).join(', '):'')+
     '. Raw counts are biased toward whichever literature is largest — here <b>'+
     esc(LABS[big])+'</b>, at '+ratio+'× — so on raw degrees almost everything looks like '+
     'that type’s, and the answer becomes a restatement of which corpus is biggest rather '+
     'than a claim about biology.</p></div>';

  // The significance section, generated from the live model like the rest.
  const tested=DATA.nodes.filter(n=>n.q[si]<1||n.d.some(r=>r[si]>0));
  const sig=tested.filter(n=>n.q[si]<0.05);
  const excl=tested.filter(n=>P[n.id]&&P[n.id].tau>=0.999);
  const exsig=excl.filter(n=>n.q[si]<0.05).length;
  h+='<h3><span class=n>5.</span>How sure is it? <span class=mut>— the q column</span></h3>'+
     '<p>T is an <b>effect size</b>. It says how concentrated a gene is and nothing about '+
     'whether that concentration could have arisen by chance — which is why a gene resting '+
     'on a single paper scores 1.00 beside a well-attested marker. The <b>q</b> column is '+
     'the other half.</p>'+
     '<span class=fml>q = Benjamini-Hochberg( P(X ≥ papers in dominant type) )\n'+
     '    X ~ Binomial( all this gene’s papers, dominant type’s share of all papers )</span>'+
     '<p>The null is that a gene has no preference, so its papers land across the corpora in '+
     'proportion to how big those corpora are. Counting is done in <b>publications, not '+
     'partners</b>: one sentence naming three genes creates three pairs at once, so partner '+
     'counts treat a single author’s phrasing as several independent observations. On this '+
     'data the partner-level test calls about nine times as many genes significant as the '+
     'paper-level one.</p>'+
     '<p>At this score <b>'+fmtInt(sig.length)+'</b> of '+fmtInt(tested.length)+' genes clear '+
     'a 5% false discovery rate. Of the '+fmtInt(excl.length)+' at T = 1.00 — fully exclusive '+
     'to one type — only <b>'+fmtInt(exsig)+'</b> do.</p>'+
     '<div class=why><p><b>Power is deeply asymmetric, and that is not a defect.</b> Where '+
     'one corpus holds most of the literature, “found only there” is barely surprising and '+
     'takes tens of exclusive papers to establish, while two can suffice in a corpus holding '+
     'a fraction of a percent. A gene that fails the FDR in the dominant literature has not '+
     'been shown to be unspecific — it has not been shown to be anything.</p></div>'+
     (OVDUP[si]>0
       ? '<div class=why><p><b>⚠ On this data the test’s own precondition fails.</b> It '+
         'assumes the corpora are independent samples, which requires that no publication '+
         'be counted twice — and <b>'+fmtInt(OVDUP[si])+'</b> papers here appear in more '+
         'than one corpus. Each of those is counted as two independent observations, so '+
         'the q values are optimistic. The fix is upstream: make the queries mutually '+
         'exclusive.</p></div>'
       : '<p>The test needs the corpora to be independent samples, which requires that no '+
         'publication be counted twice. That was checked: <b>no paper here appears in more '+
         'than one corpus</b>, so the assumption holds.</p>')+
     '<p>What q does <b>not</b> license: the null concerns publishing, not biology. A small q '+
     'says this gene is written about disproportionately in this subtype’s literature, which '+
     'is confounded by research fashion, by how the queries were drawn, and by extraction '+
     'error. It is evidence about a corpus, not a claim about a tumour.</p>';

  h+='<h3>Three things to know</h3>'+
     '<p>• T is recomputed at <b>every score step</b> — degrees and totals both move, so a '+
     'gene’s label and colour can shift as you drag the slider.</p>'+
     '<p>• It is always measured over the <b>full</b> graphs, never the subgraph on screen. '+
     'Focusing a gene, unticking a type or matching sentence text changes what is drawn, '+
     'but never what a gene <i>is</i>.</p>'+
     '<p>• T says nothing about <b>how much</b> evidence there is. A gene seen once, in one '+
     'corpus, scores 1.00 — the same as a well-attested marker. Magnitude is carried '+
     'separately, by node size and by edge width, and it is what the <b>publication '+
     'sliders</b> filter on. Raising a type’s minimum is the way to ask which of its '+
     'markers are actually attested rather than merely exclusive. The sliders open at '+
     '<b>'+DEFPUB+'</b>, so every claim a corpus makes is on the table to begin with — '+
     'including its single-paper ones, which in these corpora is most of what it has. '+
     'Raising one to 2 is the way to ask that type for corroboration.</p>';
  document.getElementById('doc').innerHTML=h;
}

// --- events and state ------------------------------------------------------
let tdeb=null;   // rebuilding restarts physics, so don't do it on every keystroke
document.getElementById('textfilter').addEventListener('input',()=>{
  clearTimeout(tdeb);tdeb=setTimeout(()=>{saveFilters();build();},280);});
let gdeb=null;
document.getElementById('genefilter').addEventListener('input',()=>{
  clearTimeout(gdeb);gdeb=setTimeout(()=>{saveFilters();build();},280);});
document.getElementById('hops').addEventListener('change',()=>{saveFilters();build();});
(()=>{const dl=document.getElementById('genelist');
  DATA.nodes.forEach(n=>{const o=document.createElement('option');o.value=n.id;dl.appendChild(o);});
  // name a gene that is actually in these graphs rather than a hardcoded example
  const d=n=>Math.max.apply(null,n.d.map(r=>r[DEFSTEP]));
  const top=DATA.nodes.slice().sort((a,b)=>d(b)-d(a))[0];
  if(top)document.getElementById('genefilter').placeholder='e.g. '+top.id;})();
document.getElementById('mincluster').addEventListener('input',()=>{
  document.getElementById('mcval').textContent=minCluster();});
document.getElementById('mincluster').addEventListener('change',()=>{saveFilters();build();});
['spechi','speclo'].forEach(id=>{
  const el=document.getElementById(id);
  el.addEventListener('input',()=>{const r=specRange();
    document.getElementById('spechiv').textContent=r[1].toFixed(2);
    document.getElementById('speclov').textContent=r[0].toFixed(2);
    paintRange(r[0],r[1]);});
  el.addEventListener('change',()=>{saveFilters();build();});
});
for(let g=0;g<N;g++){const el=document.getElementById('pub'+g);
  if(!el)continue;
  // the readout tracks the drag; the rebuild waits for release, as with score
  el.addEventListener('input',()=>{const v=document.getElementById('pubv'+g);
    if(v)v.textContent=el.value;
    document.getElementById('pubsum').textContent=pubMins().join(' · ');});
  el.addEventListener('change',()=>{saveFilters();build();});}
for(let g=0;g<N;g++){const el=document.getElementById('tchk'+g);
  if(el)el.addEventListener('change',()=>{
    // never let the last type be unticked: an empty canvas with no message
    // reads as a broken page rather than as a filter
    if(!typesOn().length){el.checked=true;return;}
    saveFilters();build();});}
['yrfrom','yrto'].forEach(id=>{
  const el=document.getElementById(id);
  el.addEventListener('input',()=>{const r=yearRange();
    document.getElementById('yrfromv').textContent=r[0];
    document.getElementById('yrtov').textContent=r[1];});
  el.addEventListener('change',()=>{saveFilters();build();});
});
document.getElementById('yrnull').addEventListener('change',()=>{saveFilters();build();});
document.getElementById('fdr').addEventListener('input',()=>{
  const v=fdrMax();
  document.getElementById('fdrval').textContent=v>=1?'off':(v>=0.01?v.toFixed(2):v.toString());});
document.getElementById('fdr').addEventListener('change',()=>{saveFilters();build();});
document.getElementById('sigbtn').addEventListener('click',()=>{
  const b=document.getElementById('sigbtn');
  b.setAttribute('aria-pressed',sigOnly()?'false':'true');
  saveFilters();build();});
document.getElementById('spreadbtn').addEventListener('click',()=>{
  const b=document.getElementById('spreadbtn');
  b.setAttribute('aria-pressed',spreadOn()?'false':'true');
  saveFilters();build();});
document.getElementById('isobtn').addEventListener('click',()=>{
  const b=document.getElementById('isobtn');
  b.setAttribute('aria-pressed',removeIsolated()?'false':'true');
  saveFilters();build();});
document.getElementById('conf').addEventListener('input',()=>{
  document.getElementById('confval').textContent=STEPS[stepIdx()].toFixed(2);});
document.getElementById('conf').addEventListener('change',()=>{saveScore();build();});
// the three overlays share the canvas, so opening one closes the others
let PANEL=null;
const PANELS={tbl:['tbtn','tblwrap','Show table','Hide table'],
              sum:['sbtn','sumwrap','Summary','Hide summary'],
              doc:['dbtn','docwrap','How specificity works','Hide explainer']};
function showPanel(which){
  const opening=(PANEL!==which);
  PANEL=opening?which:null;
  // Opening the table sorts it by q ascending — most significant first — so the
  // supported genes are what a reader sees before scrolling. dir -1 reads as
  // ascending here because the q key is negated in table()'s comparator, which
  // is what makes "descending" mean "most significant" for this column.
  if(which==='tbl'&&opening){SORT={k:'q',dir:-1};if(RETABLE)RETABLE();}
  for(const k in PANELS){const [btn,wrap,off,on]=PANELS[k], b=document.getElementById(btn);
    document.getElementById(wrap).classList.toggle('on',PANEL===k);
    b.textContent=PANEL===k?on:off;
    b.setAttribute('aria-expanded',PANEL===k);}
}
// Remember what the reader left set. Wrapped in try/catch throughout: this page
// is opened over file://, where localStorage can be unavailable or throw
// outright — a saved preference must never take the graph down with it.
const LEGENDS=['legpub','legyear','legspec','legrel'], LS_KEY='type-analysis.legends',
      LS_SCORE='type-analysis.score', LS_FILTERS='type-analysis.filters';
// The page opens in the default view every time, so nothing is carried across
// opens: the restore calls below the save functions are gone, and PERSIST turns
// the savers into no-ops so no dead state is written that will never be read.
// Flip this to true to bring cross-open persistence back — the machinery is
// intact, only unhooked.
const PERSIST=false;
// A restored filter is never silent — the query sits in its own input, the
// unticked types show in the legend, and the hints say what is being shown.
function saveFilters(){
  if(!PERSIST||QUIET)return;
  try{const types=[];
    for(let g=0;g<N;g++)types.push(typeOn(g));
    localStorage.setItem(LS_FILTERS,JSON.stringify({
      pubs:pubMins(),
      fdr:String(fdrIdx()),
      sigonly:sigOnly(),
      // read from the DOM, not spreadOn(): that returns false whenever
      // "significant only" is off, which would wipe the preference on save
      spread:document.getElementById('spreadbtn').getAttribute('aria-pressed')==='true',
      iso:removeIsolated(),
      yrfrom:document.getElementById('yrfrom').value,
      yrto:document.getElementById('yrto').value,
      yrnull:keepUndated(),
      gene:document.getElementById('genefilter').value,
      hops:document.getElementById('hops').value,
      text:document.getElementById('textfilter').value,
      cluster:document.getElementById('mincluster').value,
      spechi:document.getElementById('spechi').value,
      speclo:document.getElementById('speclo').value,
      types:types}));}catch(err){}
}
function restoreFilters(){
  try{const raw=localStorage.getItem(LS_FILTERS);
    if(!raw)return;                              // first visit: empty filters
    const st=JSON.parse(raw), hops=document.getElementById('hops');
    if(typeof st.sigonly==='boolean')
      document.getElementById('sigbtn').setAttribute('aria-pressed',st.sigonly?'true':'false');
    if(typeof st.spread==='boolean')
      document.getElementById('spreadbtn').setAttribute('aria-pressed',st.spread?'true':'false');
    if(typeof st.iso==='boolean')
      document.getElementById('isobtn').setAttribute('aria-pressed',st.iso?'true':'false');
    if(typeof st.gene==='string')document.getElementById('genefilter').value=st.gene;
    if(typeof st.text==='string')document.getElementById('textfilter').value=st.text;
    // only accept a hops value the select actually offers
    if(typeof st.hops==='string'&&[...hops.options].some(o=>o.value===st.hops))
      hops.value=st.hops;
    // clamp: the slider's range may have changed since this was written
    if(typeof st.cluster==='string'){
      const mc=document.getElementById('mincluster'), v=parseInt(st.cluster);
      if(!isNaN(v)){mc.value=String(Math.max(+mc.min,Math.min(+mc.max,v)));
        document.getElementById('mcval').textContent=mc.value;}
    }
    ['spechi','speclo'].forEach(id=>{
      if(typeof st[id]!=='string')return;
      const el=document.getElementById(id), v=parseFloat(st[id]);
      if(!isNaN(v))el.value=String(Math.max(0,Math.min(1,v)));
    });
    // Only honour a type list of the right length — the same reader may have a
    // saved state from a page built over a different set of graphs, and index 3
    // would then mean a type that no longer exists. Refuse an all-off list too.
    if(Array.isArray(st.types)&&st.types.length===N&&st.types.some(Boolean))
      st.types.forEach((v,g)=>{const el=document.getElementById('tchk'+g);
        if(el)el.checked=!!v;});
    // Years are clamped to the span of whatever graphs are loaded now, which is
    // not the span they were saved against if the reader has since built a page
    // over a different set of corpora.
    ['yrfrom','yrto'].forEach(id=>{
      if(typeof st[id]!=='string')return;
      const el=document.getElementById(id), v=parseInt(st[id]);
      if(!isNaN(v)){el.value=String(Math.max(YMIN,Math.min(YMAX,v)));
        const o=document.getElementById(id==='yrfrom'?'yrfromv':'yrtov');
        if(o)o.textContent=el.value;}
    });
    if(typeof st.yrnull==='boolean')document.getElementById('yrnull').checked=st.yrnull;
    // stored as an index into FDRS, clamped in case the ladder was edited
    if(typeof st.fdr==='string'){
      const el=document.getElementById('fdr'), v=parseInt(st.fdr);
      if(!isNaN(v)){el.value=String(Math.max(0,Math.min(FDRS.length-1,v)));
        const q=fdrMax();
        document.getElementById('fdrval').textContent=
          q>=1?'off':(q>=0.01?q.toFixed(2):q.toString());}
    }
    // Same length guard, and clamp each to its own slider: the ceilings are
    // per-type and data-derived, so a threshold saved against a different set
    // of graphs could sit far outside the control it is being restored into.
    if(Array.isArray(st.pubs)&&st.pubs.length===N)
      st.pubs.forEach((v,g)=>{const el=document.getElementById('pub'+g);
        const n=parseInt(v);
        if(el&&!isNaN(n)){el.value=String(Math.max(+el.min,Math.min(+el.max,n)));
          const o=document.getElementById('pubv'+g);if(o)o.textContent=el.value;}});
  }catch(err){}
}
// The score is stored as the value (0.97), never the slider index: an index
// would quietly resolve to a different threshold if SCORE_STEPS is ever edited.
// Restored to the nearest step, the same way DEFAULT_SCORE is.
function saveScore(){
  if(!PERSIST||QUIET)return;
  try{localStorage.setItem(LS_SCORE,String(STEPS[stepIdx()]));}catch(err){}
}
function restoreScore(){
  try{const raw=localStorage.getItem(LS_SCORE);
    if(raw===null)return;                        // first visit: keep DEFAULT_SCORE
    const v=parseFloat(raw);
    if(!isFinite(v))return;
    let best=0;
    for(let i=1;i<STEPS.length;i++){
      if(Math.abs(STEPS[i]-v)<Math.abs(STEPS[best]-v))best=i;}
    document.getElementById('conf').value=best;
    document.getElementById('confval').textContent=STEPS[best].toFixed(2);
  }catch(err){}
}
// Reset writes the defaults back into the DOM, which fires the same events the
// save handlers listen on — QUIET holds them off so the wipe isn't immediately
// undone. It outlives the current task because <details> fires `toggle`
// asynchronously; a same-task flag would already be cleared by the time the
// legend's save handler ran.
let QUIET=0;
// Snapshot the markup's defaults before restoreFilters() can overwrite them, so
// Reset always agrees with a first visit.
const LEGEND_DEFAULTS={};
LEGENDS.forEach(id=>{LEGEND_DEFAULTS[id]=document.getElementById(id).open;});
function saveLegends(){
  if(!PERSIST||QUIET)return;
  try{const st={};
    LEGENDS.forEach(id=>{st[id]=document.getElementById(id).open;});
    localStorage.setItem(LS_KEY,JSON.stringify(st));}catch(err){}
}
function resetAll(){
  QUIET++;
  document.getElementById('conf').value=DEFSTEP;
  document.getElementById('confval').textContent=STEPS[DEFSTEP].toFixed(2);
  document.getElementById('genefilter').value='';
  document.getElementById('textfilter').value='';
  document.getElementById('hops').value=document.getElementById('hops').options[0].value;
  document.getElementById('mincluster').value=String(DEFCLUSTER);
  document.getElementById('mcval').textContent=String(DEFCLUSTER);
  document.getElementById('spechi').value=String(DEFTMAX);
  document.getElementById('speclo').value=String(DEFTMIN);
  document.getElementById('spechiv').textContent=DEFTMAX.toFixed(2);
  document.getElementById('speclov').textContent=DEFTMIN.toFixed(2);
  for(let g=0;g<N;g++){const el=document.getElementById('tchk'+g);if(el)el.checked=true;}
  for(let g=0;g<N;g++){const el=document.getElementById('pub'+g);
    if(el){el.value=String(DEFPUB);
      const o=document.getElementById('pubv'+g);if(o)o.textContent=String(DEFPUB);}}
  document.getElementById('yrfrom').value=String(YMIN);
  document.getElementById('yrto').value=String(YMAX);
  document.getElementById('yrfromv').textContent=YMIN;
  document.getElementById('yrtov').textContent=YMAX;
  document.getElementById('yrnull').checked=true;
  document.getElementById('fdr').value=String(DEFFDR);
  document.getElementById('fdrval').textContent=FDRS[DEFFDR]>=1?'off':String(FDRS[DEFFDR]);
  document.getElementById('sigbtn').setAttribute('aria-pressed','false');
  document.getElementById('spreadbtn').setAttribute('aria-pressed','false');
  document.getElementById('spreadbtn').hidden=true;
  document.getElementById('isobtn').setAttribute('aria-pressed','false');
  // the graph is the default view, so any overlay opened over it closes
  if(PANEL)showPanel(PANEL);
  SORT={k:'tau',dir:-1};
  LEGENDS.forEach(id=>{document.getElementById(id).open=LEGEND_DEFAULTS[id];});
  build();
  // queued after the toggle tasks, so nothing re-saves before the keys go
  setTimeout(()=>{
    QUIET--;
    try{[LS_KEY,LS_SCORE,LS_FILTERS].forEach(k=>localStorage.removeItem(k));}catch(err){}
  },0);
}
function restoreLegends(){
  try{const raw=localStorage.getItem(LS_KEY);
    if(!raw)return;                              // first visit: keep the markup's defaults
    const st=JSON.parse(raw);
    LEGENDS.forEach(id=>{if(typeof st[id]==='boolean')document.getElementById(id).open=st[id];});
  }catch(err){}
}
// PERSIST is off, so this block is skipped and the page paints the markup's
// default view on every open. The controls stay fully live within the session —
// every handler reads the DOM directly — but nothing from a previous open is
// carried in. The calls stay here, gated, so flipping PERSIST restores both
// halves of persistence at once. Order matters: both must precede build(),
// which reads the slider and filter inputs.
if(PERSIST){
  restoreLegends();
  restoreScore();
  restoreFilters();
}
LEGENDS.forEach(id=>document.getElementById(id).addEventListener('toggle',saveLegends));
document.getElementById('rbtn').addEventListener('click',resetAll);
document.getElementById('tbtn').addEventListener('click',()=>showPanel('tbl'));
document.getElementById('sbtn').addEventListener('click',()=>showPanel('sum'));
document.getElementById('dbtn').addEventListener('click',()=>showPanel('doc'));
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&PANEL)showPanel(PANEL);});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',build);
build();
</script></body></html>"""


def _esc(text):
    """Escape user text destined for HTML."""
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def _theme_vars(ramps, mode, indent):
    """The per-type custom properties for one theme block.

    Gradients and edge inks have to be CSS variables rather than inline styles:
    an inline style cannot answer to prefers-color-scheme, and the legend has to
    change with the theme like everything else does.
    """
    lines = []
    for g, arm in enumerate(ramps[mode]):
        stops = ",".join(arm["fill"][i] for i in range(0, RAMP_STEPS, 5))
        lines.append("--grad%d:linear-gradient(to right,%s);" % (g, stops))
        lines.append("--etype%d:%s;" % (g, arm["fill"][-1]))
    lines.append("--edge-shared:%s;" % SHARED_EDGE[mode])
    return "\n".join(indent + line for line in lines)


def _arms_html(labels):
    """One ramp row per type, each with its own visibility tick and cut masks."""
    out = []
    for g, lab in enumerate(labels):
        out.append(
            '        <div class="arm">\n'
            '          <div class="armhd">\n'
            '            <label><input type="checkbox" id="tchk%d" checked>'
            '<span class="nmx">%s</span></label>\n'
            '            <span class="cnt" id="acnt%d"></span>\n'
            '          </div>\n'
            '          <span class="hramp" style="background:var(--grad%d)" role="img"\n'
            '                aria-label="%s: neutral at the left for a ubiquitous gene, '
            'full colour at the right for one exclusive to it"><i class="cut" id="cutL%d">'
            '</i><i class="cut" id="cutR%d"></i></span>\n'
            '        </div>' % (g, _esc(lab), g, g, _esc(lab), g, g))
    return "\n".join(out)


def _pub_rows_html(labels, ceilings, trues):
    """One publication slider per type, each on its own data-derived range."""
    out = []
    for g, lab in enumerate(labels):
        # the tooltip carries the true maximum, which the capped slider cannot
        tip = "%s — pairs reach %d publication%s" % (
            lab, trues[g], "" if trues[g] == 1 else "s")
        out.append(
            '        <div class="prow">\n'
            '          <span class="pdot" style="background:var(--etype%d)"></span>\n'
            '          <label for="pub%d" title="%s">%s</label>\n'
            '          <input id="pub%d" type="range" min="1" max="%d" step="1" value="%d"\n'
            '                 title="%s" aria-label="Minimum publications for %s">\n'
            '          <span class="sval" id="pubv%d">%d</span>\n'
            '        </div>' % (g, g, _esc(tip), _esc(lab), g, ceilings[g],
                                DEFAULT_PUB, _esc(tip), _esc(lab), g, DEFAULT_PUB))
    return "\n".join(out)


def _edge_legend_html(labels):
    return "\n".join(
        '          <span class="lg"><span class="ln" style="background:var(--etype%d)">'
        '</span><span class="nmx">%s only</span></span>' % (g, _esc(lab))
        for g, lab in enumerate(labels))


def render(nodes, edges, totals, floors, ptotals, tested, ovdup, ovmat,
           years, ramps, vis_runtime, title, labels, ceilings, trues):
    ymin, ymax, ynull = years
    # A corpus with no dated evidence at all would otherwise render a slider
    # whose two ends are the same value. Collapse it to a one-year span; the
    # section still shows, and its own hint reports that everything is undated.
    if ymin is None:
        ymin = ymax = 0
    payload = json.dumps({"nodes": nodes, "edges": edges}, separators=(",", ":"))
    dots = "".join('<i style="background:var(--etype%d)"></i>' % g
                   for g in range(len(labels)))
    keys = dots + '<i style="background:var(--edge-shared)"></i>'
    html = TEMPLATE
    for token, value in (
        ("__VIS__", vis_runtime),
        ("__PAYLOAD__", payload),
        ("__RAMPJS__", json.dumps(ramps, separators=(",", ":"))),
        ("__STEPSJS__", json.dumps(SCORE_STEPS, separators=(",", ":"))),
        ("__TOTALSJS__", json.dumps(totals, separators=(",", ":"))),
        ("__FLOORSJS__", json.dumps(floors, separators=(",", ":"))),
        ("__PTOTJS__", json.dumps(ptotals, separators=(",", ":"))),
        ("__NTESTJS__", json.dumps(tested, separators=(",", ":"))),
        ("__OVDUPJS__", json.dumps(ovdup, separators=(",", ":"))),
        ("__OVMATJS__", json.dumps(ovmat, separators=(",", ":"))),
        ("__FDRSJS__", json.dumps(FDR_STEPS, separators=(",", ":"))),
        ("__MAXFDR__", str(len(FDR_STEPS) - 1)),
        ("__DEFFDR__", str(DEFAULT_FDR_STEP)),
        ("__VARSL__", _theme_vars(ramps, "light", "  ")),
        ("__VARSD__", _theme_vars(ramps, "dark", "    ")),
        ("__MAXSTEP__", str(len(SCORE_STEPS) - 1)),
        ("__DEFSTEP__", str(DEFAULT_STEP)),
        ("__DEFCONF__", "%.2f" % SCORE_STEPS[DEFAULT_STEP]),
        ("__DEFPUB__", str(DEFAULT_PUB)),
        ("__DEFCLUSTER__", str(DEFAULT_CLUSTER)),
        ("__DEFTMIN__", "%.2f" % DEFAULT_TMIN),
        ("__DEFTMAX__", "%.2f" % DEFAULT_TMAX),
        ("__DEFFDRV__", str(DEFAULT_FDR)),
        ("__LOWSCORE__", repr(LOW_SCORE)),
        ("__YMIN__", str(ymin)),
        ("__YMAX__", str(ymax)),
        ("__YNULL__", str(ynull)),
        ("__BD__", str(NODE_BORDER)),
        ("__BDSIG__", str(NODE_BORDER_SIG)),
        ("__BDSEL__", str(NODE_BORDER_SEL)),
        ("__BDSIGSEL__", str(NODE_BORDER_SIG_SEL)),
        ("__PUBTRUEJS__", json.dumps(trues, separators=(",", ":"))),
        ("__NTYPES__", str(len(labels))),
        ("__MDOTS__", dots),
        ("__MKEYS__", keys),
        # last: these carry user text, so substituting them first would let a
        # label containing another token corrupt the following replacements
        ("__ARMS__", _arms_html(labels)),
        ("__PUBROWS__", _pub_rows_html(labels, ceilings, trues)),
        ("__ELEG__", _edge_legend_html(labels)),
        ("__LABSJS__", json.dumps(labels)),
        ("__TITLE__", _esc(title)),
    ):
        html = html.replace(token, value)
    return html


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
#
# Every name comes from stdin; there are no defaults to fall back on except the
# per-file labels, which are only a convenience — with a dozen inputs, typing
# out a dozen names that are already in the file names is friction, not care.
# All of this lives behind main() rather than at module level so
# `import type_analysis` stays silent — importing must never block on a prompt.

def _ask(label):
    """One line from stdin. EOF is an abort, not an empty answer."""
    try:
        return input(label).strip().strip('"\'')
    except EOFError:
        raise SystemExit("\naborted: %s needs a value" % label.strip().rstrip(":"))


def _resolve(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def default_label(path):
    """A readable type name guessed from a file name.

    'lung_squamous_2026_07_19_G.html' -> 'lung squamous'. Trailing date parts and
    the pipeline's _G suffix carry no meaning to a reader, so they go; anything
    that is not obviously a date is kept, because guessing harder would start
    discarding real words.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = [p for p in stem.split("_") if p]
    while parts and (parts[-1] == "G" or parts[-1].isdigit()):
        parts.pop()
    return " ".join(parts) if parts else stem


def reject_reason(path):
    """None if this is a usable source graph, otherwise why it is not.

    Validation is by parsing, not by trusting the name. Everything this pipeline
    produces is called *_G.html — including diff_two.py's own output, which
    carries a DATA block of exactly the right shape but with the edges already
    collapsed to f/t/s/l. It looked enough like a source graph to be accepted at
    the prompt, survive the whole input phase, and then die on a bare
    KeyError: 'from' with nothing said about which file was at fault.

    Paying one parse per candidate here buys a message at the point the name was
    typed, which is the only point the reader can do anything about it.
    """
    try:
        data = extract_data(path)
    except ValueError:
        # covers both a missing `const DATA=` and malformed JSON after it
        return "no readable `const DATA=` block — not a graph from this pipeline"
    except (IOError, OSError, UnicodeDecodeError):
        return "could not be read as text"
    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        return "DATA has no nodes/edges"
    edges = data["edges"]
    if not edges:
        return "graph has no relationships"
    first = edges[0]
    if "from" not in first:
        if "f" in first and "t" in first:
            return ("a diff_two.py output, not a source graph — this needs the "
                    "per-query graphs the pipeline produces, not a comparison "
                    "built from them")
        return "edges are not in this pipeline's format"
    missing = [k for k in ("to", "sents") if k not in first]
    if missing:
        return "edges lack %s" % " and ".join(missing)
    # sentence shape decides whether any evidence survives collapse_to_pairs, so
    # check a real one rather than the first edge, which may carry none
    for edge in edges:
        for sent in edge.get("sents", ()):
            gaps = [k for k in ("pmid", "sc", "text") if k not in sent]
            return ("sentences lack %s" % " and ".join(gaps)) if gaps else None
    return "no sentences on any relationship"


def ask_input_graphs():
    """Collect two or more source graphs, by name or by glob, blank to finish."""
    print("input graphs — one name or glob per line, blank line when done:")
    chosen = []
    while True:
        name = _ask("  graph %d: " % (len(chosen) + 1))
        if not name:
            if len(chosen) >= MIN_INPUTS:
                return chosen
            print("    %d more needed (this compares types, so it needs at least %d)"
                  % (MIN_INPUTS - len(chosen), MIN_INPUTS))
            continue
        pattern = _resolve(name)
        # A glob is the point of this prompt — `lung_*_G.html` in one line rather
        # than five. Sorted, so a run is reproducible and the type order on the
        # page is not down to whatever order the filesystem felt like.
        if any(ch in name for ch in "*?["):
            hits = sorted(p for p in globmod.glob(pattern) if os.path.isfile(p))
            if not hits:
                print("    nothing matches: %s" % pattern)
                continue
            fresh = [p for p in hits if p not in chosen]
            # A sweep that happens to catch a diff graph should drop it and say
            # so, not refuse the whole pattern — *_G.html matching both kinds is
            # the normal case, not a mistake worth restarting for.
            good = []
            for path in fresh:
                why = reject_reason(path)
                if why:
                    print("    - %s — %s" % (os.path.basename(path), why))
                else:
                    print("    + %s" % os.path.basename(path))
                    good.append(path)
            if len(fresh) < len(hits):
                print("    (%d already listed)" % (len(hits) - len(fresh)))
            if not good and fresh:
                print("    nothing usable in that pattern")
            chosen.extend(good)
            continue
        if not os.path.isfile(pattern):
            print("    no such file: %s" % pattern)
            continue
        if pattern in chosen:
            print("    already listed")
            continue
        why = reject_reason(pattern)
        if why:
            print("    not usable: %s" % why)
            continue
        chosen.append(pattern)


def ask_output_path(inputs):
    """Prompt until given a writable name that is not one of the inputs."""
    while True:
        name = _ask("output file name: ")
        if not name:
            print("  a file name is required")
            continue
        if not os.path.splitext(name)[1]:
            name += ".html"
        path = _resolve(name)
        # The inputs are read, not written. A typo here would destroy megabytes
        # of upstream pipeline output with no way back.
        if any(os.path.abspath(path) == os.path.abspath(p) for p in inputs):
            print("  that is an input graph — pick another name")
            continue
        return path


def ask_graph_name():
    while True:
        name = _ask("graph name: ")
        if name:
            return name
        print("  a graph name is required")


def ask_labels(paths):
    """What to call each type, everywhere in the finished page."""
    print("labels — blank accepts the name shown:")
    labels = []
    for path in paths:
        fallback = default_label(path)
        name = _ask("  label for %s [%s]: " % (os.path.basename(path), fallback))
        labels.append(name or fallback)
    return labels


def main():
    srcs = ask_input_graphs()
    out = ask_output_path(srcs)
    title = ask_graph_name()
    labels = ask_labels(srcs)

    if len(srcs) > len(POLES):
        print("note: %d types exceeds the %d hand-picked colours; the rest are "
              "spaced round the hue circle and may be hard to tell apart"
              % (len(srcs), len(POLES)))

    per_graph = [collapse_to_pairs(extract_data(p)) for p in srcs]
    nodes, edges, totals, ptotals, tested = build_model(per_graph)
    floors = share_floors(totals)
    ovdup, ovmat = paper_overlap(per_graph)
    years = year_span(per_graph)
    # each type's slider gets its own range, from its own evidence
    trues = [max([pubs_at(s, FLOOR) for s in pairs.values()] or [1])
             for pairs in per_graph]
    ceilings = [pub_ceiling(pairs) for pairs in per_graph]

    html = render(nodes, edges, totals, floors, ptotals, tested, ovdup, ovmat,
                  years, build_ramps(len(srcs)), extract_vis_runtime(srcs[-1]),
                  title, labels, ceilings, trues)
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote %s (%.1f MB) — %d genes, %d relationships over %d types"
          % (out, len(html) / 1048576.0, len(nodes), len(edges), len(srcs)))
    if years[0] is None:
        print("publication years: none — every sentence is undated")
    else:
        print("publication years: %d-%d (%d sentences undated)"
              % (years[0], years[1], years[2]))
    print("publication sliders: %s"
          % ", ".join("%s 1-%d (reaches %d)" % (lab, ceilings[g], trues[g])
                      for g, lab in enumerate(labels)))
    # Naming the floored corpora is the point: a share correction that changes
    # which type owns a gene should never be silent at the command line either.
    at = DEFAULT_STEP
    floored = [labels[g] for g in range(len(labels)) if totals[g][at] < floors[at]]
    print("share floor at %.2f: %d endpoints — %s"
          % (SCORE_STEPS[at], floors[at],
             ("floors " + ", ".join(floored)) if floored
             else "binds on no corpus"))
    tested = [n for n in nodes if any(r[at] for r in n["d"])]
    sig = [n for n in tested if n["q"][at] < 0.05]
    print("significance at %.2f: %d of %d genes clear a 5%% FDR"
          % (SCORE_STEPS[at], len(sig), len(tested)))
    # Silence here would read as "checked and fine", so the clean case is stated
    # too — this is the assumption the whole q column rests on.
    papers = sum(t[at] for t in ptotals)
    if ovdup[at]:
        worst, pair = 0, None
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                if ovmat[at][a][b] > worst:
                    worst, pair = ovmat[at][a][b], (a, b)
        print("WARNING: %d of %d papers at %.2f (%.1f%%) appear in more than one "
              "corpus" % (ovdup[at], papers, SCORE_STEPS[at],
                          100.0 * ovdup[at] / max(1, papers)))
        if pair:
            print("         worst pair: %s and %s share %d"
                  % (labels[pair[0]], labels[pair[1]], worst))
        print("         q assumes disjoint corpora — a paper counted twice is "
              "counted as two")
        print("         independent observations, so q values are optimistic.")
    else:
        print("corpora are disjoint at %.2f: no paper is counted twice, so q is sound"
              % SCORE_STEPS[at])


if __name__ == "__main__":
    main()
