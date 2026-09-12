"""F4 -- deterministic grounding score. Pure code, no model call.

Principle 3 from the brief: "deterministic code decides, the model proposes."
Every cited claim in the report gets a word-overlap score against the real
source text it cites -- this is the mechanism that makes a manipulated or
hallucinated citation visibly fail its grounding check.

Deliberately simple and auditable: a stopword-filtered word-overlap ratio,
not embeddings or a second model call. If this were itself a model call, a
faithfulness bug in the report-writing model could plausibly also infect the
scoring model in a correlated way -- the whole point is an independent,
inspectable check.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

# Words too common to be evidence of grounding either way -- excluding them
# stops near-universal function words from inflating every score toward 1.0.
STOPWORDS = frozenset(
    """
    a an the of and or to in is are was were be been being this that these
    those for with on as by at from into over under between among within
    without not no nor but if then than so such it its it's their there
    here which who whom whose what when where why how all any both each
    few more most other some such only own same can will just should now
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


@dataclass
class GroundingResult:
    score: float  # 0.0-1.0: fraction of the claim's substantive words found in source
    matched_words: list[str]
    unmatched_words: list[str]
    claim_word_count: int
    source_word_count: int


def score_claim(claim_text: str, source_text: str) -> GroundingResult:
    """How much of `claim_text`'s substantive vocabulary appears in `source_text`.

    Recall-style, measured against the claim (not symmetric Jaccard): a claim
    that adds specifics absent from the source -- a fabricated number, an
    unsupported drug name -- loses score even if the source is much longer
    than the claim. An empty claim scores 0.0 rather than dividing by zero.
    """
    claim_words = _tokenize(claim_text)
    source_words = set(_tokenize(source_text))

    if not claim_words:
        return GroundingResult(0.0, [], [], 0, len(source_words))

    matched = [w for w in claim_words if w in source_words]
    unmatched = [w for w in claim_words if w not in source_words]
    # de-dupe while preserving first-seen order, for a stable/readable UI list
    matched_unique = list(dict.fromkeys(matched))
    unmatched_unique = list(dict.fromkeys(unmatched))

    score = len(matched) / len(claim_words)
    return GroundingResult(
        score=round(score, 4),
        matched_words=matched_unique,
        unmatched_words=unmatched_unique,
        claim_word_count=len(claim_words),
        source_word_count=len(source_words),
    )


def score_claim_against_sources(claim_text: str, source_texts: list[str]) -> GroundingResult:
    """Score against the best-matching of several source texts (multiple citations)."""
    if not source_texts:
        return score_claim(claim_text, "")
    results = [score_claim(claim_text, src) for src in source_texts]
    return max(results, key=lambda r: r.score)


def excerpt_window(source_text: str, result: GroundingResult, *, window_words: int = 45) -> str:
    """~window_words words of `source_text` centered on the first run of the
    claim's matched words, so a long retrieved excerpt (up to 1500 chars) can
    be quoted as a short window in a provenance callout rather than
    reproduced in full.

    Deliberately plain code, same as the rest of this module: token position
    lookup, no model call. Falls back to the first `window_words` words if no
    matched word is found (e.g. an empty claim).
    """
    words = source_text.split()
    if not words:
        return source_text
    matched_lower = {w.lower() for w in result.matched_words}
    idx = None
    for i, w in enumerate(words):
        token = re.sub(r"[^a-z0-9\-]", "", w.lower())
        if token in matched_lower:
            idx = i
            break
    if idx is None:
        idx = 0

    half = window_words // 2
    start = max(0, idx - half)
    end = min(len(words), start + window_words)
    start = max(0, end - window_words)
    snippet = " ".join(words[start:end])
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(words) else ""
    return f"{prefix}{snippet}{suffix}"


def highlight_html(claim_text: str, result: GroundingResult) -> str:
    """Render `claim_text` with matched words wrapped for CSS highlighting.

    Used by F9: matched words get class="grounded", unmatched words get
    class="ungrounded" -- the visible red/green highlighting shown next to
    each cited claim. Matching is done on tokens, not substrings, to avoid
    highlighting "her" inside "hers".

    `claim_text` is model-generated (an interpretation agent's own words), so
    it is escaped BEFORE the span-wrapping regex runs, and the result is safe
    to render with Jinja's `| safe` filter -- escaping first doesn't change
    which words match, since HTML escaping never touches alphanumeric runs.
    """
    escaped_text = html.escape(claim_text)
    matched_set = set(result.matched_words)
    unmatched_set = set(result.unmatched_words)

    def _wrap(match: re.Match[str]) -> str:
        original = match.group(0)
        lowered = original.lower()
        if lowered in matched_set:
            return f'<span class="grounded">{original}</span>'
        if lowered in unmatched_set:
            return f'<span class="ungrounded">{original}</span>'
        return original

    return _WORD_RE.sub(_wrap, escaped_text)
