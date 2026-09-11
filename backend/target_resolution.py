"""F6 -- Target resolution and class mapping. The load-bearing hop.

Architectural finding (verified against the live openFDA API, 2026-09-11):
openFDA cannot be searched by target name. A restricted search against
`nonclinical_toxicology` for "HER2", "KRAS", or "EGFR" returns zero hits for
all three -- but drug names ARE indexed (trastuzumab -> 13 hits, semaglutide
-> 10). Every downstream feature (F1 retrieval, F3 interpretation swarm, F4
grounding score, F5 section writers) searches by drug name, never by target
name. If this resolver returns the wrong drug list, or an empty one, nothing
downstream errors -- it just produces a valid-looking, silently empty report.

Two paths, both verified live:
  PRIMARY  -- openfda.pharm_class_moa lookup. The MoA class string IS the
              target link (e.g. searching "HER2" matches labels tagged
              "HER2/Neu/cerbB2 Antagonists [MoA]"). openFDA's controlled
              vocabulary rarely matches a user's target name verbatim
              ("Kinase Inhibitor" -> NOT_FOUND despite kinase-inhibitor
              drugs existing), so this needs an alias table, not a raw pass-
              through. TARGET_ALIASES below lists only phrases individually
              confirmed against the live API -- an unlisted target is not a
              bug, it is exactly the case the FALLBACK path exists for.
  FALLBACK -- unqualified full-text search across the whole label
              (`search=<term>`, no field qualifier). Broader recall, noisier
              (matches any mention of the term, not just the MoA field).
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend.config import OPENFDA_LABEL_ENDPOINT
from backend.http_utils import get_json

# phrase -> confirmed non-zero total at verification time (informational only;
# the live call is always re-checked, this is not used as a cached count).
# Each phrase was checked individually against
# openfda.pharm_class_moa:"<phrase>" on 2026-09-11.
TARGET_ALIASES: dict[str, list[str]] = {
    "HER2": ["HER2"],
    "ERBB2": ["HER2"],
    "GLP-1R": ["Glucagon-Like Peptide-1"],
    "GLP1R": ["Glucagon-Like Peptide-1"],
    "GLP-1": ["Glucagon-Like Peptide-1"],
    "PD-L1": ["Programmed Death Ligand 1"],
    "PDL1": ["Programmed Death Ligand 1"],
    "PD-1": ["Programmed Death Receptor-1"],
    "PD1": ["Programmed Death Receptor-1"],
    "VEGF": ["Vascular Endothelial Growth Factor"],
    "CD20": ["CD20"],
    "CD19": ["CD19"],
    "CD38": ["CD38"],
    "CD3": ["CD3"],
    "TNF": ["Tumor Necrosis Factor"],
    "TNF-ALPHA": ["Tumor Necrosis Factor"],
    "TNFA": ["Tumor Necrosis Factor"],
    "IL-6": ["Interleukin-6"],
    "IL6": ["Interleukin-6"],
    "IL-6R": ["Interleukin-6 Receptor"],
    "IL-17": ["Interleukin-17"],
    "IL17": ["Interleukin-17"],
    "IL-23": ["Interleukin-23"],
    "IL23": ["Interleukin-23"],
    "JAK": ["Janus Kinase"],
    "BTK": ["Bruton's Tyrosine Kinase"],
    "CTLA-4": ["CTLA-4"],
    "CTLA4": ["CTLA-4"],
    "DPP-4": ["Dipeptidyl Peptidase-4"],
    "DPP4": ["Dipeptidyl Peptidase-4"],
    # KRAS deliberately absent: no confirmed pharm_class_moa phrase exists.
    # It falls through to the full-text fallback by design -- this is the
    # demo's honest-gap target (Section 5 edge case 2).
}


@dataclass
class SearchAttempt:
    path: str  # "pharm_class_moa" | "full_text"
    phrase: str
    url: str
    status: str
    total: int


@dataclass
class TargetResolution:
    target_query: str
    resolution_path: str  # "pharm_class_moa" | "full_text" | "not_found"
    matched_phrase: Optional[str]
    drug_names: list[str]
    total_labels: int
    searched_at: str
    search_trail: list[SearchAttempt] = field(default_factory=list)
    data_source_status: str = "live"  # "live" | "cached" | "error"

    @property
    def found(self) -> bool:
        return self.resolution_path != "not_found" and bool(self.drug_names)


def _extract_drug_names(records: list[dict], limit: int = 25) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for rec in records:
        openfda = rec.get("openfda", {})
        for field_name in ("generic_name", "brand_name"):
            for name in openfda.get(field_name, []):
                key = name.strip().upper()
                if key and key not in seen:
                    seen.add(key)
                    names.append(name.strip())
        if len(names) >= limit:
            break
    return names[:limit]


def _search_label(query: str, limit: int) -> tuple[FetchResultStatus, dict, str]:
    url = f"{OPENFDA_LABEL_ENDPOINT}?search={urllib.parse.quote(query)}&limit={limit}"
    result = get_json(url)
    return result.status, (result.data or {}), url


FetchResultStatus = str  # alias for readability in the tuple above


def resolve_target(target: str, *, record_limit: int = 25) -> TargetResolution:
    """Resolve a target name to a list of approved drug names via two paths."""
    target = target.strip()
    normalized = target.upper().replace(" ", "")
    trail: list[SearchAttempt] = []
    now = datetime.now(timezone.utc).isoformat()

    phrases = TARGET_ALIASES.get(normalized, [])
    for phrase in phrases:
        query = f'openfda.pharm_class_moa:"{phrase}"'
        status, data, url = _search_label(query, record_limit)
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        trail.append(SearchAttempt("pharm_class_moa", phrase, url, status, total))
        if status in ("live", "cached") and total > 0:
            drug_names = _extract_drug_names(data.get("results", []))
            if drug_names:
                return TargetResolution(
                    target_query=target,
                    resolution_path="pharm_class_moa",
                    matched_phrase=phrase,
                    drug_names=drug_names,
                    total_labels=total,
                    searched_at=now,
                    search_trail=trail,
                    data_source_status=status,
                )

    # FALLBACK: unqualified full-text search. Must be phrase-quoted -- an
    # unquoted multi-word query is OR-tokenized by openFDA's search engine,
    # so garbage input like "ZZZ-NOT-A-REAL-TARGET-12345" was observed
    # matching on stray common tokens ("NOT", "A") and returning the entire
    # 250k+ label corpus instead of zero (verified 2026-09-11). Quoting
    # forces an exact-phrase match and correctly returns NOT_FOUND for
    # non-existent targets while preserving the verified real-target counts
    # (HER2 164, KRAS 12, EGFR 2581).
    query = f'"{target}"'
    status, data, url = _search_label(query, record_limit)
    total = data.get("meta", {}).get("results", {}).get("total", 0)
    trail.append(SearchAttempt("full_text", query, url, status, total))
    if status in ("live", "cached") and total > 0:
        drug_names = _extract_drug_names(data.get("results", []))
        if drug_names:
            return TargetResolution(
                target_query=target,
                resolution_path="full_text",
                matched_phrase=target,
                drug_names=drug_names,
                total_labels=total,
                searched_at=now,
                search_trail=trail,
                data_source_status=status,
            )

    # Honest empty result -- Section 5 edge case 1: clean empty state with
    # the attempted search path, never a fabricated section.
    return TargetResolution(
        target_query=target,
        resolution_path="not_found",
        matched_phrase=None,
        drug_names=[],
        total_labels=0,
        searched_at=now,
        search_trail=trail,
        data_source_status=status,
    )
