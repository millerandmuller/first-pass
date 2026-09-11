"""F8 -- curated genetic/knockout precedent dataset (Section 2 of the report).

Explicitly [GEMOCKT/KURATIERT], not [ECHT]: there is no free, queryable API
for knockout-mouse phenotype data suitable for a hackathon build (a real
integration would use MGI/IMPC), so this is a small hand-curated dataset
drawn from well-established developmental-biology literature. It is always
registered in the citation ledger with `curated=True` so the report's
[GEMOCKT/KURATIERT] badge renders wherever it is cited -- this data source is
never presented as an openFDA/FDA-sourced [ECHT] claim.

Unlisted targets return None -- Section 5 edge case 1 (clean empty state,
never a fabricated section) applies here too.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "demo_data", "genetics_precedent.json")


@dataclass
class GeneticPrecedent:
    target: str
    gene_symbol: str
    knockout_viability: str
    phenotype_summary: str
    literature_note: str
    relevance_to_target_safety: str


def _normalize(target: str) -> str:
    return target.strip().upper().replace(" ", "")


def load_all() -> dict[str, GeneticPrecedent]:
    with open(_DATA_PATH) as fh:
        raw = json.load(fh)

    precedents: dict[str, GeneticPrecedent] = {}
    for target, entry in raw["targets"].items():
        precedents[_normalize(target)] = GeneticPrecedent(
            target=target,
            gene_symbol=entry["gene_symbol"],
            knockout_viability=entry["knockout_viability"],
            phenotype_summary=entry["phenotype_summary"],
            literature_note=entry["literature_note"],
            relevance_to_target_safety=entry["relevance_to_target_safety"],
        )
    return precedents


def get_precedent(target: str) -> Optional[GeneticPrecedent]:
    return load_all().get(_normalize(target))
