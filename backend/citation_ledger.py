"""F2 -- citation ledger and aggregated bibliography.

Principle 1 from the brief: "no sentence without a clickable source." Every
claim in the report carries an inline reference ID; every reference ID
resolves to one entry in the final bibliography with a full document title,
FDA application number, and date. This module is the single registry that
guarantees that round-trip -- a section writer calls `register()` once per
source it draws on and gets back a stable ID it can inline into report text
as `[R-3]`; the same (source_type, url) pair always returns the same ID, so
citing the same label twice from different sections does not fragment the
bibliography.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Source:
    reference_id: str
    source_type: str  # "label" | "review_pdf" | "drugsfda" | "curated" | "evaluation"
    title: str
    application_number: Optional[str]
    date: Optional[str]
    url: str
    accessed_at: str
    curated: bool = False  # True for [GEMOCKT/KURATIERT] sources (F8) -- never hidden


class CitationLedger:
    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._key_to_id: dict[tuple[str, str], str] = {}
        self._order: list[str] = []
        self._counter = 0

    def register(
        self,
        *,
        source_type: str,
        title: str,
        url: str,
        application_number: Optional[str] = None,
        date: Optional[str] = None,
        curated: bool = False,
    ) -> str:
        key = (source_type, url)
        if key in self._key_to_id:
            return self._key_to_id[key]

        self._counter += 1
        reference_id = f"R-{self._counter}"
        source = Source(
            reference_id=reference_id,
            source_type=source_type,
            title=title,
            application_number=application_number,
            date=date,
            url=url,
            accessed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            curated=curated,
        )
        self._sources[reference_id] = source
        self._key_to_id[key] = reference_id
        self._order.append(reference_id)
        return reference_id

    def get(self, reference_id: str) -> Optional[Source]:
        return self._sources.get(reference_id)

    def bibliography(self) -> list[Source]:
        return [self._sources[rid] for rid in self._order]

    def __len__(self) -> int:
        return len(self._order)

    def cite(self, reference_id: str) -> str:
        """Inline citation marker for report text, e.g. "[R-3]"."""
        if reference_id not in self._sources:
            raise KeyError(f"unregistered reference id: {reference_id}")
        return f"[{reference_id}]"

    def to_markdown(self) -> str:
        lines = ["## References & Regulatory Sources", ""]
        for source in self.bibliography():
            parts = [f"**[{source.reference_id}]**", source.title]
            if source.application_number:
                parts.append(f"({source.application_number})")
            if source.date:
                parts.append(f"-- {source.date}")
            line = " ".join(parts) + f" -- [{source.url}]({source.url})"
            if source.curated:
                line += " *[GEMOCKT/KURATIERT]*"
            lines.append(f"- {line}")
        return "\n".join(lines)

    def to_html(self) -> str:
        items = []
        for source in self.bibliography():
            badge = (
                '<span class="curated-badge">GEMOCKT/KURATIERT</span>' if source.curated else ""
            )
            app_no = f" ({source.application_number})" if source.application_number else ""
            date = f" &mdash; {source.date}" if source.date else ""
            items.append(
                f'<li id="ref-{source.reference_id}">'
                f'<span class="ref-id">[{source.reference_id}]</span> '
                f'<a href="{source.url}" target="_blank" rel="noopener">{source.title}</a>'
                f"{app_no}{date} {badge}</li>"
            )
        return '<ol class="bibliography">' + "".join(items) + "</ol>"
