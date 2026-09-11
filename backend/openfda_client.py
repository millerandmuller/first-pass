"""F1 -- openFDA retrieval layer, two-hop.

Hop 1: the openFDA label endpoint gives structured label text per drug --
including `nonclinical_toxicology` where present -- and the drugsfda endpoint
gives the approval history and a list of `application_docs`.

Hop 2: an `application_docs` entry of type "Review" does not point to the
review text. It points to a TOC HTML page on accessdata.fda.gov whose actual
review PDFs live at `<same-path-minus-"TOC.html"><Suffix>.pdf` (verified
2026-09-11 against NDA 214801: the TOC page's own embedded JS hardcodes
`pdfBaseName` to exactly that string, so it can be derived from the URL
without executing or parsing the page). Which suffix exists varies by
application -- modern ones consolidate into MultidisciplineR.pdf, older ones
split into PharmR.pdf (non-clinical) / MedR.pdf (clinical) -- so multiple
candidates are tried in priority order and the first 200 wins.

accessdata.fda.gov additionally blocks requests without a browser-like
User-Agent (returns an Akamai "apology" page, HTTP 200, that looks like real
HTML at a glance) -- see ACCESSDATA_USER_AGENT in config.py.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import requests
from pypdf import PdfReader

from backend.config import (
    ACCESSDATA_USER_AGENT,
    CACHE_DIR,
    MAX_DRUGS_FOR_DEEP_REVIEW,
    OPENFDA_DRUGSFDA_ENDPOINT,
    OPENFDA_LABEL_ENDPOINT,
    OPENFDA_MAX_RETRIES,
    OPENFDA_RETRY_BACKOFF_SECONDS,
    OPENFDA_TIMEOUT_SECONDS,
    REVIEW_DOC_SUFFIXES,
)
from backend.http_utils import get_json


@dataclass
class LabelRecord:
    drug_name: str
    application_number: Optional[str]
    manufacturer_name: Optional[str]
    nonclinical_toxicology: list[str]
    boxed_warning: list[str]
    warnings: list[str]
    adverse_reactions: list[str]
    indications_and_usage: list[str]
    source_status: str  # "live" | "cached" | "zero_matches" | "error"
    source_url: str


@dataclass
class ApplicationDoc:
    doc_type: str
    url: str
    date: Optional[str]


@dataclass
class ApprovalRecord:
    application_number: Optional[str]
    sponsor_name: Optional[str]
    submissions_count: int
    review_docs: list[ApplicationDoc]
    source_status: str
    source_url: str


@dataclass
class ReviewDocument:
    application_number: Optional[str]
    pdf_url: str
    suffix_used: str
    page_count: int
    text_by_page: list[str]
    fetch_status: str  # "live" | "cached" | "not_found"


def get_label(drug_name: str) -> LabelRecord:
    query = f'openfda.generic_name:"{drug_name}"'
    url = f"{OPENFDA_LABEL_ENDPOINT}?search={urllib.parse.quote(query)}&limit=1"
    result = get_json(url)

    if not result.ok or result.status == "zero_matches" or not (result.data or {}).get("results"):
        return LabelRecord(
            drug_name=drug_name,
            application_number=None,
            manufacturer_name=None,
            nonclinical_toxicology=[],
            boxed_warning=[],
            warnings=[],
            adverse_reactions=[],
            indications_and_usage=[],
            source_status=result.status,
            source_url=url,
        )

    record = result.data["results"][0]
    openfda = record.get("openfda", {})
    return LabelRecord(
        drug_name=drug_name,
        application_number=(openfda.get("application_number") or [None])[0],
        manufacturer_name=(openfda.get("manufacturer_name") or [None])[0],
        nonclinical_toxicology=record.get("nonclinical_toxicology", []),
        boxed_warning=record.get("boxed_warning", []),
        warnings=record.get("warnings", record.get("warnings_and_cautions", [])),
        adverse_reactions=record.get("adverse_reactions", []),
        indications_and_usage=record.get("indications_and_usage", []),
        source_status=result.status,
        source_url=url,
    )


def get_approval_history(drug_name: str) -> ApprovalRecord:
    query = f'openfda.generic_name:"{drug_name}"'
    url = f"{OPENFDA_DRUGSFDA_ENDPOINT}?search={urllib.parse.quote(query)}&limit=1"
    result = get_json(url)

    if not result.ok or result.status == "zero_matches" or not (result.data or {}).get("results"):
        return ApprovalRecord(
            application_number=None,
            sponsor_name=None,
            submissions_count=0,
            review_docs=[],
            source_status=result.status,
            source_url=url,
        )

    record = result.data["results"][0]
    submissions = record.get("submissions", [])
    review_docs: list[ApplicationDoc] = []
    for submission in submissions:
        for doc in submission.get("application_docs", []):
            if doc.get("type") == "Review":
                review_docs.append(
                    ApplicationDoc(doc_type="Review", url=doc["url"], date=doc.get("date"))
                )

    return ApprovalRecord(
        application_number=record.get("application_number"),
        sponsor_name=record.get("sponsor_name"),
        submissions_count=len(submissions),
        review_docs=review_docs,
        source_status=result.status,
        source_url=url,
    )


def _derive_pdf_base(toc_url: str) -> str:
    """Strip the "TOC.html" suffix from a Review doc's TOC URL.

    Verified 2026-09-11: the TOC page's embedded JS hardcodes `pdfBaseName`
    to exactly this string, so it does not need to be scraped from the page.
    """
    if toc_url.endswith("TOC.html"):
        return toc_url[: -len("TOC.html")]
    # Some older records point straight at a document instead of a TOC page.
    return toc_url.rsplit(".", 1)[0]


def _pdf_cache_path(pdf_url: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    digest = hashlib.sha256(pdf_url.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"review_{digest}.pdf")


def fetch_review_document(
    toc_or_doc_url: str, application_number: Optional[str] = None
) -> ReviewDocument:
    """Hop 2: resolve a Review application_doc's TOC URL to real review text."""
    base = _derive_pdf_base(toc_or_doc_url)

    for suffix in REVIEW_DOC_SUFFIXES:
        pdf_url = f"{base}{suffix}"
        cache_path = _pdf_cache_path(pdf_url)

        if os.path.exists(cache_path):
            reader = PdfReader(cache_path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return ReviewDocument(
                application_number=application_number,
                pdf_url=pdf_url,
                suffix_used=suffix,
                page_count=len(pages),
                text_by_page=pages,
                fetch_status="cached",
            )

        last_exc: Optional[Exception] = None
        for attempt in range(1, OPENFDA_MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    pdf_url,
                    headers={"User-Agent": ACCESSDATA_USER_AGENT},
                    timeout=OPENFDA_TIMEOUT_SECONDS * 2,  # review PDFs run multi-MB
                )
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(OPENFDA_RETRY_BACKOFF_SECONDS * attempt)
                continue

            if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith(
                "application/pdf"
            ):
                with open(cache_path, "wb") as fh:
                    fh.write(resp.content)
                reader = PdfReader(cache_path)
                pages = [p.extract_text() or "" for p in reader.pages]
                return ReviewDocument(
                    application_number=application_number,
                    pdf_url=pdf_url,
                    suffix_used=suffix,
                    page_count=len(pages),
                    text_by_page=pages,
                    fetch_status="live",
                )
            # 404 (this suffix doesn't exist for this application) -- try
            # the next suffix rather than retrying the same one.
            break

    # Every candidate suffix 404'd or failed -- honest empty state, not a crash.
    return ReviewDocument(
        application_number=application_number,
        pdf_url=base,
        suffix_used="",
        page_count=0,
        text_by_page=[],
        fetch_status="not_found",
    )


@dataclass
class DrugEvidence:
    label: LabelRecord
    approval: ApprovalRecord
    review_document: Optional[ReviewDocument] = None


def gather_evidence_for_drugs(drug_names: list[str]) -> list[DrugEvidence]:
    """Run the full two-hop retrieval for a resolved list of drug names.

    Every drug gets hop 1 (label + approval history, cheap). Only the first
    MAX_DRUGS_FOR_DEEP_REVIEW get hop 2 (review PDF fetch+extract, expensive)
    -- keeps a live demo run fast while still showing the two-hop flow.
    """
    evidence: list[DrugEvidence] = []
    for i, name in enumerate(drug_names):
        label = get_label(name)
        approval = get_approval_history(name)
        review_doc = None
        if i < MAX_DRUGS_FOR_DEEP_REVIEW and approval.review_docs:
            review_doc = fetch_review_document(
                approval.review_docs[0].url, approval.application_number
            )
        evidence.append(DrugEvidence(label=label, approval=approval, review_document=review_doc))
    return evidence
