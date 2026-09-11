"""Shared configuration constants for the First Pass backend.

Region is pinned to us-west-2: the Anthropic model use-case form has not been
submitted for this account in us-east-1, so direct Bedrock invokes fail there
with ResourceNotFoundException. us-west-2 and us-east-2 both work; us-west-2
was chosen since AgentCore Runtime deployment targets it too.
"""

import os

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

# Direct model ID (no "us." prefix) fails: Bedrock requires the cross-region
# inference profile ID for this model.
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

OPENFDA_BASE_URL = "https://api.fda.gov"
OPENFDA_LABEL_ENDPOINT = f"{OPENFDA_BASE_URL}/drug/label.json"
OPENFDA_DRUGSFDA_ENDPOINT = f"{OPENFDA_BASE_URL}/drug/drugsfda.json"

# openFDA rate limit without an API key: 240 requests/min, 1000/day.
OPENFDA_TIMEOUT_SECONDS = 15
OPENFDA_MAX_RETRIES = 3
OPENFDA_RETRY_BACKOFF_SECONDS = 1.5

CACHE_DIR = os.environ.get("FIRST_PASS_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "cache"))

# accessdata.fda.gov returns an Akamai "apology"/rate-limit page (HTTP 200,
# looks like real HTML) to requests without a browser-like User-Agent --
# verified 2026-09-11. A default `requests` UA gets blocked; this does not.
ACCESSDATA_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Review-document PDF suffixes on accessdata.fda.gov, most nonclinical-
# relevant / most likely to exist first. Modern applications (~2015+)
# consolidate everything into MultidisciplineR.pdf; older ones split into
# PharmR.pdf (non-clinical review) and MedR.pdf (clinical review) instead --
# verified: NDA 214801 (2022) has MultidisciplineR.pdf, not PharmR.pdf.
REVIEW_DOC_SUFFIXES = [
    "MultidisciplineR.pdf",
    "PharmR.pdf",
    "IntegratedR.pdf",
    "MedR.pdf",
]

# Deep review-PDF fetch is expensive (multi-MB, 100+ pages) and openFDA's
# unauthenticated rate limit is 240 req/min -- cap how many resolved drugs
# per target get the full two-hop treatment. The rest still get their
# structured label (hop 1 only).
MAX_DRUGS_FOR_DEEP_REVIEW = 5

# AgentCore built-in evaluators used for F15.
EVALUATOR_FAITHFULNESS = "Builtin.Faithfulness"
EVALUATOR_CORRECTNESS = "Builtin.Correctness"

REPORT_SECTIONS = [
    "Target Profile & Biological Function",
    "Genetic & Knockout Precedent",
    "Class Effects & Known Target Toxicities",
    "Regulatory & Clinical Precedents",
    "Adversity & Evidence Weight Analysis",
    "Nonclinical Safety Recommendations & CTD-M2.4-Bridge",
    "References & Regulatory Sources",
]

INTERPRETATION_STANCES = ["adverse", "non_adverse", "adaptive", "artifact"]
