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
