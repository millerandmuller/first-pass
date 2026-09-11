"""Runs on the first `from backend... import ...` anywhere in the process.

Strands caches its Tracer as a true process-wide singleton on first use
(strands/telemetry/tracer.py: `_tracer_instance`, created once from whatever
`opentelemetry.trace.get_tracer_provider()` returns at that moment). F15's
trace capture only works if our capturing TracerProvider is installed as the
OTel global default *before* that first use -- so it happens here, at
package import time, rather than lazily inside evaluations.py, where it
would already be too late once any other module has constructed an Agent.
See backend/evaluations.py's module docstring for the full story.
"""

from backend.evaluations import ensure_trace_capture_installed

ensure_trace_capture_installed()
