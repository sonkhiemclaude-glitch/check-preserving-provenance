"""provpub: privacy-preserving release of agentic pipeline provenance with
verdict-equivalence guarantees.

Subpackages
-----------
data        graph schema, declared specification, synthetic workloads W1-W3,
            PROV-JSON / OpenTelemetry I/O, adapters for real traces
methods     operators VG, TC, BS, DPN and their composition
checks      verifier checks and the verdict-equivalence comparison
baselines   suppress all, module hiding, structural abstraction
attacks     inference attacks A1-A5
metrics     evaluation metrics and intervals
experiments E1-E4 runners, tables and figures
"""
__version__ = "0.2.0"

from .checks.checks import align, compare, run_checks
from .data.schemas import FAIL, PASS, SUPP, UNDET, Graph
from .data.spec import Constraint, Spec, ThresholdRule, TimedRule, VocabRule
from .methods.operators import KeyRing, bs, dpn, sanitise, tc, vg

__all__ = ["Graph", "PASS", "FAIL", "UNDET", "SUPP", "Spec", "Constraint", "VocabRule", "ThresholdRule", "TimedRule",
           "KeyRing", "vg", "tc", "bs", "dpn", "sanitise", "run_checks", "align", "compare", "__version__"]
