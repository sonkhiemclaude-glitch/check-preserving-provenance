"""ADAPTER STUBS for the original systems behind the specified baselines.

The evaluation does not run these systems; it uses simplified adaptations
(baselines/structural.py) and cites the originals. These
wrappers fix the expected interface so that an implementation can be added
later without changing the experiment code. Each raises NotImplementedError.

Expected input : provpub Graph (raw) + Spec + release key.
Expected output: provpub Graph whose runs can be mapped back to raw runs via
                 meta['run_map_inv'] (dict released_run -> raw_run) if runs
                 are relabelled.
"""
from __future__ import annotations

from ..data.schemas import Graph
from ..data.spec import Spec
from .base import ReleaseMechanism


class _Stub(ReleaseMechanism):
    kind = "adapter-stub"
    reference = ""
    todo = ""

    def release(self, g: Graph, spec: Spec, key: bytes) -> Graph:
        raise NotImplementedError(f"{self.name}: adapter stub for {self.reference}. {self.todo}")


class DavidsonModulePrivacy(_Stub):
    """Minimum-cost attribute hiding achieving Gamma-privacy (Davidson et al.,
    PODS 2011). Requires the module input-output relations and attribute costs,
    which agentic traces do not provide; the module-hiding baseline hides
    at least as much."""

    name = "davidson_gamma_privacy"
    reference = "Davidson et al., PODS 2011 (Gamma-privacy)"
    todo = "Needs module relations R(I,O), attribute costs and Gamma."


class ProvAbstraction(_Stub):
    """Validity-preserving PROV abstraction (Missier et al., FGCS 2020) or ZOOM
    user views (Biton et al., ICDE 2008)."""

    name = "prov_abstraction"
    reference = "Missier et al., FGCS 111 (2020); Biton et al., ICDE 2008"
    todo = "Needs the published algorithm/tool and a policy of sensitive nodes."


class DPEventLogRelease(_Stub):
    """Differentially private event-log release (e.g. Elkoumy et al. 2023,
    PRIPEL). Operates on event logs (case, activity, timestamp), not PROV graphs."""

    name = "dp_event_log_release"
    reference = "Elkoumy et al. 2023; Fahrenkrog-Petersen et al. 2020"
    todo = "Needs the original implementation and a PROV-to-event-log mapping."
