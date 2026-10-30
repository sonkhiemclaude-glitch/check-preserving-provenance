"""Common interface for release mechanisms compared in the evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..data.schemas import Graph
from ..data.spec import Spec


class ReleaseMechanism(ABC):
    """A function from a raw provenance graph to a release.

    ``kind`` distinguishes what a class is, so a dummy or a stub is never
    mistaken for a real method:
      'operator'  the release operators (provpub.methods)
      'baseline'  fully specified baseline adaptations
      'adapter-stub'    wrapper for an external system that is not available
      'dummy'           test double, never a result
    """

    kind: ClassVar[str] = "baseline"
    name: ClassVar[str] = "abstract"

    @abstractmethod
    def release(self, g: Graph, spec: Spec, key: bytes) -> Graph:
        """Return the released graph. Must keep meta['run_map_inv'] if runs are relabelled."""
