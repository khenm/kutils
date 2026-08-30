"""Provider loaders: construct a backend `(model, processor)` from a spec.

Backends are imported lazily inside each loader, so optional packages are
only needed when a provider is actually used — kutils never depends on them
and adds no extras for them. A paper that wants an optional backend adds
the package to its own pyproject.
"""

from kutils.models.providers.huggingface import load_huggingface
from kutils.models.providers.local import load_local
from kutils.models.providers.open_clip import load_open_clip
from kutils.models.providers.timm import load_timm
from kutils.models.providers.torchvision import load_torchvision

__all__ = [
    "load_huggingface",
    "load_local",
    "load_open_clip",
    "load_timm",
    "load_torchvision",
]
