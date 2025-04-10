from __future__ import annotations

from ._core import __doc__, __version__, permanent, registrations

try:
    from sooki import gpu_ops
except ImportError:
    pass

__all__ = ["__doc__", "__version__"]
