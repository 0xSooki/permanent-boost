from __future__ import annotations

from ._core import __doc__, __version__, registrations

try:
    from sooki import gpu_ops
except ImportError:
    pass

from .permanent import perm

__all__ = ["__doc__", "__version__", "perm"]
